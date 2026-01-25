#ifndef CUPY_INLINE_COMPILE
#include "f-attn2.cuh"
#else
#define FLT_MAX 3.402823466e+38F
#include <cuda_fp16.h>
#endif

__host__ __device__ __forceinline__ unsigned int next_power_of_2(unsigned int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// shared memory layout for flash attention 2 backward pass (fp16)
template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int HEAD_DIM>
struct shm_t{
    __half q_buff[BLOCK_SIZE_R*HEAD_DIM]; // q tile, then reused for d_o tile
    __half k_buff[BLOCK_SIZE_C*HEAD_DIM];
    __half v_buff[BLOCK_SIZE_C*HEAD_DIM]; // V transposed for dP @ V^T

    __half d_k_buff[BLOCK_SIZE_C*HEAD_DIM]; // gradient accumulator for K
    __half d_v_buff[BLOCK_SIZE_C*HEAD_DIM]; // gradient accumulator for V

    __half logsumexp[BLOCK_SIZE_R]; // logsumexp_i, then reused for D_i

    __half p_buff[BLOCK_SIZE_R*BLOCK_SIZE_C]; // P_ij (recomputed), then dS_ij
};

template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int HEAD_DIM>
__global__ void flash_attention2_backward_kernel_fp16(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ output,
    const float* __restrict__ d_output,
    const float* __restrict__ logsumexp,
    const float* __restrict__ d,
    float* __restrict__ d_query,
    float* __restrict__ d_key,
    float* __restrict__ d_value,
    int batch_size,
    int num_heads,
    int seq_len
) {
    extern __shared__ unsigned char shm[];
    __half *q_buff = (__half*)shm;
    __half *k_buff = (__half*)(shm + sizeof(__half)*BLOCK_SIZE_R*HEAD_DIM);
    __half *v_buff = (__half*)(shm + sizeof(__half)*BLOCK_SIZE_R*HEAD_DIM + sizeof(__half)*BLOCK_SIZE_C*HEAD_DIM);
    __half *d_k_buff = (__half*)(shm + sizeof(__half)*BLOCK_SIZE_R*HEAD_DIM + sizeof(__half)*BLOCK_SIZE_C*HEAD_DIM*2);
    __half *d_v_buff = (__half*)(shm + sizeof(__half)*BLOCK_SIZE_R*HEAD_DIM + sizeof(__half)*BLOCK_SIZE_C*HEAD_DIM*3);
    __half *logsumexp_d_shm = (__half*)(shm + sizeof(__half)*BLOCK_SIZE_R*HEAD_DIM + sizeof(__half)*BLOCK_SIZE_C*HEAD_DIM*4);
    __half *p_buff = (__half*)(shm + sizeof(__half)*BLOCK_SIZE_R*HEAD_DIM + sizeof(__half)*BLOCK_SIZE_C*HEAD_DIM*4 + sizeof(__half)*BLOCK_SIZE_R);
    __half *d_o_buff = q_buff;

    unsigned int tid = threadIdx.x;
    unsigned int T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
    unsigned int T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C;

    unsigned int BATCH_IDX = blockIdx.x / (num_heads * T_c);
    unsigned int HEAD_IDX = (blockIdx.x / T_c) % num_heads;
    unsigned int KV_TILE_IDX = blockIdx.x % T_c;

    int WARP_ID = tid / 32;
    int LANE_ID = tid % 32;
    int warps_per_block = blockDim.x / 32;
    int rows_per_warp = (BLOCK_SIZE_R + warps_per_block - 1) / warps_per_block;
    int warp_start_row = WARP_ID * rows_per_warp;
    int warp_end_row = min(warp_start_row + rows_per_warp, BLOCK_SIZE_R);

    const float sqrt_head_dim = sqrtf((float)HEAD_DIM);

    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                                HEAD_IDX * (seq_len * HEAD_DIM) + 
                                KV_TILE_IDX * BLOCK_SIZE_C * HEAD_DIM;
    __half2 *k_buff_h2 = reinterpret_cast<__half2 *>(k_buff);
    __half2 *d_k_buff_h2 = reinterpret_cast<__half2 *>(d_k_buff);
    __half2 *d_v_buff_h2 = reinterpret_cast<__half2 *>(d_v_buff);
    // load K,V tile into shm (V is transposed for later dP @ V^T computation)
    #pragma unroll
    for (int x=tid; x < BLOCK_SIZE_C * HEAD_DIM / 2; x += blockDim.x)
    {
        int local_row = (x*2) / HEAD_DIM;  // which row within the block [0, BLOCK_SIZE_C)
        int local_col = (x*2) % HEAD_DIM;  // which head dimension [0, head_dim)
        
        int global_seq_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_row;
        
        // check bounds
        if (global_seq_idx < seq_len && local_col + 1 < HEAD_DIM) {
            int idx = base_hbm_offset + local_row * HEAD_DIM + local_col;
            float k0 = __ldg(&key[idx]);
            float k1 = __ldg(&key[idx + 1]);
            float v0 = __ldg(&value[idx]);
            float v1 = __ldg(&value[idx + 1]);
            k_buff_h2[x] = make_half2(__float2half(k0), __float2half(k1));
            v_buff[(local_col + 0) * BLOCK_SIZE_C + local_row] = __float2half(v0);
            v_buff[(local_col + 1) * BLOCK_SIZE_C + local_row] = __float2half(v1);
        } else {
            v_buff[(local_col + 0) * BLOCK_SIZE_C + local_row] = __float2half(0.0f);                                                                       
            v_buff[(local_col + 1) * BLOCK_SIZE_C + local_row] = __float2half(0.0f); 
            k_buff_h2[x] = make_half2(__float2half(0.0f), __float2half(0.0f));
        }

        // also initialize the derivative buffers to zero
        d_k_buff_h2[x] = make_half2(__float2half(0.0f), __float2half(0.0f));
        d_v_buff_h2[x] = make_half2(__float2half(0.0f), __float2half(0.0f));
    }

    __syncthreads();

    const int q_hbm_base = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                            HEAD_IDX * (seq_len * HEAD_DIM);
    __half2 *q_buff_h2 = reinterpret_cast<__half2 *>(q_buff);
    // iterate over q tiles
    for (int i = 0; i < T_r; ++i)
    {
        // Load q, d_o, logsumexp into shm
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_R * HEAD_DIM / 2; x += blockDim.x)
        {
            int local_row = (x*2) / HEAD_DIM;  // Which row within the block [0, BLOCK_SIZE_R)
            int local_col = (x*2) % HEAD_DIM;  // Which head dimension [0, head_dim)
            
            int global_seq_idx = i * BLOCK_SIZE_R + local_row;
            
            if (global_seq_idx < seq_len && local_col + 1 < HEAD_DIM) {
                int q_idx = q_hbm_base +
                            global_seq_idx * HEAD_DIM + local_col;
                float q0 = __ldg(&query[q_idx]);
                float q1 = __ldg(&query[q_idx + 1]);
                q_buff_h2[x] = make_half2(__float2half(q0), __float2half(q1));
            } else {
                q_buff_h2[x] = make_half2(__float2half(0.0f), __float2half(0.0f));
            }
            if (local_col == 0) {
                logsumexp_d_shm[local_row] = __float2half(logsumexp[BATCH_IDX * num_heads * seq_len + 
                                                        HEAD_IDX * seq_len + 
                                                        global_seq_idx]);
            }
        }
        __syncthreads();

        // recompute S_ij = Q @ K^T / sqrt(d), then P_ij = softmax(S_ij)
        for (int row = warp_start_row; row < warp_end_row; ++row)
        {
            int global_q_idx = i * BLOCK_SIZE_R + row;
            if (global_q_idx >= seq_len) continue;

            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + col;
                if (global_kv_idx >= seq_len) {
                    continue;
                }

                float dot_product = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d+=2)
                {
                    __half2 q_val_h2 = *reinterpret_cast<__half2*>(&q_buff[row * HEAD_DIM + d]);
                    __half2 k_val_h2 = *reinterpret_cast<__half2*>(&k_buff[col * HEAD_DIM + d]);
                    float2 q_val_f2 = __half22float2(q_val_h2);
                    float2 k_val_f2 = __half22float2(k_val_h2);
                    dot_product = __fmaf_rn(q_val_f2.x, k_val_f2.x, dot_product);
                    dot_product = __fmaf_rn(q_val_f2.y, k_val_f2.y, dot_product);
                }
                dot_product /= sqrt_head_dim;
                // P_ij = exp(S_ij - logsumexp) = softmax(S_ij)
                dot_product = __expf(dot_product - __half2float(logsumexp_d_shm[row]));
                p_buff[row * BLOCK_SIZE_C + col] = __float2half(dot_product);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_R * HEAD_DIM / 2; x += blockDim.x)
        {
            int local_row = (x*2) / HEAD_DIM;  // Which row within the block [0, BLOCK_SIZE_R)
            int local_col = (x*2) % HEAD_DIM;  // Which head dimension [0, head_dim)
            
            int global_seq_idx = i * BLOCK_SIZE_R + local_row;
            
            if (global_seq_idx < seq_len && local_col + 1 < HEAD_DIM) {
                int o_idx = q_hbm_base +
                            global_seq_idx * HEAD_DIM + local_col;
                float do0 = __ldg(&d_output[o_idx]);
                float do1 = __ldg(&d_output[o_idx + 1]);
                q_buff_h2[x] = make_half2(__float2half(do0), __float2half(do1));
            }
            else
            {
                q_buff_h2[x] = make_half2(__float2half(0.0f), __float2half(0.0f));
            }

            // load D_i = rowsum(dO * O) for dS computation
            if (local_col == 0) {
                logsumexp_d_shm[local_row] = __float2half(d[BATCH_IDX * num_heads * seq_len +
                                                        HEAD_IDX * seq_len +
                                                        global_seq_idx]);
            }
        }
        __syncthreads();

        // dV_j = sum_i(P_ij^T @ dO_i)
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_C * HEAD_DIM; x += blockDim.x)
        {
            int local_c = x / HEAD_DIM;  // Which row within the block [0, BLOCK_SIZE_C)
            int local_h = x % HEAD_DIM;  // Which head dimension [0, head_dim)

            float d_v_sum = 0.0f;
            #pragma unroll
            for (int r=0; r < BLOCK_SIZE_R; ++r)
            {
                int global_q_idx = i * BLOCK_SIZE_R + r;
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_c;

                if (global_q_idx < seq_len && global_kv_idx < seq_len) {
                    float p_ij = __half2float(p_buff[r * BLOCK_SIZE_C + local_c]);
                    d_v_sum += p_ij * __half2float(d_o_buff[r * HEAD_DIM + local_h]);
                }
            }

            d_v_buff[local_c * HEAD_DIM + local_h] += __float2half(d_v_sum);
        }
        __syncthreads();

        // dS_ij = P_ij * (dP_ij - D_i) / sqrt(d), where dP_ij = dO @ V^T
        for (int row = warp_start_row; row < warp_end_row; ++row)
        {
            int global_q_idx = i * BLOCK_SIZE_R + row;
            if (global_q_idx >= seq_len) continue;

            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + col;
                if (global_kv_idx >= seq_len) {
                    continue;
                }

                // dP_ij = dO @ V^T
                float d_p_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    d_p_ij += __half2float(d_o_buff[row * HEAD_DIM + d]) * __half2float(v_buff[d * BLOCK_SIZE_C + col]);
                }
                // dS_ij = P_ij * (dP_ij - D_i) / sqrt(d)
                p_buff[row * BLOCK_SIZE_C + col] = __float2half((d_p_ij - __half2float(logsumexp_d_shm[row])) * __half2float(p_buff[row * BLOCK_SIZE_C + col]) / sqrt_head_dim);
            }
        }
        __syncthreads();

        // dQ_i = sum_j(dS_ij @ K_j) - accumulated via atomics across K,V tiles
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_R * HEAD_DIM; x += blockDim.x)
        {
            int local_r = x / HEAD_DIM;
            int local_h = x % HEAD_DIM;

            int global_q_idx = i * BLOCK_SIZE_R + local_r;

            if (global_q_idx < seq_len) {
                int q_idx = q_hbm_base +  
                            global_q_idx * HEAD_DIM + local_h;
                q_buff[x] = __float2half(query[q_idx]);
            }
            else {
                q_buff[x] = __float2half(0.0f);
            }

            float d_s_k_sum = 0.0f;
            #pragma unroll
            for (int c=0; c < BLOCK_SIZE_C; ++c)
            {
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + c;
                if (global_q_idx < seq_len && global_kv_idx < seq_len) {
                    d_s_k_sum += __half2float(p_buff[local_r * BLOCK_SIZE_C + c]) * __half2float(k_buff[c * HEAD_DIM + local_h]);
                }
            }

            if (global_q_idx < seq_len) {
                atomicAdd(&d_query[q_hbm_base + global_q_idx * HEAD_DIM + local_h], d_s_k_sum);
            }
        }
        __syncthreads();

        // dK_j = sum_i(dS_ij^T @ Q_i) - accumulated in shared mem across Q tiles
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_C * HEAD_DIM; x += blockDim.x)
        {
            int local_c = x / HEAD_DIM;
            int local_h = x % HEAD_DIM;

            int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_c;
            float d_s_q_sum = 0.0f;
            #pragma unroll
            for (int r=0; r < BLOCK_SIZE_R; ++r)
            {
                int global_q_idx = i * BLOCK_SIZE_R + r;
                if (global_q_idx < seq_len && global_kv_idx < seq_len) {
                    d_s_q_sum += __half2float(p_buff[r * BLOCK_SIZE_C + local_c]) * __half2float(q_buff[r * HEAD_DIM + local_h]);
                }
            }

            d_k_buff[local_c * HEAD_DIM + local_h] += __float2half(d_s_q_sum);
        }
        __syncthreads();
    }

    // write accumulated dK, dV gradients to global memory
    #pragma unroll
    for(int x = tid; x < BLOCK_SIZE_C * HEAD_DIM; x += blockDim.x)
    {
        int local_c = x / HEAD_DIM;
        int local_h = x % HEAD_DIM;
        int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_c;
        if (global_kv_idx < seq_len) {
            int idx = base_hbm_offset + local_c * HEAD_DIM + local_h;
            d_key[idx] = __half2float(d_k_buff[local_c * HEAD_DIM + local_h]);
            d_value[idx] = __half2float(d_v_buff[local_c * HEAD_DIM + local_h]);
        }
    }
}

// D_i = rowsum(dO_i * O_i) - needed for softmax backward
template<int HEAD_DIM>
__global__ void D_computation_reduction_kernel(
    const float *d_output, 
    const float *output, 
    int batch_size, 
    int num_heads, 
    int seq_len, 
    float *D)
{
    extern __shared__ float shmm[];
    unsigned int tid = threadIdx.x;
    unsigned int BATCH_IDX = blockIdx.x / (num_heads * seq_len);
    unsigned int HEAD_IDX = (blockIdx.x / seq_len) % num_heads;
    unsigned int SEQ_IDX = blockIdx.x % seq_len;
    
    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                                HEAD_IDX * (seq_len * HEAD_DIM) + 
                                SEQ_IDX * HEAD_DIM;
    
    float thread_sum = 0.0f;
    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        thread_sum += d_output[base_hbm_offset + i] * output[base_hbm_offset + i];
    }
    
    shmm[tid] = thread_sum;
    __syncthreads();
    
    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmm[tid] += shmm[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        D[BATCH_IDX * num_heads * seq_len + HEAD_IDX * seq_len + SEQ_IDX] = shmm[0];
    }
}


#ifndef CUPY_INLINE_COMPILE
template<int head_dim>
void host_flash_attention2_backward_fp16(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    const float* h_O,
    const float* h_deriv_O,
    const float* h_logsumexp,
    float *h_deriv_Q,
    float *h_deriv_K,
    float *h_deriv_V,
    int batch_size,
    int seq_len,
    int num_heads,
    TimerManager* tm
) {
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    const float *d_Q, *d_K, *d_V, *d_O, *d_deriv_O;
    const float *d_logsumexp;
    float *d_D;
    float *d_deriv_Q, *d_deriv_K, *d_deriv_V;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_deriv_O, qkv_size * sizeof(float));
    cudaMalloc(&d_logsumexp, (size_t)(qkv_size/head_dim) * sizeof(float));
    cudaMalloc(&d_D, (size_t)(qkv_size/head_dim) * sizeof(float));
    cudaMalloc(&d_deriv_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_deriv_K, qkv_size * sizeof(float));
    cudaMalloc(&d_deriv_V, qkv_size * sizeof(float));
    
    cudaMemcpy((void *)d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_O, h_O, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_deriv_O, h_deriv_O, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_logsumexp, h_logsumexp, (size_t)(qkv_size/head_dim) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_deriv_Q, 0, qkv_size * sizeof(float));
    cudaMemset(d_deriv_K, 0, qkv_size * sizeof(float));
    cudaMemset(d_deriv_V, 0, qkv_size * sizeof(float));

    const int HEAD_DIM = head_dim;
    const int BLOCK_SIZE_C = 32;
    const int BLOCK_SIZE_R = 32;

    printf("Computing D_i for backward pass...\n");
    const size_t threads_per_block_di = next_power_of_2(head_dim);
    const size_t total_blocks_di = batch_size * num_heads * seq_len;
    const size_t shared_mem_size_di = sizeof(float) * threads_per_block_di;
    tm->Start();
    D_computation_reduction_kernel<HEAD_DIM><<<total_blocks_di, threads_per_block_di, shared_mem_size_di>>>(
        d_deriv_O, d_O, batch_size, num_heads, seq_len, d_D
    );
    tm->Stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Running Flash Attention Backward pass...\n");
    printf("Batch: %d, Heads: %d, SeqLen: %d, HeadDim: %d\n", 
           batch_size, num_heads, seq_len, head_dim);
    
    const size_t T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C;
    const size_t total_blocks = batch_size * num_heads * T_c;
    const size_t num_threads_per_block = 256;
    const size_t shared_mem_size = sizeof(shm_t<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>);
    tm->Start();
    flash_attention2_backward_kernel_fp16<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>
                                  <<<total_blocks, num_threads_per_block, shared_mem_size>>>(
                                    d_Q, d_K, d_V, d_O, d_deriv_O, d_logsumexp, d_D,
                                    d_deriv_Q, d_deriv_K, d_deriv_V,
                                    batch_size, num_heads, seq_len
                                  );
    tm->Stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_deriv_Q, d_deriv_Q, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_deriv_K, d_deriv_K, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_deriv_V, d_deriv_V, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree((void *)d_Q));
    CUDA_CHECK(cudaFree((void *)d_K));
    CUDA_CHECK(cudaFree((void *)d_V));
    CUDA_CHECK(cudaFree((void *)d_O));
    CUDA_CHECK(cudaFree((void *)d_deriv_O));
    CUDA_CHECK(cudaFree((void *)d_logsumexp));
    CUDA_CHECK(cudaFree(d_deriv_Q));
    CUDA_CHECK(cudaFree(d_deriv_K));
    CUDA_CHECK(cudaFree(d_deriv_V));
    CUDA_CHECK(cudaFree(d_D));
}
using FA2F32BCKWRDFunc = void(const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, int, int, int, TimerManager*);
template FA2F32BCKWRDFunc host_flash_attention2_backward_fp16<32>;
template FA2F32BCKWRDFunc host_flash_attention2_backward_fp16<64>;
#else
// wrapper for instantiating the kernel using cupy
extern "C" __global__
void flash_attention2_backward_kernel_wrapper(
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const float* d_output,
    const float* logsumexp,
    const float* d,
    float* d_query,
    float* d_key,
    float* d_value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    flash_attention2_backward_kernel_fp16<32, 32, 64>(
        query, key, value, output, d_output, logsumexp, d,
        d_query, d_key, d_value,
        batch_size, num_heads, seq_len
    );
}

extern "C" __global__
void D_computation_reduction_kernel_wrapper(
    const float* d_output,
    const float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float* d
) {
    D_computation_reduction_kernel<64>(
        d_output, output, batch_size, num_heads, seq_len, d
    );
}
#endif