#ifndef CUPY_INLINE_COMPILE
#include "f-attn.cuh"
#else
#define FLT_MAX 3.402823466e+38F
#endif

template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int HEAD_DIM>
struct shm_t{
    float q_buff[BLOCK_SIZE_R*HEAD_DIM];
    float k_buff[BLOCK_SIZE_C*HEAD_DIM];
    float v_buff[BLOCK_SIZE_C*HEAD_DIM];
    float s_buff[BLOCK_SIZE_R*BLOCK_SIZE_C];
    float o_buff[BLOCK_SIZE_R*HEAD_DIM];
    float logsumexp[BLOCK_SIZE_R];
    float maxes[BLOCK_SIZE_R];
};

template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    float* maxes,
    int batch_size,
    int num_heads,
    int seq_len
) {
    extern __shared__ unsigned char shm[];
    float *q_buff = (float*)shm;
    float *k_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM);
    float *v_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM);
    float *s_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM*2);
    float *o_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C);
    float *logsumexp_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C);
    float *maxes_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C + sizeof(float)*BLOCK_SIZE_R);
    
    int tid = threadIdx.x;
    int T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R; //q,o
    int T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C; //k,v
    
    const int BATCH_IDX = blockIdx.x / num_heads;
    const int HEAD_IDX = blockIdx.x % num_heads;
    
    const int base_offset = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                           HEAD_IDX * (seq_len * HEAD_DIM);
    
    for(int j = 0; j < T_c; ++j)
    {
        // Load k,v block to shared memory
        int kv_block_start = j * BLOCK_SIZE_C;
        
        for(int x = tid; x < BLOCK_SIZE_C * HEAD_DIM; x += blockDim.x)
        {
            int local_row = x / HEAD_DIM;  // Which row within the block [0, BLOCK_SIZE_C)
            int local_col = x % HEAD_DIM;  // Which head dimension [0, head_dim)
            
            int global_seq_idx = kv_block_start + local_row;
            
            // Check bounds
            if (global_seq_idx < seq_len) {
                int idx = base_offset + global_seq_idx * HEAD_DIM + local_col;
                k_buff[x] = key[idx];
                v_buff[x] = value[idx];
            } else {
                k_buff[x] = 0.0f;
                v_buff[x] = 0.0f;
            }
        }
        __syncthreads();
        
        // For i = 0...T_r
        for(int i = 0; i < T_r; ++i)
        {
            // Load q block and o block to shared memory
            int q_block_start = i * BLOCK_SIZE_R;
            
            for(int x = tid; x < BLOCK_SIZE_R * HEAD_DIM; x += blockDim.x)
            {
                int local_row = x / HEAD_DIM;
                int local_col = x % HEAD_DIM;
                
                int global_seq_idx = q_block_start + local_row;
                
                if (global_seq_idx < seq_len) {
                    int idx = base_offset + global_seq_idx * HEAD_DIM + local_col;
                    q_buff[x] = query[idx];
                    o_buff[x] = output[idx];
                } else {
                    q_buff[x] = 0.0f;
                    o_buff[x] = 0.0f;
                }
            }
            
            // logsumexp and maxes load
            for(int x = tid; x < BLOCK_SIZE_R; x += blockDim.x)
            {
                int local_row = x;
                int global_seq_idx = q_block_start + local_row;
                
                if (global_seq_idx < seq_len) {
                    int idx = BATCH_IDX * (num_heads * seq_len) + HEAD_IDX * seq_len + global_seq_idx;
                    logsumexp_shm[local_row] = logsumexp[idx];
                    maxes_shm[local_row] = maxes[idx];
                } else {
                    logsumexp_shm[local_row] = 0.0f;
                    maxes_shm[local_row] = -FLT_MAX;
                }
            }
            __syncthreads();

            // float s_ij[BLOCK_SIZE_R*BLOCK_SIZE_C]; // register for s_ij
            float m_prev = 0.0f;  // max values for tid i, where i <= BLOCK_SIZE_R
            float l_prev = 0.0f;  // sum of exp for tid i, where i <= BLOCK_SIZE_R
            // float O_local[BLOCK_SIZE_R*HEAD_DIM];  // local output
            // compute attention for this (i,j) tile pair
            for(int x=tid; x < BLOCK_SIZE_R * BLOCK_SIZE_C; x += blockDim.x)
            {
                int local_row = x / BLOCK_SIZE_C;
                int local_col = x % BLOCK_SIZE_C;
                
                float dot_product = 0.0f;
                for(int d = 0; d < HEAD_DIM; ++d)
                {
                    // S_ij = Q_i @ (K_j)^T
                    dot_product += q_buff[local_row * HEAD_DIM + d] * k_buff[local_col * HEAD_DIM + d]; // not coalesced mem access
                }
                dot_product /= sqrtf((float)HEAD_DIM); // scale

                s_buff[local_row * BLOCK_SIZE_C + local_col] = dot_product;
            }

            __syncthreads();
            // max update - reduction 
            for(int x = tid; x < BLOCK_SIZE_R; x += blockDim.x)
            {
                int global_row_idx = i * BLOCK_SIZE_R + x;
                if (global_row_idx >= seq_len) continue;
                
                float row_max = -FLT_MAX;
                for(int col = 0; col < BLOCK_SIZE_C; ++col)
                {
                    int global_col_idx = j * BLOCK_SIZE_C + col;
                    if (global_col_idx >= seq_len) continue;
                    row_max = fmaxf(row_max, s_buff[x * BLOCK_SIZE_C + col]);
                }

                float row_l = 0.0f;
                for(int col = 0; col < BLOCK_SIZE_C; ++col)
                {
                    int global_col_idx = j * BLOCK_SIZE_C + col;
                    if (global_col_idx >= seq_len) {
                        s_buff[x * BLOCK_SIZE_C + col] = 0.0f;
                        continue;
                    }
                    float val = s_buff[x * BLOCK_SIZE_C + col];
                    // expf__ faster then expf and is hardware op (cavecats: possible loss of precision)
                    float expval = expf(val - row_max);
                    row_l += expval;
                    s_buff[x * BLOCK_SIZE_C + col] = expval; // now s_buff have P values (local unnormalized softmax)
                } 

                m_prev = maxes_shm[x];
                maxes_shm[x] = fmaxf(maxes_shm[x], row_max); //mi new
                l_prev = logsumexp_shm[x];
                logsumexp_shm[x] = expf(m_prev-maxes_shm[x])*l_prev + expf(row_max-maxes_shm[x])*row_l;

                for(int d = 0; d < HEAD_DIM; ++d)
                {
                    // o_buff update
                    float o_curr = o_buff[x * HEAD_DIM + d];
                    o_curr = expf(m_prev - maxes_shm[x])*o_curr;
                    float s_sum = 0.0f;
                    for(int col = 0; col < BLOCK_SIZE_C; ++col)
                    {
                        int global_col_idx = j * BLOCK_SIZE_C + col;
                        if (global_col_idx >= seq_len) continue;
                        s_sum += s_buff[x * BLOCK_SIZE_C + col] * v_buff[col * HEAD_DIM + d];
                    }
                    o_curr = l_prev * o_curr + expf(row_max-maxes_shm[x])*s_sum;
                    o_buff[x * HEAD_DIM + d] = o_curr/logsumexp_shm[x];
                    
                }

            }

            for(int row = tid; row < BLOCK_SIZE_R; row += blockDim.x) {
                int global_seq_idx = q_block_start + row;
                if (global_seq_idx < seq_len) {
                    
                    for(int d = 0; d < HEAD_DIM; ++d) {
                        int idx = base_offset + global_seq_idx * HEAD_DIM + d;
                        output[idx] = o_buff[row * HEAD_DIM + d];
                    }
                    
                    int stat_idx = BATCH_IDX * (num_heads * seq_len) + 
                                HEAD_IDX * seq_len + global_seq_idx;
                    maxes[stat_idx] = maxes_shm[row];
                    logsumexp[stat_idx] = logsumexp_shm[row];
                }
            }
            __syncthreads();
        }
    }
}

#ifndef CUPY_INLINE_COMPILE
template<int head_dim>
void host_flash_attention_forward(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_O,
    float* h_logsumexp,
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
    const float *d_Q, *d_K, *d_V;
    float *d_O;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMemset(d_O, 0, qkv_size * sizeof(float));
    float *d_logsumexp, *d_maxes;
    cudaMalloc(&d_logsumexp, batch_size * num_heads * seq_len * sizeof(float));
    cudaMalloc(&d_maxes, batch_size * num_heads * seq_len * sizeof(float));
    cudaMemset(d_logsumexp, 0, batch_size * num_heads * seq_len * sizeof(float));

    float* h_maxes_init = new float[batch_size * num_heads * seq_len];
    for(int i = 0; i < batch_size * num_heads * seq_len; i++) {
        h_maxes_init[i] = -FLT_MAX;
    }
    cudaMemcpy(d_maxes, h_maxes_init, batch_size * num_heads * seq_len * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_maxes_init;
    
    cudaMemcpy((void *)d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Running Flash Attention...\n");
    printf("Batch: %d, Heads: %d, SeqLen: %d, HeadDim: %d\n", 
           batch_size, num_heads, seq_len, head_dim);
    
    const size_t num_threads_per_block = 128;
    const int HEAD_DIM = head_dim;
    const int BLOCK_SIZE_C = 32;
    const int BLOCK_SIZE_R = 32;
    const int shared_mem_size = sizeof(shm_t<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>);
    tm->Start();
    flash_attention_forward_kernel<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>
                                  <<<batch_size*num_heads, num_threads_per_block, shared_mem_size>>>(
                                    d_Q, d_K, d_V, d_O, d_logsumexp, d_maxes,
                                    batch_size, num_heads, seq_len
                                  );
    tm->Stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logsumexp, d_logsumexp, batch_size * num_heads * seq_len * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree((void *)d_Q));
    CUDA_CHECK(cudaFree((void *)d_K));
    CUDA_CHECK(cudaFree((void *)d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_logsumexp));
    CUDA_CHECK(cudaFree(d_maxes));
}
using FA1ForwardFunc = void(const float*, const float*, const float*, float*, float*, int, int, int, TimerManager*);
template FA1ForwardFunc host_flash_attention_forward<32>;
template FA1ForwardFunc host_flash_attention_forward<64>;
#else
extern "C" __global__
void flash_attention_forward_kernel_wrapper(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    float *maxes,
    int batch_size,
    int num_heads,
    int seq_len
) {
    flash_attention_forward_kernel<32, 32, 64>(
        query, key, value, output, logsumexp, maxes,
        batch_size, num_heads, seq_len
    );
}
#endif
