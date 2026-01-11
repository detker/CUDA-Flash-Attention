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
    float maxes_prev[BLOCK_SIZE_R];
};

template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int HEAD_DIM>
__global__ void flash_attention2_forward_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
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
    float *maxes_prev_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C + sizeof(float)*2*BLOCK_SIZE_R);
    
    int tid = threadIdx.x; // thread id (each thread is assigned at least one element of q tile :>)
    const int T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R; //q,o
    const int T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C; //k,v
    
    const int BATCH_IDX = blockIdx.x / (num_heads*T_r);
    const int HEAD_IDX = (blockIdx.x / T_r) % num_heads;
    const int Q_TILE_IDX = blockIdx.x % T_r;

    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                                HEAD_IDX * (seq_len * HEAD_DIM) + 
                                Q_TILE_IDX * BLOCK_SIZE_R * HEAD_DIM;
    // we load q tile to SHM
    for(int x = tid; x < BLOCK_SIZE_R * HEAD_DIM; x += blockDim.x)
    {
        int local_row = x / HEAD_DIM;  // Which row within the block [0, BLOCK_SIZE_R)
        int local_col = x % HEAD_DIM;  // Which head dimension [0, head_dim)
        
        int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
        
        // check bounds
        if (global_seq_idx < seq_len) {
            int idx = base_hbm_offset + local_row * HEAD_DIM + local_col;
            q_buff[x] = query[idx];
        } else {
            q_buff[x] = 0.0f;
        }

        o_buff[x] = 0.0f; // initialize output buffer to zero
        if(local_col == 0)
        {
            logsumexp_shm[local_row] = 0.0f;
            maxes_shm[local_row] = -FLT_MAX;
        }
    }

    __syncthreads();

    const int base_hbm_offset_kv = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                                   HEAD_IDX * (seq_len * HEAD_DIM);
    // for j = 0 ... T_c-1, we loop over k,v tiles
    for(int j=0; j < T_c; ++j)
    {
        // load k,v tiles into SHM
        int kv_block_start = j * BLOCK_SIZE_C;
        for(int x = tid; x < BLOCK_SIZE_C * HEAD_DIM; x += blockDim.x)
        {
            int local_row = x / HEAD_DIM;  // Which row within the block [0, BLOCK_SIZE_C)
            int local_col = x % HEAD_DIM;  // Which head dimension [0, head_dim)
            
            int global_seq_idx = kv_block_start + local_row;
            
            if (global_seq_idx < seq_len) {
                int idx = base_hbm_offset_kv + global_seq_idx * HEAD_DIM + local_col;
                k_buff[x] = key[idx];
                v_buff[x] = value[idx];
            } else {
                k_buff[x] = 0.0f;
                v_buff[x] = 0.0f;
            }
        } 
        __syncthreads();

        // compute attention for this (i,j) tile pair
        for(int x=tid; x < BLOCK_SIZE_R * BLOCK_SIZE_C; x += blockDim.x)
        {
            int local_row = x / BLOCK_SIZE_C;
            int local_col = x % BLOCK_SIZE_C;
            
            // Check if this position is valid (not padding)
            int global_q_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
            int global_k_idx = kv_block_start + local_col;
            
            float dot_product;
            if (global_q_idx < seq_len && global_k_idx < seq_len) {
                dot_product = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    // S_ij = Q_i @ (K_j)^T
                    dot_product += q_buff[local_row * HEAD_DIM + d] * k_buff[local_col * HEAD_DIM + d]; // k_buff not coalesced mem access, surely bank conflicts
                }
                dot_product /= sqrtf((float)HEAD_DIM); // scale
            } else {
                // mask out padded positions with -inf so they become 0 after softmax
                dot_product = -FLT_MAX;
            }
            s_buff[local_row * BLOCK_SIZE_C + local_col] = dot_product;
        }
        __syncthreads();

        // max update
        for(int x = tid; x < BLOCK_SIZE_R; x += blockDim.x)
        {
            int global_q_idx = Q_TILE_IDX * BLOCK_SIZE_R + x;
            
            // only process valid query rows
            if (global_q_idx < seq_len) {
                float row_max = -FLT_MAX;
                for(int col = 0; col < BLOCK_SIZE_C; ++col)
                {
                    row_max = fmaxf(row_max, s_buff[x * BLOCK_SIZE_C + col]);
                }
                maxes_prev_shm[x] = maxes_shm[x];
                maxes_shm[x] = fmaxf(maxes_shm[x], row_max); // update max_i

                for (int col = 0; col < BLOCK_SIZE_C; ++col)
                {
                    float val = s_buff[x * BLOCK_SIZE_C + col];
                    // expf__ faster than expf and is hardware op (caveats: possible loss of precision)
                    float expval = expf(val - maxes_shm[x]);
                    s_buff[x * BLOCK_SIZE_C + col] = expval; // now s_buff have P values (local unnormalized softmax)
                }

                float row_sum = 0.0f;
                for(int col = 0; col < BLOCK_SIZE_C; ++col)
                {
                    row_sum += s_buff[x * BLOCK_SIZE_C + col];
                }
                // l_prev = logsumexp_shm[x];
                logsumexp_shm[x] = expf(maxes_prev_shm[x] - maxes_shm[x]) * logsumexp_shm[x] + row_sum;
            }
        }
        __syncthreads();

        for(int x = tid; x < BLOCK_SIZE_R * HEAD_DIM; x += blockDim.x)
        {
            int local_row = x / HEAD_DIM;
            int local_col = x % HEAD_DIM;
            int global_q_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;

            // only process valid query rows
            if (global_q_idx < seq_len) {
                float o_curr = o_buff[local_row * HEAD_DIM + local_col];
                o_curr = expf(maxes_prev_shm[local_row] - maxes_shm[local_row]) * o_curr;
                float s_sum = 0.0f;
                for(int col = 0; col < BLOCK_SIZE_C; ++col)
                {
                    s_sum += s_buff[local_row * BLOCK_SIZE_C + col] * v_buff[col * HEAD_DIM + local_col];
                }
                o_curr = o_curr + s_sum;
                o_buff[local_row * HEAD_DIM + local_col] = o_curr;
            }
        }
        __syncthreads();

    }

    for(int x = tid; x < BLOCK_SIZE_R * HEAD_DIM; x += blockDim.x)
    {
        int local_row = x / HEAD_DIM;
        int local_col = x % HEAD_DIM;
        int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
        if (global_seq_idx < seq_len) {
            int idx = base_hbm_offset + local_row * HEAD_DIM + local_col;
            output[idx] = o_buff[local_row * HEAD_DIM + local_col] / logsumexp_shm[local_row];

            // check correctness of that
            if(local_col == 0)
            {
                logsumexp[BATCH_IDX * num_heads * seq_len + 
                               HEAD_IDX * seq_len + 
                               global_seq_idx] = logsumexp_shm[local_row];
            }
        }
    }
}


#ifndef CUPY_INLINE_COMPILE
void host_flash_attention2_forward(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_O,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
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
    //utils
    float *d_logsumexp;
    cudaMalloc(&d_logsumexp, batch_size * num_heads * seq_len * sizeof(float));

    cudaMemcpy((void *)d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // run Flash Attention
    printf("Running Flash Attention...\n");
    printf("Batch: %d, Heads: %d, SeqLen: %d, HeadDim: %d\n", 
           batch_size, num_heads, seq_len, head_dim);
    
    const int HEAD_DIM = 64;
    const int BLOCK_SIZE_C = 32;
    const int BLOCK_SIZE_R = 32;
    const size_t T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
    const size_t total_blocks = batch_size * num_heads * T_r;
    const size_t num_threads_per_block = 256;
    // basically every block handles one tile of Q so we parallelize over batch_size, num_heads and over seq len dimension!!! as opposed to flash-attn1
    // schema of blocks:
    // [batch_0_head_0_qtile_0, batch_0_head_0_qtile_1, ..., batch_0_head_0_qtile_T_r, batch_0_head_1_qtile_0, ..., batch_0_head_1_qtile_T_r, ...
    // ..., batch_0_head_(num_heads-1)_qtile_0, ..., batch_0_head_(num_heads-1)_qtile_T_r, ... another batch etc etc]
    // each block handles ONE q tile, so every thread in block handles at least one element of that q tile (we have BLOCK_SIZE_R*HEAD_DIM elements in q tile e.g. 32*64=2048 elements, so with 128 threads each thread handles 16 elements of q tile)
    const int shared_mem_size = sizeof(shm_t<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>);
    tm->Start();
    flash_attention2_forward_kernel<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>
                                  <<<total_blocks, num_threads_per_block, shared_mem_size>>>(
                                    d_Q, d_K, d_V, d_O, d_logsumexp,
                                    batch_size, num_heads, seq_len
                                  );
    tm->Stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree((void *)d_Q));
    CUDA_CHECK(cudaFree((void *)d_K));
    CUDA_CHECK(cudaFree((void *)d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_logsumexp));
}
#else
// wrapper for instantiating the kernel using cupy
extern "C" __global__
void flash_attention2_forward_kernel_wrapper(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    int batch_size,
    int num_heads,
    int seq_len
) {
    flash_attention2_forward_kernel<32, 32, 64>(
        query, key, value, output, logsumexp,
        batch_size, num_heads, seq_len
    );
}
#endif