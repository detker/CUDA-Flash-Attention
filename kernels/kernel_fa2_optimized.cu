#ifndef CUPY_INLINE_COMPILE
#include "f-attn2.cuh"
#else
#define FLT_MAX 3.402823466e+38F
#endif

template<int BM, int BN, int HEAD_DIM>
struct shm_t{
    float q_buff[BM*HEAD_DIM];
    float kv_buff[BN*HEAD_DIM];
    float s_buff[BM*BN];
    float o_buff[BM*HEAD_DIM];
    float logsumexp[BM];
    float maxes[BM];
    float exp_norm_coeffs[BM];
};

template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int HEAD_DIM, int BK, int TM, int TN>
__global__ void flash_attention2_forward_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    float* __restrict__ logsumexp,
    int batch_size,
    int num_heads,
    int seq_len
) {
    extern __shared__ unsigned char shm[];
    float *q_buff = (float*)shm;
    float *kv_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM);
    float *s_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM);
    float *o_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C);
    float *logsumexp_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C);
    float *maxes_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C + sizeof(float)*BLOCK_SIZE_R);
    float *exp_norm_coeffs = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*HEAD_DIM*2 + sizeof(float)*BLOCK_SIZE_C*HEAD_DIM + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C + sizeof(float)*2*BLOCK_SIZE_R);
    
    int tid = threadIdx.x;
    const int T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R; //q,o
    const int T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C; //k,v
    
    const int BATCH_IDX = blockIdx.x / (num_heads*T_r);
    const int HEAD_IDX = (blockIdx.x / T_r) % num_heads;
    const int Q_TILE_IDX = blockIdx.x % T_r;

    // thread indexing for tiled computation GEMM
    unsigned int threadRow = threadIdx.x / (BLOCK_SIZE_C / TN); // [0, ..., (BLOCK_SIZE_R/TM)-1]
    unsigned int threadCol = threadIdx.x % (BLOCK_SIZE_C / TN); // [0, ..., (BLOCK_SIZE_C/TN)-1]

    int WARP_ID = tid / 32;
    int LANE_ID = tid % 32;

    const float sqrt_head_dim = sqrtf((float)HEAD_DIM);

    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                                HEAD_IDX * (seq_len * HEAD_DIM) + 
                                Q_TILE_IDX * BLOCK_SIZE_R * HEAD_DIM;

    float4 *q_buff_f4 = reinterpret_cast<float4 *>(q_buff);
    // load Q tile to SHM
    #pragma unroll
    for(int x = tid; x < BLOCK_SIZE_R * HEAD_DIM / 4; x += blockDim.x)
    {
        int local_row = x*4 / (HEAD_DIM);
        int local_col = (x*4) % (HEAD_DIM);
        
        int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
        
        if (global_seq_idx < seq_len && local_col + 3 < HEAD_DIM) {
            int idx = base_hbm_offset + local_row * HEAD_DIM + local_col;
            q_buff_f4[x] = __ldg(reinterpret_cast<const float4*>(&query[idx]));
        } else {
            q_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if(local_col == 0 && local_row < BLOCK_SIZE_R)
        {
            logsumexp_shm[local_row] = 0.0f;
            maxes_shm[local_row] = -FLT_MAX;
            o_buff[local_row * HEAD_DIM] = 0.0f; 
        }
    }

    const int base_hbm_offset_kv = BATCH_IDX * (num_heads * seq_len * HEAD_DIM) + 
                                   HEAD_IDX * (seq_len * HEAD_DIM);

    for(int j=0; j < T_c; ++j)
    {
        // load K,V tiles into shared memory
        int kv_block_start = j * BLOCK_SIZE_C;
        
        #pragma unroll
        for(int x = tid; x < BLOCK_SIZE_C * HEAD_DIM / 4; x += blockDim.x)
        {
            int local_row = (x * 4) / HEAD_DIM;
            int local_col = (x * 4) % HEAD_DIM;
            
            int global_seq_idx = kv_block_start + local_row;
            
            if (global_seq_idx < seq_len && local_col + 3 < HEAD_DIM) {
                int idx = base_hbm_offset_kv + global_seq_idx * HEAD_DIM + local_col;

                float4 k_val_f4 = __ldg(reinterpret_cast<const float4*>(&key[idx]));

                // K transposed in shared memory
                kv_buff[(local_col + 0) * BLOCK_SIZE_C + local_row] = k_val_f4.x;
                kv_buff[(local_col + 1) * BLOCK_SIZE_C + local_row] = k_val_f4.y;
                kv_buff[(local_col + 2) * BLOCK_SIZE_C + local_row] = k_val_f4.z;
                kv_buff[(local_col + 3) * BLOCK_SIZE_C + local_row] = k_val_f4.w;
            } else if (global_seq_idx < seq_len) {
                for(int d = local_col; d < HEAD_DIM && d < local_col + 4; ++d) {
                    int idx = base_hbm_offset_kv + global_seq_idx * HEAD_DIM + d;
                    kv_buff[d * BLOCK_SIZE_C + local_row] = __ldg(&key[idx]);
                }
            } else {
                // Zero padding
                for(int d = local_col; d < local_col + 4 && d < HEAD_DIM; ++d) {
                    kv_buff[d * BLOCK_SIZE_C + local_row] = 0.0f;
                }
            }
        } 
        __syncthreads();

        float threadS[TM * TN] = {0.0f};
        float regQ[TM] = {0.0f};
        float regK[TN] = {0.0f};

        // outer loop: tile over head_dim dimension (BK chunks)
        for (unsigned int bkIdx = 0; bkIdx < HEAD_DIM; bkIdx += BK) {
            // Q is already in shared memory, K is already transposed
            // inner computation: accumulate partial dot products
            #pragma unroll
            for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                unsigned int d = bkIdx + dotIdx;
                if (d >= HEAD_DIM) break;

                // load Q values for this thread's TM registers
                #pragma unroll
                for (unsigned int i = 0; i < TM; ++i) {
                    unsigned int qRow = threadRow * TM + i;
                    if (qRow < BLOCK_SIZE_R) {
                        regQ[i] = q_buff[qRow * HEAD_DIM + d];
                    }
                }

                // load K^T values for this thread's TN registers  
                #pragma unroll
                for (unsigned int i = 0; i < TN; ++i) {
                    unsigned int kCol = threadCol * TN + i;
                    if (kCol < BLOCK_SIZE_C) {
                        regK[i] = kv_buff[d * BLOCK_SIZE_C + kCol];
                    }
                }

                // accumulate outer product into registers
                #pragma unroll
                for (unsigned int resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
                    #pragma unroll
                    for (unsigned int resIdx_N = 0; resIdx_N < TN; ++resIdx_N) {
                        threadS[resIdx_M * TN + resIdx_N] += regQ[resIdx_M] * regK[resIdx_N];
                    }
                }
            }
        }

        // write results to shared memory and apply scaling + bounds checking
        #pragma unroll
        for (unsigned int resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
            unsigned int globalRow = threadRow * TM + resIdx_M;
            if (globalRow >= BLOCK_SIZE_R) continue;
            
            // niepotrzebne i guess
           unsigned int global_q_idx = Q_TILE_IDX * BLOCK_SIZE_R + globalRow;
            if (global_q_idx >= seq_len) continue;

            #pragma unroll
            for (unsigned int resIdx_N = 0; resIdx_N < TN; ++resIdx_N) {
                unsigned int globalCol = threadCol * TN + resIdx_N;
                if (globalCol >= BLOCK_SIZE_C) continue;

                unsigned int global_kv_idx = j * BLOCK_SIZE_C + globalCol;
                if (global_kv_idx >= seq_len) {
                    s_buff[globalRow * BLOCK_SIZE_C + globalCol] = -FLT_MAX;
                } else {
                    s_buff[globalRow * BLOCK_SIZE_C + globalCol] = 
                        threadS[resIdx_M * TN + resIdx_N] / sqrt_head_dim;
                }
            }
        }
        __syncthreads();

        // each warp processes one row
        #pragma unroll
        for(int row = WARP_ID; row < BLOCK_SIZE_R; row += (blockDim.x / 32))
        {
            int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + row;
            if (global_seq_idx >= seq_len) continue;

            float row_max = -FLT_MAX;
            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                row_max = fmaxf(row_max, s_buff[row * BLOCK_SIZE_C + col]);   
            }

            // warp reduction to find max
            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));
            }

            float new_max = 0.0f;
            float coeff = 0.0f;
            if (LANE_ID == 0)
            {
                new_max = fmaxf(maxes_shm[row], row_max);
                coeff = __expf(maxes_shm[row] - new_max);
                maxes_shm[row] = new_max;
                exp_norm_coeffs[row] = coeff;
            }

            new_max = __shfl_sync(0xFFFFFFFF, new_max, 0);
            coeff = __shfl_sync(0xFFFFFFFF, coeff, 0);

            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                float val = s_buff[row * BLOCK_SIZE_C + col];
                s_buff[row * BLOCK_SIZE_C + col] = __expf(val - new_max);
            }

            float row_sum = 0.0f;
            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                row_sum += s_buff[row * BLOCK_SIZE_C + col];
            }

            // warp reduction to find sum
            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);
            }

            if (LANE_ID == 0)
            {
                logsumexp_shm[row] = coeff * logsumexp_shm[row] + row_sum;
            }
        }
        __syncthreads();




        #pragma unroll
        for(int x = tid; x < BLOCK_SIZE_C * HEAD_DIM / 4; x += blockDim.x)
        {
            int local_row = (x * 4) / HEAD_DIM;
            int local_col = (x * 4) % HEAD_DIM;
            
            int global_seq_idx = kv_block_start + local_row;
            
            if (global_seq_idx < seq_len && local_col + 3 < HEAD_DIM) {
                int idx = base_hbm_offset_kv + global_seq_idx * HEAD_DIM + local_col;
                float4 v_val_f4 = __ldg(reinterpret_cast<const float4*>(&value[idx]));
                *reinterpret_cast<float4*>(&kv_buff[local_row * HEAD_DIM + local_col]) = v_val_f4;
            } else if (global_seq_idx < seq_len) {
                for(int d = local_col; d < HEAD_DIM && d < local_col + 4; ++d) {
                    int idx = base_hbm_offset_kv + global_seq_idx * HEAD_DIM + d;
                    kv_buff[local_row * HEAD_DIM + d] = __ldg(&value[idx]);
                }
            } else {
                // zero padding
                for(int d = local_col; d < local_col + 4 && d < HEAD_DIM; ++d) {
                    kv_buff[local_row * HEAD_DIM + d] = 0.0f;
                }
            }
        } 
        __syncthreads();

        float threadO[TM * BK] = {0.0f};
        #pragma unroll
        for (unsigned int resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
            unsigned int row = threadRow * TM + resIdx_M;
            if (row >= BLOCK_SIZE_R) continue;
            
            unsigned int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + row;
            if (global_seq_idx >= seq_len) continue;

            float coeff = exp_norm_coeffs[row];
            
            #pragma unroll
            for (unsigned int dIdx = threadCol * BK; dIdx < HEAD_DIM; dIdx += (BLOCK_SIZE_C / TN) * BK) {
                float accum[BK] = {0.0f};
                
                #pragma unroll
                for (unsigned int c = 0; c < BLOCK_SIZE_C; ++c) {
                    float s_val = s_buff[row * BLOCK_SIZE_C + c];
                    
                    #pragma unroll
                    for (unsigned int i = 0; i < BK; ++i) {
                        unsigned int d = dIdx + i;
                        if (d < HEAD_DIM) {
                            accum[i] += s_val * kv_buff[c * HEAD_DIM + d];
                        }
                    }
                }

                #pragma unroll
                for (unsigned int i = 0; i < BK; ++i) {
                    unsigned int d = dIdx + i;
                    if (d < HEAD_DIM) {
                        unsigned int idx = row * HEAD_DIM + d;
                        o_buff[idx] = o_buff[idx] * coeff + accum[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int x = tid; x < BLOCK_SIZE_R * HEAD_DIM; x += blockDim.x)
    {
        int local_row = x / HEAD_DIM;
        int local_col = x % HEAD_DIM;
        int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
        if (global_seq_idx < seq_len) {
            int idx = base_hbm_offset + local_row * HEAD_DIM + local_col;
            output[idx] = o_buff[local_row * HEAD_DIM + local_col] / logsumexp_shm[local_row];

            if(local_col == 0 && global_seq_idx < seq_len)
            {
                logsumexp[BATCH_IDX * num_heads * seq_len + 
                               HEAD_IDX * seq_len + 
                               global_seq_idx] = __logf(logsumexp_shm[local_row]) + maxes_shm[local_row];
            }
        }
    }
}

#ifndef CUPY_INLINE_COMPILE
template<int head_dim>
void host_flash_attention2_forward(
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
    
    float *d_logsumexp;
    cudaMalloc(&d_logsumexp, batch_size * num_heads * seq_len * sizeof(float));
    
    cudaMemcpy((void *)d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Running Flash Attention with optimized GEMM...\n");
    printf("Batch: %d, Heads: %d, SeqLen: %d, HeadDim: %d\n", 
           batch_size, num_heads, seq_len, head_dim);
    
    const int HEAD_DIM = head_dim;
    const int BLOCK_SIZE_C = 32;
    const int BLOCK_SIZE_R = 32;
    const int BK = 4;             // tile size for head_dim dimension
    const int TM = 4;             // each thread handles TM rows
    const int TN = 4;             // each thread handles TN cols
    
    const size_t T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
    const size_t total_blocks = batch_size * num_heads * T_r;
    // const size_t num_threads_per_block = (BLOCK_SIZE_R * BLOCK_SIZE_C) / (TM * TN);
    const size_t num_threads_per_block = 128;
    
    const int shared_mem_size = sizeof(shm_t<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM>);
    
    printf("SHARED MEM SIZE PER BLOCK: %d bytes\n", shared_mem_size);
    printf("Threads per block: %zu (TM=%d, TN=%d, BK=%d)\n", num_threads_per_block, TM, TN, BK);
    printf("Total blocks: %zu\n", total_blocks);
    
    tm->Start();
    flash_attention2_forward_kernel<BLOCK_SIZE_R, BLOCK_SIZE_C, HEAD_DIM, BK, TM, TN>
                                  <<<total_blocks, num_threads_per_block, shared_mem_size>>>(
                                    d_Q, d_K, d_V, d_O, d_logsumexp,
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
}
using FA2F32Func = void(const float*, const float*, const float*, float*, float*, int, int, int, TimerManager*);
template FA2F32Func host_flash_attention2_forward<32>;
template FA2F32Func host_flash_attention2_forward<64>;
#else
extern "C" __global__
void flash_attention2_forward_kernel_wrapper(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim
) {
    flash_attention2_forward_kernel<32, 32, 64, 4, 4, 4>(
        query, key, value, output, logsumexp,
        batch_size, num_heads, seq_len
    );
}
#endif