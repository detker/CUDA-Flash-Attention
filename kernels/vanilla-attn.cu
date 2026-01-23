#ifndef CUPY_INLINE_COMPILE
#include "vanilla-attn.cuh"
#else
#define FLT_MAX 3.402823466e+38F
#endif

template<int HEAD_DIM>
__global__ void vanilla_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    float* __restrict__ attention_scores,
    int batch_size,
    int num_heads,
    int seq_len
) {
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int tid = threadIdx.x;
    
    attention_scores += blockIdx.x * seq_len * seq_len; // [seq_len, seq_len]
    
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    const int head_offset = batch_idx * (num_heads * seq_len * HEAD_DIM) + 
                            head_idx * (seq_len * HEAD_DIM);
    
    for (int i = tid; i < seq_len * seq_len; i += blockDim.x) {
        int row = i / seq_len; 
        int col = i % seq_len;
        
        float dot_product = 0.0f;
        
        for (int d = 0; d < HEAD_DIM; ++d) {
            float q_val = query[head_offset + row * HEAD_DIM + d];
            float k_val = key[head_offset + col * HEAD_DIM + d];
            dot_product += q_val * k_val;
        }
        
        attention_scores[i] = dot_product * scale;
    }
    
    __syncthreads();
    
    for (int row = tid; row < seq_len; row += blockDim.x) {
        float row_max = -FLT_MAX;
        for (int col = 0; col < seq_len; ++col) {
            row_max = fmaxf(row_max, attention_scores[row * seq_len + col]);
        }
        
        float exp_sum = 0.0f;
        for (int col = 0; col < seq_len; ++col) {
            float exp_val = expf(attention_scores[row * seq_len + col] - row_max);
            attention_scores[row * seq_len + col] = exp_val;
            exp_sum += exp_val;
        }
        
        for (int col = 0; col < seq_len; ++col) {
            attention_scores[row * seq_len + col] /= exp_sum;
        }
    }
    
    __syncthreads();
    
    for (int i = tid; i < seq_len * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;  
        int dim = i % HEAD_DIM;  
        
        float sum = 0.0f;
        
        for (int j = 0; j < seq_len; ++j) {
            float att_weight = attention_scores[row * seq_len + j];
            float v_val = value[head_offset + j * HEAD_DIM + dim];
            sum += att_weight * v_val;
        }
        
        output[head_offset + row * HEAD_DIM + dim] = sum;
    }
}

#ifndef CUPY_INLINE_COMPILE
template<int head_dim>
void host_vanilla_attention_forward(
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
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    
    float *d_Q, *d_K, *d_V, *d_O, *d_attention_scores;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    
    size_t attention_size = batch_size * num_heads * seq_len * seq_len;
    cudaMalloc(&d_attention_scores, attention_size * sizeof(float));
    
    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Running Vanilla Attention...\n");
    printf("Batch: %d, Heads: %d, SeqLen: %d, HeadDim: %d\n", 
           batch_size, num_heads, seq_len, head_dim);
    
    const int HEAD_DIM = head_dim;
    const int BLOCK_SIZE = 128;
    const int total_blocks = batch_size * num_heads;
    
    printf("Blocks: %d, Threads per block: %d\n", total_blocks, BLOCK_SIZE);
    printf("Using global memory (HBM) for attention scores\n");
    
    if (tm) tm->Start();

    vanilla_attention_kernel<HEAD_DIM><<<total_blocks, BLOCK_SIZE>>>(
        d_Q, d_K, d_V, d_O, d_attention_scores,
        batch_size, num_heads, seq_len
    );
    
    if (tm) tm->Stop();
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_attention_scores));
}
using AttnVanillaFunc = void(const float*, const float*, const float*, float*, float*, int, int, int, TimerManager*);
template AttnVanillaFunc host_vanilla_attention_forward<32>;
template AttnVanillaFunc host_vanilla_attention_forward<64>;
#else
// wrapper for CuPy
extern "C" __global__
void vanilla_attention_kernel_wrapper(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* attention_scores,
    int batch_size,
    int num_heads,
    int seq_len
) {
    vanilla_attention_kernel<64>(
        query, key, value, output, attention_scores,
        batch_size, num_heads, seq_len
    );
}
#endif
