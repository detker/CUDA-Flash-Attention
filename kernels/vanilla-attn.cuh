#include "error_utils.h"
#include "timer.h"
#include <float.h>
#include <cuda_runtime.h>

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
    TimerManager* tm = nullptr
);
