#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <cstdint>

#include "error_utils.h"
#include "timer.h"

template<int head_dim>
void host_flash_attention2_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    int batch_size,
    int seq_len,
    int num_heads,
    TimerManager* tm
);

template<int head_dim>
void host_flash_attention2_backward(
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const float *deriv_output,
    const float *logsumexp,
    float *deriv_query,
    float *deriv_key,
    float *deriv_value,
    int batch_size,
    int seq_len,
    int num_heads,
    TimerManager* tm
);

template<int head_dim>
void host_flash_attention2_forward_fp16(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_O,
    float* h_logsumexp,
    int batch_size,
    int seq_len,
    int num_heads,
    TimerManager* tm
);

template<int head_dim>
void host_flash_attention2_backward_fp16(
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const float *deriv_output,
    const float *logsumexp,
    float *deriv_query,
    float *deriv_key,
    float *deriv_value,
    int batch_size,
    int seq_len,
    int num_heads,
    TimerManager* tm
);
