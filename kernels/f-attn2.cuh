#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>

#include "error_utils.h"
#include "timer.h"

void host_flash_attention2_forward_optim(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    TimerManager* tm);

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
    int head_dim,
    TimerManager* tm);
