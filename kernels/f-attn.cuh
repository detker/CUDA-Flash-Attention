#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>

#include "error_utils.h"
#include "timer.h"

void host_flash_attention_forward(
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