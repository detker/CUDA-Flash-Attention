#pragma once

#include <stdio.h>
#include <cstdlib>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))

#define CUDA_CHECK(call) do {                                                                 \
    cudaError_t e = (call);                                                                   \
    if (e != cudaSuccess) {                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                              \
    } } while(0)

void inline usage(char* name)
{
    fprintf(stderr, "USAGE: %s <computation_method:naive|fa1|fa2> <mode:forward|backward|forward_backward> <SHM_precision:fp16|fp32> <data_folder_path>\n", name);
    exit(EXIT_FAILURE);
}
