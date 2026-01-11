#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <string>

#include "utils.h"
#include "error_utils.h"
#include "timer.h"
#include "dispatcher.h"
#include "f-attn.cuh"
#include "f-attn2.cuh"


int main(int argc, char** argv) {
    TimerManager tm;
    TimerGPU timerGPU;
    tm.SetTimer(&timerGPU);

    ComputeType compute_method;
    ModeType mode;
    char* data_path;
    parse_args(argc, argv, &compute_method, &mode, &data_path);

    int batch_size, n_heads, seq_len, head_dim;
    parse_config_string(data_path, &batch_size, &n_heads, &seq_len, &head_dim);
    const int qkv_size = batch_size * n_heads * seq_len * head_dim;

    printf("Batch size:    %d\n", batch_size);
    printf("Num heads:     %d\n", n_heads);
    printf("Sequence len:  %d\n", seq_len);
    printf("Head dim:      %d\n", head_dim);

    float *h_Q = new float[qkv_size];
    float *h_K = new float[qkv_size];
    float *h_V = new float[qkv_size];
    float *h_O = new float[qkv_size];
    float *h_logsumexp = new float[batch_size * n_heads * seq_len]; // for backward pass if needed

    // for backward pass
    float *h_dO = nullptr;
    float *h_dQ = nullptr;
    float *h_dK = nullptr;
    float *h_dV = nullptr;
    if (mode == ModeType::Backward || mode == ModeType::ForwardBackward) {
        h_dO = new float[qkv_size];
        h_dQ = new float[qkv_size];
        h_dK = new float[qkv_size];
        h_dV = new float[qkv_size];
    }

    std::string q_path = std::string(data_path) + "/Q.bin";
    std::string k_path = std::string(data_path) + "/K.bin";
    std::string v_path = std::string(data_path) + "/V.bin";
    std::string o_path = std::string(data_path) + "/O.bin";
    std::string logsumexp_path = std::string(data_path) + "/logsumexp.bin";
    std::string dO_path = std::string(data_path) + "/dO.bin";
    std::string dQ_path = std::string(data_path) + "/dQ.bin";
    std::string dK_path = std::string(data_path) + "/dK.bin";
    std::string dV_path = std::string(data_path) + "/dV.bin";

    bool files_exists = file_exists(q_path.c_str()) &&
                         file_exists(k_path.c_str()) &&
                         file_exists(v_path.c_str());

    if (mode == ModeType::Backward) {
        files_exists = files_exists && file_exists(o_path.c_str()) && file_exists(logsumexp_path.c_str());
    }

    if (files_exists) 
    {
        printf("Loading data...\n");

        load_binary_file(q_path.c_str(), h_Q, qkv_size);
        load_binary_file(k_path.c_str(), h_K, qkv_size);
        load_binary_file(v_path.c_str(), h_V, qkv_size);

        if (mode == ModeType::Backward) {
            load_binary_file(o_path.c_str(), h_O, qkv_size);
            load_binary_file(logsumexp_path.c_str(), h_logsumexp, batch_size * n_heads * seq_len);
        }

        if (mode == ModeType::Backward || mode == ModeType::ForwardBackward) {
            if (file_exists(dO_path.c_str())) {
                load_binary_file(dO_path.c_str(), h_dO, qkv_size);
            } else {
                // If dO file does not exist, we assume that dL/dO is a gradient of L - loss function - w.r.t. output O, 
                // for demo purposes we set L to sum(O), resulting in dL/dO = 1
                for (size_t i = 0; i < qkv_size; ++i) {
                    h_dO[i] = 1.0f;
                }
            }
        }

        printf("Data loaded successfully.\n\n");
    } else {
        ERR("Data files not found.\n");
    }

    printf("Running...\n");
    RunFlashAttention(
        h_Q, h_K, h_V, h_O, h_logsumexp,
        h_dO, h_dQ, h_dK, h_dV,
        batch_size, n_heads, seq_len, head_dim,
        compute_method, mode, &tm
    );
    printf("Kernel execution completed: %.4f seconds.\n\n", tm.TotalElapsedSeconds());

    printf("Saving output...\n");
    if (mode == ModeType::Forward || mode == ModeType::ForwardBackward) {
        save_binary_file(o_path.c_str(), h_O, qkv_size);
        save_binary_file(logsumexp_path.c_str(), h_logsumexp, batch_size * n_heads * seq_len);
    }
    if (mode == ModeType::Backward || mode == ModeType::ForwardBackward) {
        save_binary_file(dQ_path.c_str(), h_dQ, qkv_size);
        save_binary_file(dK_path.c_str(), h_dK, qkv_size);
        save_binary_file(dV_path.c_str(), h_dV, qkv_size);
    }
    
    printf("Output saved successfully.\n");

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_logsumexp;
    if (mode == ModeType::Backward || mode == ModeType::ForwardBackward) {
        delete[] h_dO;
        delete[] h_dQ;
        delete[] h_dK;
        delete[] h_dV;
    }

    return EXIT_SUCCESS;
}