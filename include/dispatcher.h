#pragma once

#include <stdio.h>
#include <string>
#include "f-attn.cuh"
#include "f-attn2.cuh"
#include "vanilla-attn.cuh"
#include "timer.h"
#include "enum_types.h"


template<int HEAD_DIM>
struct FlashAttentionDispatcher 
{
    static void dispatch_forward(
        const float* Q, const float* K, const float* V, float* O, float *logsumexp,
        int batch_size, int num_heads, int seq_len,
        ComputeDataType compute_data_type, ComputeType compute_method, TimerManager* tm) 
    {
        if (compute_method == ComputeType::FlashAttention2)
        {
            if (compute_data_type == ComputeDataType::FP16) {
                printf("Running Flash Attention 2 Forward (HEAD_DIM=%d) with FP16 data stored in SHM...\n", HEAD_DIM);
                host_flash_attention2_forward_fp16<HEAD_DIM>(Q, K, V, O, logsumexp, batch_size, seq_len, num_heads, tm);
            } else {
                printf("Running Flash Attention 2 Forward (HEAD_DIM=%d)...\n", HEAD_DIM);
                host_flash_attention2_forward<HEAD_DIM>(Q, K, V, O, logsumexp, batch_size, seq_len, num_heads, tm);
            }
        }
        else if (compute_method == ComputeType::FlashAttention1)
        {
            if (compute_data_type == ComputeDataType::FP16) {
                fprintf(stderr, "Error: Flash Attention 1 FP16 support not implemented\n");
                exit(EXIT_FAILURE);
            }
            printf("Running Flash Attention 1 Forward (HEAD_DIM=%d)...\n", HEAD_DIM);
            host_flash_attention_forward(Q, K, V, O, logsumexp, batch_size, seq_len, num_heads, HEAD_DIM, tm);
        }
        else if (compute_method == ComputeType::Naive)
        {
            if (compute_data_type == ComputeDataType::FP16) {
                fprintf(stderr, "Error: Vanilla Attention FP16 support not implemented\n");
                exit(EXIT_FAILURE);
            }
            printf("Running Vanilla Attention Forward (HEAD_DIM=%d)...\n", HEAD_DIM);
            host_vanilla_attention_forward(Q, K, V, O, logsumexp, batch_size, seq_len, num_heads, HEAD_DIM, tm);
        }
        else
        {
            fprintf(stderr, "Error: Unknown compute method\n");
            exit(EXIT_FAILURE);
        }
    }

    static void dispatch_backward(
        const float* Q, const float* K, const float* V, const float* O,
        const float* dO, const float* logsumexp,
        float* dQ, float* dK, float* dV,
        int batch_size, int num_heads, int seq_len,
        ComputeDataType compute_data_type, ComputeType compute_method, TimerManager* tm) 
    {
        if (compute_method == ComputeType::FlashAttention2)
        {
            if (compute_data_type == ComputeDataType::FP16) {
                printf("Running Flash Attention 2 Backward (HEAD_DIM=%d) with FP16 data stored in SHM...\n", HEAD_DIM);
                host_flash_attention2_backward_fp16<HEAD_DIM>(Q, K, V, O, dO, logsumexp, dQ, dK, dV, batch_size, seq_len, num_heads, tm);
            } 
            else 
            {
                printf("Running Flash Attention 2 Backward (HEAD_DIM=%d)...\n", HEAD_DIM);
                host_flash_attention2_backward<HEAD_DIM>(Q, K, V, O, dO, logsumexp, dQ, dK, dV, batch_size, seq_len, num_heads, tm);
            }
        }
        else if (compute_method == ComputeType::FlashAttention1)
        {
            fprintf(stderr, "Error: Flash Attention 1 backward pass not implemented\n");
            exit(EXIT_FAILURE);
        }
        else if (compute_method == ComputeType::Naive)
        {
            fprintf(stderr, "Error: Vanilla Attention backward pass not implemented\n");
            exit(EXIT_FAILURE);
        }
        else
        {
            fprintf(stderr, "Error: Unknown compute method\n");
            exit(EXIT_FAILURE);
        }
    }

    static void dispatch_forward_backward(
        const float* Q, const float* K, const float* V, float* O, float* logsumexp,
        const float* dO, float* dQ, float* dK, float* dV,
        int batch_size, int num_heads, int seq_len,
        ComputeDataType compute_data_type, ComputeType compute_method, TimerManager* tm) 
    {
        printf("Running Forward+Backward Pass (HEAD_DIM=%d)...\n", HEAD_DIM);
        
        dispatch_forward(Q, K, V, O, logsumexp, batch_size, num_heads, seq_len, compute_data_type, compute_method, tm);
        
        dispatch_backward(Q, K, V, O, dO, logsumexp, dQ, dK, dV, 
                          batch_size, num_heads, seq_len, 
                          compute_data_type, compute_method, tm);
    }
};

template<int CurrentD, int MaxD>
struct RuntimeDimDispatcher
{
    template<typename Func, typename... Args>
    static void dispatch(int head_dim, Func&& func, Args&&... args)
    {
        if (head_dim == CurrentD) 
        {
            func.template operator()<CurrentD>(std::forward<Args>(args)...);
        }
        else 
        {
            RuntimeDimDispatcher<CurrentD * 2, MaxD>::dispatch(
                head_dim, std::forward<Func>(func), std::forward<Args>(args)...);
        }
    }
};

template<int MaxD>
struct RuntimeDimDispatcher<MaxD, MaxD>
{
    template<typename Func, typename... Args>
    static void dispatch(int head_dim, Func&& func, Args&&... args)
    {
        if (head_dim == MaxD) 
        {
            func.template operator()<MaxD>(std::forward<Args>(args)...);
        } 
        else 
        {
            exit(EXIT_FAILURE);
        }
    }
};

struct ForwardPassLauncher 
{
    const float* Q;
    const float* K;
    const float* V;
    float* O;
    float* logsumexp;
    int batch_size;
    int num_heads;
    int seq_len;
    ComputeDataType compute_data_type;
    ComputeType compute_method;
    TimerManager* tm;

    template <int HEAD_DIM>
    void operator()() const 
    {
        FlashAttentionDispatcher<HEAD_DIM>::dispatch_forward(
            Q, K, V, O, logsumexp, batch_size, num_heads, seq_len, compute_data_type, compute_method, tm);
    }
};

struct BackwardPassLauncher 
{
    const float* Q;
    const float* K;
    const float* V;
    const float* O;
    const float* dO;
    const float* logsumexp;
    float* dQ;
    float* dK;
    float* dV;
    int batch_size;
    int num_heads;
    int seq_len;
    ComputeDataType compute_data_type;
    ComputeType compute_method;
    TimerManager* tm;

    template <int HEAD_DIM>
    void operator()() const 
    {
        FlashAttentionDispatcher<HEAD_DIM>::dispatch_backward(
            Q, K, V, O, dO, logsumexp, dQ, dK, dV, 
            batch_size, num_heads, seq_len, compute_data_type, compute_method, tm);
    }
};

struct ForwardBackwardPassLauncher 
{
    const float* Q;
    const float* K;
    const float* V;
    float* O;
    float* logsumexp;
    const float* dO;
    float* dQ;
    float* dK;
    float* dV;
    int batch_size;
    int num_heads;
    int seq_len;
    ComputeDataType compute_data_type;
    ComputeType compute_method;
    TimerManager* tm;

    template <int HEAD_DIM>
    void operator()() const 
    {
        FlashAttentionDispatcher<HEAD_DIM>::dispatch_forward_backward(
            Q, K, V, O, logsumexp, dO, dQ, dK, dV,
            batch_size, num_heads, seq_len, 
            compute_data_type, compute_method, tm);
    }
};

inline void RunFlashAttention(
    const float* Q, const float* K, const float* V, float* O, float* logsumexp,
    const float* dO, float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    ComputeDataType compute_data_type, ComputeType compute_method, ModeType mode, TimerManager* tm) 
{
    static constexpr int MIN_HEAD_DIM = 32;
    static constexpr int MAX_HEAD_DIM = 64;

    if (mode == ModeType::Forward)
    {
        ForwardPassLauncher launcher{Q, K, V, O, logsumexp, batch_size, num_heads, seq_len, compute_data_type, compute_method, tm};
        RuntimeDimDispatcher<MIN_HEAD_DIM, MAX_HEAD_DIM>::dispatch(head_dim, launcher);
    }
    else if (mode == ModeType::Backward)
    {
        BackwardPassLauncher launcher{Q, K, V, O, dO, logsumexp, dQ, dK, dV, 
                                     batch_size, num_heads, seq_len, compute_data_type, compute_method, tm};
        RuntimeDimDispatcher<MIN_HEAD_DIM, MAX_HEAD_DIM>::dispatch(head_dim, launcher);
    }
    else if (mode == ModeType::ForwardBackward)
    {
        ForwardBackwardPassLauncher launcher{Q, K, V, O, logsumexp, dO, dQ, dK, dV,
                                            batch_size, num_heads, seq_len, compute_data_type, compute_method, tm};
        RuntimeDimDispatcher<MIN_HEAD_DIM, MAX_HEAD_DIM>::dispatch(head_dim, launcher);
    }
}
