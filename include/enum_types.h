#pragma once

enum class ComputeType {
    Naive,           // Naive implementation
    FlashAttention1, // Flash Attention version 1
    FlashAttention2  // Flash Attention version 2
};

enum class ModeType {
    Forward,         // Forward pass only
    Backward,        // Backward pass only
    ForwardBackward  // Both Forward and backward pass
};
