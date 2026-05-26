# Transformer in C++

I built a transformer in PyTorch last month. Now I'm porting it to C++ — no frameworks, no autograd, just flat arrays and index math.

## What's implemented

Every component written from scratch in pure C++:

matmul, layernorm, softmax, embeddings, positional encoding, multi-head attention (with cross-attention support), feedforward, encoder block, decoder block.

## Why

Part of a self-directed roadmap: C++ → CUDA → Triton → FlashAttention → INT8 quantization. This is the foundation before GPU work starts.

## Reference

Original PyTorch implementation: github.com/VrajPatel105/Transformer-Implementation-from-scratch-with-custom-dataset/blob/main/model.py