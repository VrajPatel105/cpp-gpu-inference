# Transformer in C++

Porting my PyTorch encoder-decoder transformer to C++.

## What's in here

A from-scratch C++ implementation of the transformer architecture — no PyTorch, no frameworks, just flat arrays and index math.

## Components

- `matmul` — matrix multiplication
- `layernorm` — layer normalization  
- `softmax` — softmax
- `embedding` — token embedding lookup
- `attention` — multi-head attention
- `feedforward` — FFN block
- `encoder` — full encoder block
- `decoder` — full decoder block
- `transformer` — full forward pass

## Reference

PyTorch version: `https://github.com/VrajPatel105/Transformer-Implementation-from-scratch-with-custom-dataset/blob/main/model.py`