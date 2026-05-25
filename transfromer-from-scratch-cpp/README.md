# Transformer in C++

Porting my PyTorch encoder-decoder transformer to C++.

## What's in here

A from-scratch C++ implementation of the transformer architecture — no PyTorch, no frameworks, just flat arrays and index math.

## Components

- `matmul.cpp` — matrix multiplication
- `layernorm.cpp` — layer normalization  
- `softmax.cpp` — softmax
- `embedding.cpp` — token embedding lookup
- `attention.cpp` — multi-head attention
- `feedforward.cpp` — FFN block
- `encoder.cpp` — full encoder block
- `decoder.cpp` — full decoder block
- `transformer.cpp` — full forward pass

## Reference

PyTorch version: `https://github.com/VrajPatel105/Transformer-Implementation-from-scratch-with-custom-dataset/blob/main/model.py`