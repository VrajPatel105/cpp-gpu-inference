# Transformer in C++

I built a transformer in PyTorch last month. Now I'm porting it to C++ - no frameworks, no autograd, just flat arrays and index math.

## What's implemented

Every component written from scratch in pure C++:

1. `matmul` - matrix multiplication with optional bias. Accumulator pattern: local float val, write to memory once.
2. `layernorm` - per-token normalization. Two passes: compute mean + variance, then normalize and scale.
3. `softmax` - numerically stable softmax. Three passes: find max, exp + sum, normalize.
4. `embeddings` - token lookup table scaled by sqrt(d_model).
5. `positional_encoding` - sinusoidal PE added in-place using sin/cos formulas.
6. `attention_forward` - multi-head attention with fused QKV projection. Supports causal masking (bool flag) and cross-attention (separate k/v inputs).
7. `feedforward` - two matmuls with ReLU activation, 4x expansion.
8. `residual` - element-wise add of input and sublayer output.
9. `projection_forward` - final linear projection from d_model to vocab_size.
10. `encoder_block` - attention + residual + layernorm + FFN + residual + layernorm.
11. `decoder_block` - masked self-attention + residual + layernorm + cross-attention + residual + layernorm + FFN + residual + layernorm.
12. `transformer_block` - full forward pass: src/tgt embeddings + PE, N encoder blocks, N decoder blocks, projection, softmax.

## Reference

Original PyTorch implementation: github.com/VrajPatel105/Transformer-Implementation-from-scratch-with-custom-dataset/blob/main/model.py