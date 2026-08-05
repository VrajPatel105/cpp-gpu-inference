# En-De Transformer + FlashAttention-2

An English-German encoder-decoder Transformer, fused with a from-scratch
FlashAttention-2 forward+backward kernel written in Triton.

## Origin

The base Transformer implementation (model architecture, training loop,
custom dataset pipeline) is taken from my earlier project: link :  
[Transformer-Implementation-from-scratch-with-custom-dataset](https://github.com/VrajPatel105/Transformer-Implementation-from-scratch-with-custom-dataset).

This repo builds on that foundation by replacing the model's attention
mechanism with a custom FlashAttention-2 Triton kernel, implemented and
verified from scratch (see [`5. flash-attention`](../5.%20flash-attention) 
in the parent repo  for the standalone kernel work, benchmarks, and derivation notes).

## What's new here

- FlashAttention-2 forward pass (Triton), verified against PyTorch's
  `scaled_dot_product_attention` (causal and non-causal) within
  floating-point tolerance
- FlashAttention-2 backward pass (Triton), wrapped in a custom
  `torch.autograd.Function` so it drops into standard PyTorch training
- fp16 inputs with fp32 accumulation
- `@triton.autotune`-tuned block sizes
- Benchmarked against SDPA: throughput (TFLOPS, % of hardware peak) and
  peak memory across sequence lengths 512–8192

## Results

| Metric | This kernel | PyTorch SDPA |
|---|---|---|
| Peak HW utilization (8192 tokens) | ~95% | — |
| Peak memory (8192 tokens) | ~288 MB | ~288 MB |
| Notes | Matches SDPA memory scaling; SDPA edges ahead on speed at long sequences via causal block-skipping (not yet implemented here) |

*(full benchmark table in [`5. flash-attention/flash_attention_fwd.py`](../5.%20flash-attention))*

## What stayed the same

Model architecture, tokenization, dataset handling, and training loop
are unchanged from the original project — only the attention mechanism
inside multi-head attention was replaced.

## Status

- [x] Forward pass
- [x] Causal masking
- [x] fp16 / fp32 accumulation
- [x] Autotuning + benchmarking
- [ ] Backward pass
- [ ] Full training run with the integrated kernel
- [ ] Nsight profiling