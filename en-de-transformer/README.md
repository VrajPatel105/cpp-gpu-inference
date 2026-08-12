# en-de-transformer

An English-to-German transformer built from scratch, extended with a from-scratch
FlashAttention-2 implementation and from-scratch INT8 quantization — each component
built independently, then integrated and benchmarked end-to-end.

## What's here

- Full encoder-decoder transformer, implemented from scratch in PyTorch, trained on
  a 200,000-pair Tatoeba English-German dataset.
- FlashAttention-2 forward and backward passes, implemented from scratch in Triton
  and wrapped in a custom `torch.autograd.Function`, dropped in as the model's attention mechanism.
- Custom INT8 post-training quantization (`QuantizedLinear`, LLM.int8()-style vector-wise
  quantization + mixed-precision decomposition), applied model-wide with a calibration pass.

## Results

FlashAttention-2 reaches ~95% of FP16 peak throughput at sequence length 8192.

Quantization comparison (FP32 baseline vs. custom INT8 vs. bitsandbytes):

| Metric | FP32 | Custom INT8 | bitsandbytes |
|---|---|---|---|
| Translations identical to baseline | — | 15/15 | 15/15 |
| Perplexity | 21.8669 | 21.8698 | 21.8653 |
| Static memory footprint | baseline | -46.4% | -29.3% |
| Peak inference memory | baseline | -5.4% | -8.9% |

## Why from scratch

Both FlashAttention-2 and the INT8 quantization scheme were implemented without
reference to existing kernel libraries, then validated against production
implementations to confirm correctness and competitive performance.