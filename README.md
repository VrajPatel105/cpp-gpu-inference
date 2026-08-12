# cpp-gpu-inference

End-to-end GPU inference stack built from first principles: CUDA/Triton kernels,
a from-scratch FlashAttention-2 implementation, and from-scratch INT8 quantization,
each benchmarked against production baselines (cuBLAS, bitsandbytes).

## Results

| Component | What was built | Result |
|---|---|---|
| Tiled + autotuned matmul (Triton) | Custom matmul kernel, autotuned over block size / warps / stages | Within 20% of cuBLAS across 512–4096; beats cuBLAS at 512; ~60% of theoretical FP32 peak at 2048 |
| FlashAttention-2, forward + backward (Triton) | Full FA-2 kernel from scratch, wrapped in a custom `torch.autograd.Function`, integrated into a working transformer | ~95% of FP16 peak throughput at seq_len 8192 |
| INT8 post-training quantization | Custom `QuantizedLinear` (vector-wise quantization + mixed-precision decomposition, LLM.int8()-style), applied model-wide | 15/15 identical translations vs. FP32 baseline; perplexity within 0.02%; static memory -46.4% (vs. -29.3% for bitsandbytes) |

## Repository structure

- `cpp-core/` — C++ systems programming fundamentals
- `llm-c-study/` — LLM internals studied in C
- `gpu-fundamentals-pmpp/` — CUDA fundamentals (PMPP book): memory coalescing, tiling, thread coarsening, occupancy; profiled with Nsight Compute/Systems
- `triton-kernels/` — vector add, softmax, autotuned matmul in Triton
- `flash-attention/` — hand-traced FA-1/FA-2 derivations and the from-scratch Triton kernel (forward + backward)
- `quantization/` — custom INT8 PTQ built from scratch and benchmarked against bitsandbytes. SmoothQuant, GPTQ, and AWQ were studied but not yet implemented.
- `en-de-transformer/` — **flagship project.** English-German transformer built from scratch, with the FlashAttention-2 and INT8 quantization above integrated end-to-end. See its own README.
- `transformer-from-scratch-cpp/` — early C++ port, in progress

## Background

Built while working through PMPP, the FlashAttention-2 paper, and the LLM.int8() paper,
implementing each technique from scratch rather than using existing libraries, then
validating against production implementations (cuBLAS, bitsandbytes).