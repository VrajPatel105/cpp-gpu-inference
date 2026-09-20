# cpp-gpu-inference

End-to-end GPU inference stack built from first principles: CUDA and Triton kernels,
a from-scratch FlashAttention-2 (forward and backward), and from-scratch INT8 quantization,
each benchmarked against the production library it replaces (cuBLAS, PyTorch SDPA, bitsandbytes).

Full writeup with every table, the hardware each number was measured on, and the places my
implementation loses: [Building a GPU inference stack from scratch](https://vrajpatel105.github.io/gpu-inference/index.html)

## Results

| Component | What was built | Result |
|---|---|---|
| FlashAttention-2, forward + backward (Triton) | FA-2 from scratch (forward kernel plus three backward kernels), wrapped in a custom `torch.autograd.Function` and integrated into a working transformer | Peak memory matches PyTorch SDPA from seq_len 512 to 8192 (258 MB to 288 MB, no seq_len × seq_len matrix materialized). All kernels pass `allclose` against a PyTorch reference. Slower than SDPA at long sequences because causal block skipping isn't implemented yet |
| INT8 post-training quantization | Custom `QuantizedLinear` (LLM.int8()-style: outlier channels kept in full precision, per-layer dynamic threshold, INT8 for the rest), applied model-wide | Perplexity within 0.02% of the unquantized model; 15/15 test translations word-for-word identical; static memory −46.4% vs. −29.3% for bitsandbytes. Most of that gap comes from quantizing the 52K-vocab output projection, which the bitsandbytes run left unquantized. bitsandbytes wins on peak inference memory (−8.9% vs. −5.4%) |
| Tiled + autotuned matmul (Triton) | Custom matmul kernel, autotuned over block size / warps / stages | Within ~20% of cuBLAS across 512–4096 in true FP32 (TF32 disabled on both sides); ~60% of FP32 peak at 2048 |
| CUDA matmul optimization (PMPP) | Naive → shared-memory tiled → thread-coarsened kernels | Nsight Compute confirms the coalescing fix: the uncoalesced-load estimated speedup drops from 43% to 3% after tiling; 1.24× wall-clock over naive at N=1024 |

## Hardware

| Work | GPU |
|---|---|
| FlashAttention-2, INT8 quantization, Triton matmul | NVIDIA RTX 5080 Laptop, 16 GB, CUDA 12.9, WSL2 |
| CUDA / PMPP kernels | NVIDIA RTX 4070 Laptop (HP OMEN Transcend 16), sm_89, `nvcc -O3 -arch=sm_89` |

Results from the two GPUs are never mixed in the same comparison.

## Repository structure

- `cpp-core/`: C++ systems programming fundamentals
- `llm-c-study/`: LLM internals studied in C
- `gpu-fundamentals-pmpp/`: CUDA fundamentals from the PMPP book (memory coalescing, tiling, thread coarsening, occupancy), profiled with Nsight Compute and Nsight Systems. Raw `.ncu-rep` and `.nsys-rep` files are committed
- `triton-kernels/`: vector add, softmax, autotuned matmul, and Sasha Rush's GPU puzzles in Triton
- `flash-attention/`: hand-traced FA-1/FA-2 derivations and the from-scratch Triton kernel (forward and backward). Correctness checks and benchmarks run from `flash_attention.py`
- `quantization/`: custom INT8 PTQ benchmarked against bitsandbytes. SmoothQuant, GPTQ, and AWQ were studied (notes included) but not implemented
- `en-de-transformer/`: English-German transformer built from scratch, with the FlashAttention-2 kernel and INT8 quantization above integrated end-to-end. See its own README
- `transformer-from-scratch-cpp/`: early C++ port, in progress

## Open items

- Re-measure FlashAttention-2 throughput with symmetric FLOP accounting (the original harness credited my kernel and SDPA with different FLOP counts, so throughput numbers were removed)
- Skip fully masked K/V blocks in the causal FA-2 inner loop
- 2D block / register tiling to close the gap to cuBLAS

The paged KV cache and continuous batching server that builds on this work lives in
[paged-inference-engine](https://github.com/VrajPatel105/paged-inference-engine).

## Background

Built while working through PMPP, the FlashAttention-2 paper, and the LLM.int8() paper,
implementing each technique from scratch rather than importing it, then validating
against production implementations.
