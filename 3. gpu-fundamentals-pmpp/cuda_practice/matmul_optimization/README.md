# Matmul Optimization: Tiling, Coalescing, Coarsening

## This file has been summarized via claude. The screenshots of the ncu and nsys reports are in images folder of this same folder (matmul_optimization)

## Goal

Start from a naive 1024×1024 matmul kernel and apply three progressive optimizations,
measuring each with `cudaEvent` timing and Nsight Compute (`ncu`) hardware counters to
verify the mechanism actually worked, not just that the wall-clock number moved.

## Setup

- GPU: NVIDIA GeForce RTX 4070 Laptop (sm_89, Compute Capability 8.9), 36 SMs
- Compiled with `nvcc -O3 -arch=sm_89`
- N = 1024 for all kernels
- Correctness verified with an all-ones input matrix (expected output: every element = 1024.0)
- Timing: `cudaEvent` wrapping the kernel launch only, averaged over 3 runs per kernel

## Benchmark Table

| Kernel        | Time (ms) | Speedup | GFLOPs/s |
|---------------|-----------|---------|----------|
| Naive         | 3.0045    | 1x      | 714.6    |
| Tiled         | 2.5589    | 1.17x   | 838.9    |
| Coarsened     | 2.4240    | 1.24x   | 885.7    |
| torch.matmul  | 0.227     | 13.2x   | 9457.7   |

`GFLOPs/s = 2147 / time_ms` (2 × 1024³ FLOPs for a 1024×1024 matmul)

## Kernel Configurations

| Kernel     | TILE_WIDTH | COARSE_FACTOR | Block Size | Grid Size   |
|------------|------------|---------------|------------|-------------|
| Naive      | —          | —             | 16×16      | 64×64       |
| Tiled      | 16         | —             | 16×16      | 64×64       |
| Coarsened  | 16         | 2             | 16×16      | 32×64       |

TILE_WIDTH=16 was chosen because it divides N=1024 evenly (no boundary checks needed)
and gives 256 threads/block, enough for shared-memory reuse to matter. An earlier run
at TILE_WIDTH=2 (leftover from 4×4 correctness testing) gave 39ms — 15x *slower* than
naive — since 4-thread blocks have almost no data reuse to amortize and add heavy
launch overhead. COARSE_FACTOR=2 was chosen to keep register pressure modest; the grid
is asymmetric (32×64 instead of 64×64) since coarsening in this implementation only
reduces block count along the column dimension.

## Nsight Compute (`ncu --set full`) Findings

| Kernel    | Duration | Compute Throughput | Memory Throughput | Registers/thread | L1TEX Global Load Est. Speedup |
|-----------|----------|---------------------|---------------------|-------------------|-----------------------------------|
| Naive     | 1.92ms   | 98.58%              | 98.58%              | 40                | **43.13%** (uncoalesced)          |
| Tiled     | 1.47ms   | 96.64%              | 96.64%              | 40                | 3.36%                              |
| Coarsened | 1.41ms   | 96.52%              | 96.52%              | 40                | 3.48%                              |

(`ncu` durations differ from the `cudaEvent` averages above — profiling instrumentation
adds overhead — but are internally consistent for comparing the three kernels against
each other.)

### Finding 1 — Coalescing was measurably fixed, even though wall-clock speedup was modest

Naive matmul's global memory loads were flagged with a 43.13% estimated speedup left on
the table from poor L1TEX access pattern: only 18 of 32 bytes transmitted per sector were
actually being used by each thread, consistent with uncoalesced access. Tiling dropped
this to 3.36% — essentially resolved. This is the direct hardware-level confirmation of
the arithmetic-intensity math from Ch6 (naive: 0.25 OP/B, tiled: 8 OP/B): the intended
mechanism worked. The takeaway worth noting explicitly: fixing the access pattern didn't
translate into the full 5-10x speedup textbook examples predict, likely because the
RTX 4070's L2 cache is large enough (relative to the 4MB per-matrix footprint at N=1024)
to already be absorbing some of naive's redundant reads, partially masking tiling's
theoretical advantage.

### Finding 2 — Both throughput metrics sit near-saturated across all three kernels

Compute and Memory Throughput are both ~97-98% in every kernel, meaning none of the three
is cleanly "memory-bound" or "compute-bound" in the classic sense — they're bottlenecked
on both simultaneously. This caps how much further wall-clock gain any of the three
approaches alone can deliver; closing the remaining gap to `torch.matmul` (13.2x) requires
a different compute path entirely (Tensor Cores), not further tuning of these FP32 CUDA-core
kernels.

### Finding 3 — Coarsening's grid change is confirmed at the hardware level

Coarsened kernel's grid size shows as 32×64 vs 64×64 for naive/tiled — exactly matching
the intended halving of blocks along the column dimension for COARSE_FACTOR=2.

### Finding 4 — Registers/thread stayed flat at 40 across all three kernels

Expected coarsening to show a higher register count from holding `Pvalue[COARSE_FACTOR]`
per thread instead of a scalar, but it stayed identical. Likely the compiler already had
headroom at this COARSE_FACTOR; worth re-checking if pushing to COARSE_FACTOR=4, where
register pressure and occupancy tradeoffs should become more visible.

## Stretch Goal (not yet attempted)

2D block tiling / register tiling per Boehm's blog — the next step toward closing the
remaining gap to `torch.matmul`.
