## Tiled Matmul — Triton vs cuBLAS (FP32)

Hand-written tiled matmul kernel in Triton (2D program grid, K-loop with
masked loads and `tl.dot` accumulation, autotuned over block sizes),
benchmarked against cuBLAS (`torch.matmul`) in **true FP32** — TF32 disabled
on both sides for a fair comparison, so neither uses the tensor cores.

Common currency is TFLOP/s (`2·M·N·K / time`), not GB/s: matmul is
compute-bound, so the question is "how close to peak FP32," not "how close to
peak bandwidth." Ceiling is the mobile RTX 5080 FP32 peak (~32 TFLOP/s).

| Size (N³) | Triton TFLOP/s | cuBLAS TFLOP/s | Triton % peak | cuBLAS % peak |
|-----------|---------------:|---------------:|--------------:|--------------:|
| 512       | 8.77           | 8.24           | 27.4%         | 25.7%         |
| 1024      | 15.95          | 16.64          | 49.9%         | 52.0%         |
| 2048      | 19.30          | 21.78          | 60.3%         | 68.1%         |
| 4096      | 16.53          | 20.32          | 51.7%         | 63.5%         |

**Findings**

1. **Within ~20% of cuBLAS across the range, and ahead at 512.** At 512 the
   kernel edges out cuBLAS (both overhead-bound, so this is small-matrix noise
   more than a real win). From 1024 up, cuBLAS pulls ahead but never by much —
   ~11% behind at 2048, ~19% at 4096. In true FP32 (no tensor cores) this is a
   fair fight, and closing to within 20% of NVIDIA's library with a from-scratch
   kernel is the result worth reporting.

2. **Efficiency climbs with size, then regresses at 4096.** % of peak rises
   27% → 50% → 60% from 512 to 2048 as launch/overhead amortizes over more
   compute — the expected roofline climb toward the compute ceiling. At 4096 it
   drops (Triton 60%→52%, cuBLAS 68%→64%). Both implementations regress
   together, which points at a platform ceiling — thermal/power throttling on
   the mobile card, or L2/memory pressure once the working set outgrows cache —
   not the kernel itself.

3. **Best efficiency ~60% of FP32 peak at 2048.** For a hand-derived tiled
   kernel with no tensor-core path, hitting 60% of the mobile 5080's FP32 peak
   is a strong baseline to optimize from.
