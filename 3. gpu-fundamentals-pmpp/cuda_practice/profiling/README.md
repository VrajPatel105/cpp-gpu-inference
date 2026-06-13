# CUDA Practice

## Profiling CUDA Kernels

Writing a working kernel is step one. Knowing *why* it's slow is step two — that's what profiling is for.

Profiling records everything that happens when your CUDA program runs: how long each kernel takes, how long memory transfers take, whether the GPU is sitting idle, and how much of the hardware's theoretical bandwidth you're actually using. Without it you're guessing at bottlenecks. With it you have exact numbers.

The naive matmul kernel in this folder is a good example. It works correctly but every thread fetches data directly from global memory — the slowest memory on the GPU — on every loop iteration. Multiple threads redundantly load the same values. Profiling makes this visible: you'll see the kernel is memory bound, meaning it spends more time waiting for data than doing math.

The fix is tiled matmul using shared memory, which is fast per-block memory that threads can share. After that optimization, profiling the same kernel shows a measurable speedup — turning "I think it's faster" into a concrete number.

## Tools

- **Nsight Systems** — `nsys profile --output=report ./build/your_kernel` generates a `.nsys-rep` file. Open it in the Nsight Systems GUI on Windows to visualize the timeline.

## Kernels

| File | What it does |
|------|-------------|
| `hello.cu` | First kernel, device properties |
| `vec_add.cu` | 1D kernel, vector addition |
| `squareElement.cu` | 1D kernel, square every element |
| `matmul.cu` | 2D kernel, naive matrix multiplication |
| `nvtx_matmul.cu` | Naive matmul with NVTX profiling annotations |

