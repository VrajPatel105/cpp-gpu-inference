# CUDA Programming Basics

This README covers the foundational concepts needed to understand CUDA programming, including threads, thread blocks, grids, and kernel execution.

---

## Threads

- Threads are single execution units that run kernels on the GPU.
- Similar to CPU threads, but GPUs launch thousands of them in parallel.
- Often visualized as individual arrows or execution paths.

Each thread has access to built-in variables:

- `threadIdx` — Thread index within a block  
- `blockIdx` — Block index within the grid  
- `blockDim` — Dimensions of each block  
- `gridDim` — Dimensions of the grid  

All of these are of type `dim3` and can be accessed inside a kernel to assign work.

![Thread Diagram](images/thread.svg)

---

## Thread Blocks

- A thread block is a group of threads.
- Threads within the same block can communicate and synchronize with each other.

![Thread Blocks Diagram](images/threadblocks.svg)

---

## Grid

- A kernel is launched over a grid.
- A grid consists of multiple thread blocks.
- Each block contains multiple threads.

![Grid Diagram](images/grid.svg)

---

## dim3 Data Type

`dim3` is a CUDA data type used to define dimensions in 3D space.

It contains three components:
- `x`
- `y`
- `z`

If values are omitted, they default to `1`.

### Examples

```cpp
dim3 threads(256);           // x = 256, y = 1, z = 1
dim3 blocks(100, 100);       // x = 100, y = 100, z = 1
dim3 anotherOne(10, 54, 32); // x = 10, y = 54, z = 32
```

### Notes

- Commonly used for defining grid and block dimensions.
- Missing dimensions automatically default to `1`.

---

## Kernel Invocation

CUDA kernels are launched using the triple angle bracket syntax:

```cpp
Kernel<<<numBlocks, threadsPerBlock>>>(parameters);
```

### Example

```cpp
Kernel<<<100, 256>>>(parameters);
```

This launches:
- 100 blocks
- 256 threads per block  
- Total threads = 25,600

---

## System Maximums

- Maximum threads per block: **1024**
- Maximum number of blocks per grid: **\(2^{32} - 1\)**

---

## Why Use Blocks and Threads?

- GPUs have limits on threads per block but can handle a very large number of blocks.
- This abstraction allows scalability:
  - More powerful GPUs can execute more blocks in parallel.
  - No code changes are required to benefit from better hardware.

---

## Common Kernel Pattern

A common approach is for each thread to compute a unique global ID to process data.

### Example

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

### Illustration

- 5th thread of 4th block:
  ```cpp
  id = 4 * 256 + 5 = 1029
  ```

- 14th thread of 76th block:
  ```cpp
  id = 76 * 256 + 14 = 19470
  ```

This pattern ensures each thread works on a unique portion of data.

---