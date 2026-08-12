# CUDA Programming Basics

This README covers the foundational concepts needed to understand CUDA programming, including threads, thread blocks, grids, and kernel execution.

***

## Threads

- Threads are single execution units that run kernels on the GPU.
- They are similar to CPU threads, but GPUs typically launch many more of them in parallel.
- They are sometimes drawn as wavy arrows in diagrams.

Each running thread knows the following built-in values:

- `threadIdx` — Thread index within the block
- `blockIdx` — Block index within the grid
- `blockDim` — Block dimensions in threads
- `gridDim` — Grid dimensions in blocks

Each of these is a `dim3` structure and can be read inside a kernel to assign a specific workload to each thread.

<p align="center">
  <img src="images/thread.svg" alt="Thread Diagram" width="300" />
</p>

***

## Thread Blocks

- A thread block is a collection of threads.
- All threads in the same block can communicate with each other.

<p align="center">
  <img src="images/threadblocks.svg" alt="Thread Blocks Diagram" width="300" />
</p>

***

## Grid

- A kernel is launched as a collection of thread blocks called a grid.
- The grid is made up of thread blocks, and each thread block is made up of threads.

<p align="center">
  <img src="images/grid.svg" alt="Grid Diagram" width="300" />
</p>

***

## `dim3` Data Type

`dim3` is a 3D structure (or vector type) with three integer components: `x`, `y`, and `z`.
You can initialize one, two, or all three coordinates. Any coordinate that is not provided defaults to `1`.

```cpp
dim3 threads(256);           // x = 256, y = 1, z = 1
dim3 blocks(100, 100);       // x = 100, y = 100, z = 1
dim3 anotherOne(10, 54, 32); // x = 10, y = 54, z = 32
```

### Notes

- `dim3` is commonly used in CUDA to define block and grid dimensions.
- If only `x` is given, `y` and `z` are automatically set to `1`.
- If `x` and `y` are given, `z` is automatically set to `1`.

***

## Kernel Invocation

The host launches a kernel using the triple chevron syntax `<<< >>>`.
Inside the chevrons, you specify the number of blocks and the number of threads per block.

```cpp
Kernel<<<100, 256>>>(parameters);
```

This launch creates:

- 100 blocks
- 256 threads per block
- 25,600 total threads

***

## System Maximums

- You can launch up to 1024 threads per block.
- You can launch up to `2^32 - 1` blocks in a single launch.

***

## Why Blocks and Threads?

- Each GPU has a limit on the number of threads per block, but it can handle a very large number of blocks.
- A GPU runs some number of blocks concurrently, which means it executes many threads at the same time.
- This extra level of abstraction allows more powerful GPUs to run more blocks concurrently without requiring any code changes.
- NVIDIA designed CUDA this way so the same code can scale to faster hardware automatically.

***

## Common Pattern Inside a Kernel

A very common pattern is for each thread to calculate a unique global ID so it can process a specific piece of data.

If you launch the kernel like this:

```cpp
Kernel<<<100, 256>>>(...);
```

Then each thread can compute its unique ID using:

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

### Examples

- 5th thread of the 4th block:

```cpp
int id = 4 * 256 + 5 = 1029;
```

- 14th thread of the 76th block:

```cpp
int id = 76 * 256 + 14 = 19470;
```

This pattern makes it easy to map threads to data elements in an array or tensor.