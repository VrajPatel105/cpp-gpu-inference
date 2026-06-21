Here's my understanding of chapter 4


# Chapter 4 — Compute Architecture and Scheduling

## The Mental Model

The Grid is the top of the hierarchy. It is simply a collection of all blocks launched by one kernel call. Think of it as 5000 food orders arriving at the master kitchen all at once.

The GPU is the master kitchen. It has multiple independent stations called Streaming Multiprocessors (SMs). My RTX 4070 has 36 SMs. Each SM has its own hardware, its own memory, and works completely independently from other SMs. This is where true parallelism happens — all 36 SMs run at the same time.

The GPU assigns blocks from the grid to SMs. One SM can hold multiple blocks at once. Once a block is assigned to an SM, it never moves to another SM.

The SM takes each block and splits it into Warps of exactly 32 threads. All warps from all blocks on that SM go into a pool, and the SM schedules them together.

Warps are not truly parallel with each other. The SM executes one warp at a time. But when a warp is waiting for data from global memory, the SM instantly switches to another warp that is ready. This is called latency hiding — the SM is never sitting idle, it's always doing useful work.

Within a single warp, all 32 threads execute the exact same instruction at the exact same time. This is true parallelism. If threads in the same warp take different paths (if/else), the GPU runs both paths sequentially with inactive threads masked off — this is warp divergence and hurts performance.

Each thread has its own private variables called registers. These are the fastest memory on the GPU, zero latency, and managed automatically by the compiler. Every local variable you declare in a kernel lives in a register.

Shared memory is a small, fast memory space shared between all threads in a block. It is explicitly managed — you as the programmer decide what goes there using the `__shared__` keyword. It lives on the SM itself.

Global memory is where cudaMalloc lives. It is large, slow, and accessible by all threads across all blocks. Every time a thread reads from global memory it pays a high latency cost.

---

## Key Concepts

**Warp Divergence**

When threads in the same warp take different branches (if/else), the GPU cannot execute them simultaneously. It runs the if branch first with the else threads masked off, then runs the else branch with the if threads masked off. Both paths execute sequentially. The more branches inside a warp, the worse the performance hit. Good kernel design ensures threads in the same warp always take the same path.

**Occupancy**

Occupancy is the ratio of active warps on an SM to the maximum number of warps the SM can support. Higher occupancy generally means better latency hiding because the SM has more warps to switch between while others wait for memory. Occupancy is limited by three resources — the number of blocks per SM, the number of threads per SM, and the number of registers per SM. Whichever resource runs out first is the limiting factor.

**Latency Hiding**

Global memory access takes hundreds of clock cycles. Instead of stalling, the SM switches to another warp that is ready to execute. This overlap between memory latency and computation is called latency hiding and is the primary reason GPUs need thousands of threads — not just for parallelism, but to keep the hardware busy during memory waits.

**Resource Limits**

Every SM has hard limits on how many blocks, threads, and registers it can hold simultaneously. These limits interact. For example, if you use too many registers per thread, the SM cannot fit as many warps, which reduces occupancy and hurts performance. The CUDA Occupancy Calculator helps find the right balance for a given kernel.

---

## The Hierarchy

```
Grid
└── Blocks (assigned to SMs, never move)
    └── Warps (32 threads, split by SM from each block)
        └── Threads (truly parallel within a warp)

Memory fastest to slowest:
Registers → Shared Memory → L1/L2 Cache → Global Memory
```

---

## What This Means for Naive Matmul

Every thread in the naive matmul kernel reads directly from global memory on every loop iteration. For a 1024x1024 matrix, each thread makes 1024 reads from global memory. Neighboring threads redundantly load the same data independently. This is a memory bound problem — the kernel spends most of its time waiting for data, not doing math. The fix is tiled matmul using shared memory, which is implemented in W8.