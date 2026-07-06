# PMPP Chapter 6.4 — A Checklist of Optimizations (Table 6.1)

This section is the book's summary map: six optimizations that recur across the entire book, each examined through two lenses — what it does for the **compute cores** and what it does for the **memory system** — plus the concrete strategies for applying it. Most were covered in chapters 4–6; privatization is a preview of later chapters.

The two-lens framing is the real content of this table. Almost every optimization helps *both* sides, and understanding *which mechanism* delivers each benefit is what separates memorizing the table from being able to use it.

---

## Table 6.1 — The checklist

| Optimization | Benefit to compute cores | Benefit to memory | Strategies |
|---|---|---|---|
| Maximizing occupancy | More work to hide pipeline latency | More parallel memory accesses to hide DRAM latency | Tuning usage of SM resources such as threads per block, shared memory per block, and registers per thread |
| Enabling coalesced global memory accesses | Fewer pipeline stalls waiting for global memory accesses | Less global memory traffic and better utilization of bursts/cache lines | Transfer between global memory and shared memory in a coalesced manner and performing uncoalesced accesses in shared memory (e.g., corner turning); rearranging the mapping of threads to data; rearranging the layout of the data |
| Minimizing control divergence | High SIMD efficiency (fewer idle cores during SIMD execution) | — | Rearranging the mapping of threads to work and/or data; rearranging the layout of the data |
| Tiling of reused data | Fewer pipeline stalls waiting for global memory accesses | Less global memory traffic | Placing data that is reused within a block in shared memory or registers so that it is transferred between global memory and the SM only once |
| Privatization (covered later) | Fewer pipeline stalls waiting for atomic updates | Less contention and serialization of atomic updates | Applying partial updates to a private copy of the data and then updating the universal copy when done |
| Thread coarsening | Less redundant work, divergence, or synchronization | Less redundant global memory traffic | Assigning multiple units of parallelism to each thread to reduce the price of parallelism when it is incurred unnecessarily |

---

## Row-by-row explanation

### 1. Maximizing occupancy

Occupancy = how many warps are resident on an SM relative to the maximum it supports. This is the one lever that appears in **two chapters doing two different jobs**:

- **Compute-side benefit (Chapter 4's framing):** while one warp stalls waiting for an instruction's result, the scheduler switches to another resident warp. More resident warps → the compute pipeline never sits empty → pipeline latency is hidden.
- **Memory-side benefit (Chapter 6.2's framing):** DRAM latency hiding via banks and channels requires many **independent memory requests in flight simultaneously** — the memory controller staggers them across banks so the slow sensing phases overlap. Those requests come from threads. Not enough resident warps → nothing to stagger → banks idle → DRAM latency exposed.

**Strategy:** occupancy is limited by whichever SM resource runs out first — threads per block, shared memory per block, or registers per thread. Tuning means finding which one is your binding constraint and relaxing it. Note the direct tension with thread coarsening (row 6), which *raises* register usage per thread.

### 2. Enabling coalesced global memory accesses

The core of section 6.1.

- **Compute-side benefit:** a warp that issues an uncoalesced load waits on many separate DRAM transactions instead of one — that wait shows up as pipeline stalls in the compute cores. Coalescing shortens the stall.
- **Memory-side benefit:** a coalesced access uses every byte of the DRAM burst it triggers; an uncoalesced access wastes most of each burst on data nobody asked for. Coalescing means less total traffic and full utilization of every burst/cache line that gets moved.

**The three strategies, decoded:**

1. **Coalesced global↔shared transfer + uncoalesced access inside shared memory (corner turning):** when the algorithm's traversal direction and the storage layout disagree, load in the storage-contiguous direction (coalesced) and do the "wrong-direction" reads in shared memory, where there are no bursts and direction is free.
2. **Rearranging the mapping of threads to data:** change which thread reads which element so that `threadIdx.x` ends up controlling the +1 direction of the address. Corner turning is a specific instance of this.
3. **Rearranging the layout of the data:** if you own the data layout, change it — e.g., store the matrix transposed, or restructure arrays so the natural access is contiguous. Fixes the other half of the layout/access match.

### 3. Minimizing control divergence

The Chapter 4 topic, present in the checklist for completeness.

- **Compute-side benefit:** a warp executes in SIMD — all 32 threads execute the same instruction. If threads within a warp branch differently, the hardware runs each path serially while the threads on the other path idle. Minimizing divergence keeps all SIMD lanes doing useful work — high SIMD efficiency.
- **Memory-side benefit:** none listed (the dash). Divergence is purely a compute-pipeline phenomenon — this is the one row where the two-lens framing shows an optimization that lives entirely on one side.

**Strategies:** same two levers as coalescing's items 2–3 — remap threads to work/data, or relayout the data — because divergence, like coalescing, is ultimately about the correspondence between thread index and the work/data structure. (Example from earlier chapters: assigning boundary handling so whole warps take the same branch.)

### 4. Tiling of reused data

The Chapter 5 topic.

- **Compute-side benefit:** data served from shared memory or registers returns in a few cycles instead of hundreds — the pipeline stalls waiting for global memory far less often.
- **Memory-side benefit:** each value that gets reused within a block crosses the global-memory boundary **once** instead of once per use. In the tiled matmul, a 32×32 tile means each loaded value serves 32 threads: a 32× cut in global traffic (this is exactly the 0.25 → 8 OP/B jump from exercise 4).

**Strategy:** identify data that multiple threads in a block reuse, stage it in shared memory (or registers) once, synchronize, then compute from the fast copy.

### 5. Privatization (covered later in the book)

The preview row — full treatment comes with histograms and atomics in later chapters. The idea in brief:

- **The problem it solves:** when many threads atomically update the same shared data (e.g., histogram bins), the atomic operations **serialize** — every thread waits its turn on the same memory location, and the compute cores stall on those waits.
- **The move:** give each thread (or block) a **private copy** of the output data. Threads apply their updates to their private copy with no contention, then merge the private copies into the universal copy once at the end.
- **Compute-side benefit:** far fewer stalls waiting on atomic operations. **Memory-side benefit:** contention and serialization on the shared locations collapses from "every update" to "one merge per private copy."

Recognize the family resemblance: like tiling, it works by giving threads a cheap local place to work so the expensive shared resource is touched as rarely as possible.

### 6. Thread coarsening

The Chapter 6.3 topic.

- **Compute-side benefit:** merging the work of multiple threads into one eliminates *redundant* work those threads would each have done independently — redundant loads, and in other applications redundant computation, duplicated synchronization, or divergence overhead.
- **Memory-side benefit:** the redundant global memory traffic (e.g., two blocks loading the identical input tile) is eliminated outright rather than merely overlapped — the 8 → 32 OP/B jump in exercise 4.
- **The phrase worth keeping:** "reduce the **price of parallelism** when it is incurred **unnecessarily**." Fine-grained parallelism has a price (redundant loads, sync overhead). That price buys something only if the hardware actually runs the fine-grained units in parallel; if it serializes them anyway, the price bought nothing — coarsening stops paying it.
- **The cost (not in the table, but essential):** more registers per thread (multiple live accumulators) → lower occupancy → weaker latency hiding per row 1. Coarsening and occupancy pull the same resource in opposite directions; the right coarsening factor balances them.

---

## How to actually use the checklist

The table rows are not independent — they interact, and real tuning is navigating those interactions:

- **Occupancy is the currency several others spend.** Tiling spends shared memory per block; coarsening spends registers per thread; both purchases reduce occupancy, which weakens row 1's latency hiding on *both* the compute and memory sides. Every optimization below row 1 should be checked against what it does to row 1.
- **Coalescing and divergence share their strategy list** (remap threads, relayout data) because both are symptoms of the same underlying thing: a mismatch between the thread-index structure and the data/work structure.
- **Tiling, privatization, and coarsening are the same instinct at three scopes:** stop touching the expensive shared resource repeatedly. Tiling does it for global memory reads within a block; privatization does it for atomic updates to shared outputs; coarsening does it for redundant loads *across* blocks.
- **Diagnosis order in practice:** first ask whether the kernel is memory-bound or compute-bound (the OP/B ratio from exercise 4 answers this). Memory-bound → rows 2, 4, 6 attack traffic directly. Latency-bound with low utilization → row 1. Divergent control flow in profiles → row 3. Atomic contention → row 5.

This table is the book's recurring reference point — later chapters on different parallel patterns keep coming back to these same six rows applied in new contexts.
