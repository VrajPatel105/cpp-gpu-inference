# PMPP Chapter 6 — Performance Considerations
### Complete study notes (6.1 Memory Coalescing · 6.2 Hiding Memory Latency · 6.3 Thread Coarsening · Exercises)

These notes reconstruct the entire chapter the way it was actually learned: from physics up, with concrete numbers, hand-traces, and the specific mistakes made along the way (kept in on purpose — they mark the exact spots where the intuition tends to break).

---

## 6.1 — Memory Coalescing

### 1. Why DRAM is slow (the root cause of everything in this chapter)

Every performance technique in this chapter exists because of one physical fact, so it is worth understanding fully.

A DRAM cell stores one bit as a **tiny capacitor** holding a charge. The charge is so small that reading it is not a fast digital read — it is an **analog sensing operation**. The full sequence for one access:

1. The **decoder** takes the requested address and activates the correct row of cells.
2. Each activated cell shares its stored charge with a long wire (a bit line).
3. A **sense amplifier** on that wire detects whether a sufficient amount of charge is present, and amplifies that faint analog signal into a clean digital 0 or 1.

Step 3 is the killer. Detecting a minuscule voltage difference reliably takes time — the whole decode-activate-sense sequence costs **tens of nanoseconds**, and crucially, that cost is the same whether you wanted 1 byte or a whole row. The latency is a fixed price of the physics, not a function of the request size.

Why build memory this way at all? It is a **deliberate density-versus-speed trade-off**:

- DRAM: 1 transistor + 1 capacitor per bit. Extremely dense, extremely cheap per gigabyte. Slow to read.
- SRAM (what caches and shared memory are built from): ~6 transistors per bit. Very fast, but expensive and takes far more chip area per bit.

You cannot afford gigabytes of SRAM, so main memory is DRAM, and the entire rest of this chapter is about living with the consequences.

**Mistake I made here:** I initially thought DRAM is slow *because* it senses many bits at once. That is exactly backwards. The sensing is slow for physics reasons regardless of width; sensing many bits at once is the *workaround* for the slowness, not the cause of it. Getting this causal arrow right matters, because it explains why bursts exist at all.

### 2. DRAM bursts (the workaround)

Since one sensing operation costs the same fixed latency no matter what, DRAM designers amortize that fixed cost: when one location is accessed, the hardware does not sense just that bit — it fires **an entire row of sense amplifiers in parallel**, sensing a whole row of consecutive locations at once.

That sensed row of consecutive data is called a **burst**.

The consequences:

- Once a row is sensed, any access that falls **inside** that row is cheap — the data is already sitting in the sense amplifiers.
- Any access that falls in a **different** row pays the full decode-activate-sense latency all over again.
- The burst is going to be sensed whether you use all of it or one byte of it. Unused burst data is pure waste — bandwidth spent sensing data nobody asked for.

So the performance question becomes: **how much of each burst do your accesses actually use?** That question is what coalescing answers.

### 3. Memory coalescing (matching your access pattern to bursts)

The GPU executes threads in **warps**: groups of 32 consecutive threads that issue their memory requests simultaneously. So at any moment, the memory system sees 32 addresses arriving together. Two extreme outcomes:

- All 32 addresses are **consecutive** and fall inside one burst: the hardware serves the entire warp with **one** memory transaction. Every byte of the burst is used. This is a **coalesced** access.
- The 32 addresses are **scattered** across different DRAM rows: each one potentially triggers its own separate slow sensing operation. Many transactions, most of each burst wasted. This is an **uncoalesced** access.

**The one test, every time.** Write the global memory address as a function of `threadIdx.x` (call it `tx`). Increase `tx` by exactly 1, holding everything else fixed. Does the address increase by exactly **+1**?

- Yes → coalesced.
- No — it jumps by 4, 8, N, or any stride → uncoalesced.
- The array is `__shared__` → the question is not applicable at all (shared memory is on-chip SRAM; there are no bursts, no sensing, no coalescing concept there).

In practice this reduces to locating the **multiplier** sitting next to the `tx`-dependent term in the address expression. An implicit `×1` (nothing there) means coalesced. Any explicit multiplier (`i*4`, `i*8`, `row*N`) on the term that varies across the warp means uncoalesced.

**Concrete 4×4 example.** A row-major matrix with values 1–16 lives at flat addresses 0–15. A warp of 4 threads (`tx` = 0,1,2,3) uses one of two address formulas:

| Formula | Addresses (tx=0..3) | What it fetches | Verdict |
|---|---|---|---|
| `base + tx` | 0, 1, 2, 3 | a **row** (values 1,2,3,4) | coalesced — one burst, one transaction |
| `base + tx*4` | 0, 4, 8, 12 | a **column** (values 1,5,9,13) | uncoalesced — 4 separate row activations |

**The key reframe — this is the sentence to remember:** coalescing is NOT "rows are fast, columns are slow." It is entirely about whether the direction that varies with `tx` matches the direction that is **physically contiguous in storage**. In a row-major matrix, rows are contiguous, so row-walks coalesce. In a column-major matrix, columns are contiguous, so column-walks coalesce and row-walks do not. Same operation, different layout, opposite verdict. The layout is half of the equation; the access pattern is the other half; coalescing is the *match* between them.

### 4. Row-major vs column-major (the flattening game)

Everything above depends on knowing what "physically contiguous" means, which requires being precise about storage layout.

A matrix does not exist as a 2D grid in memory. Memory is **one flat line of boxes** with addresses 0, 1, 2, 3, ... "Rows and columns" is a story we tell about that line, and the story is encoded in one formula that converts a logical `(row, col)` position into a flat address. There are two standard choices:

- **Row-major:** `address = row * width + col`. Consequence: elements of the same row sit at consecutive addresses.
- **Column-major:** `address = col * height + row`. Consequence: elements of the same column sit at consecutive addresses.

Same 2×2 matrix `[[1,2],[3,4]]`, two different flat layouts:

- Row-major flat memory: `[1, 2, 3, 4]`
- Column-major flat memory: `[1, 3, 2, 4]`

Nothing about the matrix changed. Only the conversion formula changed, and with it, which neighbors-in-logic became neighbors-in-memory.

**Major order is not a property of the array or its declaration.** In C/C++/CUDA, `float* B` or `float B[16]` is just a strip of bytes with no layout metadata whatsoever. You do not declare, allocate, or `cudaMalloc` anything differently for the two orders. Major order is purely a **contract between whoever writes the array and whoever reads it**: both sides must use the same conversion formula. Break the contract and you silently read wrong values — not slow values, *wrong* ones.

**Where mixed majors actually happen in real code** (this is not a textbook contrivance):

1. **BLAS / cuBLAS / Fortran heritage.** The foundational linear algebra libraries are column-major because Fortran is column-major by language design. cuBLAS and much scientific CUDA code inherited this. The moment your kernel interoperates with cuBLAS output, you are handling column-major data.
2. **Logical transposes that never move data.** PyTorch's `.T` and NumPy's `.T` do not copy or rearrange anything — they return a view with swapped strides. A "row-major matrix used transposed" is functionally identical to a column-major matrix. This case appears constantly in ML workloads: backpropagation and attention both routinely need one operand transposed. My own derivation of `Bᵀ × A` proved the point: `D[i][j] = Σ B[k][i]·A[k][j]`, and that `B[k][i]` term walks a fixed column of B as k varies — the transpose in the *math* determines whether you need rows or columns, no matter how B is physically stored or what you relabel it as.
3. **Mixed pipelines.** A C++ preprocessing stage hands you row-major data; the next library call expects column-major. You either physically convert (expensive) or adjust your access formula to match reality (the corner-turning philosophy).

**Trap I fell into on paper:** never draw a 2D grid by "reading the flat array in order and wrapping" — that silently assumes row-major. The reliable protocol: build a decode table from the formula first (each `(row,col) → address → value`), then draw the grid *from the table*, one cell at a time. Related trap: the identity matrix is symmetric, so it produces the identical flat array under both major orders — it *cannot* expose a major-order bug in testing. Always validate with an asymmetric matrix.

### 5. Corner turning (the fix when layout and traversal disagree)

**First, when it is NOT needed.** Both matrices row-major, computing `A × B` directly: the standard tiled kernel is already coalesced for both loads, with no modification. I verified this by tracing the standard load `Nds[ty][tx] = B[(ph*TILE+ty)*N + col]`: since `col = bx*TILE + tx`, the address's `tx`-dependent term has multiplier 1 — coalesced as-is. Corner turning is not a general upgrade to every tiled kernel; applying it where nothing is broken changes nothing.

**When it IS needed:** the storage layout and the algorithm's required traversal direction genuinely mismatch. The two real triggers:

- One matrix is stored **column-major** while the natural per-thread indexing would walk it in the row direction (the Fig 6.4 case in the book), or
- The algorithm uses a matrix **transposed** (`Bᵀ × A`) — which, per the previous section, is the same situation wearing different clothes.

In these cases the naive symmetric load puts `tx` on a strided term — every warp triggers many row activations per load.

**The core insight of the fix.** A tiled kernel touches data in two separate steps: the **load** (global memory → shared memory) and the **compute** (reads from shared memory). Coalescing rules apply *only* to the load, because only global memory has bursts. Shared memory is on-chip SRAM, addressed directly — reading it "sideways," transposed, or in any scrambled order costs nothing extra. So the two steps do not have to use the same thread-to-index mapping. Corner turning exploits exactly that freedom:

> Let the global read follow whatever direction is contiguous **in storage** (coalesced), landing the data "sideways" in shared memory — then let the compute read shared memory in whatever order the **math** needs. The orientation mismatch is absorbed entirely in the place where it is free.

**The derivation method that finally made it click (the target-value approach).** The mistake pattern to avoid is fiddling with indices until something looks plausible. Instead, derive in this order:

1. **Fix what each shared-memory slot must contain.** This comes from the compute line, which is fixed and non-negotiable: `Pvalue += Mds[ty][k] * Nds[k][tx]` requires slot `Nds[a][b]` to hold `B[ph*TILE + a][bx*TILE + b]`. Note that `row` and `by` never appear anywhere in this requirement — the value B contributes depends on the output **column**, never the output row. (My first attempt put `row` into B's formula; it produced a value from the wrong place entirely — wrong answer, not just slow.)
2. **Choose who fills which slot.** This is a free choice — the block collectively fills the tile, and any thread-to-slot assignment works as long as every slot gets filled exactly once. Choose: thread `(tx, ty)` fills slot `Nds[tx][ty]` (note the swap versus the naive `Nds[ty][tx]`). Substituting into step 1's requirement: this thread must fetch `B[ph*TILE + tx][bx*TILE + ty]`.
3. **Convert that (row, col) to a flat address using B's actual storage formula.** B is column-major: `address = col*N + row = (bx*TILE + ty)*N + (ph*TILE + tx)`.
4. **Check coalescing.** In that address, `tx` appears with no multiplier — consecutive threads in a warp (same `ty`, `tx` = 0,1,2,...) read consecutive addresses. Coalesced.

**The resulting kernel — only ONE line differs from the standard tiled matmul:**

```cuda
Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];                        // unchanged (A row-major)
Nds[tx][ty] = B[(bx*TILE_WIDTH + ty)*N + (ph*TILE_WIDTH + tx)];       // corner-turned load
// compute stays exactly:  Pvalue += Mds[ty][k] * Nds[k][tx];
// syncs, phase loop, final write: all unchanged
```

The compute line stays valid *because* step 1 designed the load around it. The write-side index pair got rotated between the load step and the read step — that rotation is the "corner" being turned.

**Debugging habit that saved the hand-trace:** substitute actual numbers on the very first line of any trace. Never carry `by*TILE_WIDTH + ty` symbolically once you know `by=0, ty=0` — write `0` immediately. The intimidation of long index expressions is really just symbol-juggling; collapsing to numbers on line one removes it. Also keep the load step and compute step mentally separate: the load has no `k` (each thread fetches exactly one element, once); `k` exists only inside the compute loop.

---

## 6.2 — Hiding Memory Latency (banks and channels)

Coalescing makes each individual access efficient. This section is about a different dimension of the problem: even perfectly coalesced accesses leave the memory system mostly idle unless many of them are **in flight at once**. The hardware structures that make overlap possible are banks and channels.

### 1. The two players, precisely

- A **bank** is one complete, independent DRAM array: its own decoder, its own row of sense amplifiers, its own cell array — and therefore its own private copy of the slow sensing latency. Two banks are two fully separate decode-and-sense machines that know nothing about each other. They can run their slow sensing steps **at the same time, on different data**.
- A **channel** is the **wire (bus)** that carries already-sensed data from a group of banks out to the processor. It has a fixed width and clock, hence a fixed bandwidth (e.g. 16 GB/s). The channel itself senses nothing and is not slow — it is just a pipe. Its one constraint: **only one bank can be transferring on it at any instant.**
- Terminology: a channel *is* a bus — specifically the bus between the memory controller and its group of DRAM banks. "Bus" is the generic term for any shared wire pathway; "channel" is this particular one. In this chapter the two are interchangeable.

The division of labor: **banks provide parallelism** (independent simultaneous sensing), **channels provide bandwidth** (separate wires). Neither substitutes for the other.

### 2. Why one bank wastes almost the entire channel

Access latency (the sensing) vastly exceeds data transfer time. The book's example ratio is **20:1**. With a single bank on a channel, the timeline is forced to serialize:

```
sense (20 units) → transfer (1) → sense (20) → transfer (1) → ...
```

The channel wire is busy 1 out of every 21 units: **1/21 ≈ 4.8% utilization**. A 16 GB/s channel delivers about 0.76 GB/s of actual data. Nineteen of every twenty nanoseconds, the expensive wire sits idle waiting for one sensing operation. This is the quantitative statement of the problem the rest of the section solves.

### 3. The fix: staggered banks sharing one channel

**Correction to my first mental model:** I pictured banks racing, with "whichever bank finishes first grabbing the channel." The real mechanism is deliberate scheduling, not a race, and the difference matters for understanding the R+1 formula below.

The memory controller **staggers the start times** of accesses to different banks:

1. Bank 0 begins sensing at t = 0.
2. Shortly after — while bank 0 is still mid-sense and needs nothing from the channel — the controller starts bank 1 sensing too, on bank 1's own independent hardware.
3. Bank 0 finishes sensing first (it started first), takes the channel, transfers its burst (fast).
4. By the time that transfer completes, bank 1 has been sensing in the background for nearly the entire duration — it is ready (or nearly ready) and takes the channel next with little or no gap.
5. Repeat, with more banks staggered behind.

The structural insight: the **slow parts of many banks overlap each other in time**, and only the **fast parts** ever touch the shared, scarce resource (the wire). The channel goes from 4.8% busy to nearly continuously busy.

**The R+1 rule.** If the latency-to-transfer ratio is R : 1, you need at least **R + 1 banks** per channel to keep the wire saturated — enough staggered banks that one is always finishing its sense exactly as the previous one finishes its transfer. For the 20:1 example: at least 21 banks per channel. Fewer than that, and gaps reappear where the channel idles waiting for the next bank to catch up.

In practice you want even **more** than R+1 banks, for two reasons:

1. **Bank conflicts.** If several simultaneous requests happen to target the *same* bank, that bank serves them strictly one at a time — their latencies cannot overlap because there is only one sensing machine involved. More banks lower the probability that independent requests collide on one bank.
2. **Capacity.** Each bank's cell array is size-limited (for latency and manufacturability), so reaching the required total memory size takes many banks anyway.

### 4. Where the requests come from: occupancy (the Chapter 4 connection)

The staggering trick has a precondition that hardware cannot manufacture on its own: at any given moment there must be **multiple independent memory requests available** to distribute across banks. The controller cannot stagger requests that do not exist yet.

Those requests come from threads. If a kernel has too few resident warps issuing loads, there are not enough independent requests in flight to occupy multiple banks concurrently — banks idle, latency goes unhidden, and the channel utilization collapses back toward the single-bank picture.

This is the same **occupancy** lever from Chapter 4, doing a second job. Chapter 4's framing: enough resident warps to hide *compute pipeline* latency (while one warp waits on an instruction, another executes). This chapter's addition: enough resident warps to hide *DRAM* latency (while one request is being sensed in bank 0, others are being sensed in banks 1, 2, 3...). Same lever, two different bottlenecks it addresses. And for best results each individual access should itself be coalesced, and the traffic should spread across channels and banks — which leads to the layout question.

### 5. Interleaved data distribution (Fig 6.9)

None of the above helps if all your simultaneous requests happen to land on the same bank or channel due to unlucky address layout. The hardware prevents this by construction: consecutive burst-sized chunks of the address space are spread **across all channels first, then across banks**:

```
M[0],M[1] → channel 0, bank 0        M[2],M[3] → channel 1, bank 0
M[4],M[5] → channel 2, bank 0        M[6],M[7] → channel 3, bank 0
M[8],M[9] → channel 0, bank 1        M[10],M[11] → channel 1, bank 1
M[12],M[13] → channel 2, bank 1      M[14],M[15] → channel 3, bank 1
```

Channels first, then wrap to the next bank. The design goal: even a **small** array gets spread across every channel, so simultaneous accesses from different threads and blocks naturally scatter across channels and banks — the precondition for staggering — with zero effort in kernel code. (Consequence noted in the book: with a bigger burst size, you need proportionally bigger arrays before all channels get involved.)

### 6. The matmul payoff (Figs 6.10 / 6.11 — the whole section in one example)

The 4×4 matmul with 2×2 tiles and 2×2 blocks, phase by phase:

- **Phase 0:** Block(0,0) and Block(0,1) both need M's top tile: `M[0], M[1], M[4], M[5]`. Per the interleaved map above, `M[0],M[1]` live in channel 0 / bank 0 and `M[4],M[5]` in channel 2 / bank 0. Each block issues two coalesced accesses, and those accesses land on **two different channels working in parallel**.
- Meanwhile Block(1,0) and Block(1,1) need `M[8], M[9], M[12], M[13]` — the **same two channels, but bank 1**. So while bank 0 of channel 0 is sensing for the top blocks, bank 1 of that same channel is simultaneously sensing for the bottom blocks. This is Fig 6.8(B)'s staggering happening live, driven by real thread blocks.
- **Phase 1** shifts the traffic to channels 1 and 3 — the channels idle in phase 0 light up next.
- Block(0,0) and Block(0,1) load **identical** M elements. GPU caches exist largely to merge exactly this kind of duplicate access into one DRAM transaction when the blocks execute close together in time. (This same redundancy becomes the motivation for thread coarsening in 6.3.)

The section's closing idea, worth keeping verbatim in spirit: there is a **symbiotic relationship** between parallel thread execution and the parallel structure of DRAM. Utilizing DRAM's banks and channels requires many threads making simultaneous accesses; achieving high thread throughput requires DRAM's parallel structure to be well utilized. If all simultaneously executing threads pile onto one channel, both sides of the bargain collapse together.

---

## 6.3 — Thread Coarsening

### 1. The default, and what hardware does when it cannot honor it

Everything before this point followed one default: parallelize at the **finest grain** — one thread per output element, one block per tile. Finest-grain parallelism maximizes the hardware's freedom to schedule.

**Correction to my first mental model:** when hardware runs out of resources, it does not "make threads do more work" or redistribute anything. Threads always execute exactly the code they were given. What hardware actually does is **serialize block scheduling**: if an SM has resources (registers, shared memory, thread slots) for only 4 resident blocks and the grid has 40, the SM runs them in sequential batches — 4 at a time, finish, next 4. Each block still does its same small job; they just stop being simultaneous.

### 2. The price of fine-grain parallelism — and when it gets charged

Fine-grain parallelism is not free. In the tiled matmul, Block(0,0) and Block(0,1) load the **identical M tile** (Fig 6.11) — that is redundant global memory traffic, a price paid for making the blocks independent.

Whether that price costs *wall-clock time* depends entirely on scheduling:

- Blocks genuinely running **in parallel** on different SMs: the redundant loads happen simultaneously. Wasted bandwidth, but the time cost is hidden by the parallelism.
- Blocks **serialized** by the hardware (not enough resources to run them concurrently): the same redundant load is now paid **again, sequentially** — it lands directly on the critical path as real added time.

That is the trigger condition: coarsening pays off precisely when the parallelism it sacrifices **was going to be serialized anyway**, so the price of that parallelism (the redundancy) was being charged for nothing in return.

### 3. The move

**Thread coarsening:** the programmer manually assigns multiple units of work to each thread — one thread computes several output elements (the coarsening factor) instead of one. In the tiled matmul: merge the work of adjacent blocks so one block loads the shared M tile **once** and produces multiple blocks' worth of output from it. The redundant load does not get hidden — it gets **eliminated**.

### 4. The costs (why it is never applied blindly)

Traced through the resource budget:

- **Shared memory per block: unchanged.** The merged block holds one copy of the tile, same size as before — that single copy now just serves more output. (This is the half I got right when quizzed.)
- **Registers per thread: increased.** Each thread now keeps multiple live accumulators at once (one `Pvalue` per output element it owns) plus the extra indexing to track them. (This is the half I hesitated on.) The SM's total register file is fixed, so higher per-thread register usage means fewer threads and blocks fit resident — **occupancy drops**.

The two failure modes of over-coarsening:

1. **Sacrificing parallelism the hardware could actually use.** If the SMs had room to run all the fine-grained blocks concurrently, merging them throws away real simultaneous execution and gains nothing — the redundancy it eliminates was already time-hidden.
2. **Register pressure eating the gain.** Push the coarsening factor high enough and the occupancy loss (fewer resident warps → worse latency hiding, per 6.2) outweighs the traffic saved.

So coarsening is a targeted trade — redundant-work elimination purchased with occupancy — applied when profiling or reasoning shows the fine-grain version's blocks were serializing. It is the first optimization in the book where the programmer deliberately does the *opposite* of "more parallelism."

---

## Exercises — answers and methods

### Ex 1 — Corner-turning kernel (Fig 6.4: B stored column-major)

Full kernel = standard tiled matmul with the one-line `Nds` change from section 6.1.5. Verified by complete hand-trace (N=2, TILE_WIDTH=2 → one block, four threads, one phase):

1. Pick **asymmetric** test data: A = [[1,2],[3,4]] row-major flat `[1,2,3,4]`; B = [[6,1],[2,9]] column-major flat `[6,2,1,9]`. Build decode tables from the formulas first, then draw grids from the tables.
2. **Respect `__syncthreads()` in the trace order:** do the load step for ALL four threads first, then the compute step for all four. A thread's compute loop reads slots that *other* threads loaded — tracing one thread end-to-end before the others is structurally wrong, and the barrier is the reason.
3. Compute each `Pvalue`, write to `P[row*N + col]` with full numeric substitution.
4. Cross-check against hand-computed A×B. My first pass disagreed at one cell (kernel said 8, hand math said 10) — the mismatch traced straight back to one wrongly computed value in one `Nds` slot, demonstrating why the cross-check exists. There was also a `P[row*N+col]` arithmetic slip (wrote P[1] where 1*2+0 = 2). Final verified: `P[0]=10, P[1]=19, P[2]=26, P[3]=39`, i.e. `P = [[10, 19], [26, 39]]`, matching hand math exactly.

Lesson repeated throughout: substitute numbers immediately; never reuse symbolic labels loosely.

### Ex 2 — BLOCK_SIZE values that fully avoid uncoalesced access

- Coalescing is judged **per warp**: 32 threads grouped by linear index `threadIdx.y * blockDim.x + threadIdx.x`, taken 32 at a time.
- Requirement: warp boundaries must align with tile-row boundaries — no row split across two warps. That holds when **BLOCK_SIZE divides 32 evenly, or 32 divides BLOCK_SIZE evenly**. (My first guess, "divisible by 32," was one direction only — BLOCK_SIZE=16 also works: two full 16-thread rows exactly fill one warp.)
- Answer: **powers of 2**, practically **1, 2, 4, 8, 16, 32** for square blocks (32×32 = 1024 threads is the hardware block-size ceiling, so nothing larger is legal for a square tile).
- Caveat: alignment is necessary, not sufficient — the address formula must still place `tx` on the +1 term.

### Ex 3 — Coalesced / uncoalesced / not-applicable classification

Rules: `__shared__` array → **not applicable**, always. Global array → find the multiplier on the `threadIdx.x`-dependent term: implicit ×1 → coalesced; ×4, ×8, ×N → uncoalesced.

| Item | Access | Verdict |
|---|---|---|
| a | `a[i]`, where i = blockIdx.x*blockDim.x + tx | coalesced |
| b | `a_s[tx]` | N.A. (shared) |
| c | `b[j*blockDim.x*gridDim.x + i]` — the j term is tx-independent | coalesced |
| d | `c[i*4 + j]` — ×4 stride on the tx term | uncoalesced |
| e | `bc_s[...]` | N.A. (shared) |
| f | `a_s[tx]` | N.A. (shared) |
| g | `d[i+8]` | coalesced |
| h | `bc_s[tx*4]` | N.A. (shared) |
| i | `e[i*8]` — ×8 stride on the tx term | uncoalesced |

(My slips: swapped c and d, and called i coalesced — both from not isolating the multiplier. The multiplier IS the test.)

### Ex 4 — Arithmetic intensity (OP/B ratios)

Ops per output element = `2N` always (N multiplies + N adds); no optimization changes the math, only the byte side. Floats are 4 bytes.

| Kernel | Reuse factor | Bytes / output element | OP/B |
|---|---|---|---|
| (a) Naive (Ch 3) | 1 | N iterations × 2 floats × 4 B = `8N` | 2N / 8N = **0.25** |
| (b) Tiled 32×32 (Ch 5) | 32 | 8N / 32 = `N/4` | 2N / (N/4) = **8** |
| (c) Tiled 32×32 + coarsening ×4 | 32 × 4 = 128 | 8N / 128 = `N/16` | 2N / (N/16) = **32** |

Two takeaways:

1. `N` cancels in every case — the ratios are **constant regardless of matrix size**. The naive kernel is memory-bound at 0.25 OP/B forever; no matrix is big enough to fix it. The diagnosis is structural, not a scale problem.
2. The progression **0.25 → 8 → 32** (128× total) comes from one single lever applied at two scopes: **eliminating redundant reloads of the same global-memory value** — within a tile (tiling, ×32) and across blocks (coarsening, ×4). That lever is the throughline of the entire chapter.

---

## The chapter in four sentences

DRAM's analog sensing makes every access slow, so hardware senses whole rows at once (bursts) and you must arrange thread addresses to land inside them (coalescing — `tx` on the +1 slot of the address formula, relative to how the data is *physically stored*). When the algorithm's traversal direction and the storage layout disagree (column-major, transposed operands), corner-turn: load in the storage-friendly direction, absorb the reorientation in shared memory where it is free. Even coalesced accesses leave the memory wire idle unless many independent requests overlap across banks and channels — which requires high occupancy and interleaved data distribution. And when redundant loads across blocks would get serialized anyway, coarsen threads to eliminate the redundancy — paying with register pressure, so only when the parallelism you sacrifice was going to be serialized regardless.
