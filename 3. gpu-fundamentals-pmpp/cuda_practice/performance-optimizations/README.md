# PMPP Chapter 6.3 — Thread Coarsening: The Coarsened Tiled Matmul Kernel

In-depth notes on the coarsened matrix multiplication kernel — every line, why it exists, why it sits where it sits, and the specific confusions that had to be untangled along the way (kept in deliberately: they mark where intuition breaks).

Companion files: the main Chapter 6 README (concepts for 6.1–6.3 + exercises) and the 6.4 optimization checklist README. This file goes deep on one thing only: the coarsened kernel itself.

---

## 1. The problem, stated precisely

Start from the verified corner-turned tiled kernel. It is correct and coalesced. But it has a structural inefficiency baked into how blocks are organized.

Picture the output matrix split into tiles, one block per tile. Block(0,0) and Block(0,1) sit side by side: different output **columns**, same output **rows**. To compute those rows, both blocks need the **exact same tile of matrix A**.

Here is the critical fact that makes this a problem, and the first confusion that had to be corrected:

**Blocks do NOT share shared memory.** Each block gets its own completely separate, private shared-memory allocation. Block 0's `Mds` and Block 1's `Mds` are two physically different pieces of on-chip memory that cannot see each other. There is no such thing as "both blocks have the same shared memory."

That privacy is exactly the problem. Block 0 loads the A-tile into *its* `Mds` from global memory. Block 1 needs the identical data — but it cannot read Block 0's `Mds`, because shared memory does not cross block boundaries. So Block 1 has no choice but to go back to slow global memory and fetch the same bytes again. **Two trips to global memory for identical data, purely because the two consumers live in different blocks.**

Whether that redundancy costs wall-clock time depends on scheduling (this is the 6.3 theory):

- If the two blocks genuinely run in parallel on different SMs, the redundant loads happen simultaneously — wasted bandwidth, but the time cost is hidden.
- If the hardware **serializes** the blocks (not enough SM resources to run them concurrently), the redundant load is paid again, sequentially — real added time on the critical path.

Coarsening pays off precisely when the parallelism it sacrifices was going to be serialized anyway.

## 2. The fix, stated precisely

Since shared memory cannot be shared *across* blocks, **move the second consumer inside the first block.** Merge `COARSE_FACTOR` adjacent blocks (along one direction — columns, here) into one block. One block, one shared memory, one global A-load, several column strips served.

Two clarifications that had to be made:

**"What if there are more blocks?"** Coarsening does not collapse the grid into one block. It merges small groups of adjacent blocks, `COARSE_FACTOR` at a time, along one direction. 8 blocks along x with `COARSE_FACTOR=2` becomes 4 blocks along x, each covering the territory of 2 old ones. 100 becomes 50. The grid stays large and parallel; each unit is just wider.

**"Is the goal to reduce the number of blocks?"** No — reducing block count is the **means**, not the purpose. The purpose is eliminating the redundant global loads. Fewer, fatter blocks is just the shape the fix takes — and it is actually a *cost* (less parallelism, more registers per thread), which is why coarsening is only worth it when that parallelism was being serialized anyway.

**Which direction do you coarsen — and is it related to major order?** Not related to major order at all. The direction is a free design choice, determined by one question: *which matrix's tile stays the same as you extend in that direction?* In C = A×B, A's tile depends only on the output **row** — walk across output columns and A's tile never changes. B's tile depends only on the output **column**. So coarsening along columns makes A the "free rider" — loadable once, reusable across all strips. If you coarsened along rows instead, the roles would flip exactly: B loads once outside the loop, A reloads inside. Same idea, mirror image. Major order (row-major vs column-major storage) is a completely separate question — it decides whether each individual load is *coalesced*, not which load goes inside the loop.

## 3. The kernel, line by line

```cuda
#define TILE_WIDTH 2
#define COARSE_FACTOR 2

__global__ void coarsenedMatrixTiling(float* A, float* B, float* P, int N){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) Pvalue[c] = 0.0f;

    for (int ph = 0; ph < N / TILE_WIDTH; ++ph) {

        // A's tile does not depend on which output column we're on,
        // so it loads exactly once per phase.
        Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];

        // Since Nds is changing values every iteration (because we have to change cols),
        // we have to get fresh values — and to get those fresh values we run this loop,
        // which simply overwrites the same Nds buffer every time.
        // (Reuse over time, not growth: Nds never gets bigger, its contents get replaced per strip.)
        for (int c = 0; c < COARSE_FACTOR; ++c) {

            int col = colStart + c * TILE_WIDTH;

            // corner-turned load, same formula as the verified 6.1 kernel,
            // just using this iteration's col instead of a fixed one.
            Nds[tx][ty] = B[col*N + (ph*TILE_WIDTH + tx)];

            __syncthreads(); // Mds and this c's Nds fully loaded before anyone computes

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }

            __syncthreads(); // everyone finished READING Nds before the next c overwrites it
        }
    }

    // delivery step: move finished results from private registers to global P
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        P[row*N + col] = Pvalue[c];
    }
}
```

### 3.1 The "strip" vocabulary

"Strip" is not CUDA terminology — just a working word for "a contiguous chunk of `TILE_WIDTH` output columns." In a 4-wide output with `TILE_WIDTH=2`, columns {0,1} are one strip, columns {2,3} another. Before coarsening: one block owns one strip. After coarsening: one block owns `COARSE_FACTOR` strips glued side by side, and the loop variable `c` is simply *which strip, of the several inside my block, am I on right now.*

### 3.2 `colStart` — the three-layer column formula

The full column formula is `col = bx*TILE_WIDTH*COARSE_FACTOR + c*TILE_WIDTH + tx`, and it is built from three stacked jumps, each answering one question:

1. `bx * TILE_WIDTH * COARSE_FACTOR` — **where does my block's entire territory begin?** Blocks now claim `COARSE_FACTOR` times more columns each, so the jump between block territories must scale by that factor, or adjacent blocks' columns would overlap and be computed twice. (First wrong attempt during derivation: `bx*TILE_WIDTH + tx + c` — the `+c` shifts by 1 instead of by a strip, producing collisions: thread tx=0's c=1 and thread tx=1's c=0 both landed on column 1. Verified broken by substituting numbers.)
2. `c * TILE_WIDTH` — **within my territory, which strip am I on?** Each strip is a full `TILE_WIDTH` wide, so moving strips means jumping a full `TILE_WIDTH`, not 1.
3. `tx` — **within that strip, which exact column is mine?** Unchanged from the original kernel; a strip is just an ordinary tile.

Precision note on the variable itself: `colStart = bx*TILE_WIDTH*COARSE_FACTOR + tx` is not "the block's starting strip" — it already bakes in `+tx`, making it **this specific thread's column within the block's first strip**. Each later strip is reached by adding `c*TILE_WIDTH` on top of that already-thread-specific anchor.

Verification (always substitute real numbers): `TILE_WIDTH=2, COARSE_FACTOR=2, bx=0` — the four (tx, c) combinations give columns 0, 1, 2, 3 with no repeats and no gaps.

### 3.3 `Pvalue[COARSE_FACTOR]` — why an array, and where it lives

One thread now owns `COARSE_FACTOR` output elements. Each is its own independent dot-product-in-progress, accumulating across the same phase loop. They cannot share one variable — they are separate sums that must not mix. Hence one accumulator slot per owned column.

**Where it lives: registers.** It is a small fixed-size local array (size known at compile time via `#define`), declared with no `__shared__` qualifier — every thread gets its own private copy; thread A's `Pvalue[0]` and thread B's `Pvalue[0]` are different physical registers, invisible to each other. This array **is** the register-pressure cost from the 6.3 theory, made physical: before coarsening one thread needed 1 accumulator register; now it needs `COARSE_FACTOR`, all alive simultaneously through the entire kernel. Fixed SM register budget → fewer resident threads → occupancy drops. This is the price tag on the whole technique.

### 3.4 The phase loop — what `ph` is (and is not)

`ph` is pure tiling, nothing to do with thread scheduling. A full dot product sums over the entire K dimension, but shared memory holds only a `TILE_WIDTH`-sized chunk at a time. So the dot product is computed in installments: phase 0 loads the first chunk of the row/column and accumulates its partial contribution; phase 1 loads the next chunk and adds its contribution; and so on for `N/TILE_WIDTH` phases. `ph` = "which installment of the summation am I on."

**Is tiling parallel?** Two axes, opposite answers:

- **Across time (the ph loop): sequential.** Phase 1 cannot start before phase 0 finishes — they reuse the same `Mds`/`Nds` buffers and `Pvalue` accumulates in order.
- **Within one phase: fully parallel.** All threads simultaneously load their piece of the tile, hit the barrier together, then all simultaneously compute.

Correct one-sentence version: *tiling is a sequence of phases, each of which is internally parallel.*

### 3.5 The load asymmetry — the heart of the kernel

Why does A's load sit outside the `c` loop while B's sits inside? Trace the addresses:

- A's load: `A[row*N + (ph*TILE_WIDTH + tx)]`. Every term — `row`, `ph`, `tx` — is independent of which output column is being handled. The address is identical for every value of `c`. There is nothing to loop over; loading it more than once per phase would fetch the same bytes again for no reason.
- B's load: `B[col*N + (ph*TILE_WIDTH + tx)]`. `col` is right there in the formula, and `col` changes with `c`. Different `c` means genuinely different data — not a redundant re-fetch, but a required fresh fetch.

The generalizable test, worth keeping as a habit: **for any load inside a coarsening loop, check whether the quantity that varies with `c` appears in the load's address expression.** Doesn't appear → identical across all `c` → hoist it outside, load once. Appears → genuinely different per `c` → it must stay inside.

Plain-language version (the corrected summary from the session): the column is the thing that changes every iteration, so it cannot be cached — it must be re-fetched fresh each time. The row is the thing that repeats, so fetch it once and reuse it; fetching it again would only produce the identical value already in hand. **The thing that changes must be re-fetched precisely because it changes; the thing that stays fixed loses nothing by being fetched once.** (The tempting inversion — "col changes, so col shouldn't have the loop" — is exactly backwards: if col's load were hoisted out and fetched once, every output would be computed against the same B data, producing row·col0 twice instead of row·col0 and row·col1. Wrong answers, not just slow ones.)

### 3.6 The Nds overwrite — reuse over time, not growth

The single most important correction of the session: **the `c` loop does not store more values in shared memory.** `Nds` is declared once, `TILE_WIDTH × TILE_WIDTH`, and never grows. Each iteration completely **overwrites** it. Same physical buffer, different contents at different moments:

```
during c=0:   Nds holds B's tile for strip 0
  (sync, overwrite)
during c=1:   the SAME buffer now holds B's tile for strip 1
  (next phase)
ph+1, c=0:    the SAME buffer again, next chunk of strip 0
```

The loop trades **time for space**: instead of allocating `COARSE_FACTOR` separate B-tiles simultaneously (more shared memory per block, which would hurt occupancy a second way), it cycles one tile through different contents sequentially.

**The Nds-vs-Pvalue asymmetry — the deciding principle.** Nds gets one reused buffer; Pvalue grew into an array. Why opposite treatments? Because Nds's old contents are **disposable** — once strip 0's contribution has been accumulated, that B data is never needed again; overwriting is free. Pvalue's contents are **running totals that must survive** across every `c` iteration and every phase until the final write. You may overwrite what you are done with; you must keep separate storage for what has to persist. That single distinction — disposable vs must-persist — decides, for every piece of data in the kernel, whether it gets one reused buffer or an array of slots.

### 3.7 The two `__syncthreads()` — two different hazards

`Nds` being one reused buffer is exactly why there are two barriers per `c` iteration, guarding two different races:

- **First sync (after the loads, before compute):** the compute step reads shared-memory slots that *other* threads wrote. Every thread's write to `Mds` and this-c's `Nds` must have landed before anyone starts reading, or a fast thread reads garbage. Same reasoning as the original kernel.
- **Second sync (after compute, before the loop advances):** a hazard the original kernel never had. Before *anyone* overwrites `Nds` with the next c's values, every thread must be finished *reading* the current c's values. Without it, a fast thread races ahead, starts loading c+1's data, and stomps on data a slower thread is still using. This hazard exists *only because* Nds is reused across iterations — it is a direct consequence of the overwrite design.

### 3.8 The delivery loop — final write

When the phase loop ends, this thread holds `COARSE_FACTOR` completed dot products in private registers. Registers are private and temporary — they vanish when the kernel ends, and nothing has touched `P` yet. The final loop is the delivery step: for each `c`, recompute the column with the identical formula used during the loads (guaranteeing result `c` lands in the column it was computed for), convert `(row, col)` to a flat address via row-major `row*N + col`, and write `Pvalue[c]` out to global memory.

In the non-coarsened kernel this was one line — one thread, one result, one write. It becomes a loop for the same reason `Pvalue` became an array: one thread, several results, several distinct destinations.

**No sync needed here:** each thread writes only its own output locations. No thread reads another's writes; no shared buffer is being overwritten. The earlier syncs existed because threads cooperated through shared `Mds`/`Nds`; this loop has no cooperation — each thread independently mails out its own finished answers.

### 3.9 The launch configuration — `gridDim` shrinks

```cuda
dim3 gridDim(N / (TILE_WIDTH * COARSE_FACTOR), N / TILE_WIDTH);
```

The x-dimension divides by an extra factor of `COARSE_FACTOR`: each block now covers that many times more output columns, so proportionally fewer blocks are needed along x. This line is the visible code-level fingerprint of "fewer parallel units, each doing more work." Forgetting to shrink it would leave blocks whose column territories overlap, computing the same outputs redundantly — the exact opposite of the intent. (The y-dimension is untouched because coarsening here is along columns only.)

## 4. Everything traced back to the 6.3 theory

Every structural feature of this kernel is one of the theory's costs/benefits made literal:

| Theory (6.3) | Code |
|---|---|
| Eliminate redundant cross-block global loads | `Mds` load sitting outside the `c` loop |
| Register pressure rises with coarsening factor | `float Pvalue[COARSE_FACTOR]` in registers |
| Fewer, larger parallel units | `gridDim.x` divided by `COARSE_FACTOR` |
| Shared memory footprint unchanged | `Mds`/`Nds` still `TILE_WIDTH × TILE_WIDTH`; Nds reused via overwrite |
| Only worth it when parallelism was serializing | The trade is explicit: parallel units spent, redundancy bought back |

And the one-question habit that generates the whole kernel structure from scratch: **for each piece of data, does it depend on which coarsened unit I'm handling, and must it persist?** Independent of `c` → hoist outside the loop, load once. Depends on `c`, disposable → one buffer, overwritten per iteration, guarded by a read-complete sync. Depends on `c`, must persist → an array with one slot per unit.

## 5. Status and next step

Kernel written, every line justified, all conceptual questions closed. Not yet done: the **hand-trace** — the same verification standard corner turning passed (where hand arithmetic caught a real wrong value in an Nds slot). Setup for the trace: `N=4, TILE_WIDTH=2, COARSE_FACTOR=2` → grid is 1×2 (one block in x, two in y), 4 threads per block, 2 phases, 2 strips per thread. Protocol, same as before: build A and B decode tables from asymmetric data, trace all threads' load steps before any compute step (respecting each barrier), keep both `Pvalue` slots per thread separate, and cross-check the final P against hand-computed A×B. Until that trace agrees with hand math, the kernel is "explained," not "verified."
