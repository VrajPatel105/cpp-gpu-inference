# Tiled Matrix Multiplication — From Scratch

A single reference that builds the tiled CUDA matmul kernel from the ground up:
**matrix multiplication → threads → tiles + k-loop**. Each layer sits on the one
below it. If a layer ever feels confusing, drop down a level — the confusion is
almost always one floor lower than where you notice it.

Running example throughout: `A` and `B` are both the 4×4 matrix filled `1..16`
(row-major), and we trace the single cell **`P[2][1] = 356`**.

---

## 0. Why tiling exists at all (the motivation)

The whole reason for this kernel is one number, and it's worth keeping in view:

- The naive matmul inner loop does `Pvalue += M[..] * N[..]` — that's **2 FLOPs**
  (one multiply, one add) for **8 bytes** read from global memory (two `float`s).
- Ratio = **2 / 8 = 0.25 FLOP/B** (the *compute-to-global-memory-access ratio*,
  a.k.a. arithmetic intensity).
- On an A100: peak bandwidth ≈ 1555 GB/s, so `1555 × 0.25 = 389 GFLOPS` achievable.
  That's only **~2%** of the A100's 19,500 GFLOPS compute peak. The math units sit
  idle ~98% of the time, starved for data. This is a **memory-bound** kernel.

You can't widen the memory "road" (1555 GB/s is fixed silicon). The only lever is
to **reduce the number of trips** — fetch each value once, then **reuse** it many
times from fast on-chip memory. That reuse is what tiling buys.

> **The chain:** `0.25 FLOP/B → ~2% of peak → road is fixed → reuse instead →
> stage tiles in shared memory → collaborative load + a barrier → reuse factor
> of TILE_WIDTH → 4 OP/B → 6220 GFLOPS.`

---

## 1. Matrix multiplication itself

**The one rule:** to fill cell `P[i][j]`, take **row `i` of A** and
**column `j` of B**, multiply them position-by-position, and sum the results.
That single number goes in the cell.

```
        A (4x4)                 B (4x4)                P = A·B
   row0:  1   2   3   4     col0..3 across       each P[i][j] =
   row1:  5   6   7   8     row0:  1  2  3  4       row i of A
 > row2:  9  10  11  12     row1:  5  6  7  8       DOT
   row3: 13  14  15  16     row2:  9 10 11 12       col j of B
                            row3: 13 14 15 16
```

**Geometric picture:** put A on the left, B on top, P on the bottom-right. Row `i`
of A slides *right*, column `j` of B slides *down*, and they collide at `P[i][j]`.
The collision IS the dot product.

```
                         B
                         |  (col 1 = 2, 6, 10, 14)
                         v
                    +---------+
        A  --->     |  P[2][1]|  =  9·2 + 10·6 + 11·10 + 12·14
 (row 2 =           |  = 356  |     = 18 + 60 + 110 + 168
  9,10,11,12)       +---------+     = 356
```

**Two properties that unlock everything above:**

1. **Every cell is computed the same way** — its own row `i` dotted with its own
   column `j`. 16 cells → 16 independent dot products.
2. **No cell depends on any other.** `P[2][1]` needs nothing from `P[0][0]`. They
   could all be computed *simultaneously* by different workers without conflict.

Property 2 is the entire reason matmul runs well on a GPU.

---

## 2. Threads — one thread per P cell

Because the 16 cells are independent, hand each one to its own thread. Every thread
runs the **same kernel code**, but with its own coordinates `(row, col)`:

```
              P — one thread per cell
   +-----------+-----------+-----------+-----------+
   | row0 col0 | row0 col1 | row0 col2 | row0 col3 |
   +-----------+-----------+-----------+-----------+
   | row1 col0 | row1 col1 | row1 col2 | row1 col3 |
   +-----------+-----------+-----------+-----------+
   | row2 col0 |[row2 col1]| row2 col2 | row2 col3 |   <- this thread → 356
   +-----------+-----------+-----------+-----------+
   | row3 col0 | row3 col1 | row3 col2 | row3 col3 |
   +-----------+-----------+-----------+-----------+

   16 cells → 16 threads. Same instructions, different badge, different output.
```

**The badge → cell mapping (this is what `Row`/`Col` are for):**

```c
int row = by * TILE_WIDTH + ty;   // which row of P do I own
int col = bx * TILE_WIDTH + tx;   // which col of P do I own
```

- `by, ty` etc. are the thread's **identity badge** — *who am I in the grid*.
- `× TILE_WIDTH` turns the block-coordinate into a **jump** (skip whole blocks),
  then the thread-coordinate (`ty`/`tx`) is the **offset** inside the block.
- Rule to burn in: **y → row (vertical), x → col (horizontal).** It feels backwards
  because we say "x,y" but index "row,col" — that inversion is a classic bug source.

**Where does the block shape come from?** Not the kernel. The *host launch* sets it:

```c
dim3 blockDim(TILE_WIDTH, TILE_WIDTH);          // TILE_WIDTH² threads, one per tile cell
dim3 gridDim(N / TILE_WIDTH, N / TILE_WIDTH);   // enough blocks to cover all of P
matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
```

The kernel quietly *assumes* it was launched with a `TILE_WIDTH × TILE_WIDTH` block.
That assumption is the invisible handshake between host and device.

> **Gotcha (this is the doorway to §5.5):** `N / TILE_WIDTH` is integer division.
> It only works when `N` is a clean multiple of `TILE_WIDTH`. If `N = 5,
> TILE_WIDTH = 2`, then `5/2 = 2` blocks cover only 4 columns — column 5 is computed
> by nobody. The fix is the ceiling trick `(N + TILE_WIDTH - 1) / TILE_WIDTH`, plus
> boundary checks inside the kernel.

---

## 3. Tiles + the k-loop — chunking the dot product

A thread's dot product is **N long** (a full row of A · full column of B = 4 values
here). But a shared-memory tile only holds **TILE_WIDTH** values at a time (2 here).
So the thread can't do all 4 products at once — it does them in
**`N / TILE_WIDTH` phases**, TILE_WIDTH products per phase, pulling each chunk from
the fast shared tile.

```
   full dot product for P[2][1]  (length 4):
       9·2  +  10·6   |   11·10  +  12·14
       \___ phase 0 __/   \___ phase 1 ___/

   phase 0:  tile holds row→ 9,10   col↓ 2,6
             Pvalue += 9·2   (k=0)
             Pvalue += 10·6  (k=1)        Pvalue = 78

   phase 1:  tile holds row→ 11,12  col↓ 10,14
             Pvalue += 11·10 (k=0)
             Pvalue += 12·14 (k=1)        Pvalue = 78 + 278 = 356  ✓
```

**Two loops, two different jobs — don't merge them:**

- **Outer `ph` loop** = *time*. Steps to the next tile/phase (phase 0 → phase 1).
  Each iteration loads a new chunk. `phases = N / TILE_WIDTH`.
- **Inner `k` loop** = the TILE_WIDTH products *within* one phase. Does **zero
  loading** — it reads the already-filled shared tile.

> Phases are **time**, not space. Blocks in the grid are space (which region of P).
> One thread marches through phases *sequentially* to finish *its own* one cell.

**Collaborative loading (why the tile fills up):** each thread loads exactly **one**
M element and **one** N element into the shared tile (its own `[ty][tx]` slot). The
*other* slots are filled by the *other threads in the block*. After the barrier, the
whole tile is full and **every thread can read every cell** — that's the reuse.

```
   shared Mds tile (filled by the whole block, then read by the whole block)
   +--------+--------+
   | [0][0] | [0][1] |   each cell written by ONE thread (its [ty][tx])
   +--------+--------+   after __syncthreads(), ALL threads may read ALL cells
   | [1][0] | [1][1] |
   +--------+--------+
```

**The reuse payoff (the factor of TILE_WIDTH):** a tile costs `TILE_WIDTH²` global
loads to fill but is read `TILE_WIDTH² × TILE_WIDTH` times during the k-loops →
each fetched value is reused **TILE_WIDTH** times. So `0.25 FLOP/B × TILE_WIDTH`.
With TILE_WIDTH = 16: `0.25 × 16 = 4 OP/B`, lifting the A100 ceiling from 389 to
**6220 GFLOPS** (~32% of peak).

---

## 4. The flat index (row-major addressing)

A 2D matrix is stored as a **1D array** in memory (row-major). To reach element
`[r][c]`: skip `r` full rows (`r × Width` elements), then step `c` across:

```
   flat index = r * Width + c
```

In tiling, both `r` and `c` split into a **per-phase jump** + a **per-thread offset**:

```c
// M slides RIGHT across its row → the phase jump lands on the COLUMN
Mds[ty][tx] = A[ row * N  +  (ph*TILE_WIDTH + tx) ];
//               \_row_/      \_______ column _______/
//                            ph*TILE_WIDTH = which tile this phase
//                            tx            = my offset inside the tile

// N slides DOWN its column → the phase jump lands on the ROW
Nds[ty][tx] = B[ (ph*TILE_WIDTH + ty) * N  +  col ];
//               \________ row _________/       \col fixed/
```

**The asymmetry is the whole trick:** for M the moving `ph` term is the *column*
(bare); for N it's the *row* (gets the `× N`). That's literally "M slides right,
N slides down" written in index form.

Sanity check: phase `ph=2`, thread `tx=3`, TILE_WIDTH=16 → column
`2*16 + 3 = 35`. ✓

---

## 5. The full kernel, mapped line-by-line

```c
#define TILE_WIDTH 16

__global__ void matmulKernel(float* A, float* B, float* P, int N) {

    // identity badges — who am I in the grid
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // which P cell do I own (badge × jump + offset);  y→row, x→col
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // the block's shared scratchpads (one tile of A, one of B)
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // my private running total — OUTSIDE the loop so it survives across phases
    float Pvalue = 0.0f;

    for (int ph = 0; ph < N / TILE_WIDTH; ++ph) {     // OUTER: step to next phase (time)

        // collaborative load: each thread loads its one M and one N element
        Mds[ty][tx] = A[row * N + (ph*TILE_WIDTH + tx)];   // M slides right
        Nds[ty][tx] = B[(ph*TILE_WIDTH + ty) * N + col];   // N slides down
        __syncthreads();        // BARRIER 1 (read-after-write): wait till tile is FULL

        for (int k = 0; k < TILE_WIDTH; ++k)               // INNER: products within phase
            Pvalue += Mds[ty][k] * Nds[k][tx];             // my row of Mds · my col of Nds

        __syncthreads();        // BARRIER 2 (write-after-read): wait till everyone's DONE
    }                           //          reading before the next phase overwrites tile

    P[row * N + col] = Pvalue;  // write my one finished cell home
}
```

**Why each barrier exists:**

- **Barrier 1 (after load, before read)** — *read-after-write*. Threads don't run in
  lockstep; without this, a fast thread could read a tile cell a slow thread hasn't
  filled yet → reads stale garbage. Wait until the whole tile is loaded.
- **Barrier 2 (after read, before next load)** — *write-after-read*. Without this, a
  fast thread could overwrite the tile with the next phase's data while a slow thread
  is still reading the current phase → corrupts its neighbors' dot products.

---

## 6. Mental anchors (the things that are easy to get wrong)

- **The seed:** one thread produces **one full element of P** (a whole dot product),
  NOT "one product of two numbers." If there were no sum, there'd be no k-loop.
- **`Pvalue` is private (a register), but private ≠ can't accumulate.** It's your own
  jar — only you deposit into it, but you deposit many times. Shared *input* (tiles),
  private *output* (Pvalue).
- **There is no collective sum.** 16 threads → 16 private `Pvalue`s → 16 independent
  cells. A shared accumulator would corrupt results — its absence is correct design.
- **Phases = time, blocks = space.** Don't conflate them.
- **`Pvalue` outside the loop = it survives** across phases (78 → 356). Inside the
  loop it would reset to 0 every phase and you'd keep only the last phase's work.
- **Index asymmetry:** M's moving term is the column (bare); N's is the row (× N).

---

## 7. Where to go next

- **Medium twist — break the square.** Real matmul is `A[I×K] · B[K×J] = P[I×J]`.
  You've been using one `N` for three different dimensions. Work out which `N` in
  each index is really `I` (rows of A / P), `K` (the shared/dot dimension), or `J`
  (cols of B / P). This is the cleanest test of whether the flat-index rule is truly
  yours.
- **§5.5 Boundary checks.** Handle `N` not a multiple of `TILE_WIDTH`. Watch for the
  two *separate* guards — one on the element you **load** (is this input in bounds?),
  one on the element you **write** (does my P cell exist?) — and the rule that an
  out-of-bounds thread must still **load a 0 and still call `__syncthreads()`**, never
  skip (or it corrupts neighbors and can deadlock the block).
- **Verification habit:** use `B = identity` → output must equal `A` exactly (catches
  row/col transposition instantly); then use an asymmetric `B` and hand-check one cell.
