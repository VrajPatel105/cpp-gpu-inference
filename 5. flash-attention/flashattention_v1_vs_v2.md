# FlashAttention-1 vs FlashAttention-2

FA-2 does not change the underlying math of FA-1 — the online softmax
recurrence (running max `m`, running sum `l`, running output `O`) is
identical, and produces exactly the same numerical result. FA-2 is a
**scheduling / systems optimization** on top of the same algorithm: it
restructures *when* and *how* operations happen on the GPU to reduce
overhead that FA-1's original loop structure left on the table.

Five changes, in order:

---

## 1. Loop order swapped (K/V outer → Q outer)

**FA-1 (Algorithm 1):**
```
for j in range(T_c):        # outer: K/V blocks
    load K_j, V_j
    for i in range(T_r):    # inner: Q blocks
        load Q_i, O_i, l_i, m_i      # from HBM
        ... compute, rescale, merge ...
        write O_i, l_i, m_i          # back to HBM
```

**FA-2:**
```
for i in range(T_r):        # outer: Q blocks
    load Q_i
    init O_i, l_i, m_i in SRAM
    for j in range(T_c):    # inner: K/V blocks
        load K_j, V_j
        ... compute, rescale, merge ...
    write O_i                        # once, after inner loop
```

**Why it matters:** In FA-1, a Q block's running state `(O, l, m)` has
to leave SRAM and go back to HBM every time the outer loop moves to a
new K/V block — then get reloaded next time that Q block comes back
around. That's `T_c` HBM round-trips of pure bookkeeping state, per Q
row-block. In FA-2, the Q block and its running state stay resident in
SRAM for the *entire* inner loop over K/V, so nothing round-trips to
HBM until the block is completely finished — one write instead of `T_c`.

---

## 2. Deferred normalization

**FA-1:** at every single block, fully normalizes: computes the
combined old+new contribution and immediately divides by `l_i_new`
(`diag(l_i_new)^-1 (...)` in line 12). Next block then has to
*undo* that normalization (`diag(l_i) * ...`) to merge in the old
contribution correctly — a division followed by a multiplication that
cancel out, repeated every block.

**FA-2:** carries an **unnormalized** output accumulator through the
whole inner loop. Rescaling by `e^(m_old - m_new)` still happens each
block (that part is unavoidable — it's the actual online-softmax
correction), but the divide-by-`l` step happens **once**, after the
last K/V block, instead of once per block.

**Result:** for `T_c` total blocks, FA-1 does `T_c` divisions; FA-2
does 1.

---

## 3. Reduced non-matmul FLOPs (general)

"Non-matmul" = anything that isn't `QK^T` or `PV` (i.e. `rowmax`,
`exp`, rescaling multiplies, divisions). GPUs run matmul much faster
than these ops (Tensor Cores vs. general/special-function units), so
minimizing non-matmul work — even when it's a small fraction of total
FLOPs — has an outsized effect on wall-clock time.

FA-2 applies this beyond just normalization:
- Skips rescaling `O` on the very first block (guaranteed to be zero
  since `O_i` and `l_i` start at 0 — FA-1 still computes it and gets 0).
- Cleans up how the rescale factor `e^(m_old - m_new)` is computed and
  reused across the `l` and `O` updates.
- For causal attention, skips fully-masked Q-block/K-block pairs
  entirely rather than computing them and multiplying by zero.

---

## 4. Parallelization across thread blocks (sequence-length parallelism)

**FA-1:** parallelizes thread blocks over `batch × num_heads` only.
Each `(batch, head)` pair gets one thread block, which runs the full
nested loop (all of Q, all of K/V) sequentially inside itself.

**Problem:** when `batch × heads` is small (e.g. batch=1, long-context
inference), there aren't enough independent units of work to fill a
GPU's ~100+ SMs — most sit idle regardless of how long the sequence is.

**FA-2:** additionally parallelizes over the **Q row-block** dimension
(enabled by #1 — Q is now the outer loop, and different Q row-blocks
have zero data dependency on each other). Units of work become
`batch × heads × (N / B_r)` — scales with sequence length, so long
sequences keep the GPU fully occupied even at batch size 1.

---

## 5. Warp-level work partitioning

**FA-1:** within a thread block, splits the **K/V block** across
warps — each warp computes a partial `S_ij` / partial `O` contribution
from a slice of K/V. Since no single warp has the full picture, warps
must synchronize and combine partial results via shared memory before
softmax/merge can proceed.

**FA-2:** splits the **Q block** across warps instead, giving each
warp the *entire* K/V block. Since attention rows are independent of
each other, each warp computes its own Q rows' output fully
independently, start to finish — no cross-warp synchronization needed,
no partial results to combine.

---

## Throughline

FA-1 is mathematically correct but its original loop/partitioning
choices create avoidable overhead:
- extra HBM round-trips (#1)
- redundant normalization arithmetic (#2, #3)
- underused GPU parallelism at small batch size (#4)
- unnecessary warp-to-warp synchronization (#5)

FA-2 removes all five without touching the underlying online-softmax
math — same `(m, l, O)` recurrence, same final numbers, restructured
scheduling.
