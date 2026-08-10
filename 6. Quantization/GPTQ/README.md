# README

*This summary was written with the help of Claude.*

# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
*Frantar, Ashkboos, Hoefler, Alistarh*

## 1. What it is

GPTQ is **weight-only** quantization — it never touches activations at all, unlike LLM.int8()
(which separates outlier activation dimensions into FP16) or SmoothQuant (which rescales
activations to be smoother). GPTQ instead treats quantizing each layer's weights as a precise
optimization problem: given that quantization *will* introduce rounding error, which specific way
of rounding minimizes the damage to that layer's actual output? It's powerful enough to quantize a
model the size of GPT-3 (~350GB) in about 4 GPU hours.

## 2. Built on Optimal Brain Quantization (OBQ)

GPTQ's core idea isn't new — it's a scaled-up, re-engineered version of an older method called
**Optimal Brain Quantization**, itself descended from earlier work on pruning (Optimal Brain
Damage/Surgeon).

**OBQ's process, one row of a weight matrix at a time:**
1. Quantize one weight.
2. That introduces some rounding error (original value − quantized value).
3. Instead of leaving that error sitting there, **adjust all the other not-yet-quantized weights in
   the row** to compensate — so the row's overall output stays as close as possible to the
   pre-quantization output.
4. Repeat for the next weight, compensating whatever's left, until the whole row is quantized.

**Why the Hessian is needed:** to know exactly how much to nudge the remaining weights to
compensate, you need to know how sensitive the layer's output is to each weight, and how the
weights interact with each other. That sensitivity/interaction information is the Hessian matrix
(second-order derivatives of the output with respect to the weights). OBQ uses the Hessian's inverse
to compute the mathematically optimal compensation at each step.

**Why OBQ never scaled:** redoing this compensation computation (including a fresh matrix inversion)
for every single weight, one at a time, is far too slow for billion-parameter models. GPTQ's entire
contribution is making this same idea computationally tractable at scale — not changing the math,
but re-engineering how it's computed.

## 3. GPTQ's three tricks over OBQ

**Trick 1 — Fixed quantization order (drop the greedy choice).**
OBQ greedily picks, at every step, whichever remaining weight currently has the least error —
recomputed after every single weight, which is inherently sequential and slow. GPTQ's authors found
that for large models, a simple **fixed order** (e.g. left to right, column by column) gives nearly
identical quality to OBQ's careful greedy ordering. Knowing the order in advance is what unlocks
Trick 2.

**Trick 2 — Lazy batch updates.**
OBQ updates *every* remaining weight in the row immediately after quantizing one weight — many tiny,
slow operations. GPTQ processes weights in **blocks** (e.g. 128 columns at a time): it applies
updates within the current block as it goes, but *delays* propagating changes to everything outside
the block. Once a block finishes, it applies **one large batched update** to all remaining columns
at once. Big batched matrix operations run far faster on GPUs than many small sequential ones — this
is a systems-efficiency change, not a change to the underlying math.

**Trick 3 — Cholesky decomposition (numerical stability + reuse).**
OBQ's compensation step needs the Hessian's inverse, recomputed at every single weight — slow, and
numerically unstable at scale (floating-point errors compound). GPTQ instead computes the **Cholesky
decomposition** of the Hessian inverse **once, upfront**. Cholesky decomposition breaks a matrix into
a triangular form (a "square root" of the matrix). Once computed, the exact compensation coefficient
for each weight can be read off **sequentially, row by row of the Cholesky factor**, with no further
re-inversion needed mid-algorithm.

## 4. Toy example — the compensation step, concretely

Row of weights: `w = [2.0, 1.0, 0.5]`, with a (toy) Hessian:

```
H = [ 4   2   1 ]
    [ 2   3   1 ]
    [ 1   1   2 ]
```

**Quantize `w[0]`:** say rounding takes `2.0 → 1.8`. Error introduced: `δ = 2.0 - 1.8 = 0.2`.

**Compensate the rest:** using the OBQ update

$$
\delta w_{\text{remaining}} = -\frac{\delta}{[H^{-1}]_{00}} \cdot [H^{-1}]_{\text{remaining}, 0}
$$

`w[1]` and `w[2]` both shift slightly, by an amount computed directly from the inverse Hessian,
specifically to cancel out the damage that quantizing `w[0]` caused to the row's output.

**Repeat:** quantize `w[1]` (now slightly adjusted), measure its new error, compensate `w[2]`, and so
on until the row is fully quantized.

At real scale (thousands of weights per row instead of 3), this is exactly where Cholesky (avoids
re-inverting a huge matrix at every step) and lazy batching (avoids thousands of tiny sequential
updates) make the difference between "4 GPU hours" and "computationally infeasible."

## 5. The full GPTQ loop

1. Run calibration data through the model; compute the Hessian `H` for each layer's weights
   (approximated from that layer's input activations, roughly `H ≈ 2·XᵀX`)
2. Compute the Cholesky decomposition of `H⁻¹` once, upfront, for that layer
3. Process the weight matrix in fixed left-to-right column order, in blocks (e.g. 128 columns)
4. Within each block: quantize each weight, use the precomputed Cholesky factor to cheaply compute
   the exact compensation for the rest of the block
5. After a block finishes: batch-update all remaining not-yet-processed columns at once
6. Repeat block by block until the whole matrix is quantized

## 6. One-line summary

GPTQ's math (Hessian-guided compensation) is identical to OBQ's — the entire contribution is making
that math run at billion-parameter scale, via a fixed order (removes the sequential bottleneck of
greedy selection), lazy batching (turns many tiny updates into large fast matmuls), and Cholesky
reformulation (removes the need for repeated, unstable matrix inversion).
