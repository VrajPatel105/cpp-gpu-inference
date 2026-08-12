# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
*I have used claude to write this entire summary*

## 1. The problem: memory bottleneck

Large transformers need huge memory just to hold their weights. A 7B parameter model in FP32
(4 bytes/param) needs ~28GB just to load — before you've run a single token through it.
Quantizing to INT8 (1 byte/param) cuts that to ~7GB, a 4x reduction. Feed-forward and attention
projection layers account for ~95% of parameters and 65-85% of compute in large transformers, so
quantizing those layers is where the memory win actually comes from.

Prior 8-bit quantization methods worked fine up to ~350M parameters but degraded performance and
required post-training tuning. Multi-billion-parameter quantization without degradation was an open
problem — that's what this paper solves.

## 2. Two quantization data types (background)

**Absmax quantization** (symmetric): scale by the max absolute value in the tensor.

$$
\text{scale} = \frac{127}{\max(|x|)}, \qquad x_{i8} = \text{round}(x \cdot \text{scale})
$$

Fast (int8 × int8 multiply, no extra terms) but wastes range on asymmetric data — e.g. ReLU outputs
(always ≥ 0) only ever use half the `[-127, 127]` range.

**Zeropoint quantization** (asymmetric): scale by the actual min-max range and shift so the full
range is used, even for asymmetric data. More precise on asymmetric distributions, but multiplying
two zeropoint-quantized numbers expands into 4 terms (`AB + A·zp_b + B·zp_a + zp_a·zp_b`) instead
of a plain int8×int8 op — 4x the multiply cost without a fused hardware instruction. This is why
it's "rarely used in practice despite being more accurate."

LLM.int8() itself is built on **absmax**, not zeropoint — speed/hardware-compatibility won out.

## 3. Vector-wise quantization (handles the general case, works up to 2.7B params)

Instead of one scale for an entire tensor, give **each row of X** (each token's feature vector)
its own scale `cx`, and **each column of W** (each output neuron's weights) its own scale `cw`.
Quantize, do the matmul entirely in int8 (accumulate in int32), then dequantize the *whole output*
in one shot using the outer product of the two scale vectors:

$$
C_{f16} \approx \frac{1}{c_x \otimes c_w} \cdot C_{i32}
$$

This is cheap because the scale is constant across each row's dot product, so it factors out of the
sum cleanly — one elementwise divide over the output, not a per-dot-product unwind.

**Where it stops working:** past 6.7B parameters, systematic outlier features emerge in every
transformer layer — specific *feature dimensions* (columns of X) that are large in magnitude across
almost every token (row). Because vector-wise quantization scales per **row**, and the outlier lives
in specific **columns** that appear in nearly every row, there's no row you can isolate the outlier
to — giving each row its own scale doesn't help when the anomaly isn't row-specific to begin with.
An inflated row max shrinks that row's scale factor, crushing all its other values toward zero when
quantized.

## 4. Mixed-precision decomposition (handles the outlier case, extends to 175B)

Outliers are sparse but systematic: ~0.1% of feature dimensions, but zeroing them out degrades
perplexity by 600-1000% (vs. 0.1% for removing the same count of random dimensions). Since the
problem is column-specific, the fix is to physically split the matmul by column:

$$
C_{f16} \approx \underbrace{X^h_{f16}W^h_{f16}}_{h \in O} + \underbrace{S_{f16}\cdot(X^h_{i8}W^h_{i8})}_{h \notin O}
$$

- **Outlier columns** (set `O`, any feature dimension with magnitude ≥ threshold `α = 6.0`) → pulled
  out, multiplied in full **FP16**, no quantization at all.
- **Everything else** (99.9%+ of dimensions) → quantized and multiplied via vector-wise quantization
  above, then dequantized.
- The two results are **added together** for the final output.

`|O| ≤ 7` even at 13B parameters — this is why decomposition only costs ~0.1% extra memory despite
reintroducing full-precision math for those columns. There just aren't many outlier columns.

## 5. Worked example (small matrices, for intuition)

```
X = [ 0.5  -0.3   0.2 ]      W = [ 0.4   0.1 ]
    [ 0.1   0.6  -0.4 ]          [-0.2   0.5 ]
                                  [ 0.3  -0.6 ]
```

**Vector-wise:** row scales from X's per-row absmax, column scales from W's per-column absmax,
quantize both, int8 matmul, dequantize with the outer product `1/(cx ⊗ cw)`.

**Decomposition:** if column 1 of X contains an outlier (say `7.2`, above `α = 6.0`), split by
column — `X_outlier`/`W_outlier` (that one dimension) go through plain FP16 matmul; the remaining
columns (`X_normal`/`W_normal`) go through the vector-wise int8 path above. Add the two results.

## 6. Result

LLM.int8() = vector-wise quantization + mixed-precision decomposition. It's the only method in the
paper's benchmarks that tracks the 32-bit baseline perplexity all the way from 125M to 175B params
with zero degradation — plain absmax/row-wise/zeropoint all fail past 2.7B–13B. This is what made it
possible to run OPT-175B/BLOOM on a single server with consumer GPUs.

## 7. Why this matters for the roadmap

This paper *is* the required W14 build target: "Custom INT8 PTQ on PyTorch transformer" means
implementing vector-wise quantization + mixed-precision decomposition from scratch — the exact math
above — calibrating scale factors, quantizing weights/activations, and measuring the perplexity hit,
without bitsandbytes or AutoGPTQ.
