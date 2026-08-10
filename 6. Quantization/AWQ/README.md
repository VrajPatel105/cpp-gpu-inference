# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

*This README was written using Claude.*

## 1. What's a "channel"?

A channel = one specific index position along the hidden/feature dimension. In a `[seq_len,
hidden_dim]` activation matrix, one channel = one column. In a weight matrix `[hidden_dim,
output_dim]`, it's the matching row (input side) or column (output side). Since the hidden dimension
size stays constant through the whole residual stream (embedding -> every layer -> final output),
"channel 47" refers to the same coordinate persisting across the entire network -- which is exactly
why outlier phenomena can be "the same channel across almost every layer." Not a named architectural
component like an attention head -- just a fixed coordinate along the feature axis, threaded through
every weight matrix and activation tensor.

## 2. The core observation

In a matmul, a weight's actual contribution to the output is `weight x activation`. If a weight sits
in a channel that gets multiplied by a large-magnitude activation, even a small rounding error in
that weight gets amplified by the activation it's paired with. So a weight's importance for output
accuracy isn't just about its own size -- it's about its size combined with what it multiplies against.

**Consequence:** weight channels that line up with large-magnitude ("outlier") activation channels
matter disproportionately more for output error. These are called **salient** weight channels.

## 3. Why not just use GPTQ?

GPTQ optimizes every single weight's rounding via Hessian-based compensation -- expensive but precise.
AWQ's pitch: you don't need that much machinery. Protecting just the *salient channels* (identified
cheaply via activation magnitude) gets comparable accuracy for far less compute -- no per-weight
Hessian optimization loop at all.

## 4. The mechanism

1. Run a small calibration set through the model; measure each channel's typical activation magnitude.
2. Flag channels with consistently large activation magnitude as salient.
3. For salient weight channels only: scale the weights **up** by a factor `s` before quantizing.
4. Quantize normally (round to the integer grid).
5. After the matmul, scale the result back **down** by `s` to undo the scaling:

$$
Q(w \cdot s) \cdot \frac{x}{s} \approx w \cdot x
$$

Mathematically unchanged -- same core trick as SmoothQuant's per-channel rescaling, applied here to
protect specific important weight channels rather than to globally rebalance activations vs. weights.

**Why scaling up before rounding helps:** rounding error is roughly a fixed absolute amount relative
to the grid spacing, so small values suffer large *relative* error (e.g. 0.4 -> 0 is 100% relative
error). Scale the weight up first (say x4, so 0.4 becomes 1.6), and rounding to 2 is a much smaller
relative error. Dividing by 4 afterward recovers the correct output scale.

## 5. How `s` is chosen

Unlike SmoothQuant's closed-form `alpha`-based formula, AWQ does a small, cheap **grid search**: try a
handful of candidate `s` values, measure actual output error on calibration data for each, keep
whichever minimizes it. Lightweight compared to GPTQ's per-weight Hessian computation, but still
empirically tuned rather than derived from a formula.

## 6. Where AWQ sits relative to the other three methods

| Method | What's quantized | Outlier/error strategy | Cost |
|---|---|---|---|
| LLM.int8() | Weights + activations (mixed) | Physically separate outlier activation channels into FP16 | Low (~0.1% extra memory) |
| SmoothQuant | Weights + activations (uniform INT8) | Mathematically rebalance activation/weight difficulty via `alpha`-controlled per-channel scaling | Low (one-time calibration) |
| GPTQ | Weights only | Per-weight Hessian-based compensation, quantize one weight at a time | High (Cholesky-based optimization, but fast enough for GPT-3 in ~4 GPU hours) |
| AWQ | Weights only | Protect salient channels (activation-magnitude-guided) via cheap scale search | Low (grid search, no Hessian) |

**One-line summary:** AWQ quantizes only weights, uses activation magnitude (not weight magnitude) to
decide which weight channels matter most, and protects those cheaply via scaling -- trading some of
GPTQ's precision for a much lighter compute cost, while still empirically getting comparable results.
