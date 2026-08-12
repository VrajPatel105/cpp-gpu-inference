# Quantization Methods Comparison: LLM.int8() vs SmoothQuant vs GPTQ vs AWQ

*This README was written using Claude.*

## 1. LLM.int8()

**What it quantizes:** both weights and activations, but not uniformly -- a small set of outlier
activation channels stay in FP16, everything else (99.9%+) is quantized to INT8.

**Core mechanism -- mixed-precision decomposition:** identify feature dimensions (channels) where
activation magnitude exceeds a threshold (`alpha = 6.0`). Split the matmul in two: outlier channels
multiply in full FP16, the rest quantize via vector-wise quantization (separate scale per row of
activations, per column of weights) and multiply in INT8. Add the two results together.

**Why it's needed:** vector-wise quantization alone works up to ~2.7B parameters, but past 6.7B,
systematic outlier features emerge in every layer -- specific channels with large magnitude across
almost every token. Since these are column-specific and vector-wise quantization scales per row,
there's no row to isolate the outlier to.

**Cost:** very low overhead (~0.1% extra memory) since only ~7 channels need the FP16 path even at
13B+ parameters.

**Precision mix:** FP16 + INT8 in the same forward pass.

## 2. SmoothQuant

**What it quantizes:** both weights and activations, uniformly in INT8 -- no mixed precision.

**Core mechanism -- migration via per-channel rescaling:** instead of separating outlier channels
out, mathematically move the quantization difficulty from activations to weights. For each channel,
compute a smoothing factor `s_j = max|X_j|^alpha / max|W_j|^(1-alpha)`, divide that channel's
activations by `s_j`, multiply the matching weight channel by `s_j`. The product `X * W` is
unchanged, but activations become easier to quantize (weights absorb some of the difficulty they can
afford, since weights don't have the outlier problem to begin with).

**Why it's needed:** avoids the complexity of physically splitting the matmul (LLM.int8()'s
approach) -- everything can be quantized in plain INT8 with one calibration pass.

**Key parameter:** migration strength `alpha` (default 0.5, balances difficulty evenly between
activations and weights; push higher for models with more extreme activation outliers).

**Cost:** low -- one calibration pass to compute per-channel maxes, no retraining.

**Precision mix:** uniform INT8 throughout.

## 3. GPTQ

**What it quantizes:** weights only. Activations are left untouched.

**Core mechanism -- Hessian-based per-weight compensation:** builds on Optimal Brain Quantization
(OBQ). Quantize one weight at a time; after rounding introduces error, use the layer's Hessian
(second-order sensitivity information) to compute the mathematically optimal adjustment to the
remaining not-yet-quantized weights, so the row's overall output stays as close as possible to the
original. Repeat until the whole row is quantized. GPTQ makes this scale to billion-parameter models
via three speed tricks: fixed left-to-right quantization order (instead of picking the "best" weight
each step), Cholesky decomposition for stable reusable Hessian-inverse updates, and batched updates
across blocks of columns instead of one weight at a time.

**Why it's needed:** the most precise of the four methods -- directly optimizes each weight's
rounding rather than relying on channel-level heuristics.

**Cost:** the most expensive of the four -- still requires real compute (roughly 4 GPU hours for a
GPT-3-scale model), though vastly cheaper than OBQ's original unscaled approach, which never ran on
large models at all.

**Precision mix:** weights quantized (commonly down to INT4), activations stay full precision.

## 4. AWQ

**What it quantizes:** weights only, same as GPTQ.

**Core mechanism -- activation-aware channel protection:** observes that a weight's contribution to
output error depends not just on its own magnitude but on the magnitude of the activation it
multiplies against. Weight channels aligned with large-magnitude ("salient") activation channels
matter disproportionately more. Instead of optimizing every weight (GPTQ's approach), AWQ identifies
salient channels via calibration, scales those weight channels up by a factor `s` before quantizing
(so they land on a less error-prone part of the rounding grid), quantizes normally, then scales the
result back down by `s` afterward. The scaling factor `s` is chosen via a small, cheap grid search
that measures actual output error on calibration data, rather than GPTQ's Hessian optimization or
SmoothQuant's closed-form formula.

**Why it's needed:** gets comparable accuracy protection to GPTQ for a fraction of the compute cost
-- no per-weight optimization loop, no Hessian computation.

**Cost:** low -- grid search over a handful of candidate scale values, much cheaper than GPTQ.

**Precision mix:** weights quantized (commonly INT4), activations stay full precision.

## Side-by-side summary

| | LLM.int8() | SmoothQuant | GPTQ | AWQ |
|---|---|---|---|---|
| Quantizes activations? | Yes (except outliers) | Yes (all, smoothed) | No | No |
| Quantizes weights? | Yes | Yes | Yes | Yes |
| Outlier strategy | Physically separate into FP16 | Rescale to migrate difficulty to weights | N/A (weight-only, no activation outlier issue) | Protect weight channels aligned with activation outliers |
| Optimization cost | Low | Low | High (Hessian-based) | Low (cheap search) |
| Precision mix | FP16 + INT8 | Uniform INT8 | Weights INT4/INT8, activations full precision | Weights INT4/INT8, activations full precision |
| Typical bit-width | INT8 | INT8 | INT4 (or INT8) | INT4 (or INT8) |
| Needs calibration data? | No (threshold-based) | Yes (per-channel activation stats) | Yes (Hessian estimation) | Yes (activation magnitude + scale search) |

## The throughline

Naive INT8 fails on LLMs because of systematic activation outliers (LLM.int8()'s core finding). The
four methods respond to that same finding four different ways: physically separate the outliers
(LLM.int8()), mathematically rebalance them away (SmoothQuant), abandon activation quantization and
precisely optimize every weight instead (GPTQ), or abandon activation quantization and cheaply
protect just the weights that matter most (AWQ). Each subsequent method trades some precision or
generality for a cheaper, more practical implementation.
