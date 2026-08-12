# SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
*This was written by claude :)*

## 1. The core asymmetry: weights vs. activations

**Weights (W)** — learned parameters, static after training. Their distribution can be computed
offline, once, with no data run through the model. Empirically well-behaved, no outlier problem.

**Activations (X)** — the actual tensor of values flowing between layers at inference time.
Dynamic — depend on the specific input. This is where the systematic outlier problem (from
LLM.int8()) lives: a small number of feature dimensions consistently reach large magnitudes across
almost every token, once models are large enough.

**Where the outliers come from:** not randomness, not the input being "unknown" — they're an
emergent side effect of training. Certain hidden dimensions become load-bearing information channels
(tied to things like LayerNorm's learned scale parameters and high-frequency tokens), and because the
network relies on them heavily and consistently, their values grow large and stay large. Nobody
designed this; it's what gradient descent converged to as useful structure.

**The asymmetry in one line:** weights are easy to quantize because they're static and well-behaved;
activations are hard because they're dynamic and carry the outlier problem. Weights don't have an
outlier problem at all — only activations do.

## 2. LLM.int8()'s approach vs. SmoothQuant's approach

LLM.int8() handles the activation outlier problem by **physically separating** outlier feature
dimensions into their own FP16 matmul (mixed-precision decomposition), while quantizing the rest
normally.

SmoothQuant asks a different question: what if we didn't need to separate anything at all, and could
quantize the *entire* activation and weight tensor in plain INT8? Its answer: **mathematically
migrate the quantization difficulty from activations to weights**, since weights have "room to
spare" and activations don't.

## 3. The smoothing mechanism

For each channel `j`, compute a smoothing factor:

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
$$

Then rescale that channel on both sides — divide activations by `s_j`, multiply weights by `s_j`:

$$
\hat{X} = X \cdot \text{diag}(s)^{-1}, \qquad \hat{W} = \text{diag}(s) \cdot W
$$

The math is mathematically unchanged: `(X/s) \cdot (s \cdot W) = X \cdot W`. Only *where the
quantization difficulty sits* moves — activations get smoother (easier to quantize), weights absorb
a bit more difficulty (but can afford it, since they were easy to begin with).

**The migration strength hyperparameter `α` controls the split:**
- `α = 1` → pushes all difficulty onto weights (activations become trivial, weights may become extreme)
- `α = 0` → no migration at all (activations stay exactly as hard as before)
- `α = 0.5` → balances both sides roughly equally — found empirically to work well for most models
- Models with especially extreme activation outliers (e.g. some OPT/BLOOM variants) benefit from
  nudging `α` higher, trading harder-to-quantize weights for much easier activations.

## 4. Worked example

Channel `j` has `max|X_j| = 8`, `max|W_j| = 0.5`, `α = 0.5`:

$$
s_j = \frac{8^{0.5}}{0.5^{0.5}} = \frac{2.8284}{0.7071} \approx 4.0
$$

- Activation max: `8 / 4 = 2` → much gentler quantization target
- Weight max: `0.5 \times 4 = 2` → grew, but note it lands on the **same value (2)** as the
  activation — not a coincidence, that's exactly what `α = 0.5` does: bring both sides to an equal
  footing rather than favoring either one.

## 5. Result

Because the rescaling is exact (mathematically identity-preserving) and computed once per channel
ahead of time, SmoothQuant lets you quantize both weights and activations to INT8 using a single,
simple quantization scheme — no need for outlier-specific FP16 decomposition like LLM.int8(). It's a
post-training method (PTQ) — no retraining required, just a calibration pass to compute the `X_j`
maxes per channel before picking `s_j`.

## 6. How it compares to LLM.int8()

| | LLM.int8() | SmoothQuant |
|---|---|---|
| Outlier handling | Physically separate outlier columns into FP16 | Mathematically rescale so nothing needs separating |
| Precision mix | Mixed (FP16 + INT8 in the same matmul) | Uniform INT8 throughout |
| Extra cost | ~0.1% memory for outlier FP16 path | One-time per-channel scale computation (calibration) |
| Key parameter | Outlier threshold `α = 6.0` (magnitude cutoff) | Migration strength `α = 0.5` (unrelated meaning, same symbol) |

Note: both papers use the symbol `α` for a *different* hyperparameter — worth not conflating the two
when comparing notes later.
