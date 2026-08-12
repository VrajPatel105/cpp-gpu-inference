# W14: INT8 Post-Training Quantization on a Custom Transformer

## Objective

This is a custom INT8 post-training quantization (PTQ) from scratch, applying it to a PyTorch transformer, and comparing the result against an established library (bitsandbytes) implementation on translation quality, perplexity, and memory footprint.

The base model is an English-to-German transformer trained from scratch (encoder-decoder, 6 blocks each, 8 attention heads, d_model 512), following the LLM.int8() approach: outlier-aware mixed-precision matrix multiplication, where a small set of activation-derived outlier channels are kept in full precision while the remaining weights and activations are quantized to INT8.

## Implementation

### Calibration

A calibration pass was run over the trained model using forward hooks to track the running maximum activation magnitude per input channel, across every `nn.Linear` layer. Per-layer outlier channels were identified using a dynamic threshold (mean plus three standard deviations of per-channel activation magnitude) rather than a fixed cutoff, since activation distributions varied meaningfully across layers. The result was stored as a dictionary mapping each layer's dotted module name to a tensor of outlier channel indices.

### QuantizedLinear

A custom `nn.Module`, `QuantizedLinear`, replaces each `nn.Linear` layer. On construction, it splits the layer's weight matrix into an outlier column set (kept in full precision) and a normal column set, which is quantized to INT8 using a per-output-neuron scale factor. At inference time, the forward pass splits the input activation the same way, quantizes the normal activation columns to INT8 dynamically (per-token scale), and computes two matrix multiplications: a full-precision path for the outlier columns and an INT8 path for the rest. The INT8 result is dequantized by dividing by the product of the activation and weight scales, then summed with the outlier path and bias.

Single-layer correctness was verified by comparing the output of one real trained layer against its quantized counterpart on identical input: mean absolute error of 0.0038 against output magnitudes of roughly 0.1 to 0.5, consistent with expected INT8 rounding error rather than a implementation bug.

## Comparison Methodology

Three model variants were evaluated: the original unquantized model, the custom `QuantizedLinear` implementation, and a bitsandbytes INT8 implementation (`Linear8bitLt`), used as an independent, library-based reference point.

## Results

### Translation quality

Fifteen test sentences, covering statements, questions, and varied sentence structures, were translated through all three model variants using greedy decoding. All fifteen translations were identical, word for word, across the original, custom-quantized, and bitsandbytes-quantized models. Two sentences ("What time is it?" and "Where is the bathroom?") produced grammatically imperfect output in all three variants equally, indicating a base model limitation on question-structured sentences rather than a quantization effect.

### Perplexity

Perplexity was computed as the exponential of average cross-entropy loss over the held-out validation split.

| Model | Perplexity |
|---|---|
| Original | 21.8669 |
| Custom QuantizedLinear | 21.8698 |
| bitsandbytes | 21.8653 |

All three values fall within 0.02 percent of one another, indicating that quantization introduced no measurable degradation in predictive quality on this dataset.

### Memory footprint

Static footprint measures total bytes across all parameters and buffers. Peak GPU memory measures allocation during a single translation call.

| Model | Static footprint | Peak GPU memory (inference) |
|---|---|---|
| Original | 430.52 MB | 2534.08 MB |
| Custom QuantizedLinear | 230.77 MB (-46.4%) | 2397.27 MB (-5.4%) |
| bitsandbytes | 304.52 MB (-29.3%) | 2308.47 MB (-8.9%) |

The custom implementation achieves a larger static footprint reduction, attributable to its inclusion of the output projection layer, which bitsandbytes could not quantize due to the dimensional constraint described above; this single layer accounts for roughly 26.7 million of the model's parameters. Conversely, bitsandbytes achieves a larger reduction in peak inference-time memory, likely reflecting more mature internal memory management in its CUDA kernel path relative to the custom implementation's forward pass, which was written for correctness rather than optimized for temporary allocation overhead.

