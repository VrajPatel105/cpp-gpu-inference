# This repo covers 4 quantization methods

1. **LLM.int8()**
   - Implemented from scratch (custom `QuantizedLinear`, calibration-based outlier detection)
   - Compared against bitsandbytes' implementation on a custom EN-DE transformer
   - Findings:

   | Model | Perplexity | Static Memory | Peak GPU Memory |
   |---|---|---|---|
   | Original | 21.8669 | 430.52 MB | 2534.08 MB |
   | Custom QuantizedLinear | 21.8698 | 230.77 MB (-46.4%) | 2397.27 MB (-5.4%) |
   | bitsandbytes | 21.8653 | 304.52 MB (-29.3%) | 2308.47 MB (-8.9%) |

   - Translation quality: 15/15 test sentences identical across all three models

2. **SmoothQuant**
3. **GPTQ**
4. **AWQ**