# AI Model Quantization - Complete Reference

Comprehensive documentation covering all quantization methods, concepts, and practical techniques for deep learning models.

## Contents

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 1 | [01-fundamentals.md](01-fundamentals.md) | Core Concepts | Affine formula, symmetric/asymmetric, uniform/non-uniform |
| 2 | [02-ptq.md](02-ptq.md) | Post-Training Quantization | Static/dynamic, calibration methods, W8A8 results |
| 3 | [03-qat.md](03-qat.md) | Quantization-Aware Training | Fake quantization, STE, fine-tuning schedules |
| 4 | [04-granularity.md](04-granularity.md) | Granularity Levels | Per-tensor, per-channel, per-group, per-block |
| 5 | [05-nvidia.md](05-nvidia.md) | NVIDIA Ecosystem | TensorRT, ModelOpt, FP8, INT8 Tensor Cores |
| 6 | [06-microsoft.md](06-microsoft.md) | Microsoft Approaches | ONNX Runtime, DeepSpeed/ZeroQuant, MX formats |
| 7 | [07-google.md](07-google.md) | Google Approaches | TF Model Optimization, gemmlowp, BRECQ |
| 8 | [08-llm-methods.md](08-llm-methods.md) | LLM Methods | GPTQ, AWQ, SmoothQuant, OmniQuant, GGUF, SpQR, etc. |
| 9 | [09-hardware.md](09-hardware.md) | Hardware Formats | INT8 Tensor Cores, FP8 E4M3/E5M2, MX, BFP |
| 10 | [10-evaluation.md](10-evaluation.md) | Evaluation & Tradeoffs | Metrics, method comparison, decision framework |
| 11 | [11-mixed-precision.md](11-mixed-precision.md) | Advanced Techniques | Mixed precision, outlier handling, distillation |

## Methods Covered

### By Vendor
- **NVIDIA**: TensorRT, ModelOpt, FP8, INT8/INT4 calibration, NVFP4
- **Microsoft**: ONNX Runtime, DeepSpeed, ZeroQuant, Microscaling (MX)
- **Google**: TensorFlow QAT, gemmlowp, BRECQ

### By Algorithm
- **PTQ**: MinMax, Entropy, Percentile, MSE calibration
- **QAT**: Fake quantization, STE, learned step size
- **LLM-specific**: GPTQ, AWQ, SmoothQuant, OmniQuant, GGUF/K-quants, SqueezeLLM, QuIP, SpQR, AQLM, HQQ, EXL2

### By Hardware Format
- INT8, INT4, FP8 (E4M3/E5M2), BFloat16, Float16
- Block Floating Point (BFP), Microscaling (MX)
- NVFP4, MXFP8, MXFP6, MXFP4

## Sources

Research verified against 24 primary sources including:
- Qualcomm "A White Paper on Neural Network Quantization" (arxiv:2106.08295)
- Gholami et al. "A Survey of Quantization Methods" (arxiv:2103.13630)
- NVIDIA TensorRT documentation and developer blogs
- ONNX Runtime quantization documentation
- GPTQ (arxiv:2210.17323), AWQ (MIT HAN Lab), SmoothQuant (arxiv:2211.10438)
- FP8 Formats for Deep Learning (arxiv:2209.05433)
- Microscaling (MX) specification (arxiv:2310.10537)
- OmniQuant (arxiv:2308.13137)

Total documentation: ~160 KB across 11 files.
