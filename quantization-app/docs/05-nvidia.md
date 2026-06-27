# NVIDIA Quantization Ecosystem

## Overview

NVIDIA provides a comprehensive quantization ecosystem spanning hardware support (Tensor Cores), software tools (TensorRT, ModelOpt), and research contributions. Their approach centers on symmetric quantization with entropy-based calibration, optimized for their GPU architectures.

## TensorRT Quantization

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. It provides built-in quantization support as a core optimization strategy.

### Quantization Scheme

TensorRT uses **symmetric quantization** for both weights and activations:

```
Formula:
  scale = (2 * amax) / 256

  x_q = Clip(Round(x_f / scale), -128, 127)

  x_f ≈ x_q * scale

Where:
  amax = maximum absolute value of the tensor
  256 = total number of INT8 levels used (symmetric: -128 to 127)
```

Note: TensorRT maps to the full [-128, 127] range in its symmetric scheme, using `amax` to determine the scale.

### INT8 Workflow in TensorRT

```python
import tensorrt as trt

# 1. Build TensorRT engine with INT8 mode
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)

# 2. Provide calibrator for activation ranges
class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data):
        super().__init__()
        self.data = calibration_data
        self.batch_idx = 0
        
    def get_batch_size(self):
        return 32
    
    def get_batch(self, names):
        if self.batch_idx < len(self.data):
            batch = self.data[self.batch_idx]
            self.batch_idx += 1
            return [batch.data_ptr()]
        return None
    
    def read_calibration_cache(self):
        # Load cached calibration if available
        if os.path.exists('calibration.cache'):
            with open('calibration.cache', 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open('calibration.cache', 'wb') as f:
            f.write(cache)

config.int8_calibrator = MyCalibrator(calibration_dataset)

# 3. Build optimized engine
network = parse_onnx_model(onnx_path)
engine = builder.build_engine(network, config)
```

### Calibration Process

TensorRT's calibration collects activation histograms during a calibration run, then applies a calibration algorithm:

```
1. Run inference on calibration data (typically 500-1000 images)
2. Collect activation histograms for each tensor
3. Apply calibration algorithm to determine optimal amax:
   - IInt8EntropyCalibrator2: Entropy/KL-divergence (recommended)
   - IInt8MinMaxCalibrator: Uses observed min/max
   - IInt8EntropyCalibrator: Legacy entropy calibrator
   - IInt8LegacyCalibrator: Percentile-based
4. Cache calibration results for fast rebuilds
```

### Layer Fusion with Quantization

TensorRT fuses operations and handles quantization at fusion boundaries:

```
Before fusion:
  Conv → BN → ReLU → [dequant → quant] → Conv → BN → ReLU

After fusion:
  [INT8 Conv+BN+ReLU] → [INT8 Conv+BN+ReLU]
  
The intermediate dequant/quant is eliminated within fused blocks.
```

### Performance on Turing/Ampere GPUs

INT8 on NVIDIA GPUs with Tensor Cores provides:

- **Up to 16x math throughput** vs FP32 (Tensor Cores perform INT8 → INT32 accumulation)
- **4x bandwidth reduction** (INT8 values are 1/4 the size of FP32)
- **Combined speedup**: 2-4x end-to-end for most models (limited by memory bandwidth and non-quantized ops)

```
Theoretical throughput (A100):
  FP32: 19.5 TFLOPS
  FP16: 312 TFLOPS (Tensor Core)
  INT8: 624 TOPS (Tensor Core)
  
Ratio: INT8/FP32 = 32x theoretical (math only)
Practical: 2-4x end-to-end (memory-bound operations limit gains)
```

## NVIDIA ModelOpt (TensorRT Model Optimizer)

ModelOpt is NVIDIA's unified toolkit for model optimization, supporting quantization, pruning, distillation, and NAS.

### Supported Formats

| Format | Bits | Type | Hardware |
|--------|------|------|----------|
| FP8 (E4M3) | 8 | Float | Hopper+ (H100) |
| INT8 | 8 | Integer | Turing+ (T4, A100, H100) |
| INT4 | 4 | Integer | Ampere+ (with dequant) |
| NVFP4 | 4 | Float | Blackwell (B100+) |

### PTQ with ModelOpt

```python
import modelopt.torch.quantization as mtq

# Define quantization configuration
quant_config = mtq.INT8_DEFAULT_CFG  # or FP8_DEFAULT_CFG, INT4_AWQ_CFG

# Calibrate and quantize
def calibrate_loop(model):
    """Run calibration data through the model."""
    for batch in calibration_loader:
        model(batch)

# Apply PTQ
with mtq.quantize(model, quant_config):
    calibrate_loop(model)

# Export to TensorRT
mtq.export(model, 'model_quantized.onnx')
```

### QAT with ModelOpt

```python
import modelopt.torch.quantization as mtq

# Insert fake-quantization nodes
mtq.quantize(model, quant_config, forward_loop=calibrate_loop)

# Fine-tune with quantization-aware training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(num_qat_epochs):
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Export quantized model
mtq.export(model, 'model_qat.onnx')
```

### Compression Results

ModelOpt typically achieves:
- **2x-4x model compression** depending on format
- FP8: ~2x compression with minimal accuracy loss
- INT4: ~8x compression with moderate accuracy impact
- Combined with pruning: up to 10x+ total compression

## FP8 on Hopper (H100) and Blackwell

### FP8 Formats

NVIDIA Hopper introduced native FP8 computation with two variants:

#### E4M3 (4-bit exponent, 3-bit mantissa)

```
Bit layout: [S][E E E E][M M M]
  S: 1 sign bit
  E: 4 exponent bits (bias = 7)
  M: 3 mantissa bits

Range: [-448, 448]
Precision: ~3.6 decimal digits at magnitude 1.0
Special values:
  - Does NOT represent infinity
  - Only one NaN pattern (0b_1_1111_111 = all ones)
  - This extends usable dynamic range vs IEEE conventions

Use: Forward pass / inference (needs precision over range)
```

#### E5M2 (5-bit exponent, 2-bit mantissa)

```
Bit layout: [S][E E E E E][M M]
  S: 1 sign bit
  E: 5 exponent bits (bias = 15)
  M: 2 mantissa bits

Range: [-57344, 57344]
Precision: ~2.5 decimal digits at magnitude 1.0
Special values:
  - Follows IEEE 754 conventions
  - Represents ±infinity and multiple NaN patterns

Use: Backward pass / gradients (needs range over precision)
```

### Why Two Formats?

- **E4M3 for forward pass (inference)**: Activations and weights typically have limited range but need precision. The extra mantissa bit (3 vs 2) provides 2x better precision.
- **E5M2 for backward pass (gradients)**: Gradients can span a wide dynamic range (especially early in training or for deep networks). The extra exponent bit doubles the dynamic range.

### FP8 Performance

```
H100 FP8 Tensor Core throughput: ~2x vs FP16/BF16
  FP16: 990 TFLOPS (with sparsity)
  FP8:  1979 TFLOPS (with sparsity)

Memory bandwidth savings: 2x vs FP16 (half the bytes)
```

### FP8 Training Recipe

```python
# Typical FP8 training with NVIDIA TransformerEngine
import transformer_engine.pytorch as te

# Replace standard layers with TE layers (FP8-aware)
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = te.MultiheadAttention(
            hidden_size, num_heads,
            fuse_qkv_params=True
        )
        self.mlp = te.LayerNormMLP(
            hidden_size, hidden_size * 4,
            activation='gelu'
        )

# Training loop with FP8
with te.fp8_autocast(enabled=True):
    output = model(input_batch)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

### Per-Tensor Scaling for FP8

FP8 uses dynamic per-tensor scaling (delayed scaling) to maximize the use of the limited dynamic range:

```
1. Track amax history over recent iterations
2. Compute scale: scale = FP8_MAX / amax
3. Apply scale before FP8 cast: x_fp8 = cast_to_fp8(x * scale)
4. After computation, apply inverse scale to output

This "delayed scaling" uses the amax from the previous iteration
as an estimate for the current iteration's range.
```

## NVIDIA Quantization Best Practices

### General Recommendations

1. **Start with INT8 PTQ** — sufficient for most models
2. **Use entropy calibration** (IInt8EntropyCalibrator2) as default
3. **Calibrate with 500-1000 representative samples**
4. **Always validate accuracy** on held-out data after quantization
5. **Fall back to FP16** for sensitive layers if INT8 accuracy is insufficient
6. **Use per-channel for weights** (default in TensorRT)

### For LLMs

1. **FP8 (H100+)**: First choice — minimal accuracy impact, 2x speedup
2. **INT8 SmoothQuant**: For older GPUs, handles activation outliers
3. **INT4 weight-only (AWQ/GPTQ)**: Maximum compression for memory-bound inference
4. **NVFP4 (Blackwell)**: Next-gen 4-bit with hardware support

### Mixed Precision Strategy

```
Layer type          Recommended precision
─────────────────   ────────────────────
Embedding           FP16 (sparse access, not compute-bound)
Attention QKV       INT8 or FP8
Attention Softmax   FP32 (numerical sensitivity)
Feed-forward        INT8 or FP8
Layer Norm          FP32 (variance computation)
Output logits       FP16 (final precision matters)
```

## TensorRT-LLM Quantization

TensorRT-LLM extends TensorRT for large language model deployment:

```python
# TensorRT-LLM quantization example
from tensorrt_llm.quantization import quantize

# FP8 quantization for LLM
model = quantize(
    model_path="meta-llama/Llama-2-70b",
    quantization="fp8",
    calibration_data=calibration_dataset,
    output_path="llama-70b-fp8"
)

# INT4 AWQ quantization
model = quantize(
    model_path="meta-llama/Llama-2-70b", 
    quantization="int4_awq",
    group_size=128,
    calibration_data=calibration_dataset,
    output_path="llama-70b-int4"
)
```

## References

- NVIDIA TensorRT Developer Guide, "Working with INT8" chapter
- NVIDIA, "Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training" (2021)
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022)
- NVIDIA ModelOpt documentation
- NVIDIA TransformerEngine documentation
- NVIDIA TensorRT-LLM documentation
- Noune et al., "8-bit Numerical Formats For Deep Neural Networks" (2022)
