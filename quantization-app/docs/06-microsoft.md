# Microsoft Quantization Approaches

## Overview

Microsoft contributes to AI quantization through multiple channels: ONNX Runtime (production inference), DeepSpeed (large-scale training and inference), and collaborative industry standards (Microscaling/MX formats). Their approach spans the full spectrum from production-ready INT8 to cutting-edge sub-4-bit formats.

## ONNX Runtime Quantization

ONNX Runtime (ORT) provides a comprehensive quantization toolkit for deploying models in production across diverse hardware.

### Core Formula

ONNX Runtime uses the standard affine quantization formula:

```
val_fp32 = scale * (val_quantized - zero_point)

Quantization:
  val_quantized = Clip(Round(val_fp32 / scale) + zero_point, 0, 255)  # uint8
  or
  val_quantized = Clip(Round(val_fp32 / scale) + zero_point, -128, 127)  # int8

Where:
  scale = (max_val - min_val) / (qmax - qmin)
  zero_point = round(qmin - min_val / scale)
```

### Dynamic Quantization

No calibration needed — quantization parameters computed at runtime:

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization - simplest approach
quantize_dynamic(
    model_input='model_fp32.onnx',
    model_output='model_int8_dynamic.onnx',
    weight_type=QuantType.QInt8,
    # Optionally specify which ops to quantize
    op_types_to_quantize=['MatMul', 'Attention']
)

# Inference
session = ort.InferenceSession('model_int8_dynamic.onnx')
result = session.run(None, {'input': input_data})
```

**Characteristics:**
- Weights quantized offline, activations quantized at runtime
- No calibration dataset needed
- Best for transformer/NLP models where activation ranges vary
- Slight runtime overhead for computing activation statistics

### Static Quantization

Requires calibration dataset for pre-computing activation ranges:

```python
from onnxruntime.quantization import (
    quantize_static, 
    CalibrationDataReader,
    QuantFormat,
    QuantType
)

# Custom calibration data reader
class MyCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = iter(calibration_data)
    
    def get_next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

# Static quantization with calibration
calibration_reader = MyCalibrationReader(calibration_dataset)

quantize_static(
    model_input='model_fp32.onnx',
    model_output='model_int8_static.onnx',
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,  # QuantizeLinear/DequantizeLinear nodes
    per_channel=True,              # Per-channel for weights
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax  # or Entropy, Percentile
)
```

### QDQ Format

ONNX Runtime uses the QDQ (Quantize-Dequantize) format, where explicit QuantizeLinear and DequantizeLinear nodes mark quantization boundaries:

```
Original:        Input → Conv → ReLU → Output

QDQ format:      Input → DQ → Conv → Q → DQ → ReLU → Q → Output
                  ↑ Q                                      
                (quantized input)

Where:
  Q  = QuantizeLinear(x, scale, zero_point)
  DQ = DequantizeLinear(x_q, scale, zero_point)
```

This format allows hardware-specific backends to fuse Q/DQ operations with compute operations during graph optimization.

### Quantization-Aware Training

ONNX Runtime QAT works through model conversion from frameworks:

```python
# Approach 1: Convert QAT model from PyTorch
# (Train with PyTorch QAT, then export to ONNX)
import torch

model_qat = train_qat_model()  # PyTorch QAT training
torch.onnx.export(model_qat, dummy_input, 'model_qat.onnx',
                  opset_version=13)  # Opset 13+ for QDQ ops

# Approach 2: Convert QAT model from TensorFlow
# (Train with TF-MOT, convert via tf2onnx)
import tf2onnx
tf2onnx.convert.from_keras(qat_model, output_path='model_qat.onnx')
```

### Execution Providers and Quantization

Different hardware backends support different quantization configurations:

| Execution Provider | Format | Best For |
|-------------------|--------|----------|
| CPU (default) | Dynamic INT8 | General inference |
| CUDA | Static INT8, FP16 | NVIDIA GPUs |
| TensorRT | INT8 (symmetric) | NVIDIA optimized |
| OpenVINO | INT8, INT4 | Intel hardware |
| QNN | INT8, INT16 | Qualcomm DSPs |
| DirectML | FP16 | Windows GPU |

## DeepSpeed Quantization

DeepSpeed provides quantization capabilities optimized for large-scale models, particularly LLMs.

### ZeroQuant

ZeroQuant enables INT8 quantization for both BERT-style (encoder) and GPT-style (decoder) models:

```python
import deepspeed

# ZeroQuant configuration
ds_config = {
    "quantization": {
        "enabled": True,
        "quantize_weight": {
            "enabled": True,
            "quantizer": "symmetric",
            "bits": 8,
            "group_size": 128  # Group-wise quantization
        },
        "quantize_activation": {
            "enabled": True,
            "quantizer": "symmetric", 
            "bits": 8,
            "type": "token_wise"  # Per-token for activations
        }
    }
}

# Initialize model with quantization
model = deepspeed.init_inference(
    model,
    config=ds_config,
    dtype=torch.int8
)
```

**Key innovations in ZeroQuant:**
- **Group-wise quantization for weights**: Groups of 128 elements share parameters, balancing accuracy and efficiency
- **Token-wise quantization for activations**: Each token in a sequence gets its own scale, handling the token-level variation in transformers
- **Hardware-optimized kernels**: Custom CUDA kernels for INT8 GEMM with group-wise dequantization

### Layer-by-Layer Knowledge Distillation (LKD)

ZeroQuant introduces LKD — distillation that doesn't require the original training data:

```python
# Layer-by-layer knowledge distillation
# The FP32 model acts as teacher at each layer
def lkd_quantization(model_fp32, calibration_data):
    model_int8 = copy(model_fp32)
    
    for layer_idx in range(num_layers):
        # Get FP32 layer output (teacher)
        teacher_output = model_fp32.layers[layer_idx](input)
        
        # Quantize this layer
        quantize_layer(model_int8.layers[layer_idx])
        
        # Fine-tune quantized layer to match teacher output
        student_output = model_int8.layers[layer_idx](input)
        loss = mse_loss(student_output, teacher_output)
        
        # Brief optimization (few hundred steps)
        optimize(model_int8.layers[layer_idx], loss)
    
    return model_int8
```

**Benefits:**
- No access to original training data required
- Small calibration set sufficient (even generated data works)
- Layer-by-layer approach is memory efficient (only one layer in GPU memory for optimization)
- Works for models too large to fine-tune end-to-end

### ZeroQuant-V2

Extended ZeroQuant with:
- Low-Rank Compensation (LoRC): adds a low-rank matrix to compensate for quantization error
- Sensitivity-based mixed precision: automatically assigns different bit-widths to layers

```python
# ZeroQuant-V2 with Low-Rank Compensation
# For a weight matrix W quantized to W_q:
# W ≈ W_q + A @ B  (low-rank correction)
# Where A ∈ R^(m×r), B ∈ R^(r×n), r << min(m,n)

def zero_quant_v2(weight, bits=4, rank=32):
    # Step 1: Quantize weight
    w_q, scale, zp = quantize(weight, bits)
    w_dq = dequantize(w_q, scale, zp)
    
    # Step 2: Compute residual
    residual = weight - w_dq
    
    # Step 3: Low-rank approximation of residual
    U, S, V = svd(residual)
    A = U[:, :rank] @ torch.diag(S[:rank])
    B = V[:rank, :]
    
    return w_q, scale, zp, A, B
```

### DeepSpeed-FP6

DeepSpeed also supports FP6 inference for LLMs:
- 6-bit floating point representation
- Custom GPU kernels for FP6 dequantization fused with GEMM
- 2.67x compression vs FP16 with minimal quality loss

## Microscaling (MX) Formats

### Background

Microscaling formats were developed collaboratively by **Microsoft, AMD, Intel, Meta, NVIDIA, and Qualcomm** as an open industry standard for next-generation low-precision computing. Published as the OCP (Open Compute Project) Microscaling Specification.

### Architecture

```
MX Format Structure:
┌─────────────────────────────────────────────────────┐
│  Block of 32 elements                               │
│                                                     │
│  Shared scale: E8M0 (8-bit exponent, no mantissa)   │
│  = 2^(exponent - 127)                              │
│                                                     │
│  Element 1: [sign | data bits]                      │
│  Element 2: [sign | data bits]                      │
│  ...                                                │
│  Element 32: [sign | data bits]                     │
└─────────────────────────────────────────────────────┘
```

### E8M0 Shared Scale

The shared exponent uses E8M0 format:
- 8 bits of exponent (no sign, no mantissa)
- Bias: 127
- Represents: 2^(e - 127) where e ∈ [0, 254]
- Value 255 reserved for NaN
- Range: 2^(-127) to 2^(127) ≈ 5.88×10^(-39) to 1.70×10^(38)

### MX Format Variants

| Format | Element Bits | Element Format | Effective bits/element | Total bits for 32 elements |
|--------|-------------|----------------|----------------------|---------------------------|
| MXFP8 | 8 | E4M3 or E5M2 | 8.25 | 264 (256 + 8) |
| MXFP6 | 6 | E3M2 or E2M3 | 6.25 | 200 (192 + 8) |
| MXFP4 | 4 | E2M1 | 4.25 | 136 (128 + 8) |
| MXINT8 | 8 | INT8 | 8.25 | 264 (256 + 8) |

### How MX Quantization Works

```python
def mx_quantize(tensor, block_size=32, element_format='fp4'):
    """
    Microscaling quantization.
    
    Args:
        tensor: Input tensor (flattened or along quantization axis)
        block_size: Number of elements sharing a scale (always 32 for MX)
        element_format: 'fp8_e4m3', 'fp6_e3m2', 'fp4_e2m1', 'int8'
    """
    # Reshape into blocks
    blocks = tensor.reshape(-1, block_size)
    
    # Compute shared exponent (E8M0) per block
    # = floor(log2(max(|elements|))) adjusted to E8M0 representable value
    block_max = blocks.abs().max(dim=1, keepdim=True).values
    shared_exp = torch.floor(torch.log2(block_max)).clamp(-127, 127)
    shared_scale = (2.0 ** shared_exp)  # E8M0 value
    
    # Scale elements by shared scale
    scaled = blocks / shared_scale
    
    # Quantize each element to the target format
    quantized_elements = cast_to_format(scaled, element_format)
    
    return quantized_elements, shared_exp.to(torch.uint8)


def mx_dequantize(quantized_elements, shared_exp, block_size=32):
    """Dequantize MX format back to FP32."""
    shared_scale = 2.0 ** (shared_exp.float() - 127)
    dequantized = quantized_elements.float() * shared_scale
    return dequantized.reshape(-1)
```

### Design Rationale

1. **Block size of 32**: Matches warp size on NVIDIA GPUs, enables efficient parallel processing
2. **E8M0 scale**: 8-bit exponent gives enormous dynamic range without wasting bits on mantissa (block-level precision comes from the elements)
3. **No zero-point**: Symmetric scaling only (zero always maps to zero), simplifies hardware
4. **Standardized**: Multi-vendor agreement ensures hardware interoperability

### Hardware Targets

MX formats are designed for next-generation hardware:
- **AMD MI350+**: Native MXFP4/MXFP6/MXFP8 support
- **NVIDIA Blackwell**: Compatible with block-scaled compute
- **Intel**: Future Xeon and accelerator products
- **Custom ASICs**: Any OCP-compliant accelerator

### Comparison with Per-Group INT4

```
Per-group INT4 (group_size=128):
  - Scale: FP16 (16 bits per 128 elements = 0.125 bits/element overhead)
  - Zero-point: INT4 (4 bits per 128 elements ≈ 0.03 bits/element)
  - Total: ~4.16 bits/element
  - Requires dequantization before compute

MX FP4 (block_size=32):
  - Scale: E8M0 (8 bits per 32 elements = 0.25 bits/element overhead)
  - No zero-point (symmetric)
  - Total: 4.25 bits/element
  - Native hardware compute (no dequantization needed)
```

The MX approach trades slightly higher overhead per element for native hardware computation support and simpler implementation.

## Microsoft Olive

Olive is Microsoft's model optimization toolkit that orchestrates quantization workflows:

```python
# Olive configuration for ONNX Runtime quantization
olive_config = {
    "input_model": {
        "type": "OnnxModel",
        "model_path": "model.onnx"
    },
    "passes": {
        "quantization": {
            "type": "OnnxStaticQuantization",
            "config": {
                "data_config": "calibration_config",
                "per_channel": True,
                "calibrate_method": "MinMax",
                "quant_format": "QDQ",
                "weight_type": "QInt8",
                "activation_type": "QUInt8"
            }
        }
    }
}
```

## References

- ONNX Runtime Quantization documentation
- Yao et al., "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers" (2022)
- Yao et al., "ZeroQuant-V2: Exploring Post-Training Quantization in LLMs from Comprehensive Study to Low Rank Compensation" (2023)
- Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023)
- OCP Microscaling Formats Specification v1.0
- Microsoft Olive documentation
- DeepSpeed documentation
- Wu et al., "ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats" (2023)
