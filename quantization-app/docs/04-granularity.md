# Quantization Granularity

## Overview

Quantization granularity defines the scope over which a single set of quantization parameters (scale and zero-point) is shared. Finer granularity provides more precise quantization at the cost of additional storage for parameters and potentially reduced hardware efficiency.

The choice of granularity has a dramatic impact on accuracy — in some cases making the difference between a working model and a catastrophic failure.

## Per-Tensor Quantization

The coarsest granularity: a single scale and zero-point for an entire weight tensor or activation tensor.

```
Weight tensor W ∈ R^(C_out × C_in × H × W)
→ One S, one Z for all elements

Parameters stored: 2 values (S, Z) per tensor
```

### Characteristics

- **Simplest implementation**: Minimal metadata overhead
- **Maximum hardware efficiency**: Single scale factor for entire matrix multiplication
- **Least accurate**: Must accommodate the full range of all values in the tensor
- **Susceptible to outliers**: A single channel with large magnitudes ruins resolution for all channels

### When Per-Tensor Fails

Per-tensor quantization fails when different channels/dimensions have very different value ranges:

```
Example: Conv layer weights
  Channel 0: range [-0.01, 0.01]   (small, precise weights)
  Channel 5: range [-2.0, 2.0]     (large weights)
  
Per-tensor scale must accommodate [-2.0, 2.0]
→ Resolution = 4.0 / 255 = 0.0157
→ Channel 0 only gets ~1 quantization level! (0.02 / 0.0157 ≈ 1)
```

### Batch Normalization Folding Catastrophe

A critical interaction: when batch normalization is folded into convolution weights before per-tensor quantization, the resulting weight distribution can have vastly different scales per channel, causing catastrophic accuracy loss.

**Example results (per-tensor + BN folding):**

| Model | FP32 | Per-Tensor PTQ |
|-------|------|----------------|
| EfficientNet-B0 | 76.85% | **12.93%** |
| MobileNet-V2 | 71.72% | **0.12%** |
| Inception-V3 | 78.0% | 42.3% |

These catastrophic failures occur because BN folding creates per-channel scale variations that per-tensor quantization cannot handle.

**Solution:** Always use per-channel quantization when BN folding is applied.

## Per-Channel (Per-Axis) Quantization

Each output channel (or specified axis) gets its own scale and zero-point:

```
Weight tensor W ∈ R^(C_out × C_in × H × W)
→ One S_i, one Z_i for each output channel i (i = 0, ..., C_out-1)

Parameters stored: 2 × C_out values per tensor
```

### Implementation

```python
def per_channel_quantize(weight_tensor, axis=0, bits=8):
    """
    Quantize a weight tensor with per-channel parameters.
    
    Args:
        weight_tensor: Shape (C_out, C_in, H, W) for conv
        axis: Channel axis (typically 0 for output channels)
        bits: Quantization bit-width
    """
    q_min, q_max = -(2**(bits-1)), 2**(bits-1) - 1
    
    # Compute per-channel min/max
    # Reshape to (C_out, -1) to get min/max per channel
    flat = weight_tensor.reshape(weight_tensor.shape[axis], -1)
    ch_min = flat.min(dim=1).values
    ch_max = flat.max(dim=1).values
    
    # Per-channel scale and zero-point
    scales = (ch_max - ch_min) / (q_max - q_min)
    zero_points = torch.round(q_min - ch_min / scales).clamp(q_min, q_max)
    
    # Quantize each channel with its own parameters
    # (broadcasting handles the per-channel application)
    shape = [1] * weight_tensor.ndim
    shape[axis] = -1
    scales_shaped = scales.reshape(shape)
    zp_shaped = zero_points.reshape(shape)
    
    w_q = torch.clamp(
        torch.round(weight_tensor / scales_shaped) + zp_shaped,
        q_min, q_max
    )
    
    return w_q, scales, zero_points
```

### Accuracy Impact

Per-channel quantization provides dramatically better results, especially for architectures with BN folding:

**EfficientNet-Lite W4A8 comparison:**

| Granularity | Calibration | Top-1 Accuracy |
|-------------|-------------|----------------|
| Per-tensor | Max | 71.24% |
| **Per-channel** | **Max** | **74.01%** |
| Per-channel | Percentile 99.99% | 74.15% |
| FP32 baseline | — | 75.42% |

The per-channel vs per-tensor gap is **2.77%** — substantial at this precision level.

**Per-channel with max calibration achieves within 0.4% of FP32 for all tested standard models** (ResNet, VGG, Inception, etc.), making it the recommended default for production deployment.

### Hardware Support

Per-channel quantization is well-supported on modern hardware:
- **NVIDIA GPUs**: Full support in TensorRT (per-channel for weights, per-tensor for activations)
- **ARM CPUs**: Supported in XNNPACK and TFLite
- **x86 CPUs**: Supported in FBGEMM (PyTorch) and oneDNN (Intel)
- **Google TPUs**: Native per-channel support

**Important constraint:** Per-channel is typically applied only to weights. Activations usually remain per-tensor because per-channel activation quantization would require per-channel accumulation, which is much less hardware-efficient.

## Per-Group Quantization

A middle ground between per-tensor and per-channel: groups of N consecutive elements share quantization parameters.

```
Weight vector w ∈ R^n, group_size = g
→ n/g groups, each with its own S and Z

Parameters stored: 2 × (n/g) values per vector
```

### Typical Configuration

- **group_size = 128**: Most common choice (GPTQ, AWQ, bitsandbytes)
- **group_size = 64**: Higher accuracy, more overhead
- **group_size = 32**: Used in MX formats (block floating point)
- **group_size = 256**: Lower overhead, slightly less accuracy

### Implementation

```python
def per_group_quantize(tensor, group_size=128, bits=4):
    """
    Quantize tensor with per-group parameters.
    
    Common in LLM quantization (GPTQ, AWQ, etc.)
    """
    q_min, q_max = -(2**(bits-1)), 2**(bits-1) - 1
    
    # Reshape into groups
    assert tensor.numel() % group_size == 0
    grouped = tensor.reshape(-1, group_size)
    
    # Per-group statistics
    g_min = grouped.min(dim=1, keepdim=True).values
    g_max = grouped.max(dim=1, keepdim=True).values
    
    # Per-group scale and zero-point
    scales = (g_max - g_min) / (q_max - q_min)
    zero_points = torch.round(q_min - g_min / scales).clamp(q_min, q_max)
    
    # Quantize
    w_q = torch.clamp(
        torch.round(grouped / scales) + zero_points,
        q_min, q_max
    )
    
    return w_q.reshape(tensor.shape), scales.squeeze(), zero_points.squeeze()
```

### Storage Overhead

For a weight matrix of size M × N with group_size g and b-bit quantization:

```
Weight storage: M × N × b bits
Scale storage:  (M × N / g) × 16 bits (FP16 scales)
ZP storage:     (M × N / g) × b bits (or 16 bits)

Effective bits per weight = b + (16 + b) / g
  For g=128, b=4: 4 + 20/128 ≈ 4.16 bits per weight
  For g=32, b=4:  4 + 20/32 = 4.625 bits per weight
```

### Usage in LLM Methods

| Method | Group Size | Bit-width | Notes |
|--------|-----------|-----------|-------|
| GPTQ | 128 | 3-4 bit | Block processing with Hessian |
| AWQ | 128 | 4 bit | Activation-aware scaling |
| bitsandbytes NF4 | 64 | 4 bit | Non-uniform (normal float) |
| GGUF q4_K_M | 256 (super-blocks) | 4 bit | Mixed with importance |
| HQQ | 64-128 | 2-4 bit | Half-quadratic optimization |

## Per-Block Quantization (Block Floating Point)

In block floating point (BFP), a block of elements shares a single exponent (scale), while each element retains its own mantissa:

```
Block of N elements: [x_1, x_2, ..., x_N]
Shared scale: S = max(|x_1|, |x_2|, ..., |x_N|) → stored as E8M0
Individual: each x_i stored with reduced mantissa bits
```

### Microscaling (MX) Format

The industry-standard block floating point format:

```
Block size: 32 elements
Shared scale: E8M0 (8-bit exponent, no mantissa) = 1 byte per block
Element formats:
  - MXFP8:  8-bit float per element  (effective: 8 + 8/32 = 8.25 bits)
  - MXFP6:  6-bit float per element  (effective: 6 + 8/32 = 6.25 bits)
  - MXFP4:  4-bit float per element  (effective: 4 + 8/32 = 4.25 bits)
  - MXINT8: 8-bit int per element    (effective: 8 + 8/32 = 8.25 bits)
```

### Comparison: Per-Group vs Per-Block

| Aspect | Per-Group (integer) | Per-Block (BFP/MX) |
|--------|--------------------|--------------------|
| Scale format | FP16/FP32 | E8M0 (compact) |
| Typical block size | 64-128 | 32 |
| Element format | Integer | Float or Integer |
| Hardware support | Current GPUs | Next-gen (MI350+) |
| Overhead per element | ~0.16-0.25 bits | 0.25 bits |
| Best for | LLM weight compression | Native hardware compute |

## Granularity Comparison Summary

| Granularity | Params per Tensor | Accuracy | Hardware Efficiency | Use Case |
|-------------|-------------------|----------|--------------------|-----------| 
| Per-tensor | 2 | Low | Highest | Activations, simple deployment |
| Per-channel | 2 × C_out | High | Good | Weights (standard practice) |
| Per-group (128) | 2 × N/128 | Very High | Good | LLM weight-only quantization |
| Per-group (32) | 2 × N/32 | Highest | Moderate | Ultra-low-bit quantization |
| Per-element | 2 × N | Perfect | Low | Not practical (defeats purpose) |

## Practical Recommendations

### For Production CNNs (INT8)
```
Weights:      Per-channel, symmetric
Activations:  Per-tensor, asymmetric (or symmetric with unsigned)
```

### For LLM Weight-Only Quantization (INT4/INT3)
```
Weights:      Per-group (group_size=128), asymmetric
Activations:  Keep in FP16 (weight-only quantization)
```

### For Full W8A8 Quantization
```
Weights:      Per-channel, symmetric
Activations:  Per-tensor, symmetric (for hardware compatibility)
              or per-token for transformer models
```

### For Edge/Mobile Deployment
```
Weights:      Per-channel (if hardware supports), else per-tensor
Activations:  Per-tensor (minimize overhead)
Format:       Symmetric (simpler implementation)
```

## Impact on Matrix Multiplication

The choice of granularity affects how matrix multiplication is implemented:

```
Per-tensor (simplest):
  Y = S_x * S_w * (X_q * W_q)     # Single rescale after GEMM

Per-channel weights (standard):
  Y[i] = S_x * S_w[i] * (X_q * W_q[i,:])  # Per-output-channel rescale
  # Still one GEMM, just per-channel post-processing

Per-group weights:
  Y = Σ_g S_x * S_w[g] * (X_q * W_q[g])   # Sum of group contributions
  # May require multiple smaller GEMMs or specialized kernels
```

## References

- Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference" (2018)
- Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation" (2020)
- Nagel et al., "A White Paper on Neural Network Quantization" (2021)
- Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2022)
