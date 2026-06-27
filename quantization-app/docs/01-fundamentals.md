# Core Quantization Concepts

## What is Quantization?

Quantization is the process of mapping continuous (or high-precision) values to a discrete set of values. In the context of deep learning, it typically means converting model weights and/or activations from 32-bit floating-point (FP32) to lower-precision formats such as INT8, INT4, or FP8.

The fundamental goal is to reduce model size, memory bandwidth requirements, and computational cost while preserving model accuracy.

## Why Quantization Works

Neural network weights and activations exhibit predictable statistical properties that make quantization effective:

1. **Normal/bell-shaped distributions**: Weight tensors typically follow approximately Gaussian distributions centered near zero. This means most values cluster in a narrow range, and the full FP32 dynamic range is vastly underutilized.

2. **Redundancy**: Neural networks are heavily over-parameterized. Small perturbations to individual weights (as introduced by quantization) have minimal impact on the learned function.

3. **Noise tolerance**: Training with techniques like dropout and data augmentation means networks are inherently robust to small perturbations — quantization noise is simply another form of perturbation.

## The Affine Quantization Formula

The core mathematical relationship between a real-valued number `x` and its quantized representation `x_q` is:

```
x = S * (x_q - Z)
```

Where:
- **S (Scale)**: A positive float32 value that maps the quantized integer range to the real-value range
- **Z (Zero-point)**: An integer value in the quantized domain that corresponds to real value 0.0
- **x_q**: The quantized integer representation
- **x**: The original (or reconstructed) real value

### Derivation

Given a real-value range `[x_min, x_max]` that we want to map to a quantized range `[q_min, q_max]`:

```
S = (x_max - x_min) / (q_max - q_min)

Z = round(q_min - x_min / S)

x_q = round(x / S) + Z = Clip(round(x / S) + Z, q_min, q_max)
```

For INT8, the quantized range is typically `[0, 255]` (unsigned) or `[-128, 127]` (signed).

### Worked Example

Suppose we have activation values in the range `[-1.5, 3.0]` and want to quantize to unsigned INT8 `[0, 255]`:

```
S = (3.0 - (-1.5)) / (255 - 0) = 4.5 / 255 ≈ 0.01765

Z = round(0 - (-1.5) / 0.01765) = round(85.0) = 85

# Quantize x = 1.2:
x_q = round(1.2 / 0.01765) + 85 = round(67.99) + 85 = 68 + 85 = 153

# Dequantize back:
x_reconstructed = 0.01765 * (153 - 85) = 0.01765 * 68 = 1.2002
```

The quantization error here is `|1.2 - 1.2002| = 0.0002`.

## Symmetric Quantization

In symmetric quantization, the zero-point `Z` is fixed at 0:

```
x = S * x_q
x_q = Clip(round(x / S), -127, 127)
S = max(|x_min|, |x_max|) / 127
```

**Key characteristics:**
- Maps the real-value range symmetrically around zero: `[-|max|, +|max|]`
- Uses the signed INT8 range `[-127, 127]`, deliberately excluding `-128` to maintain perfect symmetry
- Trades one representable value for computational speedup: eliminates the zero-point addition/subtraction in inference computation
- Real zero maps exactly to quantized zero (Z=0), which is important for zero-padding in convolutions

**When to use:** Symmetric quantization is preferred for weights (which are typically centered around zero) and is the standard choice in NVIDIA TensorRT.

### Computational Advantage

For matrix multiplication `Y = X * W`:

With asymmetric quantization:
```
Y = S_x * S_w * (X_q - Z_x) * (W_q - Z_w)
  = S_x * S_w * (X_q*W_q - Z_x*W_q - X_q*Z_w + Z_x*Z_w)
```

With symmetric quantization (Z_x = Z_w = 0):
```
Y = S_x * S_w * X_q * W_q
```

The symmetric form eliminates three additional terms, significantly reducing computation.

## Asymmetric Quantization

Asymmetric quantization uses a non-zero zero-point:

```
x = S * (x_q - Z)
x_q = Clip(round(x / S) + Z, q_min, q_max)
```

**Key characteristics:**
- Preserves the full quantized range `[-128, 127]` (or `[0, 255]` for unsigned)
- Ensures real zero is exactly representable (critical for ReLU activations where zero has semantic meaning)
- Better utilization of the quantized range when the real distribution is not centered at zero
- Slightly higher computational cost due to zero-point offset

**When to use:** Asymmetric quantization is preferred for activations (especially after ReLU, which produces values in `[0, +max]`).

### Example: ReLU Activations

After ReLU, values are in `[0, max_val]`. With symmetric quantization, half the quantized range (the negative half) is wasted. Asymmetric quantization maps the full `[0, 255]` range to `[0, max_val]`, providing 2x better resolution.

## Uniform vs Non-Uniform Quantization

### Uniform Quantization

Quantization levels are evenly spaced:

```
Levels: {..., -2S, -S, 0, S, 2S, ...}
```

- Simple hardware implementation (shift + multiply)
- Optimal when values are uniformly distributed
- Most hardware accelerators (GPUs, TPUs) are designed for uniform quantization
- Standard in INT8/INT4 quantization

### Non-Uniform Quantization

Quantization levels are unevenly spaced, often denser where values are more frequent:

```
Examples:
- Logarithmic: levels at powers of 2 (used in some weight formats)
- Learned: codebook-based quantization (k-means clustering of values)
- Mixed: combination of uniform and non-uniform regions
```

**Advantages:**
- Better representation of non-uniform distributions (e.g., Gaussian weight distributions)
- Can achieve higher effective precision for the same bit-width
- Used in methods like SqueezeLLM, AQLM, and QuIP

**Disadvantages:**
- Requires lookup tables or special hardware
- More complex dequantization logic
- Limited hardware support on current accelerators

### Comparison

| Property | Uniform | Non-Uniform |
|----------|---------|-------------|
| Hardware support | Excellent | Limited |
| Implementation | Simple | Complex |
| Optimal for | Uniform distributions | Skewed distributions |
| Inference speed | Fast | Slower (lookup needed) |
| Accuracy at low bits | Moderate | Better |

## Quantization Granularity Overview

The choice of how many values share a single scale/zero-point pair significantly impacts accuracy:

1. **Per-tensor**: One S, Z pair for the entire tensor
2. **Per-channel**: One S, Z pair per output channel (e.g., per filter in conv layers)
3. **Per-group**: One S, Z pair per group of N elements (e.g., 128)
4. **Per-element**: Individual scale per element (effectively floating-point)

See [04-granularity.md](./04-granularity.md) for detailed analysis.

## Mathematical Properties

### Quantization Error

The maximum quantization error for uniform quantization is bounded by:

```
|x - Q(x)| ≤ S/2
```

where `S` is the scale factor. This means finer scale (more levels) reduces maximum error.

### Signal-to-Quantization-Noise Ratio (SQNR)

For a signal with variance σ² quantized to b bits:

```
SQNR ≈ 6.02 * b + 1.76 dB (for uniform distributions)
```

Each additional bit provides approximately 6 dB improvement in SQNR.

### Clipping vs Rounding Error Tradeoff

- **Rounding error**: Error from mapping values to nearest quantization level
- **Clipping error**: Error from saturating values outside the representable range

Optimal quantization range balances these two sources of error. Using the full min/max range minimizes clipping error but may increase rounding error (if outliers stretch the range). Clipping the range reduces rounding error for the majority of values at the cost of saturating outliers.

## Data Types Summary

| Format | Bits | Range (approximate) | Use Case |
|--------|------|---------------------|----------|
| FP32 | 32 | ±3.4×10³⁸ | Training baseline |
| FP16 | 16 | ±65504 | Mixed-precision training |
| BF16 | 16 | ±3.4×10³⁸ | Training (same range as FP32) |
| FP8 E4M3 | 8 | ±448 | Inference (Hopper+) |
| FP8 E5M2 | 8 | ±57344 | Gradients (Hopper+) |
| INT8 | 8 | [-128, 127] | Production inference |
| INT4 | 4 | [-8, 7] | Weight compression |
| NF4 | 4 | non-uniform | QLoRA weights |

## References

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
- Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference" (2018), Google whitepaper
- Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021)
- NVIDIA TensorRT Developer Guide - Working with INT8
- Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation" (2020)
