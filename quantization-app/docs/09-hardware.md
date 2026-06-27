# Hardware Considerations for Quantization

## Overview

The effectiveness of quantization depends critically on hardware support. Modern AI accelerators are designed with low-precision computation as a primary feature, but the specific formats, throughput ratios, and memory hierarchies vary significantly across platforms.

## INT8 Tensor Cores (NVIDIA Turing/Ampere/Hopper)

### Architecture

NVIDIA Tensor Cores perform matrix-multiply-accumulate operations at mixed precision:

```
INT8 Tensor Core operation:
  Input A: INT8 matrix (e.g., 8×4 fragment)
  Input B: INT8 matrix (e.g., 4×8 fragment)
  Accumulator: INT32 matrix (8×8 fragment)
  
  C_int32 += A_int8 × B_int8
  
Each Tensor Core: one 4×4 matrix multiply-accumulate per cycle
```

### Throughput Comparison

| GPU | FP32 TFLOPS | FP16 TFLOPS | INT8 TOPS | INT8/FP32 Ratio |
|-----|-------------|-------------|-----------|-----------------|
| T4 (Turing) | 8.1 | 65 | 130 | 16x |
| A100 (Ampere) | 19.5 | 312 | 624 | 32x |
| H100 (Hopper) | 67 | 990 | 1979 | ~30x |

**Key performance factors:**

1. **Math throughput: up to 16x** (Turing) to 32x (Ampere) vs FP32
   - INT8 operations are simpler → more fit on silicon
   - Tensor Cores designed for INT8 → INT32 accumulation

2. **Bandwidth reduction: 4x** vs FP32
   - INT8 values are 1 byte vs 4 bytes for FP32
   - 4x more values per memory transaction
   - Critical for memory-bound operations (which LLM inference is)

3. **Practical end-to-end speedup: 2-4x**
   - Limited by: non-quantized ops, memory latency, kernel launch overhead
   - Memory-bound models see closer to 4x (bandwidth-limited)
   - Compute-bound models see closer to 2x (need to also quantize activations)

### INT8 Accumulation Pattern

```
For GEMM Y = A × B (both INT8):

  # Each output element accumulates many INT8 × INT8 products
  Y[i][j] = Σ_k A[i][k] * B[k][j]   (accumulate in INT32)
  
  # INT32 accumulator prevents overflow:
  # max value = N × 127 × 127 = N × 16129
  # For N=4096 (typical hidden dim): max = 66M (fits in INT32)
  
  # After accumulation, requantize to INT8 or dequantize to FP16
  Y_final = Requantize(Y_int32, output_scale) → INT8
  # or
  Y_final = Dequantize(Y_int32, scale_A * scale_B) → FP16
```

## FP8 (NVIDIA Hopper/Blackwell)

### E4M3 Format

```
Bit layout: [S][EEEE][MMM]
  Sign: 1 bit
  Exponent: 4 bits, bias = 7
  Mantissa: 3 bits (implicit leading 1 for normal numbers)

Properties:
  - Range: [-448, 448]
  - Smallest positive normal: 2^(-6) = 0.015625
  - Smallest positive subnormal: 2^(-9) = 0.001953125
  - Precision at 1.0: 2^(-3) = 0.125 (8 levels between 1.0 and 2.0)
  
Special values:
  - Does NOT represent infinity (extends usable range)
  - Only ONE NaN pattern: S=1, E=1111, M=111 (0xFF)
  - All other bit patterns are valid numbers
  - This gives 240 unique representable magnitudes (vs 238 with IEEE inf/NaN)
```

### E5M2 Format

```
Bit layout: [S][EEEEE][MM]
  Sign: 1 bit
  Exponent: 5 bits, bias = 15
  Mantissa: 2 bits (implicit leading 1 for normal numbers)

Properties:
  - Range: [-57344, 57344]
  - Smallest positive normal: 2^(-14) ≈ 6.1 × 10^(-5)
  - Smallest positive subnormal: 2^(-16) ≈ 1.5 × 10^(-5)
  - Precision at 1.0: 2^(-2) = 0.25 (4 levels between 1.0 and 2.0)
  
Special values:
  - Follows IEEE 754 conventions
  - Represents ±infinity (E=11111, M=00)
  - Multiple NaN patterns (E=11111, M≠00)
```

### E4M3 vs E5M2 Comparison

| Property | E4M3 | E5M2 |
|----------|------|------|
| Max magnitude | 448 | 57,344 |
| Min normal | 0.015625 | ~6.1×10⁻⁵ |
| Precision (at 1.0) | 0.125 | 0.25 |
| Dynamic range (decades) | ~4.7 | ~8.9 |
| Infinity | No | Yes (IEEE 754) |
| NaN patterns | 1 | Multiple |
| Use case | Forward / inference | Backward / gradients |

### Why E4M3 for Inference, E5M2 for Gradients

**Forward pass (inference):**
- Weights and activations have controlled ranges (typically within [-10, 10] after normalization)
- Precision matters more than range (small differences in weights → different predictions)
- E4M3's 3 mantissa bits give 2x better precision than E5M2

**Backward pass (gradients):**
- Gradients can span many orders of magnitude (especially in early layers)
- Very small gradients (deep in the network) must not underflow to zero
- Very large gradients (from loss spikes) must not overflow
- E5M2's wider range (57K vs 448) handles these extremes better

### FP8 Performance

```
H100 Tensor Core throughput:
  FP16/BF16: 990 TFLOPS (dense) / 1979 TFLOPS (structured sparsity)
  FP8:       1979 TFLOPS (dense) / 3958 TFLOPS (structured sparsity)
  
  Ratio: FP8 = 2x FP16/BF16 throughput
  
Memory bandwidth benefit:
  FP8 = half the bytes of FP16 → 2x effective bandwidth
  
Combined benefit for memory-bound inference: up to 2x end-to-end
```

### Per-Tensor Dynamic Scaling

FP8's limited dynamic range requires per-tensor scaling:

```python
def fp8_dynamic_scaling(tensor, fp8_format='e4m3'):
    """
    Scale tensor to maximize FP8 utilization.
    """
    if fp8_format == 'e4m3':
        fp8_max = 448.0
    else:  # e5m2
        fp8_max = 57344.0
    
    # Compute scale to map tensor range into FP8 range
    amax = tensor.abs().max()
    scale = fp8_max / amax
    
    # Scale and cast
    tensor_scaled = tensor * scale
    tensor_fp8 = cast_to_fp8(tensor_scaled, fp8_format)
    
    # Store inverse scale for dequantization
    return tensor_fp8, 1.0 / scale

# Delayed scaling (uses previous iteration's amax as estimate)
class DelayedScaling:
    def __init__(self, history_length=16):
        self.amax_history = deque(maxlen=history_length)
    
    def get_scale(self, fp8_max=448.0):
        # Use max of recent history (conservative)
        amax = max(self.amax_history) if self.amax_history else 1.0
        return fp8_max / amax
    
    def update(self, tensor):
        self.amax_history.append(tensor.abs().max().item())
```

## Block Floating Point (BFP)

### Concept

Block Floating Point shares a single exponent across a group of values, with each value retaining its own mantissa:

```
Standard floating point (per-element):
  [S₁|E₁|M₁] [S₂|E₂|M₂] [S₃|E₃|M₃] ...
  Each element: 1+8+23 = 32 bits (FP32)

Block Floating Point:
  [E_shared] [S₁|M₁] [S₂|M₂] [S₃|M₃] ...
  Shared exponent: 8 bits for N elements
  Per-element: 1 + m bits (sign + mantissa only)
  
  Effective bits per element = (1 + m) + 8/N
  For N=32, m=3: 4 + 0.25 = 4.25 bits/element
```

### Advantages

1. **Reduced exponent overhead**: Instead of 8 exponent bits per element, amortize over the block
2. **Simpler hardware**: Shared exponent means simpler alignment logic in multipliers
3. **Natural granularity**: Block aligns with hardware parallelism (warp/wavefront size)

### Trade-offs

```
If elements in a block have very different magnitudes:
  Element A = 100.0  →  scale set by A
  Element B = 0.001  →  only gets ~0 mantissa bits of precision!
  
BFP works best when elements within a block have similar magnitudes.
This is why block size matters: smaller blocks → more uniform → better accuracy
                                larger blocks → less overhead → better efficiency
```

## Microscaling (MX) Formats

### Industry Standard

MX is the standardized version of block floating point, developed by Microsoft, AMD, Intel, Meta, NVIDIA, and Qualcomm under the Open Compute Project (OCP).

```
MX Block Structure (32 elements):
┌──────────────────────────────────────────┐
│ Shared scale: E8M0 (8 bits, exponent only)│
│ Element 0:  [sign][format-specific bits]  │
│ Element 1:  [sign][format-specific bits]  │
│ ...                                       │
│ Element 31: [sign][format-specific bits]  │
└──────────────────────────────────────────┘

Dequantization:
  real_value[i] = shared_scale × element_value[i]
  shared_scale = 2^(E8M0_value - 127)
```

### MX Format Details

| Format | Element | Element Bits | Block Overhead | Effective bpw |
|--------|---------|-------------|----------------|---------------|
| MXFP8 | E4M3/E5M2 | 8 | 8/32 = 0.25 | 8.25 |
| MXFP6 | E3M2/E2M3 | 6 | 8/32 = 0.25 | 6.25 |
| MXFP4 | E2M1 | 4 | 8/32 = 0.25 | 4.25 |
| MXINT8 | INT8 | 8 | 8/32 = 0.25 | 8.25 |

### MXFP4 Element Format (E2M1)

```
Bit layout: [S][EE][M]
  Sign: 1 bit
  Exponent: 2 bits, bias = 1
  Mantissa: 1 bit

Representable values (positive):
  0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
  (8 positive values + 8 negative + zero = 17 levels)

Combined with E8M0 shared scale:
  Effective dynamic range spans the full E8M0 range
  Precision: 4 levels per octave (between powers of 2)
```

### Hardware Implementation

```
MX hardware compute path:
  1. Load block (32 elements + shared scale)
  2. No explicit dequantization needed if hardware supports MX natively
  3. Multiply-accumulate in higher precision (FP32 accumulator)
  4. Requantize output to target format

Native MX support (upcoming):
  - AMD MI350: MXFP4, MXFP6, MXFP8
  - Future Intel Xeon/accelerators
  - Future NVIDIA (Blackwell compatible)
```

## INT4 and Sub-4-Bit Quantization

### Challenges

At 4 bits, only 16 distinct values are representable:
```
INT4 signed: -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
NF4 (normal float 4): non-uniformly spaced, optimized for Gaussian distributions
FP4 (E2M1): 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (+ negatives)
```

With only 16 levels, per-group quantization (group_size 32-128) is **required** for acceptable quality.

### NVIDIA NVFP4

NVIDIA's proprietary 4-bit floating-point format for Blackwell architecture:
- E2M1 format (2-bit exponent, 1-bit mantissa)
- Block-scaled (shared exponent per group)
- Native hardware support for 4-bit tensor core operations
- Theoretical 8x compression from FP32, 4x from FP16

### Hardware Support Landscape

| Format | Hardware | Status |
|--------|----------|--------|
| INT8 | All modern GPUs, CPUs, TPUs | Production |
| FP8 | NVIDIA H100/H200, AMD MI300 | Production |
| INT4 (weight-only) | GPU via dequant kernels | Production (software) |
| INT4 (compute) | Limited native support | Emerging |
| MXFP4 | AMD MI350, future hardware | 2024-2025 |
| NVFP4 | NVIDIA Blackwell | 2024-2025 |
| INT2 | Software-only | Research |

### Memory Bandwidth Analysis

For autoregressive LLM inference (batch_size=1):

```
Model: 70B parameters

FP16 weight loading per token:
  70B × 2 bytes = 140 GB
  H100 bandwidth: 3.35 TB/s
  Time: 140 / 3350 = 41.8 ms/token → ~24 tokens/sec

INT4 weight loading per token:
  70B × 0.5 bytes = 35 GB
  H100 bandwidth: 3.35 TB/s
  Time: 35 / 3350 = 10.4 ms/token → ~96 tokens/sec

Speedup: ~4x (approaches theoretical 4x bandwidth reduction)
```

This demonstrates why **weight-only quantization** is so effective for LLM inference — it's almost purely memory-bandwidth-bound.

## Comparison: Integer vs Floating-Point Quantization

### Integer (INT8, INT4)

```
Pros:
  + Simple hardware (integer ALU)
  + Deterministic arithmetic
  + Compact representation
  + Well-supported everywhere

Cons:
  - Uniform spacing doesn't match data distributions
  - Requires careful calibration
  - Zero-point handling adds complexity
  - Poor at representing very small values
```

### Floating-Point (FP8, FP4)

```
Pros:
  + Non-uniform spacing matches neural network distributions
  + Covers wide dynamic range
  + No zero-point needed (symmetric by nature)
  + Better for training (handles gradient magnitudes)

Cons:
  - More complex hardware (FP ALU needed)
  - Special value handling (NaN, inf)
  - Fewer representable values per range octave
  - Currently less hardware support (FP8 only on latest GPUs)
```

### Hybrid: Block Floating Point / MX

```
Pros:
  + Combines FP dynamic range with integer simplicity
  + Amortized exponent overhead
  + Well-suited to hardware parallelism
  + Standardized (multi-vendor support)

Cons:
  - Fixed block size (32) may not align with all architectures
  - Elements with very different magnitudes lose precision
  - Requires block-aligned access patterns
  - Limited current hardware (arriving 2024-2025)
```

## Platform-Specific Considerations

### NVIDIA GPUs
- INT8: Full Tensor Core support (Turing+), TensorRT integration
- FP8: Native Tensor Core support (Hopper+), TransformerEngine
- INT4/FP4: Dequantization kernels (current), native compute (Blackwell)

### AMD GPUs
- INT8: Matrix cores on MI200/MI300
- FP8: Native support on MI300
- MXFP4/6/8: Native support on MI350 (2025)

### Intel (CPU/GPU/Gaudi)
- INT8: VNNI (x86), AMX (Sapphire Rapids+)
- BF16: AMX support
- INT4: Software dequantization (OpenVINO)
- Future: MX format support

### ARM (Mobile/Edge)
- INT8: Full support via NEON/SVE
- INT4: Software dequant, some DSP support
- Target: TFLite, XNNPACK runtimes

### Apple Silicon
- INT8: ANE (Apple Neural Engine)
- INT4: CoreML weight compression
- FP16: GPU and ANE
- MLX framework supports quantized inference

## References

- NVIDIA, "NVIDIA A100 Tensor Core GPU Architecture" whitepaper (2020)
- NVIDIA, "NVIDIA H100 Tensor Core GPU Architecture" whitepaper (2022)
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022)
- Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023)
- OCP Microscaling Formats Specification v1.0
- NVIDIA CUTLASS documentation
- AMD ROCm documentation
- Intel oneDNN documentation
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)
