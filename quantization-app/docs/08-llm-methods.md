# LLM-Specific Quantization Methods

## Overview

Large Language Models (LLMs) present unique quantization challenges:
- **Scale**: Billions of parameters make full QAT prohibitively expensive
- **Outlier activations**: Transformer attention produces extreme activation values in specific channels
- **Memory-bound inference**: LLM inference is dominated by memory bandwidth (loading weights), making weight compression highly impactful
- **Autoregressive nature**: Token-by-token generation amplifies any per-token errors

This has driven the development of specialized methods that achieve aggressive compression (3-4 bit) with minimal quality loss.

## GPTQ (2022)

### Core Approach

GPTQ (Accurate Post-Training Quantization for Generative Pre-Trained Transformers) quantizes weights one layer at a time using second-order information (the inverse Hessian matrix) to determine optimal rounding decisions.

Based on the Optimal Brain Quantization (OBQ) framework, GPTQ dramatically improves computational efficiency to make it practical for 175B+ parameter models.

### Algorithm

For each layer with weight matrix W and calibration data producing input X:

```
Input: Weight matrix W ∈ R^(d_row × d_col), Hessian H = 2 * X * X^T
Output: Quantized weight matrix W_q

1. Compute Hessian: H = X @ X.T (using calibration data)
2. Compute inverse Hessian: H_inv = (H + λI)^(-1)  (with dampening)
3. Process columns in blocks of B=128:
   For each block of columns [i, i+B):
     a. Quantize column i: w_q[i] = quantize(w[i])
     b. Compute quantization error: δ = w[i] - w_q[i]  
     c. Compensate remaining columns: 
        W[:, i+1:] -= (δ / H_inv[i,i]) * H_inv[i, i+1:]
     d. Update Hessian inverse for next iteration
4. Apply per-group scaling (group_size = 128 typical)
```

### Key Innovations

1. **Column blocking (B=128)**: Process 128 columns at a time instead of one-by-one, enabling efficient GPU utilization through batch matrix operations
2. **Lazy batch updates**: Accumulate compensation updates and apply them in one batched operation per block
3. **Cholesky decomposition**: Efficiently compute the needed rows of the inverse Hessian without full matrix inversion
4. **Dampening**: Add small λ to Hessian diagonal for numerical stability

### Implementation

```python
import torch
from gptq import GPTQ

def gptq_quantize_layer(weight, hessian, bits=4, group_size=128):
    """
    GPTQ quantization for a single linear layer.
    
    Args:
        weight: (out_features, in_features) weight matrix
        hessian: (in_features, in_features) Hessian matrix
        bits: target bit-width
        group_size: number of elements sharing scale/zero-point
    """
    rows, cols = weight.shape
    block_size = 128
    
    # Dampening for numerical stability
    damp = 0.01 * torch.diag(hessian).mean()
    hessian += damp * torch.eye(cols, device=weight.device)
    
    # Cholesky decomposition of inverse Hessian
    H_inv = torch.linalg.cholesky(torch.linalg.inv(hessian))
    
    w = weight.clone()
    w_q = torch.zeros_like(w)
    
    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        
        # Process each column in the block
        for j in range(block_start, block_end):
            # Quantize this column
            q = quantize_column(w[:, j], bits, group_size, j)
            w_q[:, j] = q
            
            # Error from quantization
            error = (w[:, j] - q) / H_inv[j, j]
            
            # Compensate remaining columns in this block
            w[:, j+1:block_end] -= error.unsqueeze(1) * H_inv[j, j+1:block_end].unsqueeze(0)
        
        # Compensate columns after this block (lazy update)
        w[:, block_end:] -= (w[:, block_start:block_end] - w_q[:, block_start:block_end]) @ \
                            H_inv[block_start:block_end, block_end:]
    
    return w_q
```

### Results

| Model | Bits | Perplexity (FP16) | Perplexity (GPTQ) | Degradation |
|-------|------|-------------------|-------------------|-------------|
| OPT-175B | 4-bit | 8.34 | 8.68 | +0.34 |
| OPT-175B | 3-bit | 8.34 | 9.56 | +1.22 |
| BLOOM-176B | 4-bit | — | Near-FP16 | <0.5 |
| LLaMA-65B | 4-bit | 3.53 | 3.84 | +0.31 |

**Performance:**
- OPT-175B 3/4-bit: **3.24x speedup** over FP16 baseline
- Quantizing OPT-175B takes approximately **4 GPU hours** (A100)
- Memory reduction: 175B × 2 bytes (FP16) → 175B × 0.5 bytes (4-bit) = **~87 GB → ~22 GB**

**Recommendation:** 4-bit is the sweet spot. 2-bit causes severe quality degradation (perplexity often doubles or worse), while 3-bit is marginal for larger models.

## AWQ - Activation-Aware Weight Quantization

### Key Insight

Not all weights are equally important. AWQ observes that **protecting only 1% of salient weights** (keeping them at higher precision or reducing their quantization error) can greatly reduce overall quantization error.

Critically, salience is determined by **activation magnitude**, not weight magnitude:

```
Weight importance ∝ |activation| × |weight|

Intuition: A weight is "salient" if it processes large activations,
because quantization error on that weight gets amplified by the 
large activation values it multiplies.
```

### Algorithm

```
1. Identify salient weights:
   - Run calibration data through the model
   - Compute average activation magnitude per input channel
   - Channels with high average activation = salient weight columns

2. Apply per-channel scaling:
   - Scale up salient weight channels before quantization
   - Scale down corresponding activation channels (mathematically equivalent)
   - This reduces relative quantization error for salient weights

3. Quantize scaled weights:
   - Standard per-group quantization (group_size=128)
   - Salient channels now have larger values → less relative error
```

### Mathematical Formulation

For a linear layer Y = X @ W:

```
Y = X @ W = (X / s) @ (s * W)   [mathematically equivalent for any s > 0]

Choose s per-channel to minimize quantization error:
  s_i = (max(|X[:, i]|))^α   where α ∈ (0, 1), typically α = 0.5

Effect:
  - Salient channels (large activations): s is large → weights scaled up → 
    quantization error is smaller relative to weight magnitude
  - Non-salient channels: s is small → weights scaled down → 
    higher relative error, but multiplied by small activations so impact is minimal
```

### Implementation

```python
def awq_quantize(model, calibration_data, bits=4, group_size=128):
    """
    AWQ: Activation-Aware Weight Quantization
    """
    for layer in model.layers:
        # Step 1: Collect activation statistics
        act_scales = collect_activation_scales(layer, calibration_data)
        # act_scales[i] = mean(|X[:, i]|) for input channel i
        
        # Step 2: Compute per-channel scaling factors
        weight = layer.weight  # (out_features, in_features)
        w_scales = weight.abs().max(dim=0).values  # max weight per input channel
        
        # Optimal scaling: balance activation importance and weight range
        scales = (act_scales.pow(0.5) / w_scales.pow(0.5)).clamp(min=1e-5)
        
        # Step 3: Apply scaling to weights (scale up salient channels)
        scaled_weight = weight * scales.unsqueeze(0)
        
        # Step 4: Quantize the scaled weights
        quantized_weight = per_group_quantize(scaled_weight, bits, group_size)
        
        # Step 5: Store inverse scales for runtime dequantization
        layer.weight_quantized = quantized_weight
        layer.input_scales = 1.0 / scales  # Applied to activations at runtime
    
    return model
```

### Advantages over GPTQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Speed | ~4 GPU-hours for 175B | Minutes for 175B |
| Calibration data | 128 samples typical | Small set sufficient |
| Method | Second-order (Hessian) | First-order (activations) |
| Generalization | Sensitive to calibration set | Better generalization |
| Reordering needed | Column reordering helps | No reordering |
| Hardware efficiency | Group dequant | Group dequant + channel scale |

## SmoothQuant

### Problem: Activation Outliers

Transformer models produce extreme outlier activations in specific channels (sometimes 100x larger than the median). These outliers make activation quantization extremely difficult:

```
Example activation tensor (one channel):
  Normal channels: values in [-1, 1]
  Outlier channel: values in [-100, 100]
  
Per-tensor INT8 scale = 200/255 ≈ 0.78
→ Normal channel resolution: only ~2-3 distinct quantized values!
```

### Solution: Smooth the Activation Distribution

SmoothQuant migrates the quantization difficulty from activations (hard to quantize) to weights (easy to quantize) through a mathematically equivalent transformation:

```
Y = X @ W = (X @ diag(s)^(-1)) @ (diag(s) @ W)
         = X_smooth @ W_smooth

Where:
  X_smooth = X / s     (activations divided by per-channel scale)
  W_smooth = s * W     (weights multiplied by per-channel scale)
  
Choose s to balance difficulty:
  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
  
Where α is the migration strength (α = 0.5 is optimal for many LLMs)
```

### Migration Strength α

```
α = 0:   No smoothing (all difficulty on activations) — original model
α = 0.5: Balanced — optimal for most LLMs (GPT, OPT, BLOOM)
α = 1:   All difficulty migrated to weights — rarely used (weights become hard)

The optimal α depends on the relative difficulty:
- If activations have extreme outliers and weights are well-behaved: α closer to 1
- If weights have large variation: α closer to 0
- In practice, α = 0.5 works well across GPT-3, OPT, BLOOM, LLaMA
```

### Implementation

```python
def smooth_quant(model, calibration_data, alpha=0.5):
    """
    Apply SmoothQuant transformation to enable W8A8 quantization.
    """
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
            
        # Collect activation statistics
        act_max = collect_channel_max(layer, calibration_data)  # per-channel max
        weight_max = layer.weight.abs().max(dim=0).values       # per-input-channel max
        
        # Compute smoothing scales
        scales = (act_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)
        
        # Apply smoothing (offline transformation)
        layer.weight.data *= scales.unsqueeze(0)  # Scale up weights
        
        # Store scales to apply to activations at runtime
        # (or fold into preceding LayerNorm)
        apply_act_scales(model, name, 1.0 / scales)
    
    # Now both weights AND activations can be quantized to INT8
    quantized_model = quantize_w8a8(model, calibration_data)
    return quantized_model
```

### Results

SmoothQuant enables **W8A8 quantization** (both weights and activations in INT8) for LLMs that previously could not be quantized to INT8:

| Model | W8A8 without SmoothQuant | W8A8 with SmoothQuant |
|-------|-------------------------|----------------------|
| OPT-175B | Fails (perplexity explodes) | Matches FP16 |
| BLOOM-176B | Fails | Matches FP16 |
| GLM-130B | Fails | Matches FP16 |

**Speedup:** 1.5x over FP16 inference (both compute and memory bandwidth benefits)

## OmniQuant

### Overview

OmniQuant combines two learnable techniques — Learnable Weight Clipping (LWC) and Learnable Equivalent Transformation (LET) — within a differentiable optimization framework that maintains PTQ-level computational efficiency.

### Learnable Weight Clipping (LWC)

Instead of using fixed min/max for weight quantization, learn optimal clipping thresholds:

```python
def learnable_weight_clipping(weight, bits=4, group_size=128):
    """
    Learn optimal clipping bounds that minimize quantization error.
    
    Standard: clip to [min(W), max(W)]
    LWC: clip to [min(W) * (1 + α_min), max(W) * (1 + α_max)]
    where α_min, α_max are learnable parameters
    """
    # Learnable parameters (initialized to 0 = no clipping change)
    alpha_min = nn.Parameter(torch.zeros(num_groups))
    alpha_max = nn.Parameter(torch.zeros(num_groups))
    
    # Compute clipped range
    w_min = weight.min(per_group) * (1 + alpha_min).sigmoid()  # sigmoid ensures [0, 1]
    w_max = weight.max(per_group) * (1 + alpha_max).sigmoid()
    
    # Quantize with learned clip range
    scale = (w_max - w_min) / (2**bits - 1)
    w_q = torch.clamp(torch.round((weight - w_min) / scale), 0, 2**bits - 1)
    w_dq = w_q * scale + w_min
    
    return w_dq
```

**Why LWC helps:** Extreme weight values (outliers) stretch the quantization range, reducing precision for the majority. Learning to clip these outliers optimally balances clipping error (for outliers) against rounding error (for the majority).

### Learnable Equivalent Transformation (LET)

Similar to SmoothQuant but with learned (not hand-crafted) transformation parameters:

```python
def learnable_equivalent_transformation(layer, calibration_data):
    """
    Learn per-channel scales that shift quantization difficulty
    from activations to weights.
    
    Y = X @ W = (X @ diag(s)^-1) @ (diag(s) @ W)
    But s is LEARNED to minimize quantized output error.
    """
    # Learnable per-channel scales (initialized to 1 = no transformation)
    log_scales = nn.Parameter(torch.zeros(in_features))
    
    def forward_with_let(x):
        scales = log_scales.exp()  # Positive scales
        x_smooth = x / scales
        w_smooth = layer.weight * scales.unsqueeze(0)
        
        # Quantize both
        x_q = fake_quantize(x_smooth, bits=8)
        w_q = fake_quantize(w_smooth, bits=4)
        
        return x_q @ w_q.T
    
    # Optimize scales to minimize output error
    optimizer = torch.optim.Adam([log_scales], lr=1e-3)
    for step in range(num_optimization_steps):
        x = sample_calibration_batch()
        target = x @ layer.weight.T  # FP32 output
        output = forward_with_let(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return log_scales.exp().detach()
```

### OmniQuant Results

| Model | Bits | Round-to-Nearest | GPTQ | OmniQuant |
|-------|------|-----------------|------|-----------|
| LLaMA-7B | W4A16 | 5.68 | 5.63 | **5.58** |
| LLaMA-13B | W4A16 | 5.09 | 5.04 | **4.99** |
| LLaMA-7B | W3A16 | 8.07 | 6.55 | **6.03** |
| LLaMA-7B | W4A4 | — | — | **7.15** |

OmniQuant excels at aggressive quantization (W3, W2) and weight+activation quantization (W4A4).

## GGUF/GGML (llama.cpp)

### Overview

GGUF (GPT-Generated Unified Format) is the file format and quantization system used by llama.cpp, the most popular framework for CPU-based LLM inference. It enables running large models on consumer hardware without GPU.

### Quantization Types

| Type | Bits/Weight | Description | Quality |
|------|-------------|-------------|---------|
| q4_0 | 4.5 | 4-bit, block_size=32, single scale | Low |
| q4_1 | 5.0 | 4-bit, block_size=32, scale + min | Medium-Low |
| q5_0 | 5.5 | 5-bit, block_size=32, single scale | Medium |
| q5_1 | 6.0 | 5-bit, block_size=32, scale + min | Medium-High |
| q8_0 | 8.5 | 8-bit, block_size=32, single scale | High |
| q4_K_M | ~4.8 | K-quant, mixed precision | **Recommended** |
| q5_K_M | ~5.7 | K-quant, mixed precision | High quality |
| q6_K | ~6.6 | K-quant, 6-bit | Near-FP16 |

### K-Quants: Importance-Based Mixed Precision

K-quants (introduced in llama.cpp) use different bit-widths for different parts of each layer based on importance:

```
Within each layer:
- Attention weights (Q, K, V, O): higher precision (e.g., 6-bit)
- MLP weights (gate, up, down): lower precision (e.g., 4-bit)
- Layer norm: keep FP32 or FP16

The '_M' suffix means 'medium' balance between quality and size
The '_S' suffix means 'small' (more aggressive compression)
The '_L' suffix means 'large' (less compression, higher quality)
```

### Block Structure

```
q4_K_M block layout (super-block of 256 elements):
┌─────────────────────────────────────────────────┐
│ Block header:                                    │
│   d (FP16): super-block scale                    │
│   dmin (FP16): super-block minimum               │
│   scales (6-bit × 16): sub-block scales          │
│   mins (6-bit × 16): sub-block minimums          │
│ Data:                                            │
│   256 × 4-bit quantized values                   │
└─────────────────────────────────────────────────┘

Dequantization:
  val = d * sub_scale * q_val - dmin * sub_min
```

### Performance Comparison

For 13B and 30B models, q4_K_M achieves **lower perplexity than GPTQ-4bit**:

```
LLaMA-13B perplexity (lower is better):
  FP16:     5.09
  q4_K_M:   5.36 (GGUF)
  GPTQ-4b:  5.40 (128 group-size)
  q4_0:     5.72 (basic GGUF)
  
LLaMA-30B perplexity:
  FP16:     4.10
  q4_K_M:   4.23 (GGUF)
  GPTQ-4b:  4.28
```

### Advantages of GGUF

- **CPU-friendly**: No GPU required, runs on any x86/ARM CPU
- **AVX/NEON optimized**: SIMD-optimized dequantization and GEMM
- **Memory-mapped**: Models loaded via mmap, enabling models larger than RAM
- **Progressive loading**: Partial model loading for extremely large models
- **Cross-platform**: Windows, Linux, macOS, Android, iOS

## SqueezeLLM

### Approach

SqueezeLLM combines two techniques:

1. **Sensitivity-weighted non-uniform quantization**: Uses Fisher information to determine where to place quantization levels (denser where sensitivity is high)
2. **Sparse outlier storage**: Keeps extreme weight values at full precision in a sparse format

```python
# SqueezeLLM conceptual implementation
def squeeze_llm_quantize(weight, sensitivity, bits=4, sparsity=0.005):
    """
    Args:
        weight: Layer weight matrix
        sensitivity: Fisher information (importance of each weight)
        bits: Target bit-width for the dense part
        sparsity: Fraction of weights to keep as sparse outliers
    """
    # Step 1: Identify and extract outliers
    threshold = torch.quantile(weight.abs(), 1 - sparsity)
    outlier_mask = weight.abs() > threshold
    outlier_values = weight[outlier_mask]  # Stored in FP16, sparse format
    
    # Step 2: Non-uniform quantization of remaining weights
    # Use sensitivity-weighted k-means to find optimal quantization levels
    non_outlier_weights = weight[~outlier_mask]
    centroids = sensitivity_weighted_kmeans(
        non_outlier_weights, 
        sensitivity[~outlier_mask],
        n_clusters=2**bits
    )
    
    # Step 3: Assign each weight to nearest centroid
    assignments = assign_to_nearest(non_outlier_weights, centroids)
    
    return centroids, assignments, outlier_mask, outlier_values
```

## QuIP (Quantization with Incoherence Processing)

### Key Idea

Weight matrices with "incoherent" (spread-out) entries quantize better than those with coherent (structured) entries. QuIP applies random orthogonal transformations to make weights more incoherent before quantization:

```
W_quantized = R @ Quantize(R^T @ W @ R') @ R'^T

Where R, R' are random orthogonal matrices (Hadamard transforms for efficiency)
```

### QuIP# (Sharp)

The enhanced version uses the E8 lattice for optimal vector quantization:
- 8-dimensional lattice with optimal packing
- Provides better rate-distortion tradeoff than scalar quantization
- Achieves 2-bit quantization with acceptable quality for large models

## SpQR (Sparse-Quantized Representation)

### Approach

SpQR identifies and isolates outlier weights that cause disproportionate quantization error:

```
Strategy:
1. Identify "sensitive" weights (outliers that cause large output error when quantized)
2. Keep sensitive weights at 16-bit precision (sparse storage)
3. Quantize remaining weights to 3-4 bits (dense storage)

Typical: ~1% of weights are sensitive
Storage: 3-bit dense + 1% sparse FP16 ≈ 3.2-3.5 effective bits
```

### Sensitivity Detection

```python
def detect_sensitive_weights(layer, calibration_data, threshold_percentile=99):
    """
    Identify weights whose quantization causes disproportionate output error.
    """
    weight = layer.weight
    
    # Compute per-weight sensitivity
    # = how much output changes when this specific weight is quantized
    sensitivity = torch.zeros_like(weight)
    
    for batch in calibration_data:
        output_fp32 = layer(batch)
        for i, j in weight_indices:
            # Quantize just this weight
            w_q = quantize_single(weight[i, j])
            delta_w = w_q - weight[i, j]
            # Output change ≈ delta_w * input[j] (first-order approximation)
            sensitivity[i, j] += abs(delta_w * batch[:, j].mean())
    
    # Mark top sensitive weights for FP16 storage
    threshold = torch.quantile(sensitivity, threshold_percentile / 100)
    sensitive_mask = sensitivity > threshold
    
    return sensitive_mask
```

## AQLM (Additive Quantization for Language Models)

### Approach

AQLM uses vector quantization with multiple codebooks (additive quantization):

```
Instead of scalar quantization (one number → one code):
  Vector quantization: group of numbers → sum of codebook entries

W[:, i:i+g] ≈ C_1[code_1[i]] + C_2[code_2[i]] + ... + C_M[code_M[i]]

Where:
  g = group size (8-16 elements typically)
  C_k = codebook k (learned, 256 entries × g dimensions)
  code_k[i] = 8-bit index into codebook k
  M = number of codebooks (1-2 typically)
```

### Effective Bit-Width

```
For group_size=8, M=2 codebooks:
  Storage per group: 2 × 8 bits (two codebook indices) = 16 bits
  Effective bits per weight: 16 / 8 = 2 bits per weight
  
Plus codebook storage (amortized over the layer): negligible for large matrices
```

## HQQ (Half-Quadratic Quantization)

### Key Innovation

HQQ formulates quantization as a half-quadratic optimization problem that **requires no calibration data**:

```
Objective: minimize || W - Dequant(Quant(W)) ||²

Approach: Alternating optimization
  1. Fix quantized values, optimize scales/zero-points
  2. Fix scales/zero-points, optimize quantized values
  
Each step has a closed-form solution → very fast optimization
```

### Advantages

- **No calibration data**: Entirely data-free
- **Very fast**: Seconds per layer (closed-form solutions)
- **Competitive quality**: Matches GPTQ at 4-bit, better at 3-bit and 2-bit
- **Simple**: Easy to implement and use

## EXL2 (ExLlamaV2)

### Variable Bit-Width Quantization

EXL2 assigns different bit-widths to different layers (and even parts of layers) based on a quality target:

```
Configuration: target average bits per weight (e.g., 4.0, 3.5, 5.0)

Optimization:
  Given a perplexity budget, find the bit allocation across layers
  that minimizes total perplexity degradation:
  
  min Σ_layers perplexity_increase(layer, bits[layer])
  s.t. average(bits) = target

Result: sensitive layers get more bits, robust layers get fewer
```

### Supported Bit-Widths

Per layer, EXL2 can assign: 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 8 bits

```
Example allocation for a 4.0 bpw target:
  Embedding layer: 8 bits (very sensitive)
  First attention: 6 bits
  Middle attention layers: 4 bits
  Feed-forward layers: 3.5 bits
  Last layer: 6 bits
  Output head: 8 bits
```

## Method Comparison Summary

| Method | Bits | Speed to Quantize | Calibration Data | Quality | Best For |
|--------|------|-------------------|------------------|---------|----------|
| GPTQ | 3-4 | Hours | ~128 samples | Excellent | GPU inference |
| AWQ | 4 | Minutes | Small set | Excellent | GPU inference |
| SmoothQuant | 8 (W8A8) | Fast | Calibration set | Near-lossless | Throughput |
| OmniQuant | 2-4 | Hours | Calibration set | Best at low-bit | Aggressive compression |
| GGUF K-quants | 2-8 | Minutes | None | Good | CPU inference |
| SqueezeLLM | 3-4 | Hours | Calibration set | Very good | Non-uniform quantization |
| SpQR | 3-4 | Hours | Calibration set | Very good | Outlier handling |
| AQLM | 2-3 | Hours | Training data | Best at 2-bit | Extreme compression |
| HQQ | 2-4 | Seconds | None | Good | Fast quantization |
| EXL2 | Variable | Minutes | Calibration set | Excellent | Flexible targets |

## References

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2022)
- Shao et al., "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models" (2023)
- Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization" (2023)
- Chee et al., "QuIP: 2-Bit Quantization of Large Language Models With Guarantees" (2023)
- Dettmers & Zettlemoyer, "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression" (2023)
- Egiazarian et al., "AQLM: Extreme Compression of Large Language Models via Additive Quantization" (2024)
- Badri & Shaji, "HQQ: Half-Quadratic Quantization" (2023)
- llama.cpp documentation and source code
- ExLlamaV2 documentation
