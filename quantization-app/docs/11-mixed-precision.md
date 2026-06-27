# Mixed Precision and Advanced Techniques

## Overview

Mixed precision quantization assigns different bit-widths to different parts of a model, recognizing that not all layers, channels, or operations contribute equally to model quality. Combined with advanced techniques like outlier handling, knowledge distillation, and progressive quantization, mixed precision enables aggressive compression without catastrophic quality loss.

## Mixed Precision Quantization

### Concept

Rather than applying a uniform bit-width across the entire model, mixed precision assigns bit-widths based on layer sensitivity:

```
Example mixed-precision allocation (LLM):
  Embedding layer:         FP16 (8 bits minimum)
  Attention Q, K, V:       4-bit (moderate sensitivity)
  Attention output proj:   4-bit
  MLP gate projection:     3-bit (less sensitive, large)
  MLP up projection:       3-bit
  MLP down projection:     4-bit (slightly more sensitive)
  Layer normalization:      FP32 (numerical precision critical)
  Output logits head:       FP16 or 8-bit (high sensitivity)
  
Average: ~3.8 bits per weight parameter
```

### Layer Sensitivity Analysis

Different layers exhibit dramatically different sensitivity to quantization:

```
Highly Sensitive Layers:
├── First layer (embedding projection)
│   └── Interfaces directly with input; errors propagate through entire network
├── Last layer (output head / logits)
│   └── Final prediction; errors directly affect output quality
├── Attention layers (especially Q, K)
│   └── Small magnitude weights, softmax amplifies errors
├── Layers after residual connections
│   └── Quantization error accumulates through skip connections
└── Normalization layers
    └── Variance computation is numerically sensitive

Less Sensitive Layers:
├── Large feed-forward / MLP layers
│   └── High redundancy, many parameters absorb noise
├── Middle layers (neither first nor last)
│   └── Errors can be corrected by subsequent layers
└── Layers with heavy overparameterization
    └── Quantization acts as regularization
```

### Sensitivity Measurement Methods

```python
def measure_layer_sensitivity(model, calibration_data, metric='perplexity'):
    """
    Measure each layer's sensitivity to quantization by quantizing
    one layer at a time and measuring quality degradation.
    """
    baseline_score = evaluate(model, calibration_data, metric)
    sensitivities = {}
    
    for layer_name, layer in model.named_modules():
        if not is_quantizable(layer):
            continue
        
        # Quantize only this layer
        model_copy = copy_model(model)
        quantize_layer(model_copy, layer_name, bits=4)
        
        # Measure degradation
        score = evaluate(model_copy, calibration_data, metric)
        sensitivities[layer_name] = baseline_score - score
    
    return sensitivities

# Alternative: use Hessian trace as sensitivity proxy
def hessian_sensitivity(model, calibration_data):
    """
    Layers with larger Hessian trace are more sensitive.
    Hessian trace ≈ sum of squared gradients (Fisher information).
    """
    sensitivities = {}
    
    for batch in calibration_data:
        loss = model(batch).loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in sensitivities:
                    sensitivities[name] = 0
                sensitivities[name] += (param.grad ** 2).sum().item()
    
    return sensitivities
```

### Bit-Width Assignment Algorithms

**Greedy approach:**
```python
def assign_bitwidths(sensitivities, target_avg_bits, available_bits=[2,3,4,5,6,8]):
    """
    Assign bit-widths to minimize total sensitivity-weighted quantization error
    subject to average bit-width constraint.
    """
    # Sort layers by sensitivity (most sensitive first)
    layers_sorted = sorted(sensitivities.items(), key=lambda x: -x[1])
    
    # Start with minimum bits everywhere
    assignments = {name: min(available_bits) for name, _ in layers_sorted}
    
    # Iteratively increase bits for most sensitive layers
    while average_bits(assignments) < target_avg_bits:
        # Find the layer where increasing bits gives most benefit
        best_layer = None
        best_benefit = -1
        
        for name, sensitivity in layers_sorted:
            current_bits = assignments[name]
            next_bits = next_higher(current_bits, available_bits)
            if next_bits is None:
                continue
            benefit = sensitivity * (error_reduction(current_bits, next_bits))
            if benefit > best_benefit:
                best_benefit = benefit
                best_layer = name
        
        if best_layer:
            assignments[best_layer] = next_higher(assignments[best_layer], available_bits)
        else:
            break
    
    return assignments
```

### EXL2's Variable Bit-Width

EXL2 (ExLlamaV2) implements practical mixed precision with fine granularity:

```
Process:
1. Measure per-layer sensitivity (perplexity impact of quantizing each layer)
2. Formulate as optimization: minimize Σ PPL_increase(layer_i, bits_i)
   subject to: average(bits) = target (e.g., 4.0 bpw)
3. Use dynamic programming to find optimal assignment
4. Quantize each layer at its assigned bit-width

Supported per-layer bit-widths: 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 8
(fractional bits achieved through mixed coding within a layer)
```

## Weight-Only vs Weight+Activation Quantization

### Weight-Only Quantization (W4A16, W8A16)

```
Architecture:
  Stored weights: INT4/INT8
  At runtime: dequantize weights to FP16, then compute in FP16
  Activations: always FP16

Inference path:
  x_fp16 → Load(W_int4) → Dequant(W_int4 → W_fp16) → MatMul(x_fp16, W_fp16) → y_fp16

Benefits:
  - Reduces weight memory (dominant for LLMs at batch=1)
  - Preserves activation precision (no accumulation of activation errors)
  - Simpler to implement (no activation calibration)
  - Better quality than W+A quantization at same bit-width
  
Limitations:
  - Dequantization overhead at runtime
  - No compute speedup (still FP16 math)
  - Less benefit at large batch sizes (become compute-bound)
```

### Weight+Activation Quantization (W8A8, W4A4)

```
Architecture:
  Stored weights: INT8/INT4
  Activations: also quantized to INT8/INT4
  Computation: integer arithmetic (Tensor Core accelerated)

Inference path:
  x_fp16 → Quant(x → x_int8) → INT8_MatMul(x_int8, W_int8) → Dequant → y_fp16

Benefits:
  - Both memory AND compute reduction
  - Tensor Core acceleration (INT8 TOPS >> FP16 TFLOPS)
  - Better at large batch sizes (compute-bound regime)
  - Lower total memory including activations
  
Limitations:
  - Activation quantization is difficult (outliers, dynamic range)
  - Requires calibration for activation ranges
  - Quality loss from quantizing both dimensions
  - Some ops must stay in higher precision (softmax, layernorm)
```

### Tradeoff Summary

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| LLM, batch=1 | Weight-only (W4A16) | Memory-bound, quality preserved |
| LLM, batch=32+ | W8A8 (SmoothQuant) | Compute-bound, need TOPS |
| CNN classification | W8A8 static | Balanced, well-supported |
| Edge/mobile | W8A8 or W4A8 | Memory + compute constraints |
| Fine-tuning/training | W4A16 (QLoRA) | Need gradient precision |

## Outlier Handling Strategies

Activation outliers are the primary challenge for quantization of transformer models. Specific attention heads produce activation values 10-100x larger than the median, destroying quantization resolution.

### Strategy 1: Averaging (Replacing Global Min/Max)

Instead of using the global min/max (sensitive to single outliers), use statistics of per-sample min/max:

```python
def averaging_calibration(activations_per_sample):
    """
    Instead of: scale = max_over_all_samples(activations)
    Use:        scale = average_over_samples(per_sample_max)
    """
    per_sample_max = [sample.max() for sample in activations_per_sample]
    per_sample_min = [sample.min() for sample in activations_per_sample]
    
    # Average min/max is less sensitive to single extreme samples
    robust_max = np.mean(per_sample_max)
    robust_min = np.mean(per_sample_min)
    
    scale = (robust_max - robust_min) / 255
    return scale
```

### Strategy 2: Mean ± N Standard Deviations

Clip the quantization range to μ ± Nσ:

```python
def std_based_clipping(tensor, n_sigma=3.0):
    """
    Clip range to mean ± N standard deviations.
    Assumes roughly Gaussian distribution.
    
    N=3: captures 99.7% of values (some clipping)
    N=4: captures 99.99% of values (minimal clipping)
    N=6: captures 99.9999% (almost no clipping)
    """
    mean = tensor.mean()
    std = tensor.std()
    
    clip_min = mean - n_sigma * std
    clip_max = mean + n_sigma * std
    
    # Compute scale based on clipped range
    scale = (clip_max - clip_min) / 255
    zero_point = round(-clip_min / scale)
    
    return scale, zero_point
```

### Strategy 3: ACIQ (Analytical Clipping for Integer Quantization)

ACIQ derives the optimal clipping threshold analytically, assuming the tensor follows a known distribution (Gaussian or Laplace):

```python
def aciq_optimal_clip(tensor, bits=8, distribution='gaussian'):
    """
    ACIQ: find optimal clipping threshold that minimizes
    total quantization error (clipping error + rounding error).
    
    For Gaussian distribution:
      optimal_clip ≈ α * σ
      where α depends on bit-width:
        8-bit: α ≈ 5.03
        4-bit: α ≈ 2.83  
        2-bit: α ≈ 1.71
    """
    if distribution == 'gaussian':
        # Precomputed optimal α for different bit-widths
        alpha_table = {8: 5.03, 7: 4.42, 6: 3.81, 5: 3.22, 4: 2.83, 3: 2.12, 2: 1.71}
        alpha = alpha_table[bits]
        
        std = tensor.std()
        optimal_threshold = alpha * std
        
    elif distribution == 'laplace':
        # For Laplace: α values are different
        alpha_table = {8: 6.85, 4: 3.89, 2: 2.38}
        alpha = alpha_table[bits]
        
        b = tensor.abs().mean()  # Laplace scale parameter
        optimal_threshold = alpha * b
    
    # Clip and quantize
    clipped = tensor.clamp(-optimal_threshold, optimal_threshold)
    scale = 2 * optimal_threshold / (2**bits - 1)
    
    return scale, clipped
```

**Advantage:** No search or calibration needed — the optimal clip is computed in closed form from the tensor statistics. Works well when the distribution assumption holds.

### Strategy 4: SmoothQuant's Migration Approach

Instead of handling outliers after they form, prevent them from being problematic by migrating the quantization challenge from activations to weights:

```
Original:  Y = X @ W    (X has outliers, W is well-behaved)
Smoothed:  Y = (X/s) @ (s*W)  (X/s has no outliers, s*W still well-behaved)

The key insight: multiplying weights by s only slightly increases their range,
but dividing activations by s dramatically reduces their outlier magnitudes.
Because weights have more redundancy, they can absorb the scale better.
```

See [08-llm-methods.md](./08-llm-methods.md) for full SmoothQuant details.

### Strategy 5: Keeping Outlier Weights at Higher Precision (SpQR Approach)

Identify individual weights whose quantization causes disproportionate error, and keep them at FP16 in a sparse format:

```python
def spqr_mixed_precision(weight, sensitivity, outlier_fraction=0.01):
    """
    Keep top 1% of sensitive weights at FP16 (sparse),
    quantize remaining 99% to 3-4 bits (dense).
    """
    # Identify outlier weights
    threshold = torch.quantile(sensitivity, 1 - outlier_fraction)
    outlier_mask = sensitivity > threshold
    
    # Sparse storage for outliers
    outlier_indices = outlier_mask.nonzero()
    outlier_values = weight[outlier_mask]  # FP16
    
    # Dense quantization for non-outliers
    weight_no_outliers = weight.clone()
    weight_no_outliers[outlier_mask] = 0  # Will be added from sparse
    quantized_dense = quantize(weight_no_outliers, bits=3)
    
    # At inference: output = dense_part + sparse_part
    # dense_part = dequant(quantized_dense) @ x
    # sparse_part = scatter(outlier_values, outlier_indices) @ x
    
    return quantized_dense, outlier_indices, outlier_values

# Effective storage:
# 99% at 3 bits + 1% at 16 bits + index overhead
# ≈ 3 × 0.99 + 16 × 0.01 + overhead ≈ 3.2-3.5 bits per weight
```

### Comparison of Outlier Strategies

| Strategy | Complexity | Quality | Speed Impact | Calibration |
|----------|-----------|---------|--------------|-------------|
| Averaging | Low | Moderate | None | Per-sample stats |
| Mean ± Nσ | Low | Moderate | None | Distribution stats |
| ACIQ | Low | Good | None | Analytical |
| SmoothQuant | Medium | Excellent | Minimal | Activation stats |
| SpQR (sparse) | High | Excellent | Some (sparse ops) | Sensitivity analysis |

## Knowledge Distillation + Quantization

Knowledge distillation uses a large teacher model to guide the training of a smaller/quantized student model.

### Basic Distillation for Quantization

```python
def distillation_quantized_training(
    teacher_fp32, 
    student_quantized, 
    train_data,
    alpha=0.7,           # Weight for distillation loss
    temperature=4.0,     # Softens teacher predictions
    num_epochs=10
):
    """
    Train quantized student to match teacher's soft predictions.
    """
    optimizer = torch.optim.Adam(student_quantized.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch in train_data:
            # Teacher predictions (soft targets)
            with torch.no_grad():
                teacher_logits = teacher_fp32(batch['input'])
            
            # Student predictions
            student_logits = student_quantized(batch['input'])
            
            # Hard loss (standard task loss)
            hard_loss = F.cross_entropy(student_logits, batch['labels'])
            
            # Soft loss (match teacher distribution)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Combined loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Layer-wise Distillation (for LLMs)

For large models where end-to-end distillation is impractical:

```python
def layerwise_distillation(teacher, student_quantized, calibration_data):
    """
    Match each quantized layer's output to the teacher's layer output.
    More memory-efficient than end-to-end for large models.
    """
    for layer_idx in range(num_layers):
        # Get teacher's layer output
        with torch.no_grad():
            teacher_input = get_layer_input(teacher, layer_idx, calibration_data)
            teacher_output = teacher.layers[layer_idx](teacher_input)
        
        # Optimize quantized layer to match
        layer = student_quantized.layers[layer_idx]
        optimizer = torch.optim.Adam(
            get_trainable_params(layer),  # Scales, zero-points, or weight corrections
            lr=1e-4
        )
        
        for step in range(1000):
            student_output = layer(teacher_input)
            loss = F.mse_loss(student_output, teacher_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Self-Distillation

The FP32 version of the same model acts as teacher for its quantized version:

```
Teacher: Model_FP32 (same architecture, same weights before quantization)
Student: Model_INT4 (quantized version)

Benefits:
  - No separate teacher needed
  - Architecture perfectly matched
  - Focuses on recovering quantization-specific losses
  - Works with QAT (fake-quant student matches FP32 teacher)
```

## Progressive Quantization

Gradually reduce precision during training, allowing the model to adapt incrementally:

### Linear Schedule

```python
def progressive_quantization_training(model, train_data, 
                                       start_bits=8, target_bits=4,
                                       total_epochs=30):
    """
    Gradually decrease quantization precision during training.
    """
    for epoch in range(total_epochs):
        # Linear bit-width reduction
        progress = epoch / total_epochs
        current_bits = start_bits - progress * (start_bits - target_bits)
        current_bits = max(target_bits, round(current_bits))
        
        # Update fake-quant nodes with current bit-width
        update_quantization_bits(model, current_bits)
        
        # Train for one epoch
        train_one_epoch(model, train_data)
    
    # Final fine-tuning at target precision
    update_quantization_bits(model, target_bits)
    for epoch in range(5):
        train_one_epoch(model, train_data, lr=1e-5)
```

### Stage-Based Schedule

```python
stages = [
    {'bits': 8, 'epochs': 5,  'lr': 1e-4},  # Start gentle
    {'bits': 6, 'epochs': 5,  'lr': 5e-5},  # Intermediate
    {'bits': 4, 'epochs': 10, 'lr': 2e-5},  # Target precision
    {'bits': 4, 'epochs': 5,  'lr': 1e-6},  # Final fine-tuning
]

for stage in stages:
    update_quantization_bits(model, stage['bits'])
    optimizer = create_optimizer(model, lr=stage['lr'])
    train_epochs(model, train_data, optimizer, stage['epochs'])
```

### Benefits of Progressive Quantization

1. **Avoids sudden quality collapse**: Model adapts gradually instead of sudden precision change
2. **Better convergence**: Provides a curriculum from easy (high bits) to hard (low bits)
3. **Useful for extreme compression**: Essential for achieving usable 2-3 bit models
4. **Combined with other techniques**: Often paired with distillation for best results

## Advanced Mixed Precision Techniques

### Attention-Specific Quantization

Attention layers require special handling due to the softmax operation:

```python
def quantize_attention(attention_layer, config):
    """
    Mixed precision specifically for self-attention.
    """
    return {
        'Q_proj': quantize(attention_layer.q_proj, bits=config.qkv_bits),    # 4-8 bit
        'K_proj': quantize(attention_layer.k_proj, bits=config.qkv_bits),    # 4-8 bit
        'V_proj': quantize(attention_layer.v_proj, bits=config.v_bits),      # Often higher
        'O_proj': quantize(attention_layer.o_proj, bits=config.out_bits),    # 4-8 bit
        'QK_matmul': 'fp16',  # Keep in FP16 (small values, softmax sensitivity)
        'softmax': 'fp32',    # Always FP32 (numerical stability)
        'AV_matmul': config.av_precision,  # Can be INT8 or FP16
        'KV_cache': config.kv_cache_bits,  # 4-8 bit for memory savings
    }
```

### KV-Cache Quantization

For long-sequence LLM inference, the KV-cache becomes a major memory consumer:

```python
def quantize_kv_cache(key_states, value_states, bits=8, group_size=128):
    """
    Quantize KV-cache to reduce memory during long-sequence generation.
    
    Memory savings:
      Sequence 4096, hidden 4096, 32 layers, batch 1:
      FP16 KV-cache: 2 × 4096 × 4096 × 32 × 2 bytes = 2 GB
      INT4 KV-cache: 2 × 4096 × 4096 × 32 × 0.5 bytes = 0.5 GB
      Savings: 1.5 GB (75% reduction)
    """
    k_quantized = per_group_quantize(key_states, bits, group_size)
    v_quantized = per_group_quantize(value_states, bits, group_size)
    
    return k_quantized, v_quantized
```

### Activation-Aware Mixed Precision (HAWQ)

HAWQ (Hessian-Aware Quantization) uses second-order information to determine optimal bit-widths:

```
For each layer i:
  sensitivity_i = trace(H_i)  (trace of the Hessian for layer i's parameters)
  
Layers with larger Hessian trace → more sensitive → assign more bits
Layers with smaller Hessian trace → less sensitive → assign fewer bits

Optimization:
  min Σ_i sensitivity_i × quantization_error_i(bits_i)
  s.t. Σ_i size_i(bits_i) ≤ target_model_size
```

## Practical Mixed Precision Recipes

### Recipe: LLM Serving (Quality-Focused)

```
Target: 4-bit average, preserve quality
─────────────────────────────────────
Embedding:          8-bit
First 2 layers:     6-bit (sensitive to input representation)
Middle layers:      4-bit (bulk of parameters)
Last 2 layers:      6-bit (sensitive to output quality)
LM head:            8-bit
LayerNorm:          FP32
Softmax:            FP32

Effective: ~4.2 bits average
Quality: within 0.1 PPL of uniform 4-bit
```

### Recipe: Edge Deployment (Size-Focused)

```
Target: minimum model size, acceptable quality
────────────────────────────────────────────
Embedding:          4-bit (large, must compress)
All attention:      4-bit
All MLP:            3-bit (largest layers, compress most)
LM head:            4-bit
LayerNorm:          FP16
Activations:        INT8

Effective: ~3.2 bits average
Quality: ~1.0 PPL increase (acceptable for edge)
Size: ~40% of uniform 4-bit
```

### Recipe: Maximum Throughput (Latency-Focused)

```
Target: maximum inference speed
────────────────────────────────
All weights:        INT8 (per-channel, symmetric)
All activations:    INT8 (per-tensor, symmetric)
Accumulation:       INT32
Post-accumulation:  Requantize to INT8
LayerNorm:          FP32 (fused with quant/dequant)
Softmax:            FP32

Effective: W8A8 throughout
Speed: 2-4x vs FP16 (Tensor Core INT8)
Quality: <1% accuracy loss
```

## References

- Wang et al., "HAQ: Hardware-Aware Automated Quantization with Mixed Precision" (2019)
- Dong et al., "HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision" (2019)
- Dong et al., "HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks" (2020)
- Park et al., "Profit: A Novel Training Method for sub-4-bit MobileNet Models" (2020)
- Banner et al., "ACIQ: Analytical Clipping for Integer Quantization of Neural Networks" (2019)
- Dettmers & Zettlemoyer, "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression" (2023)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2022)
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Polino et al., "Model compression via distillation and quantization" (2018)
- Zafrir et al., "Q8BERT: Quantized 8Bit BERT" (2019)
