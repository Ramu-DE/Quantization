# Post-Training Quantization (PTQ)

## Overview

Post-Training Quantization (PTQ) converts a pre-trained FP32 model to lower precision without any retraining or fine-tuning. This makes it the most accessible quantization approach — requiring only a trained model and optionally a small calibration dataset.

**Key advantages:**
- No training infrastructure needed
- Fast to apply (minutes to hours vs days of retraining)
- No access to original training data required (for dynamic PTQ)
- Minimal hyperparameter tuning

**Typical results:**
- **W8A8** (8-bit weights and activations): accuracy within **0.7-1%** of FP32 baselines for standard architectures (ResNet50, MobileNetV2, BERT-base)
- INT8 is considered "drop-in" for most production models
- Quality degrades significantly below 8 bits without additional techniques

## Static vs Dynamic PTQ

### Dynamic Quantization

In dynamic quantization, weight quantization parameters are computed offline (at conversion time), but **activation quantization parameters are computed at runtime** for each input batch.

```python
# PyTorch Dynamic Quantization Example
import torch

model_fp32 = load_model()

# Apply dynamic quantization - only weights are pre-quantized
# Activations are quantized dynamically at inference time
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)
```

**Characteristics:**
- No calibration dataset required
- Computes scale and zero-point for activations at runtime based on actual input
- Slightly slower inference (overhead of computing activation statistics per batch)
- Best for models dominated by large linear layers (NLP models like BERT, GPT)
- Weights are always statically quantized

**When to use:**
- Quick deployment with minimal effort
- Models with highly variable activation ranges across inputs
- When calibration data is unavailable
- LSTM/RNN models where activation ranges vary significantly across time steps

### Static Quantization

In static quantization, both weight and activation quantization parameters are **pre-computed offline** using a calibration dataset.

```python
# PyTorch Static Quantization Example
import torch
from torch.quantization import prepare, convert, default_qconfig

model_fp32 = load_model()
model_fp32.eval()

# Specify quantization configuration
model_fp32.qconfig = default_qconfig

# Insert observers to collect activation statistics
model_prepared = prepare(model_fp32)

# Run calibration (typically ~200 examples sufficient)
calibration_loader = get_calibration_data(num_samples=200)
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

# Convert to quantized model
model_int8 = convert(model_prepared)
```

**Characteristics:**
- Requires a representative calibration dataset (~200 examples is typically sufficient)
- No runtime overhead for computing activation statistics
- Faster inference than dynamic quantization
- Better accuracy when activation distributions are consistent across inputs
- Standard for CNN models in production (image classification, detection)

**When to use:**
- Production deployment where latency matters
- Models with consistent activation distributions (most CNNs)
- When calibration data is available (doesn't need labels)

### Comparison

| Aspect | Dynamic PTQ | Static PTQ |
|--------|-------------|------------|
| Calibration data | Not needed | ~200 samples |
| Runtime overhead | Yes (compute act. stats) | None |
| Inference speed | Moderate | Fastest |
| Accuracy | Good | Better (usually) |
| Activation handling | Per-batch statistics | Pre-computed fixed stats |
| Best for | NLP, variable activations | CNNs, stable activations |

## Calibration Methods

Calibration is the process of determining optimal scale and zero-point values for static quantization. The choice of calibration method is **critical** — no single method is universally best, and a poor choice can cause catastrophic accuracy loss.

### MinMax Calibration

The simplest approach: uses the actual minimum and maximum values observed during calibration.

```python
# Pseudocode
scale = (observed_max - observed_min) / (q_max - q_min)
zero_point = round(q_min - observed_min / scale)
```

**Pros:**
- Simple to implement
- No information loss within observed range
- Preserves extreme values

**Cons:**
- Highly sensitive to outliers — a single extreme value stretches the entire range
- Results in poor resolution for the majority of values when outliers exist
- Can cause catastrophic failures for some architectures

**WARNING:** MinMax (max) calibration can cause catastrophic failures:
- **Inception v4**: drops to **0.12%** accuracy (from 80.16% FP32 baseline)
- **EfficientNet-b0**: drops to **22.3%** accuracy (from 76.85% FP32 baseline)

These failures occur because outlier activations stretch the quantization range, leaving insufficient resolution for the majority of values.

### Entropy (KL-Divergence) Calibration

Finds the clipping threshold that minimizes the KL-divergence (information loss) between the original FP32 distribution and the quantized distribution.

```python
# Pseudocode for entropy calibration
def find_optimal_threshold(histogram, num_quantized_bins=128):
    """
    Search over candidate thresholds to minimize KL-divergence
    between original and quantized distributions.
    """
    best_divergence = float('inf')
    best_threshold = None
    
    for threshold in candidate_thresholds:
        # Create reference distribution (original, clipped at threshold)
        reference = clip_and_normalize(histogram, threshold)
        
        # Create candidate quantized distribution
        quantized = quantize_distribution(reference, num_quantized_bins)
        
        # Compute KL divergence
        divergence = kl_divergence(reference, quantized)
        
        if divergence < best_divergence:
            best_divergence = divergence
            best_threshold = threshold
    
    return best_threshold
```

**Pros:**
- Principled information-theoretic approach
- Best for activations (which often have long tails)
- Used by NVIDIA TensorRT as one of its calibration options
- Robust to outliers (will clip extreme values if it reduces overall information loss)

**Cons:**
- More computationally expensive
- Requires collecting activation histograms
- May clip important extreme values in some cases

### Percentile Calibration

Clips the range at a specified percentile (e.g., 99.99% or 99.999%) of the observed values, discarding extreme outliers.

```python
# Pseudocode
def percentile_calibration(collected_values, percentile=99.99):
    sorted_values = sort(abs(collected_values))
    threshold = sorted_values[int(len(sorted_values) * percentile / 100)]
    scale = 2 * threshold / (q_max - q_min)  # symmetric
    return scale
```

**Pros:**
- Robust to outliers
- Simple and intuitive
- Configurable aggressiveness (99.99% vs 99.9%)

**Cons:**
- Percentile choice is a hyperparameter that needs tuning
- May discard important signal in the tails

**Recommended percentiles:**
- 99.99% — conservative, good default
- 99.999% — very conservative, closer to MinMax
- 99.9% — aggressive clipping, may lose information

### MSE (Mean Squared Error) Calibration

Finds the clipping threshold that minimizes the mean squared error between original and quantized values.

```python
# Pseudocode
def mse_calibration(collected_values):
    best_mse = float('inf')
    best_threshold = None
    
    for threshold in np.linspace(min_val, max_val, num_candidates):
        # Quantize with this threshold
        quantized = quantize(collected_values, threshold)
        
        # Compute MSE
        mse = mean((collected_values - dequantize(quantized))**2)
        
        if mse < best_mse:
            best_mse = mse
            best_threshold = threshold
    
    return best_threshold
```

**Pros:**
- Directly minimizes reconstruction error
- Good balance between clipping and rounding error
- Works well for both weights and activations

**Cons:**
- Computationally more expensive than MinMax/Percentile
- May not correlate perfectly with task accuracy

### Calibration Method Comparison

| Method | Best For | Risk | Compute Cost |
|--------|----------|------|--------------|
| MinMax | Well-behaved distributions | Catastrophic with outliers | Very low |
| Entropy/KL | Activations | May over-clip | Medium |
| Percentile | General purpose | Hyperparameter sensitivity | Low |
| MSE | Weights | May not optimize task metric | Medium |

### Calibration Best Practices

1. **Use multiple methods and compare**: No single method is universally best
2. **Validate on held-out data**: Always check task accuracy after quantization
3. **Collect diverse calibration data**: Ensure calibration set covers the expected input distribution
4. **200 samples is typically sufficient**: Diminishing returns beyond this for most models
5. **Never use MinMax alone for sensitive models**: Always have entropy/percentile as backup
6. **Per-channel calibration**: Combine with per-channel quantization for best results

## Layer Sensitivity

Not all layers are equally sensitive to quantization:

```
Highly sensitive:
- First and last layers (interface with input/output)
- Attention layers (small weight magnitudes, large dynamic range)
- Depthwise separable convolutions
- Batch normalization (when folded)

Less sensitive:
- Large fully-connected layers (high redundancy)
- Middle convolution layers
- Layers with ReLU activation (bounded output)
```

### Mixed-Precision Strategy

For PTQ, a common strategy is to keep sensitive layers at higher precision:

```python
# Example: Skip quantization for first/last layers
sensitive_layers = ['conv1', 'fc_final', 'attention.qkv']

for name, module in model.named_modules():
    if name in sensitive_layers:
        module.qconfig = None  # Keep FP32
    else:
        module.qconfig = default_qconfig
```

## Advanced PTQ Techniques

### AdaRound (Adaptive Rounding)

Instead of simple round-to-nearest, learns optimal rounding direction (up or down) for each weight:

```
Standard: w_q = round(w / S)
AdaRound: w_q = floor(w / S) + h(V)  where h(V) ∈ {0, 1} is learned
```

This can recover 1-2% accuracy for 4-bit quantization without any fine-tuning.

### Bias Correction

Quantization introduces a systematic bias in layer outputs. Bias correction compensates:

```python
# After quantization, correct the bias
expected_shift = E[W_q * x] - E[W * x]  # Estimated on calibration data
layer.bias -= expected_shift
```

### Cross-Layer Equalization (CLE)

Exploits the scale-equivariance of ReLU: `ReLU(sx) = s * ReLU(x)` for `s > 0`.

Rescales weight ranges across consecutive layers to make them more quantization-friendly without changing the model's function:

```
W1_new = W1 * S
W2_new = W2 / S
# Choose S to equalize ranges: s_i = sqrt(range(W1[:,i]) / range(W2[i,:]))
```

## Practical Workflow

```
1. Train model in FP32 (or start with pre-trained model)
2. Prepare calibration dataset (~200 representative samples)
3. Choose quantization configuration:
   - Bit-width: INT8 for safe, INT4 for aggressive compression
   - Granularity: Per-channel for weights, per-tensor for activations
   - Calibration: Start with percentile/entropy, validate
4. Run calibration
5. Evaluate quantized model on validation set
6. If accuracy insufficient:
   a. Try different calibration methods
   b. Use per-channel instead of per-tensor
   c. Apply mixed-precision (keep sensitive layers in FP32/FP16)
   d. Consider AdaRound or bias correction
   e. Fall back to QAT if all else fails
7. Deploy quantized model
```

## Common Pitfalls

1. **Using MinMax without validation**: Always check accuracy — MinMax can catastrophically fail
2. **Insufficient calibration data**: Too few samples may not capture the activation distribution
3. **Ignoring batch normalization folding**: BN folding before quantization is essential; folding after can cause issues
4. **Per-tensor for depthwise convolutions**: These layers have highly varied per-channel ranges; always use per-channel
5. **Quantizing everything uniformly**: Some layers need higher precision — use sensitivity analysis

## References

- Nagel et al., "A White Paper on Neural Network Quantization" (2021), Qualcomm AI Research
- Banner et al., "Post-Training 4-bit Quantization of Convolution Networks for Rapid-Deployment" (2019)
- Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias Correction" (2019)
- Li et al., "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction" (2021)
- Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation" (2020)
- Hubara et al., "Accurate Post Training Quantization With Small Calibration Sets" (2020)
