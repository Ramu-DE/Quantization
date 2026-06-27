# Google Quantization Approaches

## Overview

Google has been a pioneer in neural network quantization, driven by the need to deploy models efficiently on mobile devices (via TensorFlow Lite), TPUs, and cloud infrastructure. Their contributions include the TensorFlow Model Optimization Toolkit, the gemmlowp library, quantization research (BRECQ, learned quantization), and production quantization for their own models.

## TensorFlow Model Optimization Toolkit

### Quantization-Aware Training (QAT)

The TF Model Optimization Toolkit provides a high-level API for QAT with TensorFlow/Keras models:

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load pre-trained model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Apply QAT to the entire model
qat_model = tfmot.quantization.keras.quantize_model(base_model)

# Compile with standard training configuration
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune (typically 10% of original training)
qat_model.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=2)
    ]
)

# Convert to quantized TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite = converter.convert()

# Save
with open('model_quantized.tflite', 'wb') as f:
    f.write(quantized_tflite)
```

### Selective Quantization

Apply QAT only to specific layers:

```python
import tensorflow_model_optimization as tfmot

# Define which layers to quantize
def apply_quantization(layer):
    # Skip batch normalization and certain sensitive layers
    if isinstance(layer, (tf.keras.layers.BatchNormalization,)):
        return layer
    # Quantize Conv2D and Dense layers
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

# Clone model with selective quantization annotations
annotated_model = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_quantization
)

# Apply quantization to annotated layers
qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
```

### Custom Quantization Configuration

```python
# Define custom quantization config
class Custom8BitQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, 
                 tfmot.quantization.keras.quantizers.LastValueQuantizer(
                     num_bits=8, symmetric=True, narrow_range=False,
                     per_axis=True, axis=(-1,)  # Per-channel
                 ))]
    
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation,
                 tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                     num_bits=8, symmetric=False, narrow_range=False,
                     per_axis=False
                 ))]
    
    def get_output_quantizers(self, layer):
        return []
    
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
    
    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]
```

### Published QAT Results

Google's published results demonstrate near-lossless INT8 quantization:

| Model | Non-quantized (Top-1) | QAT Quantized (Top-1) | Difference |
|-------|----------------------|----------------------|------------|
| MobileNetV1 | 71.03% | 71.06% | **+0.03%** |
| MobileNetV2 | 71.77% | 71.56% | -0.21% |
| ResNet50 | 76.3% | 76.1% | -0.2% |
| InceptionV3 | 78.0% | 77.8% | -0.2% |
| NASNet-Mobile | 74.0% | 73.7% | -0.3% |

Notable: MobileNetV1 with QAT actually **exceeds** the FP32 baseline by 0.03%, demonstrating that QAT can act as a regularizer.

### Fake Quantization Nodes

TensorFlow inserts `tf.quantization.fake_quant_with_min_max_vars` operations during training:

```python
# What happens inside TF QAT (conceptual)
class FakeQuantLayer(tf.keras.layers.Layer):
    def __init__(self, num_bits=8, narrow_range=False):
        super().__init__()
        self.num_bits = num_bits
        self.narrow_range = narrow_range
        
    def build(self, input_shape):
        self.min_var = self.add_weight('min', initializer='zeros', trainable=True)
        self.max_var = self.add_weight('max', initializer='ones', trainable=True)
    
    def call(self, inputs, training=None):
        if training:
            # Fake quantization: quantize then dequantize
            return tf.quantization.fake_quant_with_min_max_vars(
                inputs, self.min_var, self.max_var,
                num_bits=self.num_bits, narrow_range=self.narrow_range
            )
        return inputs
```

### TFLite Quantization

TensorFlow Lite is the deployment target for quantized models on mobile/edge:

```python
# Post-training quantization options in TFLite

# 1. Dynamic range quantization (weight-only)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 2. Full integer quantization (weights + activations)
def representative_dataset():
    for data in calibration_data:
        yield [data.numpy().astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# 3. Float16 quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### TFLite Quantization Specification

TFLite uses asymmetric quantization for activations and symmetric for weights:

```
Activations (uint8):
  real_value = scale * (uint8_value - zero_point)
  scale > 0, zero_point ∈ [0, 255]

Weights (int8, per-channel):
  real_value = scale[channel] * int8_value
  zero_point = 0 (symmetric)
  scale[channel] > 0 for each output channel
```

## gemmlowp: Low-Precision Matrix Multiplication

gemmlowp is Google's library for efficient low-precision (primarily 8-bit) general matrix multiplication (GEMM), used as the computational backend for quantized inference.

### Design Philosophy

```
Key principles:
1. Quantized computation must be bitwise deterministic
2. Support for both symmetric and asymmetric quantization
3. Efficient on ARM NEON, x86 SSE/AVX, and other SIMD architectures
4. Exact integer arithmetic (no floating-point in inner loops)
```

### Quantized Matrix Multiplication

For matrices A (uint8) and B (uint8) with zero-points:

```
C_real = (A_q - ZP_A) * (B_q - ZP_B) * S_A * S_B

Expanding the integer part:
int32_accumulator = A_q * B_q 
                  - ZP_A * sum(B_q, over rows)
                  - ZP_B * sum(A_q, over columns)  
                  + ZP_A * ZP_B * depth

Final: C_real = S_A * S_B * int32_accumulator
```

gemmlowp computes the int32 accumulator efficiently using SIMD instructions, then applies the final scale in a single pass.

### Output Pipeline

```
gemmlowp output pipeline:
  INT32 accumulator
    → Add bias (INT32)
    → Multiply by output multiplier (fixed-point, no floating-point!)
    → Right-shift (efficient division by power of 2)
    → Add output zero-point
    → Clamp to [0, 255] or [-128, 127]
    → Cast to uint8/int8
```

This entire pipeline avoids floating-point operations, enabling efficient deployment on hardware without FPU.

## BRECQ: Block Reconstruction

BRECQ (Block Reconstruction for Post-Training Quantization) is a Google-affiliated research contribution that pushes PTQ accuracy to near-QAT levels.

### Core Idea

Instead of quantizing the entire network at once (which accumulates errors) or one layer at a time (which ignores inter-layer dependencies), BRECQ reconstructs **blocks** of layers:

```
Network: [Block1] → [Block2] → [Block3] → ... → [BlockN]

For each block:
  1. Feed calibration data through the FP32 model up to this block
  2. Quantize the block's weights
  3. Optimize the quantized weights to minimize block output error
  4. Move to next block (using quantized previous blocks)
```

### Mathematical Formulation

For a block with parameters θ, minimize the reconstruction loss:

```
min_θ E_x [ || f(x; θ) - f(x; θ_q) ||² ]

Where:
  f(x; θ)   = block output with FP32 weights
  f(x; θ_q) = block output with quantized weights
  x          = input to this block (from calibration data)
```

### Fisher Information for Sensitivity

BRECQ uses the Fisher Information Matrix to weight the reconstruction loss, giving more importance to parameters that significantly affect the loss:

```python
# Fisher-weighted reconstruction (simplified)
def brecq_optimize_block(block_fp32, block_quantized, calibration_inputs):
    """
    Optimize quantized block weights to minimize output reconstruction error,
    weighted by Fisher information.
    """
    optimizer = torch.optim.Adam(
        get_quantization_parameters(block_quantized), 
        lr=1e-4
    )
    
    for iteration in range(num_iterations):
        # Sample calibration batch
        x = sample_batch(calibration_inputs)
        
        # FP32 block output (target)
        with torch.no_grad():
            target = block_fp32(x)
        
        # Quantized block output
        output = block_quantized(x)
        
        # Reconstruction loss (optionally Fisher-weighted)
        loss = fisher_weighted_mse(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### BRECQ Results

BRECQ achieves state-of-the-art PTQ results, particularly at low bit-widths:

| Model | Bits (W/A) | Layer-wise PTQ | BRECQ | QAT |
|-------|-----------|----------------|-------|-----|
| ResNet-18 | W4/A4 | 51.4% | **69.0%** | 69.5% |
| ResNet-50 | W4/A4 | 20.4% | **72.4%** | 73.1% |
| MobileNetV2 | W4/A4 | 0.1% | **53.3%** | 58.2% |
| RegNetX-600M | W4/A4 | 40.1% | **65.5%** | — |

At W4/A4, BRECQ closes most of the gap between naive PTQ and QAT, without any training data or backpropagation through the full model.

### Block Selection Strategy

```
Optimal block boundaries:
- Residual blocks (ResNet blocks) are natural units
- Transformer layers (attention + FFN) form blocks
- Blocks should be small enough for effective optimization
  but large enough to capture inter-layer dependencies
  
Typical: 2-4 conv layers per block, or 1 transformer layer per block
```

## Google's Quantization in Production

### Quantized Models in Google Products

Google deploys quantized models extensively:
- **Google Translate**: Quantized on-device translation models
- **Google Assistant**: INT8 speech recognition and NLU
- **Google Photos**: Quantized image classification/search
- **Pixel phones**: TFLite quantized models for camera, voice

### TPU Quantization

Google's TPUs support:
- **BF16**: Default training format (brain floating point)
- **INT8**: Inference on TPU v4+
- **Quantization-aware training**: Integrated into TPU training pipelines

```python
# JAX/Flax quantization for TPU (conceptual)
import jax
import jax.numpy as jnp
from aqt import flax as aqt_flax  # AQT: Accurate Quantized Training

# Define quantized model
class QuantizedDense(nn.Module):
    features: int
    quant_config: aqt_flax.DenseGeneral.HParams
    
    @nn.compact
    def __call__(self, x):
        return aqt_flax.DenseGeneral(
            features=self.features,
            hparams=self.quant_config
        )(x)
```

### AQT (Accurate Quantized Training)

Google's internal quantization library for JAX/TPU:
- Supports arbitrary bit-widths
- Per-channel and per-tensor quantization
- Stochastic rounding support
- Integrated with JAX transformations (jit, vmap, pmap)

## Additional Google Research

### Learned Quantization

Google has contributed research on learning quantization parameters:
- **Learned Step Size Quantization**: Learning optimal step sizes during training
- **Differentiable Quantization**: Making the quantization function differentiable for end-to-end optimization

### Quantization for On-Device ML

```
Google's mobile quantization philosophy:
1. Design architectures that quantize well (MobileNet, EfficientNet-Lite)
2. Use QAT from the start for mobile-targeted models
3. Target INT8 for compute, INT8/uint8 for storage
4. Leverage per-channel quantization (well-supported on ARM)
5. Export via TFLite for mobile deployment
```

### EfficientNet-Lite

Specifically designed to be quantization-friendly:
- Removed squeeze-and-excitation blocks (cause quantization issues)
- Used simpler activation functions
- Designed with per-channel quantization in mind

## References

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018), Google
- Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper" (2018), Google
- Li et al., "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction" (2021)
- TensorFlow Model Optimization Toolkit documentation
- gemmlowp GitHub repository and documentation
- TensorFlow Lite quantization specification
- Abdolrashidi et al., "Pareto-Optimal Quantized ResNet Is Mostly 4-bit" (2021), Google
- AQT (Accurate Quantized Training) documentation
