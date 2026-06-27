# Quantization-Aware Training (QAT)

## Overview

Quantization-Aware Training (QAT) simulates the effects of quantization during the training (or fine-tuning) process, allowing the model to learn to compensate for quantization noise. This produces significantly better results than PTQ, especially at low bit-widths (4-bit and below).

**Key insight:** By exposing the model to quantization effects during training, the optimizer adjusts weights to be robust to quantization, resulting in a model that maintains accuracy after actual deployment-time quantization.

## When is QAT Necessary?

QAT is essential when PTQ fails to meet accuracy requirements. This commonly occurs with:

- **Quantization-sensitive architectures**: Models with depthwise separable convolutions, squeeze-and-excitation blocks, or complex normalization
- **Low bit-width targets**: 4-bit or lower quantization
- **Tight accuracy budgets**: When even 0.5% degradation is unacceptable
- **Small models**: Compact models have less redundancy to absorb quantization noise

### Dramatic Example: EfficientNet-B0

| Method | Top-1 Accuracy | Notes |
|--------|---------------|-------|
| FP32 baseline | 77.4% | Reference |
| PTQ (W8A8) | 33.9% | Catastrophic failure |
| QAT (W8A8) | 76.8% | Near-full recovery |

This 43-percentage-point gap between PTQ and QAT demonstrates why QAT is essential for sensitive architectures.

## The Straight-Through Estimator (STE)

### The Problem

Quantization is a step function — it maps continuous values to discrete levels. Step functions have zero gradient almost everywhere (and undefined gradient at the step boundaries). This means standard backpropagation cannot compute gradients through quantization operations.

```
Q(x) = round(x / S) * S  →  dQ/dx = 0 (almost everywhere)
```

### The Solution: STE

The Straight-Through Estimator (Bengio et al., 2013) approximates the gradient of the quantization function as the identity:

```
Forward pass:  y = Q(x)           (actual quantization)
Backward pass: ∂L/∂x ≈ ∂L/∂y    (pass gradient straight through)
```

More precisely, the STE for quantization with clipping:

```
∂Q/∂x ≈ {
    1,  if x_min ≤ x ≤ x_max   (within quantization range)
    0,  otherwise                (outside range, gradient is killed)
}
```

This can be thought of as: "pretend quantization didn't happen during the backward pass, but kill gradients for values that were clipped."

### Implementation

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        # Quantize and immediately dequantize (fake quantization)
        x_q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max)
        x_dq = scale * (x_q - zero_point)
        
        # Save mask for backward (STE with clipping)
        ctx.save_for_backward(x, torch.tensor([scale, q_min, q_max]))
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        x, params = ctx.saved_tensors
        scale, q_min, q_max = params[0], params[1], params[2]
        
        # STE: pass gradient through where x is within quantizable range
        x_min = scale * (q_min - 0)  # Simplified for Z=0
        x_max = scale * (q_max - 0)
        
        # Gradient is 1 inside range, 0 outside
        mask = (x >= x_min) & (x <= x_max)
        grad_input = grad_output * mask.float()
        
        return grad_input, None, None, None, None
```

### Why STE Works

Despite being a crude approximation, STE works because:
1. The quantization function is "mostly" identity (round is close to identity for most values)
2. The gradient direction is preserved even if magnitude is approximate
3. The model learns to keep weights within quantizable ranges over time
4. Clipping the gradient to zero outside the range provides useful information (tells the optimizer not to push values further out of range)

## Fake Quantization Nodes

During QAT, the model graph is augmented with "fake quantization" (or "simulated quantization") nodes. These simulate quantization effects during training but keep computations in floating-point:

```
Standard layer:    input → [Conv/Linear] → output
                   
QAT layer:         input → [FakeQuant] → [Conv/Linear] → [FakeQuant] → output
                              ↑                              ↑
                        Quantize activations          Quantize weights
                        (simulate INT8)               (simulate INT8)
```

The fake quantization operation:
```
FakeQuant(x) = Dequantize(Quantize(x))
             = S * (Clip(round(x/S) + Z, q_min, q_max) - Z)
```

This introduces quantization noise during training while keeping all computations in FP32, allowing gradient computation.

## QAT Workflow

```
1. Start with pre-trained FP32 model (or train from scratch)
2. Insert fake quantization nodes:
   - After each activation (simulates activation quantization)
   - Before each weight usage (simulates weight quantization)
3. Choose quantization parameters:
   - Bit-width, symmetric/asymmetric, per-tensor/per-channel
4. Fine-tune with STE:
   - Use ~10% of original training schedule (NVIDIA recommendation)
   - Start with learning rate from later training stages
   - Use learning rate annealing (cosine or step decay)
5. After training completes:
   - Remove fake quantization nodes
   - Export actual quantized model with final scale/zero-point values
6. Deploy the quantized model
```

## NVIDIA's QAT Recommendations

Based on extensive experimentation, NVIDIA recommends:

1. **Training duration**: Fine-tune for approximately **10% of the original training schedule**
   - Example: If original training was 90 epochs, QAT fine-tuning uses ~9 epochs

2. **Learning rate**: Start with the learning rate used in the final stages of FP32 training, with annealing
   ```
   Initial LR for QAT ≈ Final LR of FP32 training
   Schedule: Cosine annealing to 0
   ```

3. **Quantization parameter updates**: 
   - Update scale/zero-point during initial portion of training
   - Freeze quantization parameters for the final portion
   - Allows the model to fully adapt to fixed quantization parameters

4. **Initialization**: Always start from a well-trained FP32 checkpoint

5. **Batch normalization**: Fold BN layers before QAT begins, or use BN in "eval" mode during QAT

## PyTorch QAT Implementation

```python
import torch
import torch.quantization as quant

# 1. Load pre-trained model
model = load_pretrained_model()
model.train()

# 2. Specify QAT configuration
model.qconfig = quant.get_default_qat_qconfig('fbgemm')  # For x86
# Or for mobile: quant.get_default_qat_qconfig('qnnpack')

# 3. Fuse modules (Conv+BN+ReLU, Linear+ReLU)
model_fused = quant.fuse_modules(model, [
    ['conv1', 'bn1', 'relu1'],
    ['conv2', 'bn2', 'relu2'],
])

# 4. Prepare model for QAT (inserts fake-quant nodes)
model_qat = quant.prepare_qat(model_fused)

# 5. Fine-tune (10% of original schedule)
optimizer = torch.optim.SGD(model_qat.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    for batch in train_loader:
        output = model_qat(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()  # STE handles quantization nodes
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

# 6. Convert to actual quantized model
model_qat.eval()
model_quantized = quant.convert(model_qat)

# 7. Save quantized model
torch.save(model_quantized.state_dict(), 'model_quantized.pth')
```

## TensorFlow QAT Implementation

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# 1. Load pre-trained model
model = tf.keras.models.load_model('model_fp32.h5')

# 2. Apply QAT to the entire model
qat_model = tfmot.quantization.keras.quantize_model(model)

# Or selectively apply to specific layers:
def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = tf.keras.models.clone_model(
    model, clone_function=apply_quantization_to_dense
)
qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# 3. Compile and fine-tune
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

qat_model.fit(
    train_dataset,
    epochs=fine_tune_epochs,  # ~10% of original training
    validation_data=val_dataset
)

# 4. Export to TFLite with actual quantization
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model_qat.tflite', 'wb') as f:
    f.write(tflite_model)
```

## QAT Results

### Image Classification

| Model | FP32 | PTQ W8A8 | QAT W8A8 |
|-------|------|----------|-----------|
| ResNet-50 | 76.1% | 75.4% | 76.0% |
| MobileNet-V2 | 71.9% | 70.5% | 71.7% |
| EfficientNet-B0 | 77.4% | 33.9% | 76.8% |
| Inception-V3 | 78.0% | 77.3% | 77.9% |

### NLP Models

| Model | FP32 (F1) | PTQ W8A8 | QAT W8A8 |
|-------|-----------|----------|-----------|
| BERT-base (SQuAD) | 88.5 | 87.8 | 88.4 |
| DistilBERT (SST-2) | 91.3% | 90.7% | 91.1% |

## Advanced QAT Techniques

### Learned Step Size Quantization (LSQ)

Instead of fixing quantization step size, learn it as a trainable parameter:

```python
class LSQ(nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.step_size = nn.Parameter(torch.tensor(1.0))  # Learnable
        
    def forward(self, x):
        q_max = 2**(self.bits - 1) - 1
        q_min = -2**(self.bits - 1)
        
        x_q = torch.clamp(torch.round(x / self.step_size), q_min, q_max)
        x_dq = x_q * self.step_size
        
        # STE for x, gradient for step_size
        return x + (x_dq - x).detach()  # Trick for STE in PyTorch
```

### Progressive Quantization

Gradually decrease precision during training:
```
Epochs 1-3:   W8A8  (start gentle)
Epochs 4-6:   W6A8  (reduce weight bits)
Epochs 7-9:   W4A8  (target precision)
```

### Knowledge Distillation + QAT

Use the FP32 model as a teacher during QAT:

```python
def qat_with_distillation(student_qat, teacher_fp32, batch, alpha=0.7, temperature=4.0):
    student_logits = student_qat(batch['input'])
    teacher_logits = teacher_fp32(batch['input']).detach()
    
    # Hard loss (standard cross-entropy)
    hard_loss = F.cross_entropy(student_logits, batch['target'])
    
    # Soft loss (distillation)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Combined loss
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return loss
```

## QAT vs PTQ Decision Framework

```
Start with PTQ
    ↓
Accuracy acceptable? → YES → Deploy PTQ model
    ↓ NO
Try better calibration / per-channel / mixed-precision
    ↓
Accuracy acceptable? → YES → Deploy PTQ model
    ↓ NO
Apply QAT fine-tuning (~10% schedule)
    ↓
Accuracy acceptable? → YES → Deploy QAT model
    ↓ NO
Try QAT + Knowledge Distillation
    ↓
Consider higher bit-width or architecture changes
```

## Common QAT Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Training instability | LR too high for QAT | Reduce LR by 10-100x from original |
| No improvement | FakeQuant not correctly inserted | Verify all ops have observers |
| Accuracy oscillates | BN statistics mismatch | Freeze BN during QAT or fold first |
| Worse than PTQ | Insufficient training | Increase epochs or use distillation |
| Gradient explosion | Aggressive clipping in STE | Gradient clipping or wider quant range |

## References

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation" (2013)
- Esser et al., "Learned Step Size Quantization" (LSQ, 2020)
- NVIDIA, "Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT" (2021)
- TensorFlow Model Optimization Toolkit documentation
- PyTorch Quantization documentation
- Bhalgat et al., "LSQ+: Improving low-bit quantization through learnable offsets and better initialization" (2020)
