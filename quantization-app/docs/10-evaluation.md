# Evaluation Metrics and Practical Tradeoffs

## Overview

Evaluating quantized models requires measuring multiple dimensions: accuracy/quality preservation, inference speed improvement, memory reduction, and model size reduction. The right quantization strategy depends on which tradeoffs are acceptable for a given deployment scenario.

## Evaluation Metrics

### Perplexity Degradation (Primary for LLMs)

Perplexity measures how well a language model predicts text. Lower is better.

```
Perplexity = exp(-1/N × Σ log P(token_i | context_i))

Interpretation:
  - FP16 baseline perplexity: ~5.0 (LLaMA-7B on WikiText-2)
  - +0.1 perplexity: Barely noticeable quality difference
  - +0.5 perplexity: Slight quality decrease, acceptable for most uses
  - +1.0 perplexity: Noticeable degradation in some tasks
  - +2.0 perplexity: Significant quality loss
  - +5.0+ perplexity: Catastrophic failure
```

**Measurement protocol:**
- Always evaluate on a held-out dataset (WikiText-2, C4, etc.)
- Use consistent context length (typically 2048 tokens)
- Report on the same hardware/implementation as deployment
- Compare against the exact same FP16 model (not a different checkpoint)

### Task Accuracy

For classification, QA, and reasoning tasks:

```python
# Example evaluation pipeline
def evaluate_quantized_model(model_q, benchmarks):
    results = {}
    
    for benchmark in benchmarks:
        if benchmark.type == 'classification':
            results[benchmark.name] = compute_accuracy(model_q, benchmark)
        elif benchmark.type == 'qa':
            results[benchmark.name] = compute_f1(model_q, benchmark)
        elif benchmark.type == 'generation':
            results[benchmark.name] = compute_bleu_rouge(model_q, benchmark)
        elif benchmark.type == 'reasoning':
            results[benchmark.name] = compute_pass_at_k(model_q, benchmark)
    
    return results

# Common benchmarks for LLMs
benchmarks = [
    'MMLU (knowledge)',
    'HellaSwag (commonsense)',
    'ARC (reasoning)',
    'TruthfulQA (factuality)',
    'GSM8K (math)',
    'HumanEval (coding)',
    'WinoGrande (coreference)',
]
```

### Inference Latency

```
Key metrics:
  - Time to first token (TTFT): important for interactive applications
  - Time per output token (TPOT): determines generation speed
  - Tokens per second (TPS): overall throughput
  - Batch throughput: tokens/sec at various batch sizes

Measurement considerations:
  - Warm up the model (first few inferences are slower)
  - Measure over many iterations (reduce variance)
  - Report percentiles (p50, p95, p99) not just mean
  - Specify hardware, batch size, sequence length
  - Account for prefill vs decode phases separately
```

### Memory Reduction

```
Metrics:
  - Peak GPU memory usage (during inference)
  - Model weight memory (theoretical vs actual)
  - KV-cache memory (for LLMs)
  - Total runtime memory

Theoretical vs actual:
  - FP32 → INT8: theoretical 4x reduction
  - FP16 → INT4: theoretical 4x reduction
  - Actual reduction is less due to:
    * Quantization metadata (scales, zero-points)
    * Activations still in FP16
    * KV-cache still in FP16 (unless also quantized)
    * Framework overhead
    * Temporary buffers
```

### Model Size on Disk

```
Storage calculation:
  Weight storage:
    FP32: params × 4 bytes
    FP16: params × 2 bytes
    INT8: params × 1 byte + scales
    INT4: params × 0.5 bytes + scales
    
  Scale/metadata overhead for INT4 (group_size=128):
    (params / 128) × 2 bytes (FP16 scales)
    + (params / 128) × 0.5 bytes (INT4 zero-points)
    ≈ params × 0.02 bytes additional

Example: LLaMA-70B
  FP16: 140 GB
  INT8: ~70 GB  
  INT4 (GPTQ, g=128): ~35 GB + ~0.7 GB metadata ≈ 36 GB
  INT4 (GGUF q4_K_M): ~40 GB (includes K-quant overhead)
```

## Practical Results

### Compression Ratios

| Format | Theoretical Compression (vs FP32) | Practical Compression | Notes |
|--------|----------------------------------|----------------------|-------|
| FP16 | 2x | 2x | Lossless for inference |
| INT8 (W8A8) | 4x | 3-3.5x | Activations + metadata overhead |
| INT8 (W8A16) | 2x weights only | 1.8x | Activations stay FP16 |
| INT4 (W4A16) | 4x weights only | 3.5-3.8x | Metadata overhead |
| INT4 (W4A8) | 8x | 4-5x | Rare, difficult |
| INT3 | 5.3x weights only | 4-5x | Significant quality concern |
| INT2 | 8x weights only | 6-7x | Often catastrophic |

**Key insight: practical reduction is 4x-8x, not the theoretical 16x** that pure bit-width ratio would suggest. This is because:
1. Activations and KV-cache are not (or less aggressively) quantized
2. Quantization metadata adds overhead
3. Not all layers are quantized (embedding, output head)
4. Framework buffers and overhead

### INT8 W8A8 Results

W8A8 quantization (both weights and activations in INT8) achieves less than 1% accuracy loss for most standard models:

| Model | FP32 Accuracy | W8A8 Accuracy | Degradation |
|-------|--------------|---------------|-------------|
| ResNet-50 | 76.1% | 75.4-75.9% | 0.2-0.7% |
| BERT-base (SQuAD F1) | 88.5 | 87.8-88.3 | 0.2-0.7 |
| MobileNetV2 | 71.9% | 70.9-71.5% | 0.4-1.0% |
| GPT-3 175B (perplexity) | baseline | +0.1-0.3 | Minimal |
| ViT-B/16 | 81.1% | 80.5-80.8% | 0.3-0.6% |

### 4-Bit Weight-Only Results

Well-tuned 4-bit methods (GPTQ, AWQ) achieve 0.3-0.5 perplexity increase:

| Model | Method | FP16 PPL | 4-bit PPL | Δ PPL |
|-------|--------|----------|-----------|-------|
| LLaMA-7B | GPTQ | 5.68 | 5.85 | +0.17 |
| LLaMA-7B | AWQ | 5.68 | 5.78 | +0.10 |
| LLaMA-13B | GPTQ | 5.09 | 5.20 | +0.11 |
| LLaMA-30B | GPTQ | 4.10 | 4.22 | +0.12 |
| LLaMA-65B | GPTQ | 3.53 | 3.62 | +0.09 |
| Mistral-7B | AWQ | 5.25 | 5.40 | +0.15 |

**Trend:** Larger models quantize better (less relative degradation).

### 2-Bit Results: Often Catastrophic

```
2-bit quantization results (various methods):

LLaMA-7B:
  Round-to-nearest: perplexity > 10000 (unusable)
  GPTQ 2-bit:      perplexity ~15-20 (poor)
  QuIP# 2-bit:     perplexity ~7.5 (acceptable for some use cases)
  AQLM 2-bit:      perplexity ~7.0 (best 2-bit result)

Multimodal models at 2-bit:
  Vision tasks: often collapse to near-zero accuracy
  Language tasks: heavily degraded but sometimes usable
  
Conclusion: 2-bit is NOT production-ready for most applications.
Only specialized methods (AQLM, QuIP#) achieve usable results,
and only for language-only tasks on large models (13B+).
```

## Method Comparison Table

### GPTQ vs AWQ vs GGUF vs bitsandbytes

| Aspect | GPTQ | AWQ | GGUF (q4_K_M) | bitsandbytes (NF4) |
|--------|------|-----|---------------|-------------------|
| **Typical bits** | 4 (3-8) | 4 | ~4.8 effective | 4 (NF4) |
| **Method** | Hessian-based | Activation-aware | Importance-mixed | Normal float |
| **Quantize time** | Hours | Minutes | Minutes | On-load |
| **Calibration data** | ~128 samples | Small set | None | None |
| **Quality (7B, PPL)** | +0.17 | +0.10 | +0.15 | +0.25 |
| **Quality (13B+)** | Excellent | Excellent | Very good | Good |
| **Inference speed** | Fast (GPU) | Fast (GPU) | Good (CPU) | Moderate |
| **Hardware** | GPU (CUDA) | GPU (CUDA) | CPU/GPU | GPU (CUDA) |
| **Runtime** | ExLlama, vLLM | vLLM, TGI | llama.cpp | HuggingFace |
| **Memory overhead** | Low | Low | Low | Moderate |
| **Best for** | GPU serving | GPU serving | CPU inference | Training/fine-tune |

### Detailed Quality Comparison (LLaMA-2-7B)

| Method | WikiText-2 PPL | MMLU (5-shot) | HellaSwag | ARC-C |
|--------|---------------|---------------|-----------|-------|
| FP16 | 5.47 | 45.3% | 76.0% | 46.3% |
| GPTQ-4bit (g128) | 5.63 | 44.5% | 75.2% | 45.4% |
| AWQ-4bit (g128) | 5.60 | 44.8% | 75.4% | 45.6% |
| GGUF q4_K_M | 5.65 | 44.4% | 75.0% | 45.2% |
| bitsandbytes NF4 | 5.72 | 43.9% | 74.5% | 44.8% |
| GPTQ-3bit | 6.30 | 41.2% | 72.1% | 42.5% |
| GGUF q2_K | 7.85 | 35.8% | 65.3% | 38.1% |

## When to Use What

### Decision Framework

```
                    ┌─────────────────────────┐
                    │ What is your constraint? │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
         ┌──────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐
         │  Memory     │  │  Latency   │  │ Accuracy │
         │  Limited    │  │  Critical  │  │ Critical │
         └──────┬─────┘  └─────┬──────┘  └────┬─────┘
                │               │               │
         Weight-only      W8A8 or FP8      Stay FP16/BF16
         INT4/INT3       (full pipeline)   or W8A8 max
```

### Weight-Only Quantization (INT4, W4A16)

**Best for: Memory-bound inference (large batch, long sequences, single-user)**

```
Characteristics:
  - Only weights are quantized; activations remain in FP16
  - Reduces weight loading bandwidth (main bottleneck for LLMs)
  - Computation still in FP16 (dequantize weights before GEMM)
  - Best speedup at batch_size=1 (purely memory-bound)
  
Use when:
  - Deploying LLMs on limited GPU memory
  - Batch size is small (1-8)
  - Sequence lengths are long (more weight reuse)
  - Quality must be preserved (4-bit weight-only is very good)

Methods: GPTQ, AWQ, GGUF, EXL2
Typical speedup: 2-4x vs FP16 (memory-bound scenarios)
```

### Weight + Activation Quantization (INT8, W8A8)

**Best for: Compute-bound inference (large batch, high throughput)**

```
Characteristics:
  - Both weights AND activations quantized to INT8
  - Computation in INT8 (Tensor Core acceleration)
  - Maximum computational speedup
  - Best at large batch sizes (compute-bound regime)
  
Use when:
  - Serving many users simultaneously (large batch)
  - Throughput matters more than single-request latency
  - Hardware has INT8 Tensor Cores
  - Can tolerate ~0.5-1% accuracy loss

Methods: SmoothQuant, TensorRT INT8, ONNX Runtime static
Typical speedup: 2-4x vs FP16 (compute-bound scenarios)
```

### Dynamic Quantization

**Best for: Simplest deployment, no calibration data**

```
Characteristics:
  - Weights pre-quantized, activations quantized at runtime
  - No calibration dataset needed
  - Slight runtime overhead (computing activation stats)
  - Good accuracy (adaptive to each input)
  
Use when:
  - Quick deployment without optimization infrastructure
  - Calibration data unavailable
  - Model has highly variable activation distributions
  - NLP/transformer models with varying input characteristics

Methods: PyTorch quantize_dynamic, ONNX Runtime dynamic
Typical speedup: 1.5-2x vs FP32 (less than static)
```

### Static PTQ

**Best for: Production deployment with best latency/accuracy balance**

```
Characteristics:
  - All parameters pre-computed offline
  - No runtime overhead for quantization
  - Best inference latency
  - Requires calibration dataset (~200 samples)
  
Use when:
  - Production deployment with latency SLAs
  - Calibration data is available
  - Model architecture is well-supported (CNNs, standard transformers)
  - Need deterministic performance

Methods: TensorRT, ONNX Runtime static, TFLite
Typical speedup: 2-4x vs FP32
```

### Quantization-Aware Training (QAT)

**Best for: When PTQ fails or sub-4-bit needed**

```
Characteristics:
  - Requires training infrastructure (GPU time, data, etc.)
  - Best accuracy at any given bit-width
  - Can enable quantization levels that PTQ cannot (e.g., W4A4)
  - Most expensive to implement
  
Use when:
  - PTQ causes unacceptable accuracy loss
  - Targeting aggressive quantization (4-bit with activations)
  - Model is quantization-sensitive (EfficientNet, MobileNet)
  - Have access to training data and compute
  - Product justifies the engineering investment

Methods: PyTorch QAT, TF-MOT, ModelOpt QAT
Training cost: ~10% of original training compute
```

## Evaluation Checklist

### Before Deploying a Quantized Model

```
□ Accuracy Validation
  □ Evaluate on held-out test set (not calibration set)
  □ Test across multiple benchmarks (not just perplexity)
  □ Check edge cases and failure modes
  □ Compare against FP16 on the same test set
  □ Test with real user queries (not just benchmarks)

□ Performance Validation
  □ Measure end-to-end latency (not just GEMM speedup)
  □ Test at expected batch sizes
  □ Measure memory usage (peak and sustained)
  □ Test with expected input lengths
  □ Profile for bottlenecks (is it actually faster?)

□ Robustness Testing
  □ Test with out-of-distribution inputs
  □ Verify numerical stability (no NaN/Inf)
  □ Test long-running inference (memory leaks?)
  □ Test concurrent requests (thread safety)

□ Production Readiness
  □ Quantized model produces deterministic outputs
  □ Model loads reliably (no corrupted weights)
  □ Fallback plan if quality degrades in production
  □ Monitoring for quality regression over time
```

## Cost-Benefit Analysis

### When Quantization is Worth It

```
Clear wins:
  - LLM inference (memory-bound → 2-4x speedup from INT4)
  - Edge deployment (model must fit in device memory)
  - Batch inference pipelines (cost reduction at scale)
  - Real-time applications (latency requirements)

Marginal:
  - Small models (overhead may dominate)
  - Training-time (FP8 helps but setup is complex)
  - Models already fast enough
  - Research/experimentation (flexibility > speed)

Not worth it:
  - When accuracy is absolutely critical and 0.1% matters
  - Models smaller than 100M parameters (overhead dominates)
  - When serving costs are irrelevant
  - One-off inference tasks
```

### ROI Calculation

```
Monthly serving cost (example: LLaMA-70B):
  FP16: 2× A100-80GB ($4/hr each) = $5,760/month
  INT4: 1× A100-80GB ($4/hr) = $2,880/month
  Savings: $2,880/month = $34,560/year
  
  Quantization effort: ~2 engineer-days = ~$3,000
  ROI payback: < 1 month
```

## References

- Dettmers et al., "The case for 4-bit precision: k-bit Inference Scaling Laws" (2022)
- Yao et al., "A Comprehensive Study on Post-Training Quantization for Large Language Models" (2023)
- Liu et al., "LLM-QBench: A Benchmark Towards the Best Practice for Post-training Quantization of Large Language Models" (2023)
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2022)
- llama.cpp perplexity measurements (GitHub wiki)
- NVIDIA TensorRT documentation - Performance Best Practices
