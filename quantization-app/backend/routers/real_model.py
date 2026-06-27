"""
Real model quantization API endpoints
Downloads and quantizes actual LLMs from HuggingFace
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from services.real_quantization import (
    get_model_info,
    quantize_full_model,
    compare_quantization_methods,
    get_model_and_tokenizer,
    quantize_model_layer,
    compute_perplexity,
    benchmark_generation_speed,
)
import torch
import numpy as np
import time
import math

router = APIRouter()


def sanitize_for_json(obj):
    """Recursively replace inf/nan with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


class QuantizeModelRequest(BaseModel):
    bits: int = Field(8, ge=2, le=8)
    scheme: str = Field("symmetric", description="symmetric or asymmetric")
    group_size: int = Field(128, ge=0)
    test_prompt: str = Field("The meaning of life is")


class CompareRequest(BaseModel):
    test_prompt: str = Field("Artificial intelligence is")


class GenerateRequest(BaseModel):
    prompt: str = Field("Once upon a time")
    max_tokens: int = Field(50, ge=1, le=200)


class LayerInspectRequest(BaseModel):
    layer_name: str = Field(..., description="Full parameter name, e.g. model.layers.0.self_attn.q_proj.weight")


class SensitivityRequest(BaseModel):
    bits: int = Field(4, ge=2, le=8)
    num_layers_to_test: int = Field(10, ge=1, le=50)


class GPTQRequest(BaseModel):
    bits: int = Field(4, ge=2, le=8)
    group_size: int = Field(128, ge=32)
    num_calibration_samples: int = Field(16, ge=4, le=128)
    max_layers: int = Field(10, ge=1, le=22)


class ExportRequest(BaseModel):
    bits: int = Field(4, ge=2, le=8)
    scheme: str = Field("symmetric", description="symmetric, asymmetric, or gptq")
    group_size: int = Field(128, ge=0)


class MixedPrecisionRequest(BaseModel):
    default_bits: int = Field(4, ge=2, le=8)
    sensitive_bits: int = Field(8, ge=4, le=16)
    sensitive_layers: Optional[List[str]] = Field(None, description="Explicit list of layer names to keep at higher precision")


class BenchmarkRequest(BaseModel):
    prompt: str = Field("The", description="Prompt to use for generation")
    num_tokens: int = Field(50, ge=10, le=200)
    num_runs: int = Field(3, ge=1, le=10)


@router.get("/real-model/info")
async def model_info():
    """Get information about the loaded model (TinyLlama-1.1B)"""
    try:
        result = get_model_info()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/quantize")
async def quantize_model(request: QuantizeModelRequest):
    """
    Quantize TinyLlama-1.1B with specified settings.
    Returns perplexity before/after, text generation comparison, and per-layer stats.
    Takes 30-90 seconds on CPU.
    """
    try:
        result = quantize_full_model(
            bits=request.bits,
            scheme=request.scheme,
            group_size=request.group_size,
            test_text=request.test_prompt,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/compare")
async def compare_methods(request: CompareRequest):
    """
    Compare multiple quantization methods on TinyLlama-1.1B.
    Runs INT8/INT4/INT3 with different configurations.
    Takes 3-5 minutes on CPU.
    """
    try:
        result = compare_quantization_methods(test_text=request.test_prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/generate")
async def generate_text(request: GenerateRequest):
    """Generate text with the FP32 model (no quantization)"""
    try:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                do_sample=False,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {
            "prompt": request.prompt,
            "generated_text": text,
            "num_tokens_generated": int(output.shape[1] - input_ids.shape[1]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/inspect-layer")
async def inspect_layer(request: LayerInspectRequest):
    """
    Inspect a specific layer's weight distribution.
    Returns statistics and histogram of weight values.
    """
    try:
        model, _ = get_model_and_tokenizer()
        param = None
        for name, p in model.named_parameters():
            if name == request.layer_name:
                param = p
                break

        if param is None:
            raise HTTPException(status_code=404, detail=f"Layer '{request.layer_name}' not found")

        data = param.data.float().cpu().flatten().numpy()

        counts, bin_edges = np.histogram(data, bins=100)
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

        percentiles = np.percentile(data, [1, 5, 25, 50, 75, 95, 99]).tolist()

        return {
            "name": request.layer_name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "dtype": str(param.dtype),
            "statistics": {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "abs_max": float(np.max(np.abs(data))),
                "sparsity": float(np.sum(np.abs(data) < 1e-6) / len(data) * 100),
            },
            "percentiles": {
                "p1": percentiles[0], "p5": percentiles[1], "p25": percentiles[2],
                "p50": percentiles[3], "p75": percentiles[4], "p95": percentiles[5],
                "p99": percentiles[6],
            },
            "histogram": {
                "counts": [int(c) for c in counts],
                "bin_centers": [float(b) for b in bin_centers],
            },
            "sample_values": [float(x) for x in data[:50]],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/sensitivity")
async def layer_sensitivity(request: SensitivityRequest):
    """
    Quantize individual layers to find which are most sensitive.
    Returns per-layer perplexity impact when that layer alone is quantized.
    """
    try:
        model, tokenizer = get_model_and_tokenizer()
        eval_text = "The quick brown fox jumps over the lazy dog. Neural networks learn representations from data."
        baseline_ppl = compute_perplexity(model, tokenizer, eval_text)

        layers_to_test = []
        for name, param in model.named_parameters():
            if param.ndim >= 2 and param.numel() > 10000:
                layers_to_test.append((name, param))

        layers_to_test = layers_to_test[:request.num_layers_to_test]

        results = []
        for name, param in layers_to_test:
            original_data = param.data.clone()
            quant_result = quantize_model_layer(param.data, request.bits, "symmetric", 128)
            param.data = quant_result["dequantized"]
            ppl = compute_perplexity(model, tokenizer, eval_text)
            param.data = original_data

            results.append({
                "name": name,
                "shape": list(param.shape),
                "numel": int(param.numel()),
                "perplexity": round(ppl, 4),
                "perplexity_increase": round(ppl - baseline_ppl, 4),
                "mse": quant_result["mse"],
                "sensitivity": "high" if (ppl - baseline_ppl) > 0.5 else "medium" if (ppl - baseline_ppl) > 0.1 else "low",
            })

        results.sort(key=lambda x: -x["perplexity_increase"])

        return {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "bits": request.bits,
            "baseline_perplexity": round(baseline_ppl, 4),
            "num_layers_tested": len(results),
            "layers": results,
            "most_sensitive": results[:5] if results else [],
            "least_sensitive": results[-5:] if len(results) >= 5 else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/real-model/architecture")
async def model_architecture():
    """
    Returns the full model architecture tree showing all layers and their types.
    This is what you'd see when printing a PyTorch model.
    """
    try:
        model, _ = get_model_and_tokenizer()

        layers = []
        for name, param in model.named_parameters():
            parts = name.split(".")
            layer_type = "embedding" if "embed" in name else \
                         "attention" if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"]) else \
                         "mlp" if any(x in name for x in ["gate_proj", "up_proj", "down_proj", "mlp"]) else \
                         "norm" if "norm" in name else \
                         "head" if "lm_head" in name else "other"
            layers.append({
                "name": name,
                "type": layer_type,
                "shape": list(param.shape),
                "numel": int(param.numel()),
                "size_mb": round(param.numel() * 4 / (1024 * 1024), 3),
            })

        type_summary = {}
        for layer in layers:
            t = layer["type"]
            if t not in type_summary:
                type_summary[t] = {"count": 0, "params": 0, "size_mb": 0}
            type_summary[t]["count"] += 1
            type_summary[t]["params"] += layer["numel"]
            type_summary[t]["size_mb"] = round(type_summary[t]["size_mb"] + layer["size_mb"], 3)

        return {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "total_layers": len(layers),
            "layers": layers,
            "type_summary": type_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class VisualizeQuantizationRequest(BaseModel):
    layer_name: str = Field("model.layers.0.self_attn.q_proj.weight", description="Layer to visualize")
    bits: int = Field(4, ge=2, le=8)
    scheme: str = Field("symmetric")
    group_size: int = Field(128, ge=0)
    num_samples: int = Field(1000, ge=100, le=10000)


@router.post("/real-model/visualize-quantization")
async def visualize_quantization(request: VisualizeQuantizationRequest):
    """
    Show what happens to actual weight values during quantization.
    Returns before/after histograms, sample values, and error distribution.
    """
    try:
        model, _ = get_model_and_tokenizer()
        param = None
        for name, p in model.named_parameters():
            if name == request.layer_name:
                param = p
                break

        if param is None:
            raise HTTPException(status_code=404, detail=f"Layer '{request.layer_name}' not found")

        original = param.data.float().cpu()

        # Quantize this layer
        result = quantize_model_layer(original, request.bits, request.scheme, request.group_size)
        dequantized = result["dequantized"].cpu()

        # Compute error
        error = (original - dequantized).flatten().numpy()
        original_flat = original.flatten().numpy()
        dequantized_flat = dequantized.flatten().numpy()

        # Sample for scatter plot
        rng = np.random.RandomState(42)
        n = min(request.num_samples, len(original_flat))
        indices = rng.choice(len(original_flat), size=n, replace=False)
        sample_original = original_flat[indices].tolist()
        sample_quantized = dequantized_flat[indices].tolist()
        sample_error = error[indices].tolist()

        # Histograms
        orig_counts, orig_edges = np.histogram(original_flat, bins=80)
        orig_centers = ((orig_edges[:-1] + orig_edges[1:]) / 2).tolist()

        quant_counts, quant_edges = np.histogram(dequantized_flat, bins=80)
        quant_centers = ((quant_edges[:-1] + quant_edges[1:]) / 2).tolist()

        err_counts, err_edges = np.histogram(error, bins=60)
        err_centers = ((err_edges[:-1] + err_edges[1:]) / 2).tolist()

        # Quantization grid lines (the discrete values that exist after quantization)
        unique_vals = np.unique(dequantized_flat)
        # Sample grid lines evenly if too many
        if len(unique_vals) > 50:
            grid_indices = np.linspace(0, len(unique_vals) - 1, 50, dtype=int)
            grid_lines = unique_vals[grid_indices].tolist()
        else:
            grid_lines = unique_vals.tolist()

        return {
            "layer_name": request.layer_name,
            "shape": list(param.shape),
            "bits": request.bits,
            "scheme": request.scheme,
            "group_size": request.group_size,
            "num_elements": int(param.numel()),
            "num_unique_original": int(len(np.unique(original_flat[:100000]))),
            "num_unique_quantized": int(len(np.unique(dequantized_flat))),
            "mse": float(np.mean(error ** 2)),
            "max_error": float(np.max(np.abs(error))),
            "mean_abs_error": float(np.mean(np.abs(error))),
            "original_histogram": {
                "counts": [int(c) for c in orig_counts],
                "centers": [float(c) for c in orig_centers],
            },
            "quantized_histogram": {
                "counts": [int(c) for c in quant_counts],
                "centers": [float(c) for c in quant_centers],
            },
            "error_histogram": {
                "counts": [int(c) for c in err_counts],
                "centers": [float(c) for c in err_centers],
            },
            "sample_original": sample_original,
            "sample_quantized": sample_quantized,
            "sample_error": sample_error,
            "grid_lines": grid_lines,
            "statistics": {
                "original_min": float(original_flat.min()),
                "original_max": float(original_flat.max()),
                "original_std": float(original_flat.std()),
                "quantized_min": float(dequantized_flat.min()),
                "quantized_max": float(dequantized_flat.max()),
                "error_std": float(error.std()),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeepVisualizeRequest(BaseModel):
    bits: int = Field(4, ge=2, le=8)
    scheme: str = Field("symmetric")
    group_size: int = Field(128, ge=0)
    prompt: str = Field("Once upon a time")


@router.post("/real-model/deep-visualize")
async def deep_visualize(request: DeepVisualizeRequest):
    """
    Returns data for three visualizations:
    1. Number line snap - actual weight values snapping to grid
    2. Weight matrix heatmap - before/after as 2D grid
    3. Token probability shift - top-k predictions before/after quantization
    """
    try:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        # --- 1. Number Line Snap ---
        # Get weights from layer 0 q_proj
        target_param = None
        for name, p in model.named_parameters():
            if name == "model.layers.0.self_attn.q_proj.weight":
                target_param = p
                break

        original_weights = target_param.data.float().cpu()
        quant_result = quantize_model_layer(original_weights, request.bits, request.scheme, request.group_size)
        dequantized = quant_result["dequantized"].cpu()

        # Pick 30 weights from a single row to show on number line
        row_idx = 0
        row_orig = original_weights[row_idx, :60].numpy()
        row_quant = dequantized[row_idx, :60].numpy()

        # Get the unique quantized values in this row (grid lines)
        row_grid = sorted(set(row_quant[:60].tolist()))

        # For the number line, pick 25 weights that show interesting snapping
        indices = list(range(0, 60, 2))[:25]
        numberline_points = []
        for idx in indices:
            numberline_points.append({
                "original": float(row_orig[idx]),
                "quantized": float(row_quant[idx]),
                "error": float(abs(row_orig[idx] - row_quant[idx])),
            })

        # --- 2. Weight Matrix Heatmap ---
        # 16x16 slice of the weight matrix
        heatmap_size = 16
        heatmap_orig = original_weights[:heatmap_size, :heatmap_size].numpy().tolist()
        heatmap_quant = dequantized[:heatmap_size, :heatmap_size].numpy().tolist()
        heatmap_error = (original_weights[:heatmap_size, :heatmap_size] - dequantized[:heatmap_size, :heatmap_size]).abs().numpy().tolist()

        # --- 3. Token Probability Shift ---
        # Run the prompt through FP32 model to get top-k predictions
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

        # FP32 predictions
        with torch.no_grad():
            outputs_fp32 = model(input_ids)
            logits_fp32 = outputs_fp32.logits[0, -1, :]  # last token predictions
            probs_fp32 = torch.softmax(logits_fp32, dim=0)
            top_k_fp32 = torch.topk(probs_fp32, k=10)

        fp32_predictions = []
        for i in range(10):
            token_id = top_k_fp32.indices[i].item()
            fp32_predictions.append({
                "token": tokenizer.decode([token_id]).strip(),
                "token_id": token_id,
                "probability": float(top_k_fp32.values[i]),
            })

        # Apply quantization to all layers temporarily
        original_state = {}
        for name, param in model.named_parameters():
            if param.ndim >= 2 and param.numel() > 1000:
                original_state[name] = param.data.clone()
                result = quantize_model_layer(param.data, request.bits, request.scheme, request.group_size)
                param.data = result["dequantized"].to(device)

        # Quantized predictions
        with torch.no_grad():
            outputs_quant = model(input_ids)
            logits_quant = outputs_quant.logits[0, -1, :]
            probs_quant = torch.softmax(logits_quant, dim=0)

        # Get probabilities for the same top-k tokens from FP32
        quant_predictions = []
        for pred in fp32_predictions:
            prob = float(probs_quant[pred["token_id"]])
            quant_predictions.append({
                "token": pred["token"],
                "token_id": pred["token_id"],
                "probability": prob,
            })

        # Also get the quantized model's own top predictions
        top_k_quant = torch.topk(probs_quant, k=5)
        quant_own_top = []
        for i in range(5):
            token_id = top_k_quant.indices[i].item()
            quant_own_top.append({
                "token": tokenizer.decode([token_id]).strip(),
                "token_id": token_id,
                "probability": float(top_k_quant.values[i]),
            })

        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_state:
                param.data = original_state[name]

        return {
            "bits": request.bits,
            "scheme": request.scheme,
            "prompt": request.prompt,
            "numberline": {
                "points": numberline_points,
                "grid_lines": [float(g) for g in row_grid[:20]],
                "range_min": float(min(row_orig[:60])),
                "range_max": float(max(row_orig[:60])),
            },
            "heatmap": {
                "original": heatmap_orig,
                "quantized": heatmap_quant,
                "error": heatmap_error,
                "size": heatmap_size,
                "value_range": [float(original_weights[:heatmap_size, :heatmap_size].min()),
                               float(original_weights[:heatmap_size, :heatmap_size].max())],
            },
            "token_predictions": {
                "fp32_top": fp32_predictions,
                "quantized_same_tokens": quant_predictions,
                "quantized_own_top": quant_own_top,
                "next_word_fp32": fp32_predictions[0]["token"] if fp32_predictions else "",
                "next_word_quantized": quant_own_top[0]["token"] if quant_own_top else "",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== NEW ENDPOINTS =====


@router.post("/real-model/gptq")
async def run_gptq(request: GPTQRequest):
    """
    Run real GPTQ quantization with calibration data.
    Uses wikitext-2 calibration data to compute Hessian for each layer.
    Takes 2-5 minutes on CPU depending on max_layers.
    """
    try:
        from services.calibration import get_calibration_data
        from services.gptq_real import gptq_quantize_model

        model, tokenizer = get_model_and_tokenizer()

        # Load calibration data
        calib_start = time.time()
        calibration_data = get_calibration_data(
            num_samples=request.num_calibration_samples,
            seq_length=512,
            tokenizer=tokenizer,
        )
        calib_time = time.time() - calib_start

        # Compute baseline perplexity
        eval_text = "The quick brown fox jumps over the lazy dog. Machine learning models are trained using large datasets."
        baseline_ppl = compute_perplexity(model, tokenizer, eval_text)

        # Run GPTQ
        result = gptq_quantize_model(
            model=model,
            tokenizer=tokenizer,
            calibration_data=calibration_data,
            bits=request.bits,
            group_size=request.group_size,
            max_layers=request.max_layers,
        )

        # Compute post-GPTQ perplexity
        quantized_ppl = compute_perplexity(model, tokenizer, eval_text)

        # Generate text sample
        input_ids = tokenizer.encode("The meaning of life is", return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=30, do_sample=False)
        generation = tokenizer.decode(output[0], skip_special_tokens=True)

        # Restore original model weights (reload from cache)
        # Clear model cache to force reload of clean weights on next use
        from services.real_quantization import MODEL_CACHE
        MODEL_CACHE.clear()

        result["calibration_time_seconds"] = round(calib_time, 2)
        result["baseline_perplexity"] = round(baseline_ppl, 4)
        result["quantized_perplexity"] = round(quantized_ppl, 4)
        result["perplexity_increase"] = round(quantized_ppl - baseline_ppl, 4)
        result["sample_generation"] = generation
        result["num_calibration_samples"] = request.num_calibration_samples

        return sanitize_for_json(result)
    except Exception as e:
        # Clear model cache on error to ensure clean state
        from services.real_quantization import MODEL_CACHE
        MODEL_CACHE.clear()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/benchmark-speed")
async def benchmark_speed(request: BenchmarkRequest):
    """
    Measure tokens/second generation speed for the FP32 model.
    Runs multiple generation passes and averages the results.
    """
    try:
        model, tokenizer = get_model_and_tokenizer()
        result = benchmark_generation_speed(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            num_tokens=request.num_tokens,
            num_runs=request.num_runs,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/export")
async def export_model(request: ExportRequest):
    """
    Export quantized model to safetensors format.
    Quantizes the model with given settings, then saves weights to disk.
    """
    try:
        from services.model_export import export_quantized_model

        model, tokenizer = get_model_and_tokenizer()

        # Quantize the model first
        start_time = time.time()
        quantized_state = {}
        for name, param in model.named_parameters():
            if param.ndim >= 2 and param.numel() > 1000:
                result = quantize_model_layer(param.data, request.bits, request.scheme, request.group_size)
                quantized_state[name] = result["dequantized"]

        # Apply quantized weights
        original_state = {}
        for name, param in model.named_parameters():
            if name in quantized_state:
                original_state[name] = param.data.clone()
                param.data = quantized_state[name]

        quant_time = time.time() - start_time

        # Export
        export_result = export_quantized_model(
            model=model,
            tokenizer=tokenizer,
            bits=request.bits,
            scheme=request.scheme,
            group_size=request.group_size,
            metadata={"quantization_time_seconds": round(quant_time, 2)},
        )

        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_state:
                param.data = original_state[name]

        export_result["quantization_time_seconds"] = round(quant_time, 2)
        return export_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/real-model/exports")
async def list_exports():
    """List all saved quantized model exports"""
    try:
        from services.model_export import list_exported_models
        exports = list_exported_models()
        return {"exports": exports, "count": len(exports)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/mixed-precision")
async def run_mixed_precision(request: MixedPrecisionRequest):
    """
    Quantize with mixed precision - sensitive layers at higher bits, others at lower bits.
    Auto-detects sensitive layers (embeddings, lm_head, first/last transformer layers)
    unless explicitly specified.
    """
    try:
        from services.mixed_precision import quantize_mixed_precision

        model, tokenizer = get_model_and_tokenizer()

        # Compute baseline perplexity
        eval_text = "The quick brown fox jumps over the lazy dog. Machine learning models are trained using large datasets."
        baseline_ppl = compute_perplexity(model, tokenizer, eval_text)

        # Run mixed precision quantization
        layer_results, sensitive_layers, summary = quantize_mixed_precision(
            model=model,
            tokenizer=tokenizer,
            default_bits=request.default_bits,
            sensitive_bits=request.sensitive_bits,
            sensitive_layers=request.sensitive_layers,
        )

        # Compute post-quantization perplexity
        quantized_ppl = compute_perplexity(model, tokenizer, eval_text)

        # Generate text sample
        input_ids = tokenizer.encode("The meaning of life is", return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=30, do_sample=False)
        generation = tokenizer.decode(output[0], skip_special_tokens=True)

        # Restore model (clear cache to force reload)
        from services.real_quantization import MODEL_CACHE
        MODEL_CACHE.clear()

        return {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "default_bits": request.default_bits,
            "sensitive_bits": request.sensitive_bits,
            "baseline_perplexity": round(baseline_ppl, 4),
            "quantized_perplexity": round(quantized_ppl, 4),
            "perplexity_increase": round(quantized_ppl - baseline_ppl, 4),
            "sample_generation": generation,
            "summary": summary,
            "sensitive_layer_names": sensitive_layers,
            "layer_results": layer_results[:30],  # Limit response size
            "total_layers_quantized": len(layer_results),
        }
    except Exception as e:
        from services.real_quantization import MODEL_CACHE
        MODEL_CACHE.clear()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/real-model/evaluate-wikitext")
async def evaluate_wikitext():
    """
    Evaluate perplexity on wikitext-2 test set (industry standard metric).
    Downloads wikitext-2 if not cached. Takes 30-60 seconds on CPU.
    """
    try:
        from services.calibration import evaluate_perplexity_wikitext

        model, tokenizer = get_model_and_tokenizer()

        start_time = time.time()
        perplexity = evaluate_perplexity_wikitext(
            model=model,
            tokenizer=tokenizer,
            max_samples=50,
            seq_length=512,
        )
        elapsed = time.time() - start_time

        return {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
            "perplexity": round(perplexity, 4),
            "max_samples": 50,
            "seq_length": 512,
            "time_seconds": round(elapsed, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
