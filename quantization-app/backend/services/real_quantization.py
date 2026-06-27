"""
Real model quantization service - downloads and quantizes actual LLMs.
Automatically uses GPU (CUDA) when available for faster execution.
"""
import torch
import numpy as np
import time
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from services.device import get_device, get_device_info


MODEL_CACHE = {}


def get_model_and_tokenizer(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    if model_name not in MODEL_CACHE:
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        model.to(device)
        MODEL_CACHE[model_name] = (model, tokenizer)
    return MODEL_CACHE[model_name]


def get_model_info(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> Dict:
    model, tokenizer = get_model_and_tokenizer(model_name)
    params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    layer_info = []
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            layer_info.append({
                "name": name,
                "shape": list(param.shape),
                "numel": int(param.numel()),
                "dtype": str(param.dtype),
                "mean": float(param.data.mean()),
                "std": float(param.data.std()),
                "min": float(param.data.min()),
                "max": float(param.data.max()),
            })

    return {
        "model_name": model_name,
        "architecture": model.config.model_type,
        "num_parameters": int(params),
        "num_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "fp32_size_mb": round(size_bytes / (1024 * 1024), 2),
        "loaded_dtype": "float32",
        "layers": layer_info[:20],
    }


def compute_perplexity(model, tokenizer, text: str) -> float:
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss))


def quantize_model_layer(
    weights: torch.Tensor,
    bits: int = 8,
    scheme: str = "symmetric",
    group_size: int = 128
) -> Dict:
    original = weights.float().to(weights.device)

    if scheme == "symmetric":
        if group_size > 0 and weights.numel() > group_size:
            flat = original.reshape(-1, group_size) if original.numel() % group_size == 0 else original.reshape(1, -1)
            max_vals = flat.abs().amax(dim=1, keepdim=True)
            q_max = (2 ** (bits - 1)) - 1
            scales = max_vals / q_max
            scales = scales.clamp(min=1e-10)
            quantized = (flat / scales).round().clamp(-q_max - 1, q_max)
            dequantized = (quantized * scales).reshape(original.shape)
        else:
            max_val = original.abs().max()
            q_max = (2 ** (bits - 1)) - 1
            scale = max_val / q_max
            quantized = (original / scale).round().clamp(-q_max - 1, q_max)
            dequantized = quantized * scale
    elif scheme == "asymmetric":
        w_min = original.min()
        w_max = original.max()
        q_range = (2 ** bits) - 1
        scale = (w_max - w_min) / q_range
        scale = scale.clamp(min=1e-10)
        zero_point = (-w_min / scale).round()
        quantized = (original / scale + zero_point).round().clamp(0, q_range)
        dequantized = (quantized - zero_point) * scale
    else:
        dequantized = original

    error = (original - dequantized)
    mse = float(error.pow(2).mean())
    max_error = float(error.abs().max())

    return {
        "dequantized": dequantized,
        "mse": mse,
        "max_error": max_error,
        "compression_ratio": 32.0 / bits,
    }


def quantize_full_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    bits: int = 8,
    scheme: str = "symmetric",
    group_size: int = 128,
    test_text: str = "The meaning of life is",
) -> Dict:
    model, tokenizer = get_model_and_tokenizer(model_name)

    start_time = time.time()

    # Compute baseline perplexity
    eval_text = "The quick brown fox jumps over the lazy dog. Machine learning models are trained using large datasets."
    baseline_ppl = compute_perplexity(model, tokenizer, eval_text)

    # Quantize all linear layers
    layer_results = []
    total_params = 0
    total_mse = 0.0
    quantized_state = {}

    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() > 1000:
            result = quantize_model_layer(param.data, bits, scheme, group_size)
            quantized_state[name] = result["dequantized"]
            total_params += param.numel()
            total_mse += result["mse"] * param.numel()
            layer_results.append({
                "name": name,
                "shape": list(param.shape),
                "mse": result["mse"],
                "max_error": result["max_error"],
                "numel": int(param.numel()),
            })

    avg_mse = total_mse / total_params if total_params > 0 else 0

    # Apply quantized weights temporarily and measure perplexity
    original_state = {}
    for name, param in model.named_parameters():
        if name in quantized_state:
            original_state[name] = param.data.clone()
            param.data = quantized_state[name]

    quantized_ppl = compute_perplexity(model, tokenizer, eval_text)

    # Generate text with quantized model
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=30, do_sample=False)
    quantized_generation = tokenizer.decode(output[0], skip_special_tokens=True)

    # Restore original weights
    for name, param in model.named_parameters():
        if name in original_state:
            param.data = original_state[name]

    # Generate with original model
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=30, do_sample=False)
    original_generation = tokenizer.decode(output[0], skip_special_tokens=True)

    elapsed = time.time() - start_time

    params_total = sum(p.numel() for p in model.parameters())
    fp32_size = params_total * 4 / (1024 * 1024)
    quantized_size = params_total * bits / 8 / (1024 * 1024)

    return {
        "model_name": model_name,
        "bits": bits,
        "scheme": scheme,
        "group_size": group_size,
        "num_parameters": int(params_total),
        "num_layers_quantized": len(layer_results),
        "fp32_size_mb": round(fp32_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "compression_ratio": round(32.0 / bits, 1),
        "avg_mse": float(avg_mse),
        "baseline_perplexity": round(baseline_ppl, 4),
        "quantized_perplexity": round(quantized_ppl, 4),
        "perplexity_increase": round(quantized_ppl - baseline_ppl, 4),
        "perplexity_increase_pct": round((quantized_ppl - baseline_ppl) / baseline_ppl * 100, 2),
        "original_generation": original_generation,
        "quantized_generation": quantized_generation,
        "test_prompt": test_text,
        "time_seconds": round(elapsed, 2),
        "top_degraded_layers": sorted(layer_results, key=lambda x: -x["mse"])[:10],
    }


def benchmark_generation_speed(
    model=None,
    tokenizer=None,
    prompt: str = "The",
    num_tokens: int = 100,
    num_runs: int = 3,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
) -> Dict:
    """
    Measure generation speed in tokens/second.
    Runs multiple times and returns average.

    Args:
        model: Optional pre-loaded model (if None, loads default)
        tokenizer: Optional pre-loaded tokenizer
        prompt: Input prompt to start generation
        num_tokens: Number of tokens to generate per run
        num_runs: Number of measurement runs

    Returns:
        Dict with tokens_per_second, avg_time, and run details
    """
    if model is None or tokenizer is None:
        model, tokenizer = get_model_and_tokenizer(model_name)

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    # Warmup run (important for GPU to initialize kernels)
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=5, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_generated = num_tokens  # max_new_tokens is the target

    return {
        "tokens_per_second": round(tokens_generated / avg_time, 2),
        "avg_time_seconds": round(avg_time, 3),
        "num_tokens": num_tokens,
        "num_runs": num_runs,
        "run_times": [round(t, 3) for t in times],
        "prompt": prompt,
    }


def compare_quantization_methods(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    test_text: str = "Artificial intelligence is",
) -> Dict:
    model, tokenizer = get_model_and_tokenizer(model_name)

    eval_text = "The quick brown fox jumps over the lazy dog. Machine learning models are trained using large datasets."
    baseline_ppl = compute_perplexity(model, tokenizer, eval_text)

    device = next(model.parameters()).device

    methods = [
        {"bits": 8, "scheme": "symmetric", "group_size": 0, "label": "INT8 Symmetric (per-tensor)"},
        {"bits": 8, "scheme": "symmetric", "group_size": 128, "label": "INT8 Symmetric (group=128)"},
        {"bits": 8, "scheme": "asymmetric", "group_size": 0, "label": "INT8 Asymmetric"},
        {"bits": 4, "scheme": "symmetric", "group_size": 128, "label": "INT4 Symmetric (group=128)"},
        {"bits": 4, "scheme": "symmetric", "group_size": 32, "label": "INT4 Symmetric (group=32)"},
        {"bits": 3, "scheme": "symmetric", "group_size": 128, "label": "INT3 Symmetric (group=128)"},
    ]

    results = []
    params_total = sum(p.numel() for p in model.parameters())
    fp32_size = params_total * 4 / (1024 * 1024)

    for method in methods:
        start = time.time()

        total_mse = 0.0
        total_params = 0
        quantized_state = {}

        for name, param in model.named_parameters():
            if param.ndim >= 2 and param.numel() > 1000:
                result = quantize_model_layer(
                    param.data, method["bits"], method["scheme"], method["group_size"]
                )
                quantized_state[name] = result["dequantized"]
                total_mse += result["mse"] * param.numel()
                total_params += param.numel()

        avg_mse = total_mse / total_params if total_params > 0 else 0

        # Measure perplexity
        original_state = {}
        for name, param in model.named_parameters():
            if name in quantized_state:
                original_state[name] = param.data.clone()
                param.data = quantized_state[name]

        ppl = compute_perplexity(model, tokenizer, eval_text)

        # Generate
        input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        generation = tokenizer.decode(output[0], skip_special_tokens=True)

        # Restore
        for name, param in model.named_parameters():
            if name in original_state:
                param.data = original_state[name]

        elapsed = time.time() - start
        quantized_size = params_total * method["bits"] / 8 / (1024 * 1024)

        results.append({
            "label": method["label"],
            "bits": method["bits"],
            "scheme": method["scheme"],
            "group_size": method["group_size"],
            "mse": float(avg_mse),
            "perplexity": round(ppl, 4),
            "perplexity_increase": round(ppl - baseline_ppl, 4),
            "size_mb": round(quantized_size, 2),
            "compression_ratio": round(32.0 / method["bits"], 1),
            "generation": generation,
            "time_seconds": round(elapsed, 2),
        })

    return {
        "model_name": model_name,
        "num_parameters": int(params_total),
        "fp32_size_mb": round(fp32_size, 2),
        "baseline_perplexity": round(baseline_ppl, 4),
        "test_prompt": test_text,
        "original_generation": None,
        "methods": results,
    }
