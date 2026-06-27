"""
Mixed precision quantization - applies different bit-widths to different layers
based on sensitivity analysis.
"""
import torch
import time
from typing import Dict, List, Optional, Tuple

from services.real_quantization import quantize_model_layer


def quantize_mixed_precision(
    model,
    tokenizer,
    calibration_data=None,
    default_bits: int = 4,
    sensitive_bits: int = 8,
    sensitive_layers: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Quantize model with different bit-widths per layer.

    By default:
    - Embedding layer: keep at sensitive_bits (or FP16)
    - LM head: keep at sensitive_bits
    - First and last 2 transformer layers: sensitive_bits
    - All other layers: default_bits

    If sensitive_layers is provided, use that list instead.

    Args:
        model: HuggingFace causal LM model
        tokenizer: Corresponding tokenizer
        calibration_data: Optional calibration data (unused in basic mode)
        default_bits: Bit-width for normal layers
        sensitive_bits: Bit-width for sensitive layers
        sensitive_layers: Optional explicit list of parameter names to treat as sensitive

    Returns:
        Tuple of (per-layer results list, list of sensitive layer names)
    """
    start_time = time.time()

    if sensitive_layers is None:
        # Auto-detect sensitive layers based on common heuristics
        sensitive_layers = []
        num_hidden_layers = model.config.num_hidden_layers

        for name, _ in model.named_parameters():
            is_sensitive = False

            # Embedding layers
            if "embed" in name:
                is_sensitive = True
            # LM head
            elif "lm_head" in name:
                is_sensitive = True
            # First 2 transformer layers
            elif "layers.0." in name or "layers.1." in name:
                is_sensitive = True
            # Last 2 transformer layers
            elif f"layers.{num_hidden_layers - 1}." in name or f"layers.{num_hidden_layers - 2}." in name:
                is_sensitive = True

            if is_sensitive:
                sensitive_layers.append(name)

    results = []
    total_params_quantized = 0
    total_mse = 0.0
    bits_distribution = {"default": 0, "sensitive": 0, "skipped": 0}

    for name, param in model.named_parameters():
        # Skip small / 1D parameters (biases, norms)
        if param.ndim < 2 or param.numel() < 1000:
            bits_distribution["skipped"] += param.numel()
            continue

        # Determine bit-width for this layer
        bits = sensitive_bits if name in sensitive_layers else default_bits

        # Quantize
        result = quantize_model_layer(param.data, bits, "symmetric", 128)
        param.data = result["dequantized"]

        total_params_quantized += param.numel()
        total_mse += result["mse"] * param.numel()

        if name in sensitive_layers:
            bits_distribution["sensitive"] += param.numel()
        else:
            bits_distribution["default"] += param.numel()

        results.append({
            "name": name,
            "bits": bits,
            "is_sensitive": name in sensitive_layers,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "mse": result["mse"],
            "max_error": result["max_error"],
        })

    elapsed = time.time() - start_time

    # Compute effective bit-width (weighted average)
    effective_bits = 0.0
    if total_params_quantized > 0:
        for r in results:
            effective_bits += r["bits"] * r["numel"] / total_params_quantized

    avg_mse = total_mse / total_params_quantized if total_params_quantized > 0 else 0.0

    # Add summary info to results
    summary = {
        "default_bits": default_bits,
        "sensitive_bits": sensitive_bits,
        "effective_bits": round(effective_bits, 2),
        "total_params_quantized": total_params_quantized,
        "avg_mse": float(avg_mse),
        "time_seconds": round(elapsed, 2),
        "bits_distribution": {
            "default_params": bits_distribution["default"],
            "sensitive_params": bits_distribution["sensitive"],
            "skipped_params": bits_distribution["skipped"],
        },
        "num_sensitive_layers": len([r for r in results if r["is_sensitive"]]),
        "num_default_layers": len([r for r in results if not r["is_sensitive"]]),
    }

    return results, sensitive_layers, summary
