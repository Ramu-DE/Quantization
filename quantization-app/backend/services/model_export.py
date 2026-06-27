"""
Model export service - saves quantized model weights to disk in safetensors format.
"""
import os
import json
import time
from typing import Dict, Optional

import torch

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "exports")


def export_quantized_model(
    model,
    tokenizer,
    bits: int,
    scheme: str,
    group_size: int,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Save quantized model weights to disk in safetensors format.
    Also saves quantization config as JSON.

    Args:
        model: The model with quantized (dequantized float) weights
        tokenizer: The tokenizer (for reference in config)
        bits: Bit-width used for quantization
        scheme: Quantization scheme (symmetric/asymmetric/gptq)
        group_size: Group size used
        metadata: Additional metadata to save in config

    Returns:
        Dict with filepath, filename, file sizes, and config path
    """
    from safetensors.torch import save_file

    os.makedirs(EXPORT_DIR, exist_ok=True)

    start_time = time.time()

    # Collect state dict (only save tensor values, no nested structures)
    state_dict = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.contiguous()

    # Generate filename
    filename = f"tinyllama-1.1b-{bits}bit-{scheme}-g{group_size}.safetensors"
    filepath = os.path.join(EXPORT_DIR, filename)

    # Save weights in safetensors format
    save_file(state_dict, filepath)

    # Save quantization config
    num_params = sum(v.numel() for v in state_dict.values())
    config = {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "bits": bits,
        "scheme": scheme,
        "group_size": group_size,
        "format": "simulated_quantize_dequantize",
        "num_parameters": num_params,
        "fp32_size_mb": round(num_params * 4 / (1024 * 1024), 2),
        "theoretical_size_mb": round(num_params * bits / 8 / (1024 * 1024), 2),
        **(metadata or {}),
    }
    config_path = os.path.join(EXPORT_DIR, filename.replace(".safetensors", ".json"))
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start_time
    file_size = os.path.getsize(filepath)

    return {
        "filepath": filepath,
        "filename": filename,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "config_path": config_path,
        "config": config,
        "time_seconds": round(elapsed, 2),
    }


def list_exported_models() -> list:
    """
    List all exported quantized models with their configs.

    Returns:
        List of dicts with filename, size, and config for each export
    """
    if not os.path.exists(EXPORT_DIR):
        return []

    files = [f for f in os.listdir(EXPORT_DIR) if f.endswith(".safetensors")]
    files.sort()

    results = []
    for f in files:
        path = os.path.join(EXPORT_DIR, f)
        config_path = path.replace(".safetensors", ".json")
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path) as cf:
                    config = json.load(cf)
            except (json.JSONDecodeError, IOError):
                config = {"error": "could not read config"}

        results.append({
            "filename": f,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "config": config,
        })

    return results


def delete_exported_model(filename: str) -> Dict:
    """
    Delete an exported model file and its config.

    Args:
        filename: The .safetensors filename to delete

    Returns:
        Dict with status
    """
    filepath = os.path.join(EXPORT_DIR, filename)
    config_path = filepath.replace(".safetensors", ".json")

    if not os.path.exists(filepath):
        return {"status": "error", "message": f"File not found: {filename}"}

    os.remove(filepath)
    if os.path.exists(config_path):
        os.remove(config_path)

    return {"status": "deleted", "filename": filename}
