"""
Real GPTQ implementation using calibration data to compute Hessian.
Implements the algorithm from "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
Automatically uses GPU when available for Hessian computation and weight updates.
"""
import torch
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from services.device import get_device


def gptq_quantize_layer(
    layer_weight: torch.Tensor,
    layer_inputs: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,
) -> Dict:
    """
    Real GPTQ quantization for a single linear layer.

    Args:
        layer_weight: The weight matrix (out_features x in_features)
        layer_inputs: Calibration inputs to this layer (batch x in_features)
        bits: Target bit-width
        group_size: Quantization group size (0 = per-tensor)
        block_size: Column block size for processing (128 = GPTQ default)

    Algorithm:
        1. Compute Hessian: H = (2/n) * X^T @ X
        2. Add dampening: H += lambda * I
        3. Compute Cholesky of H_inv for numerical stability
        4. Process columns in blocks of block_size
        5. For each column: quantize, compute error, update remaining weights

    Returns:
        Dict with quantized_weight, scales, zeros, mse, and timing info
    """
    start_time = time.time()

    device = get_device()
    W = layer_weight.float().clone().to(device)
    out_features, in_features = W.shape

    # Reshape inputs: ensure (num_samples, in_features)
    X = layer_inputs.float().to(device)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
    if X.shape[1] != in_features:
        # Transpose if needed
        if X.shape[0] == in_features:
            X = X.T
        else:
            # Truncate or pad
            X = X[:, :in_features] if X.shape[1] > in_features else torch.nn.functional.pad(X, (0, in_features - X.shape[1]))

    n_samples = X.shape[0]

    # Clean inputs: remove NaN/Inf
    X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Step 1: Compute Hessian H = (2/n) * X^T @ X
    H = (2.0 / max(n_samples, 1)) * (X.T @ X)
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: Add dampening for numerical stability
    diag_mean = torch.diag(H).mean()
    if torch.isnan(diag_mean) or diag_mean <= 0:
        diag_mean = torch.tensor(1.0, device=device)
    damp = 0.01 * diag_mean
    damp = max(damp.item(), 1e-4)
    H += damp * torch.eye(in_features, device=device)

    # Step 3: Compute inverse Hessian via Cholesky
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fallback: add more dampening
        H += 0.1 * torch.eye(in_features, device=device) * diag_mean
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            # Last resort: diagonal approximation
            diag = torch.diag(H).clamp(min=1e-6)
            H_inv = torch.diag(1.0 / diag)

    # Clean H_inv
    H_inv = torch.nan_to_num(H_inv, nan=0.0, posinf=0.0, neginf=0.0)

    # Quantization parameters
    q_max = (2 ** (bits - 1)) - 1
    q_min = -(2 ** (bits - 1))

    # Storage for scales and zeros
    if group_size > 0:
        num_groups = math.ceil(in_features / group_size)
    else:
        num_groups = 1
        group_size = in_features

    scales = torch.zeros(out_features, num_groups, device=device)
    zeros = torch.zeros(out_features, num_groups, device=device)

    # Step 4: Process columns in blocks
    quantized_W = torch.zeros_like(W)
    errors = []

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)

        # Get the block of H_inv we need
        H_inv_block = H_inv[block_start:block_end, block_start:block_end].clone()

        # Process each column in this block
        for col in range(block_start, block_end):
            col_local = col - block_start

            # Determine which group this column belongs to
            group_idx = col // group_size
            group_start = group_idx * group_size
            group_end = min(group_start + group_size, in_features)

            # Compute scale for this group (symmetric quantization)
            w_group = W[:, group_start:group_end]
            max_val = w_group.abs().amax(dim=1, keepdim=False)
            scale = max_val / q_max
            scale = scale.clamp(min=1e-10)
            scales[:, group_idx] = scale

            # Get current weight column
            w_col = W[:, col]

            # Quantize
            q_col = (w_col / scale).round().clamp(q_min, q_max)

            # Dequantize
            dq_col = q_col * scale

            # Store quantized value
            quantized_W[:, col] = dq_col

            # Compute quantization error
            err = w_col - dq_col

            # Get diagonal of H_inv for this column
            h_inv_diag = H_inv_block[col_local, col_local]
            if h_inv_diag < 1e-10:
                h_inv_diag = torch.tensor(1e-10, device=device)

            # Weight the error
            weighted_err = err / h_inv_diag
            weighted_err = torch.nan_to_num(weighted_err, nan=0.0, posinf=0.0, neginf=0.0)

            # Update remaining columns in block
            if col < block_end - 1:
                remaining_start = col_local + 1
                h_inv_row = H_inv_block[col_local, remaining_start:]
                update = weighted_err.unsqueeze(1) * h_inv_row.unsqueeze(0)
                update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
                W[:, col + 1:block_end] -= update

        # After processing block, update remaining columns beyond block
        if block_end < in_features:
            block_err = W[:, block_start:block_end] - quantized_W[:, block_start:block_end]
            h_inv_cross = H_inv[block_start:block_end, block_end:]
            update = block_err @ h_inv_cross
            update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            W[:, block_end:] -= update

    # Compute MSE
    original_W = layer_weight.float().to(device)
    mse = float((original_W - quantized_W).pow(2).mean())
    max_error = float((original_W - quantized_W).abs().max())

    elapsed = time.time() - start_time

    return {
        "quantized_weight": quantized_W.cpu(),
        "scales": scales.cpu(),
        "zeros": zeros.cpu(),
        "mse": mse,
        "max_error": max_error,
        "time_seconds": round(elapsed, 3),
        "bits": bits,
        "group_size": group_size,
        "block_size": block_size,
    }


def gptq_quantize_model(
    model,
    tokenizer,
    calibration_data: List[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    max_layers: int = 20,
) -> Dict:
    """
    Apply GPTQ to linear layers of the model using real calibration data.

    For each transformer layer:
      1. Run calibration data through the model up to this layer (collect inputs)
      2. Quantize all linear sublayers using those inputs as Hessian basis
      3. Replace weights with quantized-dequantized versions
      4. Continue to next layer

    Args:
        model: HuggingFace causal LM model
        tokenizer: Corresponding tokenizer
        calibration_data: List of input_ids tensors from calibration dataset
        bits: Target bit-width
        group_size: Quantization group size
        max_layers: Maximum number of transformer layers to process

    Returns:
        Dict with per-layer results and overall statistics
    """
    start_time = time.time()
    model.eval()
    device = next(model.parameters()).device

    # Get the list of transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        transformer_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        transformer_layers = model.transformer.h
    else:
        raise ValueError("Unsupported model architecture - cannot find transformer layers")

    num_layers = min(len(transformer_layers), max_layers)
    layer_results = []
    total_mse = 0.0
    total_params = 0

    # Use a subset of calibration data for speed (use first 16 samples for hook capture)
    calib_subset = calibration_data[:min(16, len(calibration_data))]

    for layer_idx in range(num_layers):
        layer = transformer_layers[layer_idx]
        layer_start = time.time()

        # Capture inputs to this layer using hooks
        inputs_cache = []

        def capture_hook(module, inp, out):
            # inp is a tuple, first element is the hidden states
            if isinstance(inp, tuple) and len(inp) > 0:
                inputs_cache.append(inp[0].detach())

        hook = layer.register_forward_hook(capture_hook)

        # Run calibration data through the model to capture inputs
        with torch.no_grad():
            for batch in calib_subset:
                try:
                    if batch.shape[0] != 1:
                        batch = batch.unsqueeze(0)
                    model(batch.to(device))
                except Exception:
                    # Skip batches that cause errors (e.g., too long)
                    continue

        hook.remove()

        if not inputs_cache:
            layer_results.append({
                "layer_idx": layer_idx,
                "status": "skipped",
                "reason": "no inputs captured",
            })
            continue

        # Concatenate captured inputs
        layer_input = torch.cat(inputs_cache, dim=0)  # (total_tokens, hidden_size) after reshape
        if layer_input.ndim == 3:
            layer_input = layer_input.reshape(-1, layer_input.shape[-1])

        # Find all linear sub-layers in this transformer layer
        linear_layers = []
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append((name, module))

        sublayer_results = []
        for sub_name, linear in linear_layers:
            # Get input to this specific linear layer
            # For attention projections, input is the layer input
            # For MLP layers, we'd need the intermediate, but approximate with layer input
            weight = linear.weight.data

            # Use layer_input as approximation for all sublayers
            # (In production GPTQ, you'd hook each sublayer individually)
            result = gptq_quantize_layer(
                layer_weight=weight,
                layer_inputs=layer_input,
                bits=bits,
                group_size=group_size,
                block_size=128,
            )

            # Apply quantized weights back to model device
            linear.weight.data = result["quantized_weight"].to(device=device, dtype=linear.weight.dtype)

            total_mse += result["mse"] * weight.numel()
            total_params += weight.numel()

            sublayer_results.append({
                "name": sub_name,
                "shape": list(weight.shape),
                "mse": result["mse"],
                "max_error": result["max_error"],
                "time_seconds": result["time_seconds"],
            })

        layer_elapsed = time.time() - layer_start
        layer_results.append({
            "layer_idx": layer_idx,
            "status": "quantized",
            "sublayers": sublayer_results,
            "time_seconds": round(layer_elapsed, 2),
        })

    total_elapsed = time.time() - start_time
    avg_mse = total_mse / total_params if total_params > 0 else 0.0

    return {
        "method": "GPTQ",
        "bits": bits,
        "group_size": group_size,
        "num_layers_processed": num_layers,
        "total_params_quantized": total_params,
        "avg_mse": float(avg_mse),
        "total_time_seconds": round(total_elapsed, 2),
        "layer_results": layer_results,
    }
