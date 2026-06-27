# %% [markdown]
# # Post-Training Quantization (PTQ)
#
# This module implements PTQ: quantizing a pre-trained model without retraining.
#
# **PTQ Pipeline:**
# 1. Collect representative calibration inputs
# 2. Calibrate to find optimal scale/zero-point
# 3. Quantize weights using calibrated parameters
# 4. Perform inference using dequantized weights
#
# **Learning Objective:** Understand how to deploy a quantized model
# without modifying the training process.
#
# ## PTQ vs QAT
#
# - **PTQ**: Quantize after training, no gradient updates, fast
# - **QAT**: Quantize during training, model learns to compensate, slower but more accurate

# %%
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .quantizer import dequantize
from .calibrator import calibrate_min_max
from .granularity import (
    GranularityMode,
    quantize_per_tensor,
    quantize_per_channel,
    quantize_per_group,
)
from .schemes import QuantizationScheme


# %% [markdown]
# ## PTQ Layer Dataclass
#
# Stores all quantization parameters for a single layer.

# %%
@dataclass
class PTQLayer:
    """
    A post-training quantized linear layer.

    Attributes:
        W_q: Quantized weight integers, shape [out_features, in_features]
        scales: Scale parameter(s) for dequantization
                - Per-tensor: scalar float
                - Per-channel: array of shape [out_features]
                - Per-group: array of shape [out_features, num_groups]
        zero_points: Zero-point parameter(s)
                     - Per-tensor: scalar int
                     - Per-channel: array of shape [out_features]
                     - Per-group: array of shape [out_features, num_groups]
        b: Bias vector (unquantized), shape [out_features]
        bits: Bit-width used for quantization
        scheme: Quantization scheme used
        granularity: Granularity level used
    """
    W_q: np.ndarray
    scales: np.ndarray  # Can be scalar, 1D, or 2D
    zero_points: np.ndarray  # Can be scalar, 1D, or 2D
    b: np.ndarray
    bits: int
    scheme: QuantizationScheme
    granularity: GranularityMode


# %% [markdown]
# ## PTQ Quantize Layer
#
# Main PTQ function: takes a pre-trained layer and quantizes it.

# %%
def ptq_quantize_layer(
    W: np.ndarray,
    b: np.ndarray,
    calibration_inputs: List[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
    granularity: GranularityMode,
    group_size: int = 64,
) -> PTQLayer:
    """
    Quantize a pre-trained linear layer using PTQ.

    Args:
        W: Weight matrix, shape [out_features, in_features]
        b: Bias vector, shape [out_features]
        calibration_inputs: List of representative input tensors
                           Each of shape [batch_size, in_features]
        bits: Bit-width for quantization
        scheme: Quantization scheme
        granularity: Granularity level
        group_size: Group size for per-group quantization (ignored for other modes)

    Returns:
        PTQLayer with quantized weights and parameters

    Raises:
        ValueError: If calibration_inputs is empty

    Examples:
        >>> W = np.random.randn(16, 32).astype(np.float32) * 0.1
        >>> b = np.zeros(16, dtype=np.float32)
        >>> inputs = [np.random.randn(4, 32).astype(np.float32) for _ in range(10)]
        >>> layer = ptq_quantize_layer(W, b, inputs, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_TENSOR)
        >>> layer.W_q.shape
        (16, 32)
    """
    if not calibration_inputs or len(calibration_inputs) == 0:
        raise ValueError("At least 1 calibration sample is required")

    # Note: For PTQ, we typically calibrate on the weights themselves
    # (not the activations) for weight quantization.
    # The calibration_inputs are kept for API consistency but not used
    # in this simple implementation. In practice, you might use them
    # to calibrate activation quantization.

    # Quantize based on granularity
    if granularity == GranularityMode.PER_TENSOR:
        W_q, scale, zero_point = quantize_per_tensor(W, bits, scheme)
        scales = np.array([scale], dtype=np.float32)
        zero_points = np.array([zero_point], dtype=np.int32)

    elif granularity == GranularityMode.PER_CHANNEL:
        W_q, scales, zero_points = quantize_per_channel(W, bits, scheme)

    elif granularity == GranularityMode.PER_GROUP:
        W_q, scales, zero_points = quantize_per_group(W, bits, scheme, group_size)

    else:
        raise ValueError(f"Unknown granularity mode: {granularity}")

    return PTQLayer(
        W_q=W_q,
        scales=scales,
        zero_points=zero_points,
        b=b,
        bits=bits,
        scheme=scheme,
        granularity=granularity,
    )


# %% [markdown]
# ## PTQ Inference
#
# Run inference using a PTQ-quantized layer.

# %%
def ptq_infer(layer: PTQLayer, x: np.ndarray) -> np.ndarray:
    """
    Perform inference with a PTQ-quantized layer.

    Inference formula:
        output = dequantize(W_q) @ x.T + b

    Args:
        layer: PTQ-quantized layer
        x: Input tensor, shape [batch_size, in_features]

    Returns:
        Output tensor, shape [out_features, batch_size]

    Examples:
        >>> W = np.random.randn(16, 32).astype(np.float32) * 0.1
        >>> b = np.zeros(16, dtype=np.float32)
        >>> inputs = [np.random.randn(4, 32).astype(np.float32) for _ in range(10)]
        >>> layer = ptq_quantize_layer(W, b, inputs, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_TENSOR)
        >>> x_test = np.random.randn(8, 32).astype(np.float32)
        >>> output = ptq_infer(layer, x_test)
        >>> output.shape
        (16, 8)
    """
    out_features, in_features = layer.W_q.shape
    batch_size = x.shape[0]

    # Dequantize weights based on granularity
    if layer.granularity == GranularityMode.PER_TENSOR:
        W_dequant = dequantize(layer.W_q, layer.scales[0], layer.zero_points[0])

    elif layer.granularity == GranularityMode.PER_CHANNEL:
        W_dequant = np.zeros_like(layer.W_q, dtype=np.float32)
        for i in range(out_features):
            W_dequant[i, :] = dequantize(
                layer.W_q[i, :],
                layer.scales[i],
                layer.zero_points[i]
            )

    elif layer.granularity == GranularityMode.PER_GROUP:
        W_dequant = np.zeros_like(layer.W_q, dtype=np.float32)
        num_groups = layer.scales.shape[1]
        group_size = in_features // num_groups

        for i in range(out_features):
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                W_dequant[i, start:end] = dequantize(
                    layer.W_q[i, start:end],
                    layer.scales[i, g],
                    layer.zero_points[i, g]
                )

    # Compute output: W @ x.T + b
    # Output shape: [out_features, batch_size]
    output = W_dequant @ x.T + layer.b[:, np.newaxis]

    return output


# %% [markdown]
# ## Demo: PTQ on a Random Layer

# %%
if __name__ == "__main__":
    print("=" * 80)
    print("Demo: Post-Training Quantization (PTQ)")
    print("=" * 80)

    # Create a random linear layer
    np.random.seed(42)
    out_features = 16
    in_features = 64
    batch_size = 8

    W_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    b_fp32 = np.zeros(out_features, dtype=np.float32)

    print(f"\nLayer configuration:")
    print(f"  Weight shape: {W_fp32.shape}")
    print(f"  Bias shape: {b_fp32.shape}")
    print(f"  Weight stats: min={np.min(W_fp32):.4f}, max={np.max(W_fp32):.4f}, std={np.std(W_fp32):.4f}")

    # Generate calibration data
    num_calibration_samples = 20
    calibration_inputs = [
        np.random.randn(batch_size, in_features).astype(np.float32)
        for _ in range(num_calibration_samples)
    ]

    print(f"\nCalibration dataset:")
    print(f"  Number of samples: {num_calibration_samples}")
    print(f"  Batch size: {batch_size}")

    # PTQ with different granularities
    bits = 8
    scheme = QuantizationScheme.SYMMETRIC

    print(f"\nQuantization configuration:")
    print(f"  Bits: {bits}")
    print(f"  Scheme: {scheme.value}")

    print(f"\n{'Granularity':<15} {'Num_Scales':<12} {'Mean_Abs_Error':<18} {'Max_Abs_Error':<18}")
    print("-" * 80)

    for granularity in [GranularityMode.PER_TENSOR, GranularityMode.PER_CHANNEL, GranularityMode.PER_GROUP]:
        # PTQ quantize
        ptq_layer = ptq_quantize_layer(
            W_fp32, b_fp32, calibration_inputs,
            bits, scheme, granularity, group_size=16
        )

        # Test inference
        x_test = np.random.randn(batch_size, in_features).astype(np.float32)

        # FP32 baseline
        output_fp32 = W_fp32 @ x_test.T + b_fp32[:, np.newaxis]

        # PTQ inference
        output_ptq = ptq_infer(ptq_layer, x_test)

        # Compute error
        error = np.abs(output_fp32 - output_ptq)
        mae = np.mean(error)
        max_error = np.max(error)

        num_scales = ptq_layer.scales.size

        print(f"{granularity.value:<15} {num_scales:<12} {mae:<18.6f} {max_error:<18.6f}")

    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    print("1. PTQ requires no retraining - quantize once, deploy")
    print("2. Finer granularity → more scales → better accuracy")
    print("3. Typical choice: per-channel (good balance)")
    print("4. Per-group useful for very large layers (e.g., LLMs with 4096+ dims)")
