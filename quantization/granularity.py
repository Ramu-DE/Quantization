# %% [markdown]
# # Quantization Granularity
#
# This module implements different granularity levels for quantization:
#
# 1. **Per-Tensor**: Single (scale, zero-point) for entire tensor
# 2. **Per-Channel**: One (scale, zero-point) per output channel (row)
# 3. **Per-Group**: One (scale, zero-point) per fixed-size group within each channel
#
# **Learning Objective:** Understand the tradeoff between quantization accuracy
# and parameter overhead. Finer granularity → better accuracy but more parameters.
#
# ## Granularity Hierarchy
#
# ```
# Per-Tensor:   [===================]  1 scale/zp
# Per-Channel:  [===][===][===][===]  N scales/zps (N = out_channels)
# Per-Group:    [=][=][=][=][=][=]    N×M scales/zps (M = in_ch / group_size)
# ```

# %%
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

from .quantizer import quantize, dequantize
from .schemes import QuantizationScheme, compute_scale_zero_point


# %% [markdown]
# ## Granularity Mode Enum

# %%
class GranularityMode(Enum):
    """Quantization granularity level."""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


# %% [markdown]
# ## Granularity Comparison Report

# %%
@dataclass
class GranularityComparisonReport:
    """Report for comparing different granularity levels."""
    granularity: GranularityMode
    num_scale_params: int
    mean_absolute_error: float
    max_absolute_error: float


# %% [markdown]
# ## Per-Tensor Quantization
#
# Simplest: one scale and zero-point for the entire tensor.

# %%
def quantize_per_tensor(
    W: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
) -> Tuple[np.ndarray, float, int]:
    """
    Quantize a weight tensor with per-tensor granularity.

    Args:
        W: Weight tensor of shape [out_channels, in_channels]
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme

    Returns:
        Tuple of (W_quantized, scale, zero_point)

    Examples:
        >>> W = np.random.randn(8, 32).astype(np.float32)
        >>> W_q, scale, zp = quantize_per_tensor(W, 8, QuantizationScheme.SYMMETRIC)
        >>> W_q.shape == W.shape
        True
    """
    # Compute single scale/zp for entire tensor
    scale, zero_point = compute_scale_zero_point(W.flatten(), bits, scheme)

    # Quantize
    W_q = quantize(W, scale, zero_point, bits, signed=(scheme == QuantizationScheme.SYMMETRIC))

    return W_q, scale, zero_point


# %% [markdown]
# ## Per-Channel Quantization
#
# One scale and zero-point per output channel (row of weight matrix).
#
# **Rationale:** Different output channels often have different magnitude distributions.

# %%
def quantize_per_channel(
    W: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize a weight tensor with per-channel granularity.

    Args:
        W: Weight tensor of shape [out_channels, in_channels]
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme

    Returns:
        Tuple of (W_quantized, scales, zero_points)
        - W_quantized: shape [out_channels, in_channels]
        - scales: shape [out_channels]
        - zero_points: shape [out_channels]

    Examples:
        >>> W = np.random.randn(8, 32).astype(np.float32)
        >>> W_q, scales, zps = quantize_per_channel(W, 8, QuantizationScheme.SYMMETRIC)
        >>> W_q.shape == W.shape
        True
        >>> len(scales) == 8
        True
    """
    out_channels, in_channels = W.shape

    scales = np.zeros(out_channels, dtype=np.float32)
    zero_points = np.zeros(out_channels, dtype=np.int32)
    W_q = np.zeros_like(W, dtype=np.int32)

    # Quantize each channel (row) independently
    for i in range(out_channels):
        channel_data = W[i, :]

        # Compute scale/zp for this channel
        scale, zp = compute_scale_zero_point(channel_data, bits, scheme)
        scales[i] = scale
        zero_points[i] = zp

        # Quantize this channel
        W_q[i, :] = quantize(
            channel_data,
            scale,
            zp,
            bits,
            signed=(scheme == QuantizationScheme.SYMMETRIC)
        )

    return W_q, scales, zero_points


# %% [markdown]
# ## Per-Group Quantization
#
# One scale and zero-point per fixed-size group within each channel.
#
# **Rationale:** Even within a single channel, different regions may have
# different magnitude distributions. Useful for very large channels.

# %%
def quantize_per_group(
    W: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize a weight tensor with per-group granularity.

    Args:
        W: Weight tensor of shape [out_channels, in_channels]
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme
        group_size: Number of elements per group (must divide in_channels evenly)

    Returns:
        Tuple of (W_quantized, scales, zero_points)
        - W_quantized: shape [out_channels, in_channels]
        - scales: shape [out_channels, num_groups_per_channel]
        - zero_points: shape [out_channels, num_groups_per_channel]

    Raises:
        ValueError: If group_size does not divide in_channels evenly

    Examples:
        >>> W = np.random.randn(8, 32).astype(np.float32)
        >>> W_q, scales, zps = quantize_per_group(W, 8, QuantizationScheme.SYMMETRIC, group_size=8)
        >>> W_q.shape == W.shape
        True
        >>> scales.shape
        (8, 4)
    """
    out_channels, in_channels = W.shape

    if in_channels % group_size != 0:
        raise ValueError(
            f"group_size must divide in_channels evenly: "
            f"in_channels={in_channels}, group_size={group_size}"
        )

    num_groups_per_channel = in_channels // group_size

    scales = np.zeros((out_channels, num_groups_per_channel), dtype=np.float32)
    zero_points = np.zeros((out_channels, num_groups_per_channel), dtype=np.int32)
    W_q = np.zeros_like(W, dtype=np.int32)

    # Quantize each group independently
    for i in range(out_channels):
        for g in range(num_groups_per_channel):
            # Extract group
            start_idx = g * group_size
            end_idx = start_idx + group_size
            group_data = W[i, start_idx:end_idx]

            # Compute scale/zp for this group
            scale, zp = compute_scale_zero_point(group_data, bits, scheme)
            scales[i, g] = scale
            zero_points[i, g] = zp

            # Quantize this group
            W_q[i, start_idx:end_idx] = quantize(
                group_data,
                scale,
                zp,
                bits,
                signed=(scheme == QuantizationScheme.SYMMETRIC)
            )

    return W_q, scales, zero_points


# %% [markdown]
# ## Compare Granularity Levels
#
# Helper function to compare all three granularity levels on the same tensor.

# %%
def compare_granularity(
    W: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
    group_size: int = 64,
) -> list[GranularityComparisonReport]:
    """
    Compare all three granularity levels on the same tensor.

    Args:
        W: Weight tensor
        bits: Bit-width
        scheme: Quantization scheme
        group_size: Group size for per-group quantization

    Returns:
        List of GranularityComparisonReport, one per granularity level
    """
    reports = []

    # Per-Tensor
    W_q_t, scale_t, zp_t = quantize_per_tensor(W, bits, scheme)
    W_recon_t = dequantize(W_q_t, scale_t, zp_t)
    mae_t = np.mean(np.abs(W - W_recon_t))
    max_e_t = np.max(np.abs(W - W_recon_t))
    reports.append(GranularityComparisonReport(
        granularity=GranularityMode.PER_TENSOR,
        num_scale_params=1,
        mean_absolute_error=float(mae_t),
        max_absolute_error=float(max_e_t),
    ))

    # Per-Channel
    W_q_c, scales_c, zps_c = quantize_per_channel(W, bits, scheme)
    W_recon_c = np.zeros_like(W, dtype=np.float32)
    for i in range(W.shape[0]):
        W_recon_c[i, :] = dequantize(W_q_c[i, :], scales_c[i], zps_c[i])
    mae_c = np.mean(np.abs(W - W_recon_c))
    max_e_c = np.max(np.abs(W - W_recon_c))
    reports.append(GranularityComparisonReport(
        granularity=GranularityMode.PER_CHANNEL,
        num_scale_params=int(len(scales_c)),
        mean_absolute_error=float(mae_c),
        max_absolute_error=float(max_e_c),
    ))

    # Per-Group (if valid group size)
    if W.shape[1] % group_size == 0:
        W_q_g, scales_g, zps_g = quantize_per_group(W, bits, scheme, group_size)
        W_recon_g = np.zeros_like(W, dtype=np.float32)
        num_groups_per_ch = W.shape[1] // group_size
        for i in range(W.shape[0]):
            for g in range(num_groups_per_ch):
                start = g * group_size
                end = start + group_size
                W_recon_g[i, start:end] = dequantize(
                    W_q_g[i, start:end],
                    scales_g[i, g],
                    zps_g[i, g]
                )
        mae_g = np.mean(np.abs(W - W_recon_g))
        max_e_g = np.max(np.abs(W - W_recon_g))
        reports.append(GranularityComparisonReport(
            granularity=GranularityMode.PER_GROUP,
            num_scale_params=int(scales_g.size),
            mean_absolute_error=float(mae_g),
            max_absolute_error=float(max_e_g),
        ))

    return reports


# %% [markdown]
# ## Pretty-Print Granularity Comparison

# %%
def format_granularity_comparison(reports: list[GranularityComparisonReport]) -> str:
    """Format granularity comparison as a table."""
    lines = []
    lines.append("=" * 90)
    lines.append("Quantization Granularity Comparison")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Granularity':<15} {'Num_Params_Scale':<18} {'Mean_Absolute_Error':<20} {'Max_Absolute_Error':<20}")
    lines.append("-" * 90)

    for report in reports:
        lines.append(
            f"{report.granularity.value:<15} "
            f"{report.num_scale_params:<18} "
            f"{report.mean_absolute_error:<20.6f} "
            f"{report.max_absolute_error:<20.6f}"
        )

    lines.append("-" * 90)
    lines.append("")

    return "\n".join(lines)


# %% [markdown]
# ## Demo: Granularity Comparison on a Weight Matrix

# %%
if __name__ == "__main__":
    print("=" * 90)
    print("Demo: Quantization Granularity Comparison")
    print("=" * 90)

    # Create a weight matrix with non-uniform channel distributions
    np.random.seed(42)
    out_ch, in_ch = 8, 64

    W = np.zeros((out_ch, in_ch), dtype=np.float32)
    for i in range(out_ch):
        # Each channel has different std (simulating real weight distributions)
        std = 0.01 * (i + 1)  # std ranges from 0.01 to 0.08
        W[i, :] = np.random.randn(in_ch).astype(np.float32) * std

    print(f"\nWeight matrix shape: {W.shape}")
    print(f"Per-channel std variation:")
    for i in range(out_ch):
        print(f"  Channel {i}: std={np.std(W[i, :]):.4f}")

    bits = 8
    scheme = QuantizationScheme.SYMMETRIC
    group_size = 16

    reports = compare_granularity(W, bits, scheme, group_size)

    print("\n" + format_granularity_comparison(reports))

    print("=" * 90)
    print("Key Insights:")
    print("=" * 90)
    print("1. Per-Tensor: Coarsest granularity, highest error (one scale for all)")
    print("2. Per-Channel: Better accuracy, adapts to per-channel distributions")
    print("3. Per-Group: Finest granularity, lowest error, but most parameters")
    print("4. Tradeoff: Accuracy vs Parameter Count")
    print(f"   - Per-Tensor: 1 scale")
    print(f"   - Per-Channel: {out_ch} scales")
    print(f"   - Per-Group: {out_ch * (in_ch // group_size)} scales")
