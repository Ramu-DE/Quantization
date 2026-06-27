# %% [markdown]
# # Property-Based Tests for Quantization Granularity
#
# **Test Coverage:**
# - Property 6: Granularity monotonicity (per-tensor ≥ per-channel ≥ per-group MAE)

# %%
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import sys
sys.path.insert(0, '..')

from quantization.granularity import (
    GranularityMode,
    quantize_per_tensor,
    quantize_per_channel,
    quantize_per_group,
    compare_granularity,
    format_granularity_comparison,
)
from quantization.quantizer import dequantize
from quantization.schemes import QuantizationScheme


# %% [markdown]
# ## Hypothesis Strategies

# %%
# Strategy: valid bit-widths
valid_bits_gran = st.sampled_from([2, 4, 8])

# Strategy: non-uniform weight tensors (different per-row std)
@st.composite
def non_uniform_weight_tensor(draw):
    """Generate a weight tensor with non-uniform row distributions."""
    out_ch = draw(st.integers(min_value=4, max_value=16))
    in_ch = draw(st.integers(min_value=32, max_value=128))

    # Ensure in_ch is divisible by common group sizes
    in_ch = (in_ch // 16) * 16  # Make divisible by 16

    W = np.zeros((out_ch, in_ch), dtype=np.float32)

    # Generate each row with different std
    for i in range(out_ch):
        std = draw(st.floats(min_value=0.01, max_value=0.5))
        W[i, :] = np.random.randn(in_ch).astype(np.float32) * std

    # Verify non-uniformity: at least 2 rows have different std
    row_stds = [np.std(W[i, :]) for i in range(out_ch)]
    if len(set([round(s, 3) for s in row_stds])) < 2:
        # Force variation
        W[0, :] *= 0.1
        W[-1, :] *= 2.0

    return W


# %% [markdown]
# ## Property 6: Granularity Monotonicity
#
# **Property:** For weight tensors with non-uniform row distributions,
# increasing granularity SHALL produce monotonically non-increasing MAE:
# `MAE(per-tensor) >= MAE(per-channel) >= MAE(per-group)`
#
# **Validates:** Requirements 5.5, 5.6, 9.4

# %%
@given(
    W=non_uniform_weight_tensor(),
    bits=valid_bits_gran,
)
@settings(max_examples=100)
def test_property_6_granularity_monotonicity(W, bits):
    """
    Feature: ai-model-quantization
    Property 6: Granularity Monotonicity

    For non-uniform weight tensors, MAE(per-tensor) >= MAE(per-channel) >= MAE(per-group).
    """
    scheme = QuantizationScheme.SYMMETRIC
    group_size = 16

    # Skip if shape doesn't allow per-group
    if W.shape[1] % group_size != 0:
        return

    # Per-Tensor
    W_q_t, scale_t, zp_t = quantize_per_tensor(W, bits, scheme)
    W_recon_t = dequantize(W_q_t, scale_t, zp_t)
    mae_t = np.mean(np.abs(W - W_recon_t))

    # Per-Channel
    W_q_c, scales_c, zps_c = quantize_per_channel(W, bits, scheme)
    W_recon_c = np.zeros_like(W, dtype=np.float32)
    for i in range(W.shape[0]):
        W_recon_c[i, :] = dequantize(W_q_c[i, :], scales_c[i], zps_c[i])
    mae_c = np.mean(np.abs(W - W_recon_c))

    # Per-Group
    W_q_g, scales_g, zps_g = quantize_per_group(W, bits, scheme, group_size)
    W_recon_g = np.zeros_like(W, dtype=np.float32)
    num_groups = W.shape[1] // group_size
    for i in range(W.shape[0]):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            W_recon_g[i, start:end] = dequantize(
                W_q_g[i, start:end],
                scales_g[i, g],
                zps_g[i, g]
            )
    mae_g = np.mean(np.abs(W - W_recon_g))

    # Verify monotonicity (with small tolerance for numerical precision)
    tol = 1e-6

    assert mae_t >= mae_c - tol, (
        f"Per-tensor MAE should be >= per-channel MAE: "
        f"mae_t={mae_t:.6f}, mae_c={mae_c:.6f}"
    )

    assert mae_c >= mae_g - tol, (
        f"Per-channel MAE should be >= per-group MAE: "
        f"mae_c={mae_c:.6f}, mae_g={mae_g:.6f}"
    )


# %% [markdown]
# ## Edge Case Tests

# %%
def test_quantize_per_group_invalid_group_size():
    """quantize_per_group should raise ValueError if group_size doesn't divide in_channels."""
    W = np.random.randn(8, 32).astype(np.float32)

    with pytest.raises(ValueError, match="group_size must divide in_channels evenly"):
        quantize_per_group(W, 8, QuantizationScheme.SYMMETRIC, group_size=7)


# %% [markdown]
# ## Smoke Tests

# %%
def test_quantize_per_tensor_shape():
    """Verify per-tensor quantization preserves shape."""
    W = np.random.randn(8, 32).astype(np.float32)
    W_q, scale, zp = quantize_per_tensor(W, 8, QuantizationScheme.SYMMETRIC)

    assert W_q.shape == W.shape
    assert isinstance(scale, (float, np.floating))
    assert isinstance(zp, (int, np.integer))


def test_quantize_per_channel_shape():
    """Verify per-channel quantization preserves shape and returns correct param counts."""
    W = np.random.randn(8, 32).astype(np.float32)
    W_q, scales, zps = quantize_per_channel(W, 8, QuantizationScheme.SYMMETRIC)

    assert W_q.shape == W.shape
    assert len(scales) == 8  # One per output channel
    assert len(zps) == 8
    assert np.all(zps == 0)  # Symmetric scheme


def test_quantize_per_group_shape():
    """Verify per-group quantization preserves shape and returns correct param counts."""
    W = np.random.randn(8, 64).astype(np.float32)
    group_size = 16
    W_q, scales, zps = quantize_per_group(W, 8, QuantizationScheme.SYMMETRIC, group_size)

    assert W_q.shape == W.shape
    assert scales.shape == (8, 4)  # 8 channels × 4 groups per channel
    assert zps.shape == (8, 4)


def test_compare_granularity():
    """Verify compare_granularity returns reports for all levels."""
    W = np.random.randn(8, 64).astype(np.float32)
    reports = compare_granularity(W, 8, QuantizationScheme.SYMMETRIC, group_size=16)

    assert len(reports) == 3  # per-tensor, per-channel, per-group
    assert reports[0].granularity == GranularityMode.PER_TENSOR
    assert reports[1].granularity == GranularityMode.PER_CHANNEL
    assert reports[2].granularity == GranularityMode.PER_GROUP

    # Verify param counts
    assert reports[0].num_scale_params == 1
    assert reports[1].num_scale_params == 8
    assert reports[2].num_scale_params == 32  # 8 × 4


def test_format_granularity_comparison():
    """Verify format_granularity_comparison produces valid output."""
    W = np.random.randn(4, 32).astype(np.float32)
    reports = compare_granularity(W, 8, QuantizationScheme.SYMMETRIC, group_size=8)
    table = format_granularity_comparison(reports)

    assert "Granularity" in table
    assert "Num_Params_Scale" in table
    assert "Mean_Absolute_Error" in table
    assert "per_tensor" in table
    assert "per_channel" in table
    assert "per_group" in table


def test_granularity_improves_accuracy():
    """
    Verify that finer granularity typically improves accuracy
    (smoke test for Property 6 intuition).
    """
    np.random.seed(42)

    # Create a weight matrix with varied per-channel distributions
    W = np.zeros((8, 64), dtype=np.float32)
    for i in range(8):
        W[i, :] = np.random.randn(64).astype(np.float32) * (0.01 * (i + 1))

    reports = compare_granularity(W, 8, QuantizationScheme.SYMMETRIC, group_size=16)

    mae_t = reports[0].mean_absolute_error
    mae_c = reports[1].mean_absolute_error
    mae_g = reports[2].mean_absolute_error

    # Per-channel should be better than per-tensor
    assert mae_c < mae_t, f"Per-channel should be better: {mae_c} vs {mae_t}"

    # Per-group should be better than or equal to per-channel
    assert mae_g <= mae_c, f"Per-group should be better: {mae_g} vs {mae_c}"
