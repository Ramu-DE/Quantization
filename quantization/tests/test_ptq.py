# %% [markdown]
# # Unit Tests for Post-Training Quantization (PTQ)

# %%
import pytest
import numpy as np

import sys
sys.path.insert(0, '..')

from quantization.ptq import ptq_quantize_layer, ptq_infer, PTQLayer
from quantization.granularity import GranularityMode
from quantization.schemes import QuantizationScheme


# %% [markdown]
# ## Test PTQ Inference Output Shape

# %%
def test_ptq_infer_output_shape():
    """Verify ptq_infer produces correct output shape [out_features, batch_size]."""
    out_features = 16
    in_features = 32
    batch_size = 8

    W = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    b = np.zeros(out_features, dtype=np.float32)
    calibration = [np.random.randn(4, in_features).astype(np.float32) for _ in range(5)]

    # Test all granularities
    for granularity in [GranularityMode.PER_TENSOR, GranularityMode.PER_CHANNEL, GranularityMode.PER_GROUP]:
        layer = ptq_quantize_layer(
            W, b, calibration, 8, QuantizationScheme.SYMMETRIC, granularity, group_size=8
        )

        x_test = np.random.randn(batch_size, in_features).astype(np.float32)
        output = ptq_infer(layer, x_test)

        assert output.shape == (out_features, batch_size), (
            f"Output shape mismatch for {granularity}: "
            f"expected ({out_features}, {batch_size}), got {output.shape}"
        )


# %% [markdown]
# ## Test Empty Calibration Error

# %%
def test_ptq_empty_calibration():
    """ptq_quantize_layer should raise ValueError for empty calibration list."""
    W = np.random.randn(8, 16).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="At least 1"):
        ptq_quantize_layer(
            W, b, [], 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_TENSOR
        )


# %% [markdown]
# ## Test Granularity Error Ordering

# %%
def test_ptq_granularity_error_ordering():
    """
    Verify that with 10+ calibration samples and non-uniform weights,
    error_per_tensor >= error_per_channel >= error_per_group.
    """
    np.random.seed(42)

    # Create non-uniform weight matrix
    out_features = 8
    in_features = 64
    W = np.zeros((out_features, in_features), dtype=np.float32)
    for i in range(out_features):
        W[i, :] = np.random.randn(in_features).astype(np.float32) * (0.01 * (i + 1))

    b = np.zeros(out_features, dtype=np.float32)

    # Generate 10+ calibration samples
    calibration = [
        np.random.randn(4, in_features).astype(np.float32)
        for _ in range(12)
    ]

    bits = 8
    scheme = QuantizationScheme.SYMMETRIC

    # Test input
    x_test = np.random.randn(8, in_features).astype(np.float32)
    output_fp32 = W @ x_test.T + b[:, np.newaxis]

    # PTQ with each granularity
    errors = {}
    for granularity in [GranularityMode.PER_TENSOR, GranularityMode.PER_CHANNEL, GranularityMode.PER_GROUP]:
        layer = ptq_quantize_layer(W, b, calibration, bits, scheme, granularity, group_size=16)
        output_ptq = ptq_infer(layer, x_test)
        mae = np.mean(np.abs(output_fp32 - output_ptq))
        errors[granularity] = mae

    # Verify ordering (with tolerance)
    tol = 1e-6
    assert errors[GranularityMode.PER_TENSOR] >= errors[GranularityMode.PER_CHANNEL] - tol, (
        f"Per-tensor error should be >= per-channel: "
        f"{errors[GranularityMode.PER_TENSOR]} vs {errors[GranularityMode.PER_CHANNEL]}"
    )

    assert errors[GranularityMode.PER_CHANNEL] >= errors[GranularityMode.PER_GROUP] - tol, (
        f"Per-channel error should be >= per-group: "
        f"{errors[GranularityMode.PER_CHANNEL]} vs {errors[GranularityMode.PER_GROUP]}"
    )


# %% [markdown]
# ## Smoke Tests

# %%
def test_ptq_layer_structure():
    """Verify PTQLayer dataclass has correct structure."""
    W = np.random.randn(8, 16).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)
    calibration = [np.random.randn(4, 16).astype(np.float32) for _ in range(5)]

    layer = ptq_quantize_layer(
        W, b, calibration, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_TENSOR
    )

    assert hasattr(layer, 'W_q')
    assert hasattr(layer, 'scales')
    assert hasattr(layer, 'zero_points')
    assert hasattr(layer, 'b')
    assert hasattr(layer, 'bits')
    assert hasattr(layer, 'scheme')
    assert hasattr(layer, 'granularity')

    assert layer.W_q.shape == W.shape
    assert layer.bits == 8
    assert layer.scheme == QuantizationScheme.SYMMETRIC
    assert layer.granularity == GranularityMode.PER_TENSOR


def test_ptq_per_tensor_scale_count():
    """Verify per-tensor PTQ uses single scale/zp."""
    W = np.random.randn(8, 16).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)
    calibration = [np.random.randn(4, 16).astype(np.float32) for _ in range(5)]

    layer = ptq_quantize_layer(
        W, b, calibration, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_TENSOR
    )

    assert layer.scales.size == 1
    assert layer.zero_points.size == 1


def test_ptq_per_channel_scale_count():
    """Verify per-channel PTQ uses one scale/zp per output channel."""
    out_features = 8
    in_features = 16
    W = np.random.randn(out_features, in_features).astype(np.float32)
    b = np.zeros(out_features, dtype=np.float32)
    calibration = [np.random.randn(4, in_features).astype(np.float32) for _ in range(5)]

    layer = ptq_quantize_layer(
        W, b, calibration, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_CHANNEL
    )

    assert layer.scales.size == out_features
    assert layer.zero_points.size == out_features


def test_ptq_per_group_scale_count():
    """Verify per-group PTQ uses correct number of scales/zps."""
    out_features = 8
    in_features = 64
    group_size = 16
    W = np.random.randn(out_features, in_features).astype(np.float32)
    b = np.zeros(out_features, dtype=np.float32)
    calibration = [np.random.randn(4, in_features).astype(np.float32) for _ in range(5)]

    layer = ptq_quantize_layer(
        W, b, calibration, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_GROUP, group_size=group_size
    )

    num_groups = in_features // group_size
    assert layer.scales.shape == (out_features, num_groups)
    assert layer.zero_points.shape == (out_features, num_groups)


def test_ptq_inference_matches_fp32_approximately():
    """Verify PTQ inference is close to FP32 baseline."""
    np.random.seed(42)

    W = np.random.randn(8, 16).astype(np.float32) * 0.1
    b = np.random.randn(8).astype(np.float32) * 0.01
    calibration = [np.random.randn(4, 16).astype(np.float32) for _ in range(10)]

    layer = ptq_quantize_layer(
        W, b, calibration, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_CHANNEL
    )

    x_test = np.random.randn(4, 16).astype(np.float32)

    # FP32 baseline
    output_fp32 = W @ x_test.T + b[:, np.newaxis]

    # PTQ inference
    output_ptq = ptq_infer(layer, x_test)

    # Should be reasonably close (within 1% relative error on average)
    relative_error = np.abs(output_fp32 - output_ptq) / (np.abs(output_fp32) + 1e-8)
    mean_relative_error = np.mean(relative_error)

    assert mean_relative_error < 0.05, (  # Allow 5% relative error
        f"PTQ inference too far from FP32: mean relative error = {mean_relative_error:.4f}"
    )


def test_ptq_bias_unquantized():
    """Verify that PTQ keeps bias in FP32 (unquantized)."""
    W = np.random.randn(8, 16).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    calibration = [np.random.randn(4, 16).astype(np.float32) for _ in range(5)]

    layer = ptq_quantize_layer(
        W, b, calibration, 8, QuantizationScheme.SYMMETRIC, GranularityMode.PER_TENSOR
    )

    # Bias should be unchanged (FP32)
    assert np.array_equal(layer.b, b)
    assert layer.b.dtype == np.float32
