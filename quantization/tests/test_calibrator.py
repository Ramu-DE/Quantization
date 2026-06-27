# %% [markdown]
# # Property-Based Tests for Calibration Methods
#
# **Test Coverage:**
# - Property 5: Calibration scale positivity

# %%
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import sys
sys.path.insert(0, '..')

from quantization.calibrator import (
    CalibrationMethod,
    calibrate_min_max,
    calibrate_percentile,
    calibrate_entropy,
)
from quantization.schemes import QuantizationScheme


# %% [markdown]
# ## Hypothesis Strategies

# %%
# Strategy: valid bit-widths
valid_bits_cal = st.sampled_from([2, 4, 8])

# Strategy: quantization schemes
quantization_scheme = st.sampled_from([QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC])

# Strategy: list of non-empty calibration tensors
@st.composite
def calibration_tensor_list(draw, min_tensors=1, max_tensors=10):
    """Generate a list of calibration tensors."""
    num_tensors = draw(st.integers(min_value=min_tensors, max_value=max_tensors))
    tensors = []

    for _ in range(num_tensors):
        size = draw(st.integers(min_value=10, max_value=200))
        tensor = draw(arrays(
            dtype=np.float32,
            shape=(size,),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            )
        ))
        tensors.append(tensor)

    return tensors


# %% [markdown]
# ## Property 5: Calibration Scale Positivity
#
# **Property:** For any non-empty list of calibration tensors, any bits ∈ {2, 4, 8},
# and any valid scheme, each of the three calibrators SHALL return a scale value
# that is strictly greater than zero.
#
# **Validates:** Requirements 4.4, 4.7, 9.3

# %%
@given(
    tensors=calibration_tensor_list(min_tensors=1, max_tensors=5),
    bits=valid_bits_cal,
    scheme=quantization_scheme,
)
@settings(max_examples=100)
def test_property_5_calibration_scale_positivity(tensors, bits, scheme):
    """
    Feature: ai-model-quantization
    Property 5: Calibration Scale Positivity

    For any non-empty calibration list and all three methods, scale > 0.
    """
    # MinMax
    try:
        scale_mm, zp_mm = calibrate_min_max(tensors, bits, scheme)
        assert scale_mm > 0, f"MinMax scale must be positive, got {scale_mm}"
    except ValueError as e:
        # If it's an all-zeros tensor, that's expected
        if "all-zeros" not in str(e).lower() and "constant" not in str(e).lower():
            raise

    # Percentile (99%)
    try:
        scale_p, zp_p = calibrate_percentile(tensors, bits, scheme, percentile=99.0)
        assert scale_p > 0, f"Percentile scale must be positive, got {scale_p}"
    except ValueError as e:
        if "all-zeros" not in str(e).lower() and "constant" not in str(e).lower():
            raise

    # Entropy
    try:
        scale_e, zp_e = calibrate_entropy(tensors, bits, scheme)
        assert scale_e > 0, f"Entropy scale must be positive, got {scale_e}"
    except ValueError as e:
        if "all-zeros" not in str(e).lower() and "constant" not in str(e).lower():
            raise


# %% [markdown]
# ## Edge Case Tests

# %%
def test_calibrate_empty_list():
    """All calibrators should raise ValueError for empty tensor list."""
    with pytest.raises(ValueError, match="At least one"):
        calibrate_min_max([], 8, QuantizationScheme.SYMMETRIC)

    with pytest.raises(ValueError, match="At least one"):
        calibrate_percentile([], 8, QuantizationScheme.SYMMETRIC)

    with pytest.raises(ValueError, match="At least one"):
        calibrate_entropy([], 8, QuantizationScheme.SYMMETRIC)


def test_calibrate_percentile_invalid_percentile():
    """calibrate_percentile should raise ValueError for invalid percentile."""
    tensors = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

    with pytest.raises(ValueError, match="percentile must be in range"):
        calibrate_percentile(tensors, 8, QuantizationScheme.SYMMETRIC, percentile=0.0)

    with pytest.raises(ValueError, match="percentile must be in range"):
        calibrate_percentile(tensors, 8, QuantizationScheme.SYMMETRIC, percentile=100.0)

    with pytest.raises(ValueError, match="percentile must be in range"):
        calibrate_percentile(tensors, 8, QuantizationScheme.SYMMETRIC, percentile=-10.0)


# %% [markdown]
# ## Smoke Tests

# %%
def test_calibrate_min_max_known_values():
    """Verify MinMax calibration with known values."""
    tensors = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
    ]

    scale, zp = calibrate_min_max(tensors, 8, QuantizationScheme.SYMMETRIC)

    # Symmetric: scale = max(|1|, |6|) / 127 = 6 / 127
    expected_scale = 6.0 / 127
    assert np.isclose(scale, expected_scale, rtol=1e-5)
    assert zp == 0


def test_calibrate_percentile_outlier_handling():
    """Verify Percentile calibration ignores outliers."""
    # 98 normal values + 2 outliers
    normal = np.random.randn(98).astype(np.float32) * 0.5
    outliers = np.array([100.0, -100.0], dtype=np.float32)
    tensor = np.concatenate([normal, outliers])

    tensors = [tensor]

    # MinMax includes outliers
    scale_mm, _ = calibrate_min_max(tensors, 8, QuantizationScheme.SYMMETRIC)

    # Percentile excludes outliers
    scale_p, _ = calibrate_percentile(tensors, 8, QuantizationScheme.SYMMETRIC, percentile=99.0)

    # Percentile scale should be significantly smaller (more precision for main distribution)
    assert scale_p < scale_mm, f"Percentile should ignore outliers: scale_p={scale_p}, scale_mm={scale_mm}"


def test_calibrate_entropy_basic():
    """Verify Entropy calibration produces valid results."""
    np.random.seed(42)
    tensors = [np.random.randn(100).astype(np.float32) for _ in range(3)]

    scale, zp = calibrate_entropy(tensors, 8, QuantizationScheme.SYMMETRIC)

    assert scale > 0, f"Scale must be positive, got {scale}"
    assert zp == 0, f"Symmetric scheme must have zp=0, got {zp}"


def test_all_methods_return_valid_params():
    """Verify all three methods return valid (scale, zp) pairs."""
    np.random.seed(42)
    tensors = [np.random.randn(50).astype(np.float32) for _ in range(2)]
    bits = 8
    scheme = QuantizationScheme.ASYMMETRIC

    # MinMax
    scale_mm, zp_mm = calibrate_min_max(tensors, bits, scheme)
    assert scale_mm > 0
    assert 0 <= zp_mm <= 255  # Asymmetric unsigned range

    # Percentile
    scale_p, zp_p = calibrate_percentile(tensors, bits, scheme, percentile=95.0)
    assert scale_p > 0
    assert 0 <= zp_p <= 255

    # Entropy
    scale_e, zp_e = calibrate_entropy(tensors, bits, scheme)
    assert scale_e > 0
    assert 0 <= zp_e <= 255
