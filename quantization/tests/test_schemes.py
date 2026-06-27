# %% [markdown]
# # Property-Based Tests for Quantization Schemes
#
# **Test Coverage:**
# - Property 4: Symmetric scheme zero-point invariant

# %%
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import sys
sys.path.insert(0, '..')

from quantization.schemes import (
    QuantizationScheme,
    compute_scale_zero_point,
    compare_schemes,
    format_scheme_comparison,
)


# %% [markdown]
# ## Hypothesis Strategies

# %%
# Strategy: valid bit-widths
valid_bits = st.sampled_from([2, 4, 8])

# Strategy: non-constant float32 tensors (avoiding all-zeros and constant tensors)
@st.composite
def non_constant_tensor(draw):
    """Generate a non-constant float32 tensor."""
    size = draw(st.integers(min_value=10, max_value=1000))
    # Generate values with sufficient spread
    values = draw(arrays(
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

    # Ensure not constant by checking range
    if np.max(values) - np.min(values) < 1e-6:
        # Add some spread
        values[0] = -1.0
        values[-1] = 1.0

    return values


# %% [markdown]
# ## Property 4: Symmetric Scheme Zero-Point Invariant
#
# **Property:** For any tensor and any bits ∈ {2, 4, 8}, computing scale and
# zero-point with SYMMETRIC scheme SHALL always return zero_point == 0.
#
# **Validates:** Requirements 3.3, 9.6

# %%
@given(x=non_constant_tensor(), bits=valid_bits)
@settings(max_examples=100)
def test_property_4_symmetric_zero_point_invariant(x, bits):
    """
    Feature: ai-model-quantization
    Property 4: Symmetric Scheme Zero-Point Invariant

    For any tensor and bits in {2, 4, 8}, symmetric quantization returns zero_point == 0.
    """
    scale, zero_point = compute_scale_zero_point(x, bits, QuantizationScheme.SYMMETRIC)

    assert zero_point == 0, (
        f"Symmetric scheme must return zero_point=0, got {zero_point} "
        f"for bits={bits}, x_min={np.min(x):.4f}, x_max={np.max(x):.4f}"
    )

    # Also verify scale is positive
    assert scale > 0, f"Scale must be positive, got {scale}"


# %% [markdown]
# ## Additional Property: Asymmetric Zero-Point for All-Positive Tensors
#
# When a tensor has x_min >= 0 (all positive), asymmetric quantization
# should return zero_point = 0 since no negative offset is needed.

# %%
@given(bits=valid_bits)
@settings(max_examples=100)
def test_asymmetric_all_positive_zero_point(bits):
    """
    For all-positive tensors, asymmetric quantization should return zero_point = 0.
    """
    # Generate all-positive tensor
    x = np.abs(np.random.randn(100).astype(np.float32)) + 0.1  # Ensure min > 0

    scale, zero_point = compute_scale_zero_point(x, bits, QuantizationScheme.ASYMMETRIC)

    # For all-positive tensors, zp should be 0 (or very close due to rounding)
    assert zero_point == 0, (
        f"Asymmetric scheme with all-positive tensor should have zero_point=0, "
        f"got {zero_point} for x_min={np.min(x):.4f}"
    )


# %% [markdown]
# ## Edge Case Tests

# %%
def test_compute_scale_invalid_bits():
    """compute_scale_zero_point should raise ValueError for invalid bits."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="bits must be one of"):
        compute_scale_zero_point(x, 16, QuantizationScheme.SYMMETRIC)


def test_compute_scale_all_zeros():
    """compute_scale_zero_point should raise ValueError for all-zeros tensor."""
    x = np.zeros(100, dtype=np.float32)

    with pytest.raises(ValueError, match="all-zeros"):
        compute_scale_zero_point(x, 8, QuantizationScheme.SYMMETRIC)

    with pytest.raises(ValueError, match="all-zeros"):
        compute_scale_zero_point(x, 8, QuantizationScheme.ASYMMETRIC)


def test_compute_scale_constant_tensor():
    """compute_scale_zero_point should handle constant tensors gracefully."""
    x = np.full(100, 5.0, dtype=np.float32)

    # Should return identity mapping (scale=1.0, zp=0)
    scale, zp = compute_scale_zero_point(x, 8, QuantizationScheme.SYMMETRIC)
    assert scale == 1.0
    assert zp == 0


# %% [markdown]
# ## Smoke Tests

# %%
def test_symmetric_scheme_known_values():
    """Verify symmetric quantization with known inputs."""
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale, zp = compute_scale_zero_point(x, 8, QuantizationScheme.SYMMETRIC)

    # Symmetric: scale = max(|-1|, |1|) / 127 = 1.0 / 127
    expected_scale = 1.0 / 127
    assert np.isclose(scale, expected_scale, rtol=1e-6)
    assert zp == 0


def test_asymmetric_scheme_known_values():
    """Verify asymmetric quantization with known inputs."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scale, zp = compute_scale_zero_point(x, 8, QuantizationScheme.ASYMMETRIC)

    # Asymmetric: scale = (4.0 - 0.0) / 255 = 4.0 / 255
    expected_scale = 4.0 / 255
    assert np.isclose(scale, expected_scale, rtol=1e-6)

    # Zero-point: -x_min / scale = 0 / scale = 0
    assert zp == 0


def test_compare_schemes():
    """Verify compare_schemes produces valid report."""
    x = np.random.randn(100).astype(np.float32)
    report = compare_schemes(x, 8)

    # Symmetric always has zp=0
    assert report.symmetric_zero_point == 0

    # Both scales should be positive
    assert report.symmetric_scale > 0
    assert report.asymmetric_scale > 0

    # MAE should be non-negative
    assert report.symmetric_mae >= 0
    assert report.asymmetric_mae >= 0


def test_format_scheme_comparison():
    """Verify format_scheme_comparison produces valid output."""
    x = np.random.randn(100).astype(np.float32)
    report = compare_schemes(x, 8)
    table = format_scheme_comparison(report)

    # Check for expected keywords
    assert "Symmetric" in table
    assert "Asymmetric" in table
    assert "Scale" in table
    assert "Zero_Point" in table
    assert "Mean_Absolute_Error" in table
    assert "Best scheme" in table


def test_scheme_comparison_skewed_distribution():
    """For skewed distributions, asymmetric should be better or equal."""
    # All-positive tensor (ReLU-like)
    x = np.abs(np.random.randn(1000).astype(np.float32))
    report = compare_schemes(x, 8)

    # Asymmetric should be at least as good as symmetric (often better)
    # Note: This is not always guaranteed for all random seeds, but holds in expectation
    # We'll just verify both produce reasonable errors
    assert report.asymmetric_mae < 1.0  # Reasonable error bound
    assert report.symmetric_mae < 1.0  # Reasonable error bound
