# %% [markdown]
# # Property-Based Tests for FP32 Format Utilities
#
# **Test Coverage:**
# - Property 10: IEEE 754 decompose-reconstruct round-trip

# %%
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st, assume

import sys
sys.path.insert(0, '..')

from quantization.fp32_format import decompose_fp32, reconstruct_fp32


# %% [markdown]
# ## Hypothesis Strategies

# %%
# Strategy: finite float32 values (excluding NaN, Inf, and very small subnormals)
finite_float32 = st.floats(
    min_value=-1e10,
    max_value=1e10,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,  # Avoid precision issues with tiny numbers
    width=32,
)


# %% [markdown]
# ## Property 10: IEEE 754 Decompose-Reconstruct Round-Trip
#
# **Property:** For any finite float32 value, decomposing it into sign/exponent/mantissa
# and then reconstructing SHALL produce a value equal to the original within float32 precision.
#
# **Validates:** Requirements 10.4, 10.5

# %%
@given(value=finite_float32)
@settings(max_examples=100)
def test_property_10_fp32_round_trip(value):
    """
    Feature: ai-model-quantization
    Property 10: IEEE 754 Decompose-Reconstruct Round-Trip

    For any finite float32, decompose→reconstruct preserves value within float32 epsilon.
    """
    # Decompose
    result = decompose_fp32(value)

    # Reconstruct
    reconstructed = reconstruct_fp32(
        result['sign'],
        result['exponent_bits'],
        result['mantissa_bits']
    )

    # Compute epsilon for this value
    eps = np.finfo(np.float32).eps * abs(value) if value != 0 else np.finfo(np.float32).eps

    # Verify round-trip
    error = abs(reconstructed - value)

    # Also check the 'reconstructed' field in the decompose result
    decompose_reconstructed = result['reconstructed']
    decompose_error = abs(decompose_reconstructed - value)

    assert error <= eps or np.isclose(reconstructed, value, rtol=1e-6), (
        f"Round-trip error too large for value={value}: "
        f"reconstructed={reconstructed}, error={error}, eps={eps}"
    )

    assert decompose_error <= eps or np.isclose(decompose_reconstructed, value, rtol=1e-6), (
        f"Decompose internal reconstruction error too large for value={value}: "
        f"reconstructed={decompose_reconstructed}, error={decompose_error}, eps={eps}"
    )


# %% [markdown]
# ## Edge Case Tests

# %%
def test_decompose_nan():
    """decompose_fp32 should raise ValueError for NaN."""
    with pytest.raises(ValueError, match="NaN"):
        decompose_fp32(np.nan)


def test_decompose_infinity():
    """decompose_fp32 should raise ValueError for infinity."""
    with pytest.raises(ValueError, match="infinite"):
        decompose_fp32(np.inf)

    with pytest.raises(ValueError, match="infinite"):
        decompose_fp32(-np.inf)


def test_reconstruct_invalid_sign():
    """reconstruct_fp32 should raise ValueError for invalid sign."""
    with pytest.raises(ValueError, match="sign must be 0 or 1"):
        reconstruct_fp32(2, '01111111', '00000000000000000000000')


def test_reconstruct_invalid_exponent():
    """reconstruct_fp32 should raise ValueError for invalid exponent bits."""
    with pytest.raises(ValueError, match="exponent_bits"):
        reconstruct_fp32(0, '0111111', '00000000000000000000000')  # 7 bits

    with pytest.raises(ValueError, match="exponent_bits"):
        reconstruct_fp32(0, '011111112', '00000000000000000000000')  # invalid char


def test_reconstruct_invalid_mantissa():
    """reconstruct_fp32 should raise ValueError for invalid mantissa bits."""
    with pytest.raises(ValueError, match="mantissa_bits"):
        reconstruct_fp32(0, '01111111', '0000000000000000000000')  # 22 bits

    with pytest.raises(ValueError, match="mantissa_bits"):
        reconstruct_fp32(0, '01111111', '00000000000000000000002')  # invalid char


# %% [markdown]
# ## Smoke Tests

# %%
def test_decompose_known_values():
    """Verify decomposition of known values."""
    # 1.0 in FP32
    result = decompose_fp32(1.0)
    assert result['sign'] == 0
    assert result['biased_exponent'] == 127
    assert result['true_exponent'] == 0
    assert result['mantissa_bits'] == '0' * 23

    # -1.0 in FP32
    result = decompose_fp32(-1.0)
    assert result['sign'] == 1
    assert result['biased_exponent'] == 127
    assert result['true_exponent'] == 0

    # 0.0 in FP32 (exponent and mantissa both zero)
    result = decompose_fp32(0.0)
    assert result['sign'] == 0
    assert result['biased_exponent'] == 0
    assert result['mantissa_bits'] == '0' * 23


def test_reconstruct_known_values():
    """Verify reconstruction of known values."""
    # 1.0
    value = reconstruct_fp32(0, '01111111', '00000000000000000000000')
    assert np.isclose(value, 1.0)

    # -1.0
    value = reconstruct_fp32(1, '01111111', '00000000000000000000000')
    assert np.isclose(value, -1.0)

    # 2.0 (exponent = 128)
    value = reconstruct_fp32(0, '10000000', '00000000000000000000000')
    assert np.isclose(value, 2.0)

    # 0.5 (exponent = 126)
    value = reconstruct_fp32(0, '01111110', '00000000000000000000000')
    assert np.isclose(value, 0.5)


def test_bit_length():
    """Verify bit string lengths are correct."""
    result = decompose_fp32(3.14159)
    assert len(result['exponent_bits']) == 8
    assert len(result['mantissa_bits']) == 23
    assert all(c in '01' for c in result['exponent_bits'])
    assert all(c in '01' for c in result['mantissa_bits'])
