# %% [markdown]
# # Core Quantization Functions
#
# This module implements the fundamental quantization and dequantization operations
# that map floating-point tensors to integer representations and back.
#
# **Learning Objective:** Understand the role of scale, zero-point, and clamping
# in the float-to-integer mapping.
#
# ## Quantization Equation
#
# **Forward (quantize):**
# ```
# q = clamp(round(x / scale) + zero_point, q_min, q_max)
# ```
#
# **Reverse (dequantize):**
# ```
# x_reconstructed = (q - zero_point) * scale
# ```
#
# ## Parameters
#
# - **scale (s):** Maps integer steps to float range (step size)
# - **zero_point (z):** Integer offset to align ranges
# - **bits:** Bit-width (2, 4, or 8)
# - **signed:** Whether to use signed or unsigned integers

# %%
import numpy as np
from typing import Tuple

# %% [markdown]
# ## Integer Range Lookup
#
# For a given bit-width and signedness, compute the valid integer range [q_min, q_max].

# %%
def q_min_max(bits: int, signed: bool = True) -> Tuple[int, int]:
    """
    Compute the valid integer range for quantized values.

    Args:
        bits: Bit-width (one of {2, 4, 8})
        signed: Whether to use signed integers

    Returns:
        Tuple of (q_min, q_max)

    Raises:
        ValueError: If bits is not in {2, 4, 8}

    Examples:
        >>> q_min_max(8, signed=True)
        (-128, 127)
        >>> q_min_max(8, signed=False)
        (0, 255)
        >>> q_min_max(4, signed=True)
        (-8, 7)
    """
    if bits not in [2, 4, 8]:
        raise ValueError(f"bits must be one of {{2, 4, 8}}, got {bits}")

    if signed:
        # Signed range: [-(2^(bits-1)), 2^(bits-1) - 1]
        q_min = -(2 ** (bits - 1))
        q_max = (2 ** (bits - 1)) - 1
    else:
        # Unsigned range: [0, 2^bits - 1]
        q_min = 0
        q_max = (2 ** bits) - 1

    return q_min, q_max


# %% [markdown]
# ## Quantize: Float → Integer
#
# Map a floating-point tensor to an integer tensor using the quantization equation.

# %%
def quantize(
    x: np.ndarray,
    scale: float,
    zero_point: int,
    bits: int,
    signed: bool = True,
) -> np.ndarray:
    """
    Quantize a floating-point tensor to integer representation.

    The quantization formula:
        q = clamp(round(x / scale) + zero_point, q_min, q_max)

    Args:
        x: Input tensor (float32 or float64)
        scale: Positive float multiplier that maps integer range to float range
        zero_point: Integer offset within [q_min, q_max]
        bits: Bit-width (one of {2, 4, 8})
        signed: Whether to use signed integers

    Returns:
        Quantized tensor (int32)

    Raises:
        ValueError: If scale <= 0, bits invalid, zero_point out of range, or x contains NaN/Inf

    Examples:
        >>> x = np.array([1.5, -0.5, 0.0, 2.3], dtype=np.float32)
        >>> q = quantize(x, scale=0.02, zero_point=0, bits=8, signed=True)
        >>> q.dtype
        dtype('int32')
    """
    # Validate scale
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    # Validate bits
    if bits not in [2, 4, 8]:
        raise ValueError(f"bits must be one of {{2, 4, 8}}, got {bits}")

    # Get integer range
    q_min, q_max = q_min_max(bits, signed)

    # Validate zero_point
    if not (q_min <= zero_point <= q_max):
        raise ValueError(
            f"zero_point must be in range [{q_min}, {q_max}], got {zero_point}"
        )

    # Check for NaN or Inf
    if np.any(np.isnan(x)):
        raise ValueError("Input tensor contains NaN values")

    if np.any(np.isinf(x)):
        raise ValueError("Input tensor contains infinite values")

    # Quantization formula: q = clamp(round(x / scale) + zero_point, q_min, q_max)
    x_scaled = x / scale
    x_rounded = np.round(x_scaled)  # Banker's rounding (round-half-to-even)
    x_shifted = x_rounded + zero_point
    q = np.clip(x_shifted, q_min, q_max).astype(np.int32)

    return q


# %% [markdown]
# ## Dequantize: Integer → Float
#
# Reconstruct a floating-point tensor from its quantized integer representation.

# %%
def dequantize(
    q: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """
    Dequantize an integer tensor back to floating-point.

    The dequantization formula:
        x_reconstructed = (q - zero_point) * scale

    Args:
        q: Quantized tensor (integer type)
        scale: Scale factor used during quantization
        zero_point: Zero-point used during quantization

    Returns:
        Reconstructed tensor (float32)

    Examples:
        >>> q = np.array([75, -25, 0, 115], dtype=np.int32)
        >>> x = dequantize(q, scale=0.02, zero_point=0)
        >>> x.shape == q.shape
        True
    """
    # Dequantization formula: x = (q - zero_point) * scale
    q_shifted = q.astype(np.float32) - zero_point
    x_reconstructed = q_shifted * scale

    return x_reconstructed.astype(np.float32)


# %% [markdown]
# ## Demo: Quantize and Dequantize Round-Trip

# %%
if __name__ == "__main__":
    print("=" * 80)
    print("Demo: Quantization Round-Trip")
    print("=" * 80)

    # Example tensor
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.5], dtype=np.float32)

    scale = 0.02
    zero_point = 0
    bits = 8
    signed = True

    print(f"\nOriginal tensor:")
    print(f"  {x}")

    print(f"\nQuantization parameters:")
    print(f"  scale = {scale}")
    print(f"  zero_point = {zero_point}")
    print(f"  bits = {bits}")
    print(f"  signed = {signed}")

    q_min, q_max = q_min_max(bits, signed)
    print(f"  Integer range: [{q_min}, {q_max}]")

    # Quantize
    q = quantize(x, scale, zero_point, bits, signed)
    print(f"\nQuantized (int8):")
    print(f"  {q}")

    # Dequantize
    x_reconstructed = dequantize(q, scale, zero_point)
    print(f"\nReconstructed (float32):")
    print(f"  {x_reconstructed}")

    # Compute error
    error = x - x_reconstructed
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))

    print(f"\nQuantization error:")
    print(f"  Per-element: {error}")
    print(f"  Mean absolute error: {mae:.6f}")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Theoretical max (0.5 * scale): {0.5 * scale:.6f}")

    print("\n" + "=" * 80)
    print("Demo: Clipping Behavior")
    print("=" * 80)

    # Values that will be clipped
    x_extreme = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    print(f"\nExtreme values:")
    print(f"  {x_extreme}")

    q_extreme = quantize(x_extreme, scale, zero_point, bits, signed)
    print(f"\nQuantized (clamped to [{q_min}, {q_max}]):")
    print(f"  {q_extreme}")

    x_recon_extreme = dequantize(q_extreme, scale, zero_point)
    print(f"\nReconstructed:")
    print(f"  {x_recon_extreme}")

    # Show which values were clipped
    was_clipped = (q_extreme == q_min) | (q_extreme == q_max)
    print(f"\nClipped values:")
    for i, (orig, recon, clipped) in enumerate(zip(x_extreme, x_recon_extreme, was_clipped)):
        marker = "⚠️ CLIPPED" if clipped else "✓ In range"
        print(f"  {orig:6.2f} → {recon:6.2f}  {marker}")

    print("\n" + "=" * 80)
    print("Demo: Different Bit-Widths")
    print("=" * 80)

    x_test = np.linspace(-1.0, 1.0, 10, dtype=np.float32)

    for test_bits in [2, 4, 8]:
        q_min_test, q_max_test = q_min_max(test_bits, signed=True)
        scale_test = 1.0 / ((2 ** (test_bits - 1)) - 1)

        q_test = quantize(x_test, scale_test, 0, test_bits, signed=True)
        x_recon_test = dequantize(q_test, scale_test, 0)
        mae_test = np.mean(np.abs(x_test - x_recon_test))

        print(f"\n{test_bits}-bit quantization:")
        print(f"  Levels: {2 ** test_bits}")
        print(f"  Range: [{q_min_test}, {q_max_test}]")
        print(f"  Scale: {scale_test:.6f}")
        print(f"  MAE: {mae_test:.6f}")
