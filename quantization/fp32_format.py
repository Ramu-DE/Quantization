# %% [markdown]
# # IEEE 754 FP32 Format Utilities
#
# This module provides utilities for decomposing and reconstructing IEEE 754
# single-precision (FP32) floating-point numbers at the bit level.
#
# **Learning Objective:** Understand what information is represented in each
# bit of a float32 value, and what is lost during quantization to integer.
#
# ## IEEE 754 FP32 Format
#
# A float32 value consists of 32 bits:
# ```
# | Sign (1 bit) | Exponent (8 bits) | Mantissa (23 bits) |
# ```
#
# - **Sign**: 0 = positive, 1 = negative
# - **Exponent**: Biased by 127 (actual exponent = stored - 127)
# - **Mantissa**: Fractional part (with implicit leading 1)
#
# **Reconstruction formula:**
# ```
# value = (-1)^sign × 2^(exponent - 127) × (1 + mantissa / 2^23)
# ```

# %%
import struct
import numpy as np
from typing import Dict

# %% [markdown]
# ## FP32 Decomposition
#
# Extract the sign, exponent, and mantissa bits from a float32 value.

# %%
def decompose_fp32(value: float) -> Dict[str, any]:
    """
    Decompose a float32 value into its IEEE 754 components.

    Args:
        value: Float32 value to decompose

    Returns:
        Dictionary with keys:
        - 'sign': 0 or 1
        - 'exponent_bits': 8-character binary string
        - 'mantissa_bits': 23-character binary string
        - 'biased_exponent': Integer value of exponent (0-255)
        - 'true_exponent': Actual exponent (biased_exponent - 127)
        - 'reconstructed': Value reconstructed from components (for verification)

    Raises:
        ValueError: If value is NaN or infinite

    Examples:
        >>> result = decompose_fp32(-0.10985)
        >>> result['sign']
        1
        >>> len(result['exponent_bits'])
        8
        >>> len(result['mantissa_bits'])
        23
    """
    # Convert float32 to 32-bit integer representation
    try:
        # Pack as float, unpack as unsigned int (big-endian)
        bits_int = struct.unpack('>I', struct.pack('>f', np.float32(value)))[0]
    except (struct.error, OverflowError) as e:
        raise ValueError(f"Cannot decompose value {value}: {e}")

    # Check for NaN or infinity
    if np.isnan(value):
        raise ValueError("Cannot decompose NaN values")
    if np.isinf(value):
        raise ValueError("Cannot decompose infinite values")

    # Extract bit fields using bitwise operations
    sign = (bits_int >> 31) & 1  # Most significant bit
    biased_exponent = (bits_int >> 23) & 0xFF  # Next 8 bits
    mantissa = bits_int & 0x7FFFFF  # Least significant 23 bits

    # Convert to binary strings (with leading zeros)
    exponent_bits = format(biased_exponent, '08b')
    mantissa_bits = format(mantissa, '023b')

    # Compute true exponent
    true_exponent = biased_exponent - 127

    # Reconstruct value for verification
    if biased_exponent == 0:
        # Subnormal number (exponent = 0)
        reconstructed = (-1) ** sign * 2 ** (-126) * (mantissa / (2 ** 23))
    else:
        # Normal number
        reconstructed = (-1) ** sign * 2 ** true_exponent * (1 + mantissa / (2 ** 23))

    return {
        'sign': sign,
        'exponent_bits': exponent_bits,
        'mantissa_bits': mantissa_bits,
        'biased_exponent': biased_exponent,
        'true_exponent': true_exponent,
        'reconstructed': reconstructed,
    }


# %% [markdown]
# ## FP32 Reconstruction
#
# Reconstruct a float32 value from its sign, exponent, and mantissa bits.

# %%
def reconstruct_fp32(sign: int, exponent_bits: str, mantissa_bits: str) -> float:
    """
    Reconstruct a float32 value from IEEE 754 components.

    Args:
        sign: 0 or 1
        exponent_bits: 8-character binary string
        mantissa_bits: 23-character binary string

    Returns:
        Reconstructed float32 value

    Raises:
        ValueError: If inputs are invalid

    Examples:
        >>> # Reconstruct 1.0
        >>> reconstruct_fp32(0, '01111111', '00000000000000000000000')
        1.0
    """
    # Validate inputs
    if sign not in [0, 1]:
        raise ValueError(f"sign must be 0 or 1, got {sign}")

    if len(exponent_bits) != 8 or not all(c in '01' for c in exponent_bits):
        raise ValueError(f"exponent_bits must be 8-char binary string, got '{exponent_bits}'")

    if len(mantissa_bits) != 23 or not all(c in '01' for c in mantissa_bits):
        raise ValueError(f"mantissa_bits must be 23-char binary string, got '{mantissa_bits}'")

    # Convert binary strings to integers
    biased_exponent = int(exponent_bits, 2)
    mantissa = int(mantissa_bits, 2)

    # Compute true exponent
    true_exponent = biased_exponent - 127

    # Reconstruct value
    if biased_exponent == 0:
        # Subnormal number
        value = (-1) ** sign * 2 ** (-126) * (mantissa / (2 ** 23))
    elif biased_exponent == 255:
        # Infinity or NaN (not handled in this educational implementation)
        raise ValueError("Cannot reconstruct infinity or NaN")
    else:
        # Normal number
        value = (-1) ** sign * 2 ** true_exponent * (1 + mantissa / (2 ** 23))

    return np.float32(value)


# %% [markdown]
# ## Demo: Decompose and Reconstruct -0.10985
#
# This value is used in the reference slides.

# %%
if __name__ == "__main__":
    test_value = -0.10985

    print("=" * 60)
    print(f"IEEE 754 FP32 Decomposition: {test_value}")
    print("=" * 60)

    result = decompose_fp32(test_value)

    print(f"\nOriginal value: {test_value}")
    print(f"\n32-bit breakdown:")
    print(f"  Sign (1 bit):     {result['sign']}")
    print(f"  Exponent (8 bits): {result['exponent_bits']} (biased: {result['biased_exponent']}, true: {result['true_exponent']})")
    print(f"  Mantissa (23 bits): {result['mantissa_bits']}")

    print(f"\nFull bit pattern (32 bits):")
    full_bits = f"{result['sign']}{result['exponent_bits']}{result['mantissa_bits']}"
    print(f"  {full_bits}")
    print(f"  {'↑' + ' ' * 8 + '↑' + ' ' * 22 + '↑'}")
    print(f"  {'sign' + ' ' * 4 + 'exponent' + ' ' * 14 + 'mantissa'}")

    print(f"\nReconstructed value: {result['reconstructed']}")
    print(f"Round-trip error: {abs(test_value - result['reconstructed']):.15f}")

    # Show INT8 quantized representation
    print(f"\n" + "=" * 60)
    print(f"INT8 Quantization Comparison")
    print(f"=" * 60)

    # Simple quantization: scale to [-128, 127] range
    scale = 0.01  # Example scale
    quantized_int8 = int(round(test_value / scale))
    quantized_int8 = np.clip(quantized_int8, -128, 127)

    print(f"\nFP32 (32 bits): {full_bits}")
    print(f"INT8 (8 bits):  {format(quantized_int8 & 0xFF, '08b')} (value: {quantized_int8})")
    print(f"\n🔥 Compression: 32 bits → 8 bits (4x smaller)")
    print(f"⚠️  Information lost: Exponent and mantissa precision")
