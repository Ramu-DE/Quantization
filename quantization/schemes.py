# %% [markdown]
# # Quantization Mapping Schemes
#
# This module implements symmetric and asymmetric quantization schemes
# for computing scale and zero-point parameters.
#
# **Learning Objective:** Understand when to use symmetric vs asymmetric
# quantization and their tradeoffs.
#
# ## Symmetric Quantization
#
# - Zero-point is always 0
# - Range is centered around zero
# - Formula: `scale = max(|x_min|, |x_max|) / (2^(bits-1) - 1)`
# - **Best for:** Weight tensors (usually symmetric distribution)
#
# ## Asymmetric Quantization
#
# - Zero-point computed to align ranges
# - Can handle skewed distributions efficiently
# - Formula: `scale = (x_max - x_min) / (2^bits - 1)`
# - **Best for:** Activation tensors (often ReLU → all positive)

# %%
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

# %% [markdown]
# ## Quantization Scheme Enum
#
# Explicit enum for scheme selection (extensible design).

# %%
class QuantizationScheme(Enum):
    """Quantization mapping scheme."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


# %% [markdown]
# ## Scheme Comparison Report
#
# Dataclass for storing comparison results between schemes.

# %%
@dataclass
class SchemeComparisonReport:
    """Report comparing symmetric and asymmetric quantization schemes."""
    symmetric_scale: float
    symmetric_zero_point: int
    symmetric_mae: float
    asymmetric_scale: float
    asymmetric_zero_point: int
    asymmetric_mae: float


# %% [markdown]
# ## Scale and Zero-Point Computation
#
# Core function that computes quantization parameters based on scheme.

# %%
def compute_scale_zero_point(
    x: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
) -> Tuple[float, int]:
    """
    Compute scale and zero-point for a tensor using the specified scheme.

    Args:
        x: Input tensor (float32)
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme (SYMMETRIC or ASYMMETRIC)

    Returns:
        Tuple of (scale, zero_point)

    Raises:
        ValueError: If bits is invalid or tensor is all zeros

    Examples:
        >>> x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        >>> scale, zp = compute_scale_zero_point(x, 8, QuantizationScheme.SYMMETRIC)
        >>> zp
        0
        >>> scale, zp = compute_scale_zero_point(x, 8, QuantizationScheme.ASYMMETRIC)
        >>> zp == 0  # For all-positive tensors, asymmetric zp is also 0
        True
    """
    if bits not in [2, 4, 8]:
        raise ValueError(f"bits must be one of {{2, 4, 8}}, got {bits}")

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    # Handle constant tensors
    if x_min == x_max:
        # All elements are identical
        if x_min == 0.0:
            raise ValueError("Cannot compute scale for all-zeros tensor")
        # Return identity mapping
        return 1.0, 0

    if scheme == QuantizationScheme.SYMMETRIC:
        # Symmetric: zero-point = 0, range centered at zero
        abs_max = max(abs(x_min), abs(x_max))

        # Handle edge case: tensor is all zeros
        if abs_max == 0.0:
            raise ValueError("Cannot compute scale for all-zeros tensor")

        # Symmetric uses signed range: [-(2^(bits-1)), 2^(bits-1) - 1]
        q_max_symmetric = (2 ** (bits - 1)) - 1
        scale = abs_max / q_max_symmetric
        zero_point = 0

    elif scheme == QuantizationScheme.ASYMMETRIC:
        # Asymmetric: zero-point computed, unsigned range
        range_x = x_max - x_min

        if range_x == 0.0:
            # Already handled above, but double-check
            raise ValueError("Cannot compute scale for constant tensor")

        # Asymmetric uses unsigned range: [0, 2^bits - 1]
        q_max_asymmetric = (2 ** bits) - 1
        scale = range_x / q_max_asymmetric

        # Compute zero-point to align x_min with 0
        zp_float = -x_min / scale
        zero_point = int(np.clip(np.round(zp_float), 0, q_max_asymmetric))

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Ensure scale is positive
    if scale <= 0:
        raise ValueError(f"Computed scale must be positive, got {scale}")

    return scale, zero_point


# %% [markdown]
# ## Scheme Comparison
#
# Compare symmetric and asymmetric schemes on the same tensor.

# %%
def compare_schemes(x: np.ndarray, bits: int) -> SchemeComparisonReport:
    """
    Compare symmetric and asymmetric quantization on the same tensor.

    Args:
        x: Input tensor (float32)
        bits: Bit-width (one of {2, 4, 8})

    Returns:
        SchemeComparisonReport with scale, zero_point, and MAE for both schemes

    Examples:
        >>> x = np.random.randn(100).astype(np.float32)
        >>> report = compare_schemes(x, 8)
        >>> report.symmetric_zero_point
        0
    """
    # Import quantize/dequantize (will be available after quantizer.py is implemented)
    # For now, we'll implement a simple round-trip error calculation
    def _quantize_dequantize(x, scale, zero_point, bits, scheme):
        """Helper for computing quantization error."""
        if scheme == QuantizationScheme.SYMMETRIC:
            q_min = -(2 ** (bits - 1))
            q_max = (2 ** (bits - 1)) - 1
        else:
            q_min = 0
            q_max = (2 ** bits) - 1

        # Quantize
        q = np.clip(np.round(x / scale) + zero_point, q_min, q_max)
        # Dequantize
        x_reconstructed = (q - zero_point) * scale

        return x_reconstructed

    # Symmetric scheme
    sym_scale, sym_zp = compute_scale_zero_point(x, bits, QuantizationScheme.SYMMETRIC)
    x_sym = _quantize_dequantize(x, sym_scale, sym_zp, bits, QuantizationScheme.SYMMETRIC)
    sym_mae = float(np.mean(np.abs(x - x_sym)))

    # Asymmetric scheme
    asym_scale, asym_zp = compute_scale_zero_point(x, bits, QuantizationScheme.ASYMMETRIC)
    x_asym = _quantize_dequantize(x, asym_scale, asym_zp, bits, QuantizationScheme.ASYMMETRIC)
    asym_mae = float(np.mean(np.abs(x - x_asym)))

    return SchemeComparisonReport(
        symmetric_scale=sym_scale,
        symmetric_zero_point=sym_zp,
        symmetric_mae=sym_mae,
        asymmetric_scale=asym_scale,
        asymmetric_zero_point=asym_zp,
        asymmetric_mae=asym_mae,
    )


# %% [markdown]
# ## Pretty-Print Comparison Report

# %%
def format_scheme_comparison(report: SchemeComparisonReport) -> str:
    """
    Format a scheme comparison report as a human-readable table.

    Args:
        report: SchemeComparisonReport to format

    Returns:
        Formatted table string

    Examples:
        >>> x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        >>> report = compare_schemes(x, 8)
        >>> table = format_scheme_comparison(report)
        >>> "Symmetric" in table
        True
        >>> "Asymmetric" in table
        True
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Quantization Scheme Comparison")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Scheme':<12} {'Scale':<12} {'Zero_Point':<12} {'Mean_Absolute_Error':<20}")
    lines.append("-" * 80)

    lines.append(
        f"{'Symmetric':<12} {report.symmetric_scale:<12.6f} "
        f"{report.symmetric_zero_point:<12} {report.symmetric_mae:<20.6f}"
    )

    lines.append(
        f"{'Asymmetric':<12} {report.asymmetric_scale:<12.6f} "
        f"{report.asymmetric_zero_point:<12} {report.asymmetric_mae:<20.6f}"
    )

    lines.append("-" * 80)

    # Add winner indication
    if report.symmetric_mae < report.asymmetric_mae:
        winner = "Symmetric"
        improvement = (report.asymmetric_mae - report.symmetric_mae) / report.asymmetric_mae * 100
    else:
        winner = "Asymmetric"
        improvement = (report.symmetric_mae - report.asymmetric_mae) / report.symmetric_mae * 100

    lines.append(f"✓ Best scheme: {winner} ({improvement:.1f}% lower error)")
    lines.append("")

    return "\n".join(lines)


# %% [markdown]
# ## Demo: Symmetric vs Asymmetric on ReLU-Shaped Tensor
#
# ReLU activations are all positive, making asymmetric quantization more efficient.

# %%
if __name__ == "__main__":
    print("=" * 80)
    print("Demo 1: Gaussian-Distributed Weights (Symmetric Distribution)")
    print("=" * 80)

    # Weights: typically Gaussian-distributed, mean ≈ 0
    weights = np.random.randn(1000).astype(np.float32) * 0.02
    print(f"\nTensor statistics:")
    print(f"  Min: {np.min(weights):.4f}")
    print(f"  Max: {np.max(weights):.4f}")
    print(f"  Mean: {np.mean(weights):.4f}")
    print(f"  Std: {np.std(weights):.4f}")

    report_weights = compare_schemes(weights, 8)
    print("\n" + format_scheme_comparison(report_weights))

    print("\n" + "=" * 80)
    print("Demo 2: ReLU Activations (All Positive → Skewed Distribution)")
    print("=" * 80)

    # Activations: ReLU output, all positive
    activations = np.abs(np.random.randn(1000).astype(np.float32)) * 0.5
    print(f"\nTensor statistics:")
    print(f"  Min: {np.min(activations):.4f}")
    print(f"  Max: {np.max(activations):.4f}")
    print(f"  Mean: {np.mean(activations):.4f}")
    print(f"  Std: {np.std(activations):.4f}")

    report_activations = compare_schemes(activations, 8)
    print("\n" + format_scheme_comparison(report_activations))

    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    print("1. For symmetric distributions (weights): Both schemes perform similarly")
    print("2. For skewed distributions (ReLU): Asymmetric is more efficient")
    print("3. Symmetric always has zero_point = 0 (simplifies computation)")
    print("4. Asymmetric adapts to the actual data range")
