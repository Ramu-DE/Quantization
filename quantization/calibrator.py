# %% [markdown]
# # Calibration Methods for Quantization
#
# This module implements three calibration methods for determining optimal
# quantization parameters (scale, zero-point) from representative data:
#
# 1. **MinMax**: Uses global min/max across calibration data
# 2. **Percentile**: Uses percentile clipping to ignore outliers
# 3. **Entropy**: Minimizes KL-divergence between original and quantized distributions
#
# **Learning Objective:** Understand how representative data is used to find
# the optimal quantization range, and why different methods exist.

# %%
import numpy as np
from enum import Enum
from typing import Tuple, List
from scipy.special import kl_div

from .schemes import QuantizationScheme, compute_scale_zero_point

# %% [markdown]
# ## Calibration Method Enum

# %%
class CalibrationMethod(Enum):
    """Calibration method for quantization range selection."""
    MIN_MAX = "min_max"
    PERCENTILE = "percentile"
    ENTROPY = "entropy"


# %% [markdown]
# ## MinMax Calibration
#
# The simplest method: use the global minimum and maximum across all calibration tensors.
#
# **Pros:** Simple, no hyperparameters
# **Cons:** Sensitive to outliers (one extreme value can ruin the range)

# %%
def calibrate_min_max(
    tensors: List[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
) -> Tuple[float, int]:
    """
    Calibrate using global min/max across all tensors.

    Args:
        tensors: List of calibration tensors (at least 1 required)
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme (SYMMETRIC or ASYMMETRIC)

    Returns:
        Tuple of (scale, zero_point)

    Raises:
        ValueError: If tensors list is empty

    Examples:
        >>> tensors = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        >>> scale, zp = calibrate_min_max(tensors, 8, QuantizationScheme.SYMMETRIC)
        >>> zp
        0
    """
    if not tensors or len(tensors) == 0:
        raise ValueError("At least one calibration tensor is required")

    # Concatenate all tensors and find global min/max
    all_values = np.concatenate([t.flatten() for t in tensors])

    # Use the scheme-specific scale/zp computation
    scale, zero_point = compute_scale_zero_point(all_values, bits, scheme)

    return scale, zero_point


# %% [markdown]
# ## Percentile Calibration
#
# Clip to a percentile range to ignore outliers.
#
# For `percentile=99`, use the range `[0.5th, 99.5th]` percentile.
#
# **Pros:** Robust to outliers
# **Cons:** Requires tuning the percentile parameter

# %%
def calibrate_percentile(
    tensors: List[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
    percentile: float = 99.0,
) -> Tuple[float, int]:
    """
    Calibrate using percentile clipping to ignore outliers.

    Args:
        tensors: List of calibration tensors (at least 1 required)
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme
        percentile: Percentile to use (0 < p < 100), default 99.0
                   Uses [(100-p)/2, (100+p)/2] range

    Returns:
        Tuple of (scale, zero_point)

    Raises:
        ValueError: If tensors list is empty or percentile out of range

    Examples:
        >>> tensors = [np.array([1.0, 2.0, 100.0])]  # 100.0 is outlier
        >>> scale, zp = calibrate_percentile(tensors, 8, QuantizationScheme.SYMMETRIC, percentile=99.0)
        >>> # scale will ignore the 100.0 outlier
    """
    if not tensors or len(tensors) == 0:
        raise ValueError("At least one calibration tensor is required")

    if not (0 < percentile < 100):
        raise ValueError(
            f"percentile must be in range (0, 100), got {percentile}"
        )

    # Concatenate all tensors
    all_values = np.concatenate([t.flatten() for t in tensors])

    # Compute percentile range
    p_low = (100 - percentile) / 2  # e.g., 0.5 for p=99
    p_high = (100 + percentile) / 2  # e.g., 99.5 for p=99

    r_min = np.percentile(all_values, p_low)
    r_max = np.percentile(all_values, p_high)

    # Clip values to this range
    clipped_values = np.clip(all_values, r_min, r_max)

    # Compute scale/zp from clipped range
    scale, zero_point = compute_scale_zero_point(clipped_values, bits, scheme)

    return scale, zero_point


# %% [markdown]
# ## Entropy Calibration (KL-Divergence Minimization)
#
# Find the quantization range that minimizes the KL-divergence between
# the original and quantized distributions.
#
# **Algorithm:**
# 1. Compute histogram of original values (2048 bins)
# 2. For each candidate range in a search grid:
#    - Quantize with that range
#    - Compute histogram of reconstructed values
#    - Compute KL-divergence between original and reconstructed histograms
# 3. Select the range with minimum KL-divergence
#
# **Pros:** Theoretically optimal (minimizes information loss)
# **Cons:** Computationally expensive, requires search

# %%
def calibrate_entropy(
    tensors: List[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
) -> Tuple[float, int]:
    """
    Calibrate using KL-divergence minimization (entropy method).

    Searches over candidate quantization ranges and selects the one that
    minimizes the KL-divergence between original and quantized distributions.

    Args:
        tensors: List of calibration tensors (at least 1 required)
        bits: Bit-width (one of {2, 4, 8})
        scheme: Quantization scheme

    Returns:
        Tuple of (scale, zero_point)

    Raises:
        ValueError: If tensors list is empty

    Examples:
        >>> tensors = [np.random.randn(100).astype(np.float32)]
        >>> scale, zp = calibrate_entropy(tensors, 8, QuantizationScheme.SYMMETRIC)
        >>> scale > 0
        True
    """
    if not tensors or len(tensors) == 0:
        raise ValueError("At least one calibration tensor is required")

    # Import quantize/dequantize for reconstruction
    from .quantizer import quantize, dequantize, q_min_max

    # Concatenate all tensors
    all_values = np.concatenate([t.flatten() for t in tensors])

    # Compute histogram of original distribution
    num_bins = 2048
    hist_orig, bin_edges = np.histogram(all_values, bins=num_bins, density=True)
    hist_orig = hist_orig + 1e-8  # Smoothing to avoid log(0)

    # Define search grid: from full range to tight percentile range
    # We'll search 32 logarithmically-spaced percentile values from 99.9 to 60
    percentiles = np.logspace(np.log10(99.9), np.log10(60), 32)

    best_kl = float('inf')
    best_scale = None
    best_zp = None

    for p in percentiles:
        # Compute candidate range at this percentile
        p_low = (100 - p) / 2
        p_high = (100 + p) / 2
        r_min = np.percentile(all_values, p_low)
        r_max = np.percentile(all_values, p_high)

        # Clip to this range
        clipped_values = np.clip(all_values, r_min, r_max)

        # Compute scale/zp for this range
        try:
            scale_candidate, zp_candidate = compute_scale_zero_point(
                clipped_values, bits, scheme
            )
        except ValueError:
            # Skip if this range is invalid (e.g., all zeros)
            continue

        # Quantize and dequantize to get reconstructed distribution
        q_min, q_max = q_min_max(bits, signed=(scheme == QuantizationScheme.SYMMETRIC))
        try:
            q_vals = quantize(
                all_values,
                scale_candidate,
                zp_candidate,
                bits,
                signed=(scheme == QuantizationScheme.SYMMETRIC)
            )
            x_recon = dequantize(q_vals, scale_candidate, zp_candidate)
        except (ValueError, OverflowError):
            # Skip if quantization fails
            continue

        # Compute histogram of reconstructed distribution
        hist_recon, _ = np.histogram(x_recon, bins=bin_edges, density=True)
        hist_recon = hist_recon + 1e-8  # Smoothing

        # Compute KL-divergence: KL(P || Q) where P=original, Q=reconstructed
        kl = np.sum(kl_div(hist_orig, hist_recon))

        # Check for NaN (can happen if histograms are pathological)
        if np.isnan(kl) or np.isinf(kl):
            continue

        # Update best if this is better
        if kl < best_kl:
            best_kl = kl
            best_scale = scale_candidate
            best_zp = zp_candidate

    # If no valid range found, fall back to min-max
    if best_scale is None:
        return calibrate_min_max(tensors, bits, scheme)

    return best_scale, best_zp


# %% [markdown]
# ## Demo: Compare All Three Calibration Methods

# %%
if __name__ == "__main__":
    print("=" * 80)
    print("Demo: Calibration Methods Comparison")
    print("=" * 80)

    # Generate calibration data with outliers
    np.random.seed(42)
    tensors = []
    for _ in range(5):
        # Gaussian distribution + a few outliers
        t = np.random.randn(200).astype(np.float32) * 0.5
        # Add 2% outliers
        outlier_indices = np.random.choice(200, size=4, replace=False)
        t[outlier_indices] = np.random.uniform(-5, 5, size=4)
        tensors.append(t)

    print(f"\nCalibration dataset:")
    print(f"  Number of tensors: {len(tensors)}")
    all_vals = np.concatenate(tensors)
    print(f"  Total elements: {len(all_vals)}")
    print(f"  Min: {np.min(all_vals):.4f}")
    print(f"  Max: {np.max(all_vals):.4f}")
    print(f"  Mean: {np.mean(all_vals):.4f}")
    print(f"  Std: {np.std(all_vals):.4f}")
    print(f"  99th percentile: [{np.percentile(all_vals, 0.5):.4f}, {np.percentile(all_vals, 99.5):.4f}]")

    bits = 8
    scheme = QuantizationScheme.SYMMETRIC

    print(f"\n{'Method':<15} {'Scale':<12} {'Zero_Point':<12}")
    print("-" * 80)

    # MinMax
    scale_mm, zp_mm = calibrate_min_max(tensors, bits, scheme)
    print(f"{'MinMax':<15} {scale_mm:<12.6f} {zp_mm:<12}")

    # Percentile (99%)
    scale_p, zp_p = calibrate_percentile(tensors, bits, scheme, percentile=99.0)
    print(f"{'Percentile-99':<15} {scale_p:<12.6f} {zp_p:<12}")

    # Entropy
    scale_e, zp_e = calibrate_entropy(tensors, bits, scheme)
    print(f"{'Entropy':<15} {scale_e:<12.6f} {zp_e:<12}")

    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    print("1. MinMax uses full range → largest scale (includes outliers)")
    print("2. Percentile clips outliers → smaller scale (more precision for main distribution)")
    print("3. Entropy optimizes information preservation → data-adaptive")
    print("4. For data with outliers, Percentile and Entropy typically outperform MinMax")
