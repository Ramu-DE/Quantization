"""
Quantization service - core quantization logic
"""
import numpy as np
from ml_dtypes import bfloat16
from typing import Tuple, Dict, List


def quantization_error_stats(actual: np.ndarray, quantized: np.ndarray) -> Tuple[float, float]:
    """Compute Mean Squared Error and Residual Sum of Squares"""
    squared_diffs = (actual - quantized) ** 2
    mse = float(np.mean(squared_diffs))
    rss = float(np.sum(squared_diffs))
    return mse, rss


def quantize_symmetric_int8(weights: np.ndarray, bits: int = 8) -> Dict:
    """
    Symmetric INT8 quantization

    Args:
        weights: Input weights as numpy array
        bits: Bit-width (default 8)

    Returns:
        Dictionary with quantized weights and metadata
    """
    # Find absolute maximum value
    max_value = float(np.max(np.abs(weights)))

    # Compute scale factor (using bits-1 for symmetric range)
    q_max = (2 ** (bits - 1)) - 1  # 127 for 8-bit
    scale = max_value / q_max

    # Scale and round
    weights_scaled = weights * (1 / scale)
    weights_quantized = np.int8(np.rint(weights_scaled))

    # Dequantize for error computation
    weights_dequantized = weights_quantized * scale

    # Compute errors
    mse, rss = quantization_error_stats(weights, weights_dequantized)

    # Error distribution
    errors = weights - weights_dequantized

    return {
        "quantized_weights": weights_quantized.tolist(),
        "scale": float(scale),
        "zero_point": 0,
        "mse": mse,
        "rss": rss,
        "min_value": float(np.min(weights_quantized)),
        "max_value": float(np.max(weights_quantized)),
        "error_distribution": {
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
    }


def quantize_asymmetric_int8(weights: np.ndarray, bits: int = 8) -> Dict:
    """
    Asymmetric INT8 quantization

    Args:
        weights: Input weights as numpy array
        bits: Bit-width (default 8)

    Returns:
        Dictionary with quantized weights and metadata
    """
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))

    # Compute scale factor for full range
    q_range = (2 ** bits) - 1  # 255 for 8-bit
    scale = (w_max - w_min) / q_range

    # Compute zero point
    q_min = -(2 ** (bits - 1))  # -128 for 8-bit
    zero_point = int(np.rint(w_min * (-1 / scale) - q_min))

    # Quantize
    weights_scaled = weights * (1 / scale) + zero_point
    weights_quantized = np.int8(np.rint(weights_scaled))

    # Dequantize
    weights_dequantized = (weights_quantized.astype(np.float32) - zero_point) * scale

    # Compute errors
    mse, rss = quantization_error_stats(weights, weights_dequantized)
    errors = weights - weights_dequantized

    return {
        "quantized_weights": weights_quantized.tolist(),
        "scale": float(scale),
        "zero_point": int(zero_point),
        "mse": mse,
        "rss": rss,
        "min_value": float(np.min(weights_quantized)),
        "max_value": float(np.max(weights_quantized)),
        "error_distribution": {
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
    }


def quantize_clipped_int8(
    weights: np.ndarray,
    clip_min: float,
    clip_max: float,
    bits: int = 8
) -> Dict:
    """
    Clipped symmetric quantization

    Args:
        weights: Input weights
        clip_min: Minimum clip value
        clip_max: Maximum clip value
        bits: Bit-width

    Returns:
        Dictionary with quantized weights and metadata
    """
    # Clip weights
    weights_clipped = np.clip(weights, clip_min, clip_max)

    # Find max after clipping
    max_value_clip = float(np.max(np.abs(weights_clipped)))

    # Compute scale
    q_max = (2 ** (bits - 1)) - 1
    scale = max_value_clip / q_max

    # Quantize
    weights_scaled = weights_clipped * (1 / scale)
    weights_quantized = np.int8(np.rint(weights_scaled))

    # Dequantize
    weights_dequantized = weights_quantized * scale

    # Compute errors (compare with ORIGINAL weights, not clipped)
    mse, rss = quantization_error_stats(weights, weights_dequantized)
    errors = weights - weights_dequantized

    return {
        "quantized_weights": weights_quantized.tolist(),
        "scale": float(scale),
        "zero_point": 0,
        "mse": mse,
        "rss": rss,
        "min_value": float(np.min(weights_quantized)),
        "max_value": float(np.max(weights_quantized)),
        "error_distribution": {
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
    }


def compute_weight_distribution(weights: np.ndarray, num_bins: int = 100) -> Dict:
    """
    Compute histogram and cumulative distribution

    Args:
        weights: Input weights
        num_bins: Number of histogram bins

    Returns:
        Dictionary with histogram and cumulative data
    """
    # Histogram
    counts, bin_edges = np.histogram(weights, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Cumulative distribution
    cumulative = np.cumsum(counts)
    cumulative_normalized = (cumulative / cumulative[-1]) * 100

    # Statistics
    stats = {
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "median": float(np.median(weights)),
        "range": float(np.max(weights) - np.min(weights)),
    }

    return {
        "histogram": {
            "counts": counts.tolist(),
            "bin_centers": bin_centers.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
        "cumulative": {
            "values": cumulative_normalized.tolist(),
            "bins": bin_centers.tolist(),
        },
        "statistics": stats,
    }


def quantize_bfloat16(weights: np.ndarray) -> Dict:
    """
    BFloat16 quantization - cast float32 weights to bfloat16 and compute error.

    Args:
        weights: Input weights as float32 numpy array

    Returns:
        Dictionary with quantized weights and error stats
    """
    # Cast to bfloat16
    weights_bf16 = weights.astype(bfloat16)

    # Cast back to float32 for error computation
    weights_dequantized = weights_bf16.astype(np.float32)

    # Compute errors
    mse, rss = quantization_error_stats(weights, weights_dequantized)
    errors = weights - weights_dequantized

    return {
        "original_weights": [float(x) for x in weights],
        "quantized_weights": [float(x) for x in weights_dequantized],
        "mse": mse,
        "rss": rss,
        "error_distribution": {
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
    }


def quantize_symmetric_steps(weights: np.ndarray) -> Dict:
    """
    Step-by-step symmetric INT8 quantization showing intermediate values.

    Args:
        weights: Input weights as float32 numpy array

    Returns:
        Dictionary with each step's intermediate values
    """
    bits = 8
    q_max = (2 ** (bits - 1)) - 1  # 127

    # Step 1: Find max absolute value
    max_abs = float(np.max(np.abs(weights)))

    # Step 2: Compute scale factor
    scale = max_abs / q_max  # scale = max_value / 127

    # Step 3: Multiply weights by scale (divide by scale to get quantized space)
    weights_scaled = weights / scale  # still float, reversible

    # Step 4: Round to nearest integer (information lost here)
    weights_rounded = np.rint(weights_scaled)

    # Step 5: Cast to int8
    weights_int8 = np.clip(weights_rounded, -128, 127).astype(np.int8)

    steps = [
        {
            "step": 1,
            "name": "Find max absolute value",
            "description": f"max(|weights|) = {max_abs}",
            "values": [float(x) for x in weights],
            "reversible": True,
        },
        {
            "step": 2,
            "name": "Compute scale factor",
            "description": f"scale = max_abs / 127 = {max_abs} / 127 = {scale}",
            "values": [float(scale)],
            "reversible": True,
        },
        {
            "step": 3,
            "name": "Divide weights by scale",
            "description": "weights_scaled = weights / scale (still float, reversible)",
            "values": [float(x) for x in weights_scaled],
            "reversible": True,
        },
        {
            "step": 4,
            "name": "Round to nearest integer",
            "description": "weights_rounded = round(weights_scaled) (information lost here)",
            "values": [float(x) for x in weights_rounded],
            "reversible": False,
        },
        {
            "step": 5,
            "name": "Cast to int8",
            "description": "weights_int8 = int8(weights_rounded) (clipped to [-128, 127])",
            "values": [float(x) for x in weights_int8],
            "reversible": False,
        },
    ]

    return {
        "steps": steps,
        "original_weights": [float(x) for x in weights],
        "final_quantized": [int(x) for x in weights_int8],
        "scale": float(scale),
    }


def compute_error_distribution(
    weights: np.ndarray,
    method: str = "symmetric",
    bits: int = 8,
    clip_min: float = None,
    clip_max: float = None,
    num_bins: int = 50,
) -> Dict:
    """
    Compute detailed error histogram for a given quantization method.

    Args:
        weights: Input weights
        method: Quantization method (symmetric, asymmetric, clipped, bfloat16)
        bits: Bit-width
        clip_min: Min clip value (for clipped method)
        clip_max: Max clip value (for clipped method)
        num_bins: Number of histogram bins

    Returns:
        Dictionary with error histogram data
    """
    # Quantize based on method to get dequantized values
    if method == "bfloat16":
        weights_bf16 = weights.astype(bfloat16)
        weights_dequantized = weights_bf16.astype(np.float32)
    elif method == "asymmetric":
        result = quantize_asymmetric_int8(weights, bits)
        # Dequantize
        q_weights = np.array(result["quantized_weights"], dtype=np.float32)
        weights_dequantized = (q_weights - result["zero_point"]) * result["scale"]
    elif method == "clipped":
        if clip_min is None or clip_max is None:
            raise ValueError("clip_min and clip_max required for clipped method")
        result = quantize_clipped_int8(weights, clip_min, clip_max, bits)
        q_weights = np.array(result["quantized_weights"], dtype=np.float32)
        weights_dequantized = q_weights * result["scale"]
    else:
        # Default: symmetric
        result = quantize_symmetric_int8(weights, bits)
        q_weights = np.array(result["quantized_weights"], dtype=np.float32)
        weights_dequantized = q_weights * result["scale"]

    # Compute errors
    errors = weights - weights_dequantized

    # Build histogram
    counts, bin_edges = np.histogram(errors, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "bin_centers": [float(x) for x in bin_centers],
        "counts": [int(x) for x in counts],
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "num_elements": int(len(weights)),
    }


def compute_outlier_analysis(weights: np.ndarray, threshold: float = 0.10) -> Dict:
    """
    Analyze outlier weights beyond a given threshold.

    Args:
        weights: Input weights
        threshold: Threshold for outlier detection (absolute value)

    Returns:
        Dictionary with outlier information and histograms
    """
    negative_outliers = weights[weights < -threshold]
    positive_outliers = weights[weights > threshold]

    total_weights = len(weights)
    num_outliers = len(negative_outliers) + len(positive_outliers)
    outlier_percentage = float(num_outliers / total_weights * 100) if total_weights > 0 else 0.0

    # Histograms for each tail
    neg_hist = {"counts": [], "bin_centers": [], "bin_edges": []}
    if len(negative_outliers) > 0:
        neg_counts, neg_edges = np.histogram(negative_outliers, bins=min(30, len(negative_outliers)))
        neg_centers = (neg_edges[:-1] + neg_edges[1:]) / 2
        neg_hist = {
            "counts": [int(x) for x in neg_counts],
            "bin_centers": [float(x) for x in neg_centers],
            "bin_edges": [float(x) for x in neg_edges],
        }

    pos_hist = {"counts": [], "bin_centers": [], "bin_edges": []}
    if len(positive_outliers) > 0:
        pos_counts, pos_edges = np.histogram(positive_outliers, bins=min(30, len(positive_outliers)))
        pos_centers = (pos_edges[:-1] + pos_edges[1:]) / 2
        pos_hist = {
            "counts": [int(x) for x in pos_counts],
            "bin_centers": [float(x) for x in pos_centers],
            "bin_edges": [float(x) for x in pos_edges],
        }

    return {
        "negative_outliers": [float(x) for x in negative_outliers],
        "positive_outliers": [float(x) for x in positive_outliers],
        "total_weights": total_weights,
        "num_outliers": num_outliers,
        "outlier_percentage": outlier_percentage,
        "negative_histogram": neg_hist,
        "positive_histogram": pos_hist,
    }


def compute_memory_comparison(num_elements: int) -> Dict:
    """
    Compute memory usage for different data formats.

    Args:
        num_elements: Number of weight elements

    Returns:
        Dictionary with memory usage for each format
    """
    formats = [
        {"name": "float32", "bits": 32},
        {"name": "float16", "bits": 16},
        {"name": "bfloat16", "bits": 16},
        {"name": "int8", "bits": 8},
        {"name": "int4", "bits": 4},
    ]

    baseline_bytes = num_elements * 4  # float32 baseline

    results = []
    for fmt in formats:
        total_bytes = num_elements * fmt["bits"] // 8
        megabytes = total_bytes / (1024 * 1024)
        compression_ratio = baseline_bytes / total_bytes if total_bytes > 0 else 0.0

        results.append({
            "format_name": fmt["name"],
            "bits_per_element": fmt["bits"],
            "bytes": total_bytes,
            "megabytes": round(megabytes, 4),
            "compression_ratio": round(compression_ratio, 2),
        })

    return {
        "num_elements": num_elements,
        "formats": results,
    }


def get_format_info() -> Dict:
    """
    Return information about number formats used in quantization.

    Returns:
        Dictionary with format information
    """
    formats = [
        {
            "name": "float32",
            "total_bits": 32,
            "components": {"sign": 1, "exponent": 8, "mantissa": 23},
            "range_min": "-3.4e38",
            "range_max": "3.4e38",
            "description": "IEEE 754 single-precision floating point. Full precision, standard training format.",
        },
        {
            "name": "float16",
            "total_bits": 16,
            "components": {"sign": 1, "exponent": 5, "mantissa": 10},
            "range_min": "-65504",
            "range_max": "65504",
            "description": "IEEE 754 half-precision floating point. Good precision but limited range.",
        },
        {
            "name": "bfloat16",
            "total_bits": 16,
            "components": {"sign": 1, "exponent": 8, "mantissa": 7},
            "range_min": "-3.4e38",
            "range_max": "3.4e38",
            "description": "Brain floating point. Same range as float32 but fewer mantissa bits. Developed by Google Brain.",
        },
        {
            "name": "int8",
            "total_bits": 8,
            "components": {"bits": 8},
            "range_min": "-128",
            "range_max": "127",
            "description": "8-bit signed integer. 4x compression over float32. Most common quantization target.",
        },
        {
            "name": "int4",
            "total_bits": 4,
            "components": {"bits": 4},
            "range_min": "-8",
            "range_max": "7",
            "description": "4-bit signed integer. 8x compression over float32. Used in aggressive quantization (GPTQ, AWQ).",
        },
    ]

    return {"formats": formats}
