# %% [markdown]
# # Quantization Benefits Analysis
#
# This module provides utilities for analyzing the memory and compute costs
# of floating-point versus integer models, demonstrating why quantization
# is necessary for edge deployment.
#
# **Learning Objective:** Understand the concrete memory and throughput
# benefits of quantization with real numbers (e.g., Llama-70B).

# %%
import numpy as np
from typing import Dict, Tuple

# %% [markdown]
# ## Data Type Size Mapping
#
# Each data type consumes a fixed number of bytes per element:
# - **FP32**: 4 bytes (32 bits)
# - **FP16**: 2 bytes (16 bits)
# - **INT8**: 1 byte (8 bits)
# - **INT4**: 0.5 bytes (4 bits, packed)

# %%
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "int8": 1,
    "int4": 0.5,
}

# %% [markdown]
# ## Memory Footprint Calculation
#
# Given a tensor with `n` elements, the memory footprint is:
#
# ```
# memory = n × bytes_per_element
# ```

# %%
def memory_footprint(n_elements: int, dtype: str) -> float:
    """
    Compute the memory footprint in bytes for a tensor.

    Args:
        n_elements: Number of elements in the tensor
        dtype: Data type name (one of: float32, float16, int8, int4)

    Returns:
        Memory footprint in bytes

    Raises:
        ValueError: If dtype is not supported

    Examples:
        >>> memory_footprint(1000, "float32")
        4000.0
        >>> memory_footprint(1000, "int8")
        1000.0
    """
    if dtype not in DTYPE_BYTES:
        supported = ", ".join(DTYPE_BYTES.keys())
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported types: {supported}"
        )

    if n_elements < 0:
        raise ValueError(f"n_elements must be non-negative, got {n_elements}")

    return n_elements * DTYPE_BYTES[dtype]


# %% [markdown]
# ## Compression Ratio
#
# The compression ratio quantifies how much smaller a quantized model is
# compared to the original floating-point model.
#
# ```
# compression_ratio = size_dtype_a / size_dtype_b
# ```
#
# For identical shapes:
# ```
# float32 / int8 = 4 bytes / 1 byte = 4.0x
# ```

# %%
def compression_ratio(
    shape: Tuple[int, ...],
    dtype_a: str,
    dtype_b: str
) -> float:
    """
    Compute the compression ratio between two data types for a given shape.

    Args:
        shape: Tensor shape (e.g., (1000, 500))
        dtype_a: First data type (numerator)
        dtype_b: Second data type (denominator)

    Returns:
        Compression ratio (dtype_a size / dtype_b size)

    Raises:
        ValueError: If dtypes are not supported

    Examples:
        >>> compression_ratio((1000,), "float32", "int8")
        4.0
        >>> compression_ratio((512, 512), "float16", "int4")
        4.0
    """
    if dtype_a not in DTYPE_BYTES:
        supported = ", ".join(DTYPE_BYTES.keys())
        raise ValueError(
            f"Unsupported dtype_a '{dtype_a}'. Supported types: {supported}"
        )

    if dtype_b not in DTYPE_BYTES:
        supported = ", ".join(DTYPE_BYTES.keys())
        raise ValueError(
            f"Unsupported dtype_b '{dtype_b}'. Supported types: {supported}"
        )

    n_elements = int(np.prod(shape))
    size_a = memory_footprint(n_elements, dtype_a)
    size_b = memory_footprint(n_elements, dtype_b)

    if size_b == 0:
        raise ValueError("Cannot compute compression ratio with zero-size denominator")

    return size_a / size_b


# %% [markdown]
# ## MAC Count (Multiply-Accumulate Operations)
#
# For a linear layer `y = W @ x`:
# - `W` has shape `[out_features, in_features]`
# - Each output element requires `in_features` multiplications
# - Total MAC count: `out_features × in_features`
#
# **Throughput benefit:** INT8 MACs are ~4x faster than FP32 MACs on modern hardware.

# %%
def mac_count(in_features: int, out_features: int) -> int:
    """
    Compute the number of multiply-accumulate (MAC) operations for a linear layer.

    Args:
        in_features: Input dimension
        out_features: Output dimension

    Returns:
        Total MAC count

    Examples:
        >>> mac_count(512, 256)
        131072
        >>> mac_count(1024, 1024)
        1048576
    """
    if in_features < 0 or out_features < 0:
        raise ValueError("in_features and out_features must be non-negative")

    return in_features * out_features


# %% [markdown]
# ## Model Memory Table
#
# Given a model with `p` parameters, compute the total memory footprint
# for each supported data type.

# %%
def model_memory_table(param_count: int) -> Dict[str, float]:
    """
    Compute model memory in GB for all supported data types.

    Args:
        param_count: Number of model parameters (e.g., 70_000_000_000 for Llama-70B)

    Returns:
        Dictionary mapping dtype → memory in GB

    Examples:
        >>> table = model_memory_table(70_000_000_000)
        >>> table["float32"]  # doctest: +ELLIPSIS
        280.0...
        >>> table["int8"]
        70.0
    """
    if param_count < 0:
        raise ValueError(f"param_count must be non-negative, got {param_count}")

    table = {}
    for dtype in DTYPE_BYTES:
        bytes_total = memory_footprint(param_count, dtype)
        gb = bytes_total / (1024 ** 3)  # Convert bytes to GB
        table[dtype] = gb

    return table


# %% [markdown]
# ## Demo: Llama-70B Memory Analysis
#
# Let's analyze the memory requirements for Llama-70B (70 billion parameters).

# %%
if __name__ == "__main__":
    # Llama-70B has 70 billion parameters
    llama_70b_params = 70_000_000_000

    print("=" * 60)
    print("Llama-70B Memory Requirements")
    print("=" * 60)

    memory_table = model_memory_table(llama_70b_params)

    print(f"\nParameter count: {llama_70b_params:,}")
    print(f"\nMemory footprint by data type:")
    for dtype, gb in memory_table.items():
        print(f"  {dtype:8s}: {gb:8.2f} GB")

    # Compression ratios
    print(f"\nCompression ratios (vs float32):")
    baseline_gb = memory_table["float32"]
    for dtype, gb in memory_table.items():
        if dtype != "float32":
            ratio = baseline_gb / gb
            print(f"  float32 → {dtype:8s}: {ratio:.1f}x compression")

    # Example: 8-bit compression ratio on a weight tensor
    weight_shape = (4096, 4096)  # Typical transformer layer
    ratio = compression_ratio(weight_shape, "float32", "int8")
    print(f"\nSingle layer compression (float32 → int8): {ratio:.1f}x")

    # MAC count example
    macs = mac_count(4096, 4096)
    print(f"\nMAC operations for [4096, 4096] layer: {macs:,}")
    print(f"INT8 throughput advantage: ~4x faster than FP32")
