"""
AI Model Quantization - From-scratch implementation for deep learning.

This package provides a complete educational implementation of neural network
quantization, built from first principles using NumPy and PyTorch.

Main modules:
- quantizer: Core quantize/dequantize functions
- schemes: Symmetric and asymmetric mapping schemes
- calibrator: MinMax, Percentile, and Entropy calibration methods
- granularity: Per-tensor, per-channel, and per-group quantization
- ptq: Post-Training Quantization
- qat: Quantization-Aware Training (PyTorch)
- gptq: GPTQ algorithm with Hessian-guided updates
- fp32_format: IEEE 754 FP32 bit decomposition utilities
- benefits: Memory and compute analysis
"""

__version__ = "0.1.0"

# Public API exports will be added as modules are implemented
__all__ = []
