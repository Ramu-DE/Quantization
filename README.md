# AI Model Quantization - From Scratch

A complete, from-scratch implementation of AI model quantization built for deep understanding through construction.

## Overview

This project implements quantization primitives from first principles:
- **Core quantization equation** with scale, zero-point, and clamping
- **Mapping schemes**: Symmetric and asymmetric quantization
- **Calibration methods**: MinMax, Percentile, and Entropy-based
- **Granularity levels**: Per-tensor, per-channel, and per-group
- **Algorithms**: PTQ, QAT (with STE), and GPTQ (Hessian-guided)

## Features

- 🧮 **NumPy-based core**: All quantization math implemented from scratch
- 🔬 **Property-based testing**: 15 correctness properties verified with Hypothesis
- 📊 **Interactive UI**: 15+ Streamlit panels for visual learning
- 📚 **Jupyter-compatible**: All core modules use `# %%` cell markers

## Installation

```bash
# Install package and dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from quantization import quantize, dequantize

# Quantize a float32 tensor to int8
x = np.array([1.5, -0.5, 0.0, 2.3], dtype=np.float32)
scale = 0.02
zero_point = 0
bits = 8

q = quantize(x, scale, zero_point, bits, signed=True)
print(f"Quantized: {q}")  # [-128, -25, 0, 115]

x_reconstructed = dequantize(q, scale, zero_point)
print(f"Reconstructed: {x_reconstructed}")
```

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific property tests
pytest quantization/tests/test_quantizer.py -v

# Run with Hypothesis verbosity
pytest --hypothesis-verbosity=verbose
```

## Interactive Dashboard

```bash
streamlit run quantization/ui/app.py
```

Navigate through 15 interactive panels covering:
- FP32 format visualization
- Quantization pipeline
- Weight distributions
- Calibration methods
- PTQ, QAT, and GPTQ algorithms

## Project Structure

```
quantization/
├── quantizer.py          # Core quantize/dequantize
├── schemes.py            # Symmetric/asymmetric schemes
├── calibrator.py         # Calibration methods
├── granularity.py        # Per-tensor/channel/group
├── ptq.py                # Post-Training Quantization
├── qat.py                # Quantization-Aware Training
├── gptq.py               # GPTQ algorithm
├── fp32_format.py        # IEEE 754 utilities
├── benefits.py           # Memory/compute analysis
├── tests/                # Property-based test suite
└── ui/                   # Streamlit dashboard
    ├── app.py
    └── panels/
```

## Documentation

- [Requirements](requirements.md) - Detailed acceptance criteria
- [Design](design.md) - Architecture and algorithms
- [Tasks](tasks.md) - Implementation plan

## License

MIT License - Educational purposes
