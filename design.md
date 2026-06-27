# Design Document: AI Model Quantization

## Overview

This project is a from-scratch Python implementation of AI model quantization, structured as an educational, spec-driven package. The learner builds every quantization primitive themselves — no high-level quantization libraries are used for core logic. The result is a fully interactive learning environment: a Python package with well-defined modules, a property-based test suite (Hypothesis), and a Streamlit dashboard (`app.py`) with 15 interactive panels.

The system covers the full quantization learning path:

1. **Motivation** — memory/compute analysis and benefits calculator  
2. **Core equation** — quantize/dequantize with clamping and rounding  
3. **Mapping schemes** — symmetric and asymmetric scale/zero-point computation  
4. **Calibration** — MinMax, Percentile, and Entropy range calibration  
5. **Granularity** — per-tensor, per-channel, per-group quantization  
6. **Algorithms** — PTQ, QAT (with FakeQuantize + STE), and GPTQ  
7. **Visualization** — IEEE 754 bit decomposition, distribution explorers, pipeline animation  

### Key Design Decisions

- **NumPy as the primary tensor library** for all core quantization math; PyTorch is used only in the QAT module where autograd is required for the STE backward pass.
- **No external quantization libraries** (no `bitsandbytes`, no `torch.quantization`) for the core implementation.
- **Hypothesis** for property-based testing; each test maps directly to a named correctness property in this document.
- **Streamlit** for the interactive UI; sidebar navigation uses `st.session_state` so panel switching never triggers a full page reload.
- **Plotly** for all interactive charts (histograms, heatmaps, bar charts, number lines).

---

## Architecture

### Package Structure

```
quantization/
├── quantizer.py          # Core quantize/dequantize functions
├── schemes.py            # QuantizationScheme enum + scale/zero-point computation
├── calibrator.py         # MinMax, Percentile, Entropy calibrators
├── granularity.py        # Per-tensor, per-channel, per-group quantization
├── ptq.py                # Post-Training Quantization
├── qat.py                # QAT with FakeQuantize and STE (PyTorch autograd)
├── gptq.py               # GPTQ column-wise Hessian-guided update
├── fp32_format.py        # IEEE 754 FP32 bit decomposition utilities
├── benefits.py           # Memory/compute analysis (Requirement 1)
├── ui/
│   ├── app.py            # Streamlit dashboard entry point
│   └── panels/
│       ├── fp32_panel.py
│       ├── pipeline_panel.py
│       ├── distribution_panel.py
│       ├── benefits_panel.py
│       ├── rounding_clipping_panel.py
│       ├── schemes_panel.py
│       ├── calibration_panel.py
│       ├── granularity_panel.py
│       ├── dynamic_static_panel.py
│       ├── activation_panel.py
│       ├── mixed_precision_panel.py
│       ├── formula_stepper_panel.py
│       ├── ptq_panel.py
│       ├── qat_panel.py
│       └── gptq_panel.py
└── tests/
    ├── test_quantizer.py
    ├── test_schemes.py
    ├── test_calibrator.py
    ├── test_granularity.py
    ├── test_ptq.py
    ├── test_qat.py
    ├── test_gptq.py
    └── test_fp32.py
```

### Dependency Graph

```
benefits.py
    (no internal deps)

fp32_format.py
    (no internal deps)

schemes.py
    (no internal deps)

quantizer.py
    └── schemes.py

calibrator.py
    ├── quantizer.py
    └── schemes.py

granularity.py
    ├── quantizer.py
    └── schemes.py

ptq.py
    ├── quantizer.py
    ├── calibrator.py
    └── granularity.py

qat.py          (PyTorch only)
    └── quantizer.py  (logic mirrored in torch ops)

gptq.py
    ├── quantizer.py
    └── granularity.py

ui/app.py
    └── ui/panels/*.py  (each panel imports its relevant core module)
```

---

## Components and Interfaces

### `quantizer.py`

Provides the two core public functions used by all other modules.

```python
def quantize(
    x: np.ndarray,
    scale: float,
    zero_point: int,
    bits: int,
    signed: bool = True,
) -> np.ndarray:
    """
    Map floating-point tensor to integer tensor.
    Raises ValueError if scale <= 0, bits not in {2,4,8},
    or zero_point is outside [q_min, q_max].
    Returns integer ndarray with dtype int32.
    """

def dequantize(
    q: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """
    Reconstruct floating-point tensor from integer tensor.
    Returns float32 ndarray with the same shape as q.
    """

def q_min_max(bits: int, signed: bool) -> tuple[int, int]:
    """Return (q_min, q_max) for the given bit-width and signedness."""
```

### `schemes.py`

```python
from enum import Enum

class QuantizationScheme(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"

def compute_scale_zero_point(
    x: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
) -> tuple[float, int]:
    """
    Compute (scale, zero_point) for the given tensor and scheme.
    Symmetric: scale = max(|x_min|, |x_max|) / (2^(bits-1) - 1), zp = 0
    Asymmetric: scale = (x_max - x_min) / (2^bits - 1),
                zp = clamp(round(-x_min / scale), 0, 2^bits - 1)
    """

@dataclass
class SchemeComparisonReport:
    symmetric_scale: float
    symmetric_zero_point: int
    symmetric_mae: float
    asymmetric_scale: float
    asymmetric_zero_point: int
    asymmetric_mae: float

def compare_schemes(x: np.ndarray, bits: int) -> SchemeComparisonReport: ...
def format_scheme_comparison(report: SchemeComparisonReport) -> str: ...
```

### `calibrator.py`

```python
from enum import Enum

class CalibrationMethod(Enum):
    MIN_MAX = "min_max"
    PERCENTILE = "percentile"
    ENTROPY = "entropy"

def calibrate_min_max(
    tensors: list[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
) -> tuple[float, int]: ...

def calibrate_percentile(
    tensors: list[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
    percentile: float = 99.0,
) -> tuple[float, int]: ...

def calibrate_entropy(
    tensors: list[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
) -> tuple[float, int]: ...
```

### `granularity.py`

```python
from enum import Enum

class GranularityMode(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"

def quantize_per_tensor(
    W: np.ndarray, bits: int, scheme: QuantizationScheme
) -> tuple[np.ndarray, float, int]: ...

def quantize_per_channel(
    W: np.ndarray, bits: int, scheme: QuantizationScheme
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def quantize_per_group(
    W: np.ndarray, bits: int, scheme: QuantizationScheme, group_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

@dataclass
class GranularityComparisonReport:
    granularity: GranularityMode
    num_scale_params: int
    mean_absolute_error: float
    max_absolute_error: float
```

### `ptq.py`

```python
def ptq_quantize_layer(
    W: np.ndarray,
    b: np.ndarray,
    calibration_inputs: list[np.ndarray],
    bits: int,
    scheme: QuantizationScheme,
    granularity: GranularityMode,
    group_size: int = 64,
) -> "PTQLayer": ...

@dataclass
class PTQLayer:
    W_q: np.ndarray          # quantized weight integers
    scales: np.ndarray       # scale(s) for dequantization
    zero_points: np.ndarray  # zero-point(s)
    b: np.ndarray            # bias (float32, unquantized)
    bits: int
    scheme: QuantizationScheme
    granularity: GranularityMode

def ptq_infer(layer: PTQLayer, x: np.ndarray) -> np.ndarray:
    """output = dequantize(W_q) @ x.T + b"""
```

### `qat.py` (PyTorch)

```python
import torch
import torch.nn as nn

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, bits, signed=True):
        """quantize then dequantize; save clamp bounds in ctx for backward."""

    @staticmethod
    def backward(ctx, grad_output):
        """STE: pass gradient unchanged where input was in [q_min*scale, q_max*scale],
           zero otherwise."""

class FakeQuantizeLayer(nn.Module):
    """Wraps FakeQuantize with learnable or fixed scale/zero_point."""
```

### `gptq.py`

```python
def compute_hessian(X: np.ndarray) -> np.ndarray:
    """H = 2 * X.T @ X  where X is [n_samples, in_features]."""

def gptq_quantize(
    W: np.ndarray,
    H: np.ndarray,
    bits: int,
    scheme: QuantizationScheme,
    damp_factor: float = 0.01,
) -> np.ndarray:
    """
    Column-wise GPTQ quantization.
    Applies Hessian damping: H_damped = H + lambda * I
    where lambda = damp_factor * mean(diag(H)).
    Iterates exactly in_features columns.
    """
```

### `fp32_format.py`

```python
def decompose_fp32(value: float) -> dict:
    """
    Returns {
        'sign': int,           # 0 or 1
        'exponent_bits': str,  # 8-char binary string
        'mantissa_bits': str,  # 23-char binary string
        'biased_exponent': int,
        'true_exponent': int,
        'reconstructed': float,
    }
    """

def reconstruct_fp32(sign: int, exponent_bits: str, mantissa_bits: str) -> float:
    """Reconstruct float from IEEE 754 components."""
```

### `benefits.py`

```python
DTYPE_BYTES = {"float32": 4, "float16": 2, "int8": 1, "int4": 0.5}

def memory_footprint(n_elements: int, dtype: str) -> float: ...
def compression_ratio(shape: tuple, dtype_a: str, dtype_b: str) -> float: ...
def mac_count(in_features: int, out_features: int) -> int: ...
def model_memory_table(param_count: int) -> dict[str, float]: ...
```

---

## Data Models

### Enumerations

```python
class QuantizationScheme(Enum):
    SYMMETRIC  = "symmetric"
    ASYMMETRIC = "asymmetric"

class GranularityMode(Enum):
    PER_TENSOR  = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP   = "per_group"

class CalibrationMethod(Enum):
    MIN_MAX    = "min_max"
    PERCENTILE = "percentile"
    ENTROPY    = "entropy"
```

### Core Dataclasses

```python
@dataclass
class QuantizationParams:
    scale:       float                 # positive float
    zero_point:  int                   # integer in [q_min, q_max]
    bits:        int                   # one of {2, 4, 8}
    scheme:      QuantizationScheme

@dataclass
class SchemeComparisonReport:
    symmetric_scale:      float
    symmetric_zero_point: int
    symmetric_mae:        float
    asymmetric_scale:     float
    asymmetric_zero_point: int
    asymmetric_mae:       float

@dataclass
class GranularityComparisonReport:
    granularity:         GranularityMode
    num_scale_params:    int
    mean_absolute_error: float
    max_absolute_error:  float

@dataclass
class PTQLayer:
    W_q:         np.ndarray
    scales:      np.ndarray
    zero_points: np.ndarray
    b:           np.ndarray
    bits:        int
    scheme:      QuantizationScheme
    granularity: GranularityMode
```

### Type Aliases

```python
Tensor   = np.ndarray           # float32 ndarray
QTensor  = np.ndarray           # int32 ndarray (quantized)
ScaleVec = np.ndarray           # 1-D float32 array of scale values
ZPVec    = np.ndarray           # 1-D int32 array of zero-point values
```

### Integer Range Lookup

| bits | signed | q_min | q_max |
|------|--------|-------|-------|
| 2    | no     | 0     | 3     |
| 2    | yes    | -2    | 1     |
| 4    | no     | 0     | 15    |
| 4    | yes    | -8    | 7     |
| 8    | no     | 0     | 255   |
| 8    | yes    | -128  | 127   |

---

## Core Algorithm Designs

### Quantize / Dequantize

```
# Forward (quantize)
q_min, q_max = q_min_max(bits, signed)
q = clamp(round(x / scale) + zero_point, q_min, q_max)

# Reverse (dequantize)
x_reconstructed = (q - zero_point) * scale
```

The `round()` applies NumPy banker's rounding (round-half-to-even); this matches IEEE 754 default rounding mode.

### Symmetric Scale Computation

```
abs_max = max(|x_min|, |x_max|)
scale   = abs_max / (2^(bits-1) - 1)
zp      = 0
```

The signed range `[-(2^(bits-1)), 2^(bits-1)-1]` is asymmetric by 1 step, but fixing `zp=0` ensures symmetric integer representation around zero, which is preferred for weights.

### Asymmetric Scale / Zero-Point Computation

```
scale      = (x_max - x_min) / (2^bits - 1)
zp_float   = -x_min / scale
zero_point = clamp(round(zp_float), 0, 2^bits - 1)
```

Unsigned range `[0, 2^bits - 1]` is used for asymmetric by convention (activation-friendly).

### Percentile Calibration

```
p_low  = (100 - p) / 2          # e.g., 0.5 for p=99
p_high = (100 + p) / 2          # e.g., 99.5 for p=99
r_min  = percentile(all_values, p_low)
r_max  = percentile(all_values, p_high)
# then apply symmetric or asymmetric formula to (r_min, r_max)
```

### Entropy Calibration (KL-Divergence Minimization)

```
all_values = concatenate all calibration tensors
hist_orig, bin_edges = np.histogram(all_values, bins=2048)

best_range = None
best_kl    = +inf
for each candidate (r_min, r_max) in search_grid:
    scale, zp = compute_scale_zero_point_from_range(r_min, r_max, bits)
    q_vals    = quantize(all_values, scale, zp, bits)
    x_recon   = dequantize(q_vals, scale, zp)
    hist_q, _ = np.histogram(x_recon, bins=bin_edges)
    kl        = scipy.special.kl_div(hist_orig + 1e-8, hist_q + 1e-8).sum()
    if kl < best_kl:
        best_kl    = kl
        best_range = (r_min, r_max)

scale, zp = compute_scale_zero_point_from_range(*best_range, bits)
```

The search grid spans from `[p0, p100]` (full range) down to `[p40, p60]` (tight range) in 32 logarithmic steps.

### GPTQ Column-Wise Update

```
H         = 2 * X.T @ X                         # [in, in]
lambda    = 0.01 * mean(diag(H))
H_damped  = H + lambda * I                       # damping

W_out     = W.copy()
for i in range(in_features):
    if H_damped[i, i] < 1e-8:
        continue                                  # skip near-zero diagonal
    W_q_col        = quantize_col(W_out[:, i])
    q_error_col    = W_q_col - W_out[:, i]        # [out]
    for j in range(i+1, in_features):
        W_out[:, j] -= (q_error_col / H_damped[i, i]) * H_damped[i, j]
    W_out[:, i]    = W_q_col

return W_out
```

The loop executes exactly `in_features` iterations (Requirement 8.8).

### FakeQuantize Forward + STE Backward (QAT)

```python
# Forward
q_min_val = q_min * scale   # float lower bound
q_max_val = q_max * scale   # float upper bound
q = quantize(x, scale, zero_point, bits)
x_hat = dequantize(q, scale, zero_point)
ctx.save_for_backward(x, torch.tensor([q_min_val, q_max_val]))
return x_hat

# Backward (STE)
x, bounds = ctx.saved_tensors
q_min_val, q_max_val = bounds[0].item(), bounds[1].item()
mask = (x >= q_min_val) & (x <= q_max_val)  # True where in-range
grad_input = grad_output * mask.float()       # pass through in-range, zero out-of-range
return grad_input, None, None, None, None
```

### IEEE 754 FP32 Decomposition

```python
import struct

def decompose_fp32(value: float) -> dict:
    bits_int = struct.unpack('>I', struct.pack('>f', value))[0]
    sign     = (bits_int >> 31) & 1
    exponent = (bits_int >> 23) & 0xFF         # biased exponent
    mantissa =  bits_int        & 0x7FFFFF     # 23 bits
    true_exp = exponent - 127
    # Reconstruct: (-1)^sign * 2^true_exp * (1 + mantissa/2^23)
    return {
        'sign': sign,
        'exponent_bits': format(exponent, '08b'),
        'mantissa_bits': format(mantissa, '023b'),
        'biased_exponent': exponent,
        'true_exponent': true_exp,
    }
```

---

## UI Architecture

### Dashboard (`ui/app.py`)

The dashboard is a single Streamlit application launched with `streamlit run ui/app.py`. Panel switching is handled via `st.session_state["active_panel"]` — setting this key and calling `st.rerun()` is the only supported navigation mechanism, so no full page reload occurs during panel transitions.

```python
# ui/app.py skeleton
import streamlit as st
from ui.panels import (
    fp32_panel, pipeline_panel, distribution_panel, benefits_panel,
    rounding_clipping_panel, schemes_panel, calibration_panel,
    granularity_panel, dynamic_static_panel, activation_panel,
    mixed_precision_panel, formula_stepper_panel,
    ptq_panel, qat_panel, gptq_panel,
)

PANELS = {
    "FP32 Format":          fp32_panel.render,
    "Quantization Pipeline": pipeline_panel.render,
    "Weight Distribution":  distribution_panel.render,
    "Benefits Calculator":  benefits_panel.render,
    "Rounding vs Clipping": rounding_clipping_panel.render,
    "Mapping Schemes":      schemes_panel.render,
    "Calibration Methods":  calibration_panel.render,
    "Granularity Explorer": granularity_panel.render,
    "Dynamic vs Static":    dynamic_static_panel.render,
    "Activation Quantization": activation_panel.render,
    "Mixed Precision":      mixed_precision_panel.render,
    "Formula Stepper":      formula_stepper_panel.render,
    "PTQ Demo":             ptq_panel.render,
    "QAT Demo":             qat_panel.render,
    "GPTQ Demo":            gptq_panel.render,
}

def main():
    st.set_page_config(layout="wide", page_title="AI Model Quantization")
    if "active_panel" not in st.session_state:
        st.session_state["active_panel"] = "FP32 Format"
    if "visited" not in st.session_state:
        st.session_state["visited"] = set()

    with st.sidebar:
        st.title("Quantization Modules")
        for name in PANELS:
            label = f"✅ {name}" if name in st.session_state["visited"] else name
            if st.button(label, key=f"nav_{name}"):
                st.session_state["active_panel"] = name
                st.rerun()
        st.divider()
        if st.button("🗺️ Concept Map"):
            st.session_state["active_panel"] = "__concept_map__"
            st.rerun()

    active = st.session_state["active_panel"]
    st.session_state["visited"].add(active)
    if active == "__concept_map__":
        render_concept_map()
    else:
        PANELS[active]()
```

### Progress Tracker

`st.session_state["visited"]` is a `set[str]` of panel names the user has navigated to. Sidebar buttons prefix a checkmark (`✅`) for visited panels.

### Concept Map View

Rendered as a static Plotly network graph. Nodes are the 15 panels; edges group them into 5 topic clusters:

| Topic | Panels |
|---|---|
| Why Quantize | Benefits Calculator, FP32 Format |
| Equation | Quantization Pipeline, Formula Stepper, Weight Distribution |
| Mapping & Calibration | Mapping Schemes, Calibration Methods, Rounding vs Clipping |
| Granularity | Granularity Explorer, Dynamic vs Static, Activation Quantization, Mixed Precision |
| Algorithms | PTQ Demo, QAT Demo, GPTQ Demo |

### Panel Structure Convention

Each panel module exports a single `render()` function. Panels follow this layout:

```
st.header("Panel Title")        # uniform font/size
st.write("Learning objective…") # one-sentence description
# --- interactive controls (st.sidebar or inline) ---
# --- Plotly chart or metric display ---
# --- optional: code snippet or formula display ---
```

Color theme: dark background (`#0e1117`), accent blue (`#4f8bf9`), warning red (`#ff4b4b`). All charts use `plotly_dark` template.

### Panel Responsibilities (15 panels)

| Panel | Key Interactions | Core Module Used |
|---|---|---|
| FP32 Format | float text input → colored bit grid + INT8 comparison | `fp32_format.py` |
| Quantization Pipeline | r/s/z/bits/scheme inputs → animated flow diagram | `quantizer.py` |
| Weight Distribution | bit-width/scheme selectors → dual histograms | `quantizer.py`, `schemes.py` |
| Benefits Calculator | model preset/custom → GB comparison table + bar chart | `benefits.py` |
| Rounding vs Clipping | range slider → number line + 3 live error metrics | `quantizer.py` |
| Mapping Schemes | tensor slider → side-by-side number line | `schemes.py` |
| Calibration Methods | calibration data → 3 histograms with range overlays | `calibrator.py` |
| Granularity Explorer | weight heatmap → colored scale regions | `granularity.py` |
| Dynamic vs Static | input sequence + distribution shift button | `quantizer.py` |
| Activation Quantization | weight/activation toggles → 3-way output comparison | `quantizer.py` |
| Mixed Precision | per-layer bit selectors → memory chart + error score | `quantizer.py`, `benefits.py` |
| Formula Stepper | next/prev buttons → step-by-step substitution | `quantizer.py` |
| PTQ Demo | layer config + calibration samples → error table | `ptq.py` |
| QAT Demo | training run → live loss curve vs PTQ baseline | `qat.py` |
| GPTQ Demo | weight matrix → animated column heatmap | `gptq.py` |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

Before writing the final properties, redundant candidates are identified and consolidated:

- **2.4 and 9.1** both state the round-trip error bound `|x - dequantize(quantize(x))| <= 0.5 * scale`. These collapse into **Property 1**.
- **2.5, 5.4** both state the shape-invariance of quantize/dequantize. Collapsed into **Property 2**.
- **3.3 and 9.6** both state `symmetric → zp == 0`. Collapsed into **Property 4**.
- **4.4 and 4.7** both state `scale > 0` for all calibrators. Collapsed into **Property 5**.
- **5.5 and 5.6** together form the granularity monotonicity property. Unified into **Property 6**.
- **7.1 and 7.6 and 9.5** all state FakeQuantize shape invariant. Collapsed into **Property 7**.
- **7.2, 7.3, 7.4** together describe the STE gradient. Collapsed into **Property 8**.
- **9.2 and 2.1** both verify that quantize outputs lie in `[q_min, q_max]`. Collapsed into **Property 3**.
- **10.4 and 10.5** together describe the FP32 round-trip. Collapsed into **Property 10** (IEEE 754 decompose-reconstruct round-trip).
- **8.4 and 8.5** are kept distinct: 8.4 is a shape invariant (Property 9a), 8.5 is the quality property (Property 9b).

---

### Property 1: Round-Trip Error Bound

*For any* floating-point tensor where all elements satisfy `q_min * scale <= x <= q_max * scale`, and for any valid combination of `scale > 0`, `bits ∈ {2, 4, 8}`, and `zero_point ∈ [q_min, q_max]`, applying quantize then dequantize SHALL produce a tensor where every element differs from the original by at most `0.5 * scale`.

**Validates: Requirements 2.4, 9.1**

---

### Property 2: Shape Invariance of Quantize and Dequantize

*For any* floating-point tensor of arbitrary shape and any valid quantization parameters, the output of `dequantize(quantize(x))` SHALL have the same `numpy.ndarray.shape` as the input tensor `x`.

**Validates: Requirements 2.5, 5.4**

---

### Property 3: Output Range Invariant

*For any* floating-point tensor and valid parameters (`scale > 0`, `bits ∈ {2, 4, 8}`, `zero_point ∈ [q_min, q_max]`), every element of the quantized output `q` SHALL satisfy `q_min <= q <= q_max` where `q_min` and `q_max` are derived from `bits` and `signed`.

**Validates: Requirements 2.1, 9.2**

---

### Property 4: Symmetric Scheme Zero-Point Invariant

*For any* floating-point tensor and any `bits ∈ {2, 4, 8}`, computing scale and zero-point with `QuantizationScheme.SYMMETRIC` SHALL always return `zero_point == 0`, regardless of the tensor's values.

**Validates: Requirements 3.3, 9.6**

---

### Property 5: Calibration Scale Positivity

*For any* non-empty list of floating-point calibration tensors, any `bits ∈ {2, 4, 8}`, and any valid scheme, each of the three calibrators (MinMax, Percentile with `p ∈ (0, 100)`, and Entropy) SHALL return a `scale` value that is strictly greater than zero.

**Validates: Requirements 4.4, 4.7, 9.3**

---

### Property 6: Granularity Monotonicity

*For any* weight tensor of shape `[out_channels, in_channels]` where the per-row standard deviation is non-zero for at least two distinct rows, and for any group size `g` that evenly divides `in_channels`, the mean absolute quantization error SHALL satisfy: `MAE(PER_TENSOR) >= MAE(PER_CHANNEL) >= MAE(PER_GROUP)`.

**Validates: Requirements 5.5, 5.6, 9.4**

---

### Property 7: FakeQuantize Shape Invariant

*For any* floating-point input tensor of arbitrary shape and valid parameters (`scale > 0`, `bits ∈ {2, 4, 8}`, `zero_point ∈ [q_min, q_max]`), the output of `FakeQuantize.apply(x, scale, zero_point, bits)` SHALL have the same `torch.Size` as the input tensor.

**Validates: Requirements 7.1, 7.6, 9.5**

---

### Property 8: Straight-Through Estimator (STE) Gradient

*For any* scalar input `x` and valid parameters, the gradient of the `FakeQuantize` output with respect to `x` SHALL equal exactly `1.0` when `q_min * scale <= x <= q_max * scale`, and exactly `0.0` when `x < q_min * scale` or `x > q_max * scale`.

**Validates: Requirements 7.2, 7.3, 7.4**

---

### Property 9: GPTQ Output Shape and Quality

*For any* weight matrix `W` of shape `[out, in]` (with `out >= 4` and `in >= 4`) and a Hessian matrix `H` of shape `[in, in]`, the GPTQ-quantized output SHALL:
(a) have the same shape `[out, in]` as the input `W`, and  
(b) produce a mean absolute element-wise error strictly less than the mean absolute error of the same `W` quantized via naive per-tensor PTQ, when at least 10 representative input samples are used to compute the Hessian.

**Validates: Requirements 8.4, 8.5**

---

### Property 10: IEEE 754 Decompose-Reconstruct Round-Trip

*For any* finite `float32` value, decomposing it into its sign bit, 8-bit biased exponent, and 23-bit mantissa using `decompose_fp32`, and then reconstructing the value with `reconstruct_fp32`, SHALL produce a value equal to the original within `float32` precision (i.e., `abs(reconstruct(decompose(x)) - x) <= eps` for `eps = np.finfo(np.float32).eps * abs(x)`).

**Validates: Requirements 10.4, 10.5**

---

### Property 11: Memory Footprint Formula

*For any* element count `n >= 1` and any supported dtype `d ∈ {float32, float16, int8, int4}`, `memory_footprint(n, d)` SHALL equal `n * DTYPE_BYTES[d]`.

**Validates: Requirements 1.1, 1.5**

---

### Property 12: Compression Ratio Invariant

*For any* valid tensor shape, the compression ratio between `float32` and `int8` representations SHALL equal exactly `4.0`.

**Validates: Requirement 1.2**

---

### Property 13: MAC Count Formula

*For any* positive integers `in_features` and `out_features`, `mac_count(in_features, out_features)` SHALL equal `in_features * out_features`.

**Validates: Requirement 1.4**

---

### Property 14: Finer Scale Yields Lower Error

*For any* floating-point tensor and any pair of positive scales `s1 < s2` with the same `bits` and `zero_point = 0`, where all elements of the tensor lie within the representable range for both scales, quantizing with scale `s1` SHALL produce a mean absolute quantization error less than or equal to that produced by scale `s2`.

**Validates: Requirement 9.7**

---

### Property 15: Quantization Idempotence

*For any* floating-point tensor where all elements satisfy `q_min <= round(x/scale) + zero_point <= q_max`, and for any valid parameters (`scale > 0`, `bits ∈ {2, 4, 8}`, `zero_point ∈ [q_min, q_max]`), applying quantize → dequantize → quantize SHALL produce the same integer tensor as applying quantize once: `quantize(dequantize(quantize(x))) == quantize(x)`.

**Validates: Requirement 9.8**

---

## Error Handling

### ValueError Contract

All public functions follow a consistent `ValueError`-on-invalid-input contract. The error message must always identify the invalid parameter by name and state the valid range or set.

| Function | Invalid Condition | Message must contain |
|---|---|---|
| `quantize` | `scale <= 0` | `"scale"` and indication of positive requirement |
| `quantize` | `bits not in {2,4,8}` | `"bits"` and `"2, 4, 8"` |
| `quantize` | `zero_point out of [q_min, q_max]` | `"zero_point"`, `q_min`, `q_max` |
| `memory_footprint` | unsupported dtype | dtype name and list of supported dtypes |
| `compression_ratio` | mismatched shapes | both shapes |
| `calibrate_*` | empty tensor list | `"at least one"` |
| `calibrate_percentile` | `p <= 0` or `p >= 100` | `"(0, 100)"` |
| `quantize_per_group` | `in_channels % g != 0` | `in_channels`, `g`, divisibility requirement |
| `ptq_quantize_layer` | fewer than 1 calibration input | `"at least 1"` |

### Module-Level Error Propagation

- `ptq.py`, `gptq.py`, and `calibrator.py` call `quantize()` internally; any `ValueError` raised by `quantize()` propagates to the caller unchanged.
- UI panels catch `ValueError` from core modules and display them as `st.error(str(e))` rather than letting Streamlit show a raw traceback.
- GPTQ silently skips column `i` if `H_damped[i,i] < 1e-8`; this is not an error but a documented special case.

### Numerical Edge Cases

- **Scale underflow**: `max(|x_min|, |x_max|) == 0` → symmetric scale would be 0. Handle by raising `ValueError("scale would be zero: input tensor is all-zeros")`.
- **Constant tensor for asymmetric**: `x_min == x_max` → scale would be 0. Handle by setting `scale = 1.0` and `zero_point = 0` (identity mapping), and document this in the function docstring.
- **NaN/Inf in input**: `quantize` should raise `ValueError` if any element is NaN or Inf, since the clamping result would be implementation-defined.

---

## Testing Strategy

### Dual Testing Approach

The test suite uses two complementary approaches:

1. **Property-based tests** (Hypothesis) — validate universal properties across 100+ generated inputs per test. Each test maps 1:1 to a named property in this document.
2. **Example-based unit tests** (pytest) — validate specific behaviors, error conditions, UI rendering, and integration points with fixed inputs.

### Property-Based Tests (Hypothesis)

**Library**: `hypothesis` with `hypothesis.extra.numpy` strategies for array generation.  
**Minimum runs**: 100 iterations per property (configured via `@settings(max_examples=100)`).  
**Tag format**: Each test decorated with a comment `# Feature: ai-model-quantization, Property N: <property_text>`.

#### Hypothesis Strategies

```python
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes

# Strategy: valid quantization params
valid_bits    = st.sampled_from([2, 4, 8])
valid_scale   = st.floats(min_value=1e-4, max_value=1e4, allow_nan=False, allow_infinity=False)
signed_flag   = st.booleans()

# Strategy: in-range float tensor
@st.composite
def in_range_tensor(draw):
    bits   = draw(valid_bits)
    signed = draw(signed_flag)
    scale  = draw(valid_scale)
    q_lo, q_hi = q_min_max(bits, signed)
    lo = q_lo * scale
    hi = q_hi * scale
    shape = draw(st.tuples(st.integers(1, 8), st.integers(1, 8)))
    arr = draw(arrays(np.float32, shape,
                      elements=st.floats(lo, hi, allow_nan=False)))
    zp = 0 if not signed else draw(st.integers(q_lo, q_hi))
    return arr, scale, zp, bits, signed

# Strategy: 2-D weight tensor with non-uniform row std
@st.composite
def non_uniform_weight_tensor(draw):
    out_ch = draw(st.integers(2, 8))
    in_ch  = draw(st.integers(4, 16))
    stds   = draw(arrays(np.float32, out_ch,
                         elements=st.floats(0.01, 2.0, allow_nan=False)))
    rows   = [np.random.normal(0, s, in_ch).astype(np.float32) for s in stds]
    return np.stack(rows)
```

#### Property Test Mapping

| Property | Test Function | File |
|---|---|---|
| Property 1: Round-trip error bound | `test_round_trip_error_bound` | `test_quantizer.py` |
| Property 2: Shape invariance | `test_shape_invariance` | `test_quantizer.py` |
| Property 3: Output range invariant | `test_output_range_invariant` | `test_quantizer.py` |
| Property 4: Symmetric zp = 0 | `test_symmetric_zero_point` | `test_schemes.py` |
| Property 5: Scale positivity | `test_calibration_scale_positive` | `test_calibrator.py` |
| Property 6: Granularity monotonicity | `test_granularity_monotonicity` | `test_granularity.py` |
| Property 7: FakeQuantize shape invariant | `test_fake_quantize_shape` | `test_qat.py` |
| Property 8: STE gradient | `test_ste_gradient` | `test_qat.py` |
| Property 9: GPTQ shape + quality | `test_gptq_shape_and_quality` | `test_gptq.py` |
| Property 10: FP32 round-trip | `test_fp32_round_trip` | `test_fp32.py` |
| Property 11: Memory footprint formula | `test_memory_footprint_formula` | `test_quantizer.py` |
| Property 12: Compression ratio | `test_compression_ratio` | `test_quantizer.py` |
| Property 13: MAC count formula | `test_mac_count_formula` | `test_quantizer.py` |
| Property 14: Finer scale yields lower error | `test_finer_scale_lower_error` | `test_quantizer.py` |
| Property 15: Idempotence | `test_idempotence` | `test_quantizer.py` |

### Example-Based Unit Tests

These cover edge cases, error conditions, and UI presence:

- **Error conditions**: `test_invalid_scale`, `test_invalid_bits`, `test_invalid_zp`, `test_empty_calibration_list`, `test_invalid_percentile`, `test_invalid_group_size`, `test_mismatched_shapes`, `test_unsupported_dtype`
- **q_min/q_max lookup**: verify all 6 `(bits, signed)` combinations
- **Scheme comparison report**: verify `format_scheme_comparison` returns a table with correct column headers
- **Granularity report**: verify `GranularityComparisonReport` table formatting
- **GPTQ Hessian**: verify `compute_hessian` matches `2 * X.T @ X` for a small known `X`
- **GPTQ iteration count**: mock the inner loop to verify exactly `in_features` iterations execute
- **GPTQ near-zero diagonal**: verify column is skipped without error when `H[i,i] < 1e-8`
- **PTQ inference shape**: verify `ptq_infer` output shape `[out, batch_size]`
- **QAT training convergence**: integration test — 100 steps of QAT on a fixed dataset, verify QAT final error < PTQ baseline error
- **UI panels**: `test_panel_renders` — import each panel module and call `render()` under a mock Streamlit context

### Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.hypothesis]
max_examples = 100
deriving = "best_effort"
```

### Dependencies

```toml
[project]
dependencies = [
    "numpy==1.26.4",
    "scipy==1.13.0",
    "torch==2.3.0",
    "streamlit==1.35.0",
    "plotly==5.22.0",
    "hypothesis==6.103.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.2.0",
    "pytest-cov==5.0.0",
]
```
