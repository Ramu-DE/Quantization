# Implementation Plan: AI Model Quantization

## Overview

Build a from-scratch Python quantization library structured as Jupyter-style `.py` files (using `# %%` cell markers throughout), with a Hypothesis property-based test suite and a 15-panel Streamlit dashboard. Every source file — except the Streamlit UI files — uses `# %%` cell delimiters so it can run interactively in VS Code's Jupyter window and be converted to `.ipynb`.

All core modules live under `quantization/`. Tests live under `quantization/tests/`. The Streamlit app (`ui/app.py`) and its panel files are plain `.py` files without cell markers.

---

## Tasks

- [ ] 1. Project scaffold and package configuration
  - Create `quantization/` package directory with `__init__.py`
  - Create `quantization/tests/` directory with `__init__.py`
  - Create `quantization/ui/` and `quantization/ui/panels/` directories
  - Add `pyproject.toml` with pinned dependencies: `numpy==1.26.4`, `scipy==1.13.0`, `torch==2.3.0`, `streamlit==1.35.0`, `plotly==5.22.0`, `hypothesis==6.103.0`, `pytest==8.2.0`, `pytest-cov==5.0.0`
  - Configure `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `[tool.hypothesis]` with `max_examples = 100`
  - _Requirements: 2 (core equation), 9 (correctness properties)_


- [ ] 2. Implement `quantization/benefits.py` (Jupyter-style)
  - [ ] 2.1 Write `benefits.py` core functions
    - Create file with `# %% [markdown]` header cell, then `# %%` import cell (`numpy`)
    - Define `DTYPE_BYTES = {"float32": 4, "float16": 2, "int8": 1, "int4": 0.5}` in its own cell
    - Implement `memory_footprint(n_elements, dtype) -> float` — raises `ValueError` if dtype unsupported (message must contain dtype name and list of supported dtypes)
    - Implement `compression_ratio(shape_a, shape_b, dtype_a, dtype_b) -> float` — raises `ValueError` if shapes differ
    - Implement `mac_count(in_features, out_features) -> int`
    - Implement `model_memory_table(param_count) -> dict[str, float]` — returns GB for all four dtypes
    - Add a `# %% --- Demo cell ---` showing Llama-70B (70e9 params) memory table and float32-vs-int8 compression ratio
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  - [ ]* 2.2 Write property tests for `benefits.py` in `tests/test_quantizer.py`
    - **Property 11: Memory Footprint Formula** — `memory_footprint(n, d) == n * DTYPE_BYTES[d]` for all `n >= 1` and `d in DTYPE_BYTES`
    - **Validates: Requirements 1.1, 1.5**
    - **Property 12: Compression Ratio Invariant** — `compression_ratio(shape, shape, "float32", "int8") == 4.0` for any valid shape
    - **Validates: Requirement 1.2**
    - **Property 13: MAC Count Formula** — `mac_count(in_f, out_f) == in_f * out_f` for all positive integers
    - **Validates: Requirement 1.4**
    - Use `# %%` cell per test function; annotate each with `# Property N` comment


- [ ] 3. Implement `quantization/fp32_format.py` (Jupyter-style)
  - [ ] 3.1 Write `fp32_format.py` decompose/reconstruct utilities
    - Create with `# %% [markdown]` header; import `struct`, `numpy` in `# %%` cell
    - Implement `decompose_fp32(value: float) -> dict` using `struct.pack/unpack('>f')` / `'>I'`; return keys: `sign`, `exponent_bits` (8-char binary string), `mantissa_bits` (23-char binary string), `biased_exponent`, `true_exponent`, `reconstructed`
    - Implement `reconstruct_fp32(sign, exponent_bits, mantissa_bits) -> float` — inverts decompose
    - Add demo cell: decompose `-0.10985`, print all fields, verify `reconstructed == -0.10985`
    - _Requirements: 10.4, 10.5_
  - [ ]* 3.2 Write property test for `fp32_format.py` in `tests/test_fp32.py`
    - **Property 10: IEEE 754 Decompose-Reconstruct Round-Trip** — for all finite float32, `abs(reconstruct(decompose(x)) - x) <= eps * abs(x)`
    - **Validates: Requirements 10.4, 10.5**

- [ ] 4. Implement `quantization/schemes.py` (Jupyter-style)
  - [ ] 4.1 Write `schemes.py` enum, scale/zero-point computation, comparison report
    - Create with header cell; import `numpy`, `dataclasses`, `enum` in `# %%` cell
    - Define `QuantizationScheme(Enum)` with `SYMMETRIC` and `ASYMMETRIC` members
    - Define `@dataclass SchemeComparisonReport` with fields: `symmetric_scale`, `symmetric_zero_point`, `symmetric_mae`, `asymmetric_scale`, `asymmetric_zero_point`, `asymmetric_mae`
    - Implement `compute_scale_zero_point(x, bits, scheme) -> tuple[float, int]`; symmetric: `scale = max(|x_min|, |x_max|) / (2^(bits-1) - 1)`, `zp = 0`; asymmetric: `scale = (x_max - x_min) / (2^bits - 1)`, `zp = clamp(round(-x_min / scale), 0, 2^bits - 1)`; handle constant-tensor edge case (set `scale=1.0, zp=0`); handle all-zeros tensor (raise `ValueError`)
    - Implement `compare_schemes(x, bits) -> SchemeComparisonReport`
    - Implement `format_scheme_comparison(report) -> str` — returns a human-readable table with columns: Scheme, Scale, Zero_Point, Mean_Absolute_Error
    - Add demo cell comparing symmetric vs asymmetric on a ReLU-shaped tensor (all positive)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7_
  - [ ]* 4.2 Write property test for `schemes.py` in `tests/test_schemes.py`
    - **Property 4: Symmetric Scheme Zero-Point Invariant** — for all tensors and `bits in {2,4,8}`, `compute_scale_zero_point(x, bits, SYMMETRIC)` returns `zp == 0`
    - **Validates: Requirements 3.3, 9.6**


- [ ] 5. Implement `quantization/quantizer.py` (Jupyter-style)
  - [ ] 5.1 Write `quantizer.py` core quantize/dequantize functions
    - Create with header cell; import `numpy` in `# %%` cell
    - Implement `q_min_max(bits: int, signed: bool) -> tuple[int, int]` covering all 6 combinations (2/4/8 × signed/unsigned); raise `ValueError` if `bits not in {2,4,8}`
    - Implement `quantize(x, scale, zero_point, bits, signed=True) -> np.ndarray` — raises `ValueError` for `scale <= 0`, invalid `bits`, `zero_point` out of range, or NaN/Inf in `x`; applies `clamp(round(x/scale) + zero_point, q_min, q_max)`; returns `int32` ndarray
    - Implement `dequantize(q, scale, zero_point) -> np.ndarray` — `(q - zero_point) * scale`; returns `float32` ndarray of same shape as `q`
    - Add demo cell showing quantize + dequantize round-trip on `[-1.5, -0.5, 0.0, 0.5, 1.5]`
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7_
  - [ ]* 5.2 Write property tests for `quantizer.py` in `tests/test_quantizer.py`
    - **Property 1: Round-Trip Error Bound** — for in-range tensors, `|x - dequantize(quantize(x))| <= 0.5 * scale` elementwise
    - **Validates: Requirements 2.4, 9.1**
    - **Property 2: Shape Invariance** — `dequantize(quantize(x)).shape == x.shape` for all tensors
    - **Validates: Requirements 2.5, 5.4**
    - **Property 3: Output Range Invariant** — all elements of `quantize(x, ...)` satisfy `q_min <= q <= q_max`
    - **Validates: Requirements 2.1, 9.2**
    - **Property 14: Finer Scale Yields Lower Error** — for `s1 < s2`, `MAE(quantize with s1) <= MAE(quantize with s2)`
    - **Validates: Requirement 9.7**
    - **Property 15: Quantization Idempotence** — `quantize(dequantize(quantize(x))) == quantize(x)` for in-range tensors
    - **Validates: Requirement 9.8**
    - Use `# %%` cell per test function; use `@settings(max_examples=100)`

- [ ] 6. Checkpoint — Ensure all tests pass for `benefits.py`, `fp32_format.py`, `schemes.py`, and `quantizer.py`
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 7. Implement `quantization/calibrator.py` (Jupyter-style)
  - [ ] 7.1 Write `calibrator.py` calibration functions
    - Create with header cell; import `numpy`, `scipy.special`, `enum` in `# %%` cell; import `QuantizationScheme`, `compute_scale_zero_point` from `schemes`
    - Define `CalibrationMethod(Enum)` with `MIN_MAX`, `PERCENTILE`, `ENTROPY`
    - Implement `calibrate_min_max(tensors, bits, scheme) -> tuple[float, int]` — global min/max across all tensors; raise `ValueError("at least one calibration tensor is required")` if list empty
    - Implement `calibrate_percentile(tensors, bits, scheme, percentile=99.0) -> tuple[float, int]` — range is `[(100-p)/2`-th, `(100+p)/2`-th]`; raise `ValueError` if `percentile <= 0 or >= 100`
    - Implement `calibrate_entropy(tensors, bits, scheme) -> tuple[float, int]` — KL-divergence minimization with 2048-bin histogram; search grid of 32 logarithmic steps from full range to `[p40, p60]`; add `1e-8` smoothing to both histograms before `kl_div`
    - Add demo cell: calibrate a batch of 5 Gaussian tensors with all three methods, print `(scale, zp)` for each
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [ ]* 7.2 Write property test for `calibrator.py` in `tests/test_calibrator.py`
    - **Property 5: Calibration Scale Positivity** — for all non-empty calibration lists and all three methods, `scale > 0`
    - **Validates: Requirements 4.4, 4.7, 9.3**

- [ ] 8. Implement `quantization/granularity.py` (Jupyter-style)
  - [ ] 8.1 Write `granularity.py` per-tensor, per-channel, per-group quantization
    - Create with header cell; import `numpy`, `dataclasses`, `enum` in `# %%` cell; import `quantize`, `dequantize`, `q_min_max` from `quantizer`; import `QuantizationScheme`, `compute_scale_zero_point` from `schemes`
    - Define `GranularityMode(Enum)` with `PER_TENSOR`, `PER_CHANNEL`, `PER_GROUP`
    - Define `@dataclass GranularityComparisonReport` with fields: `granularity`, `num_scale_params`, `mean_absolute_error`, `max_absolute_error`
    - Implement `quantize_per_tensor(W, bits, scheme) -> tuple[np.ndarray, float, int]` — single scale/zp for entire `W`; output shape equals `W.shape`
    - Implement `quantize_per_channel(W, bits, scheme) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — one `(scale, zp)` per row (output channel); output shape equals `W.shape`
    - Implement `quantize_per_group(W, bits, scheme, group_size) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — raise `ValueError` identifying `in_channels` and `group_size` if `in_channels % group_size != 0`; compute `out_ch × (in_ch / g)` scale-zp pairs; output shape equals `W.shape`
    - Implement `format_granularity_comparison(reports: list[GranularityComparisonReport]) -> str` — table with columns: Granularity, Num_Params_Scale, Mean_Absolute_Error, Max_Absolute_Error
    - Add demo cell comparing all three modes on a `[8, 32]` weight matrix
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
  - [ ]* 8.2 Write property test for `granularity.py` in `tests/test_granularity.py`
    - **Property 6: Granularity Monotonicity** — `MAE(PER_TENSOR) >= MAE(PER_CHANNEL) >= MAE(PER_GROUP)` for non-uniform weight tensors
    - **Validates: Requirements 5.5, 5.6, 9.4**


- [ ] 9. Implement `quantization/ptq.py` (Jupyter-style)
  - [ ] 9.1 Write `ptq.py` post-training quantization
    - Create with header cell; import `numpy`, `dataclasses` in `# %%` cell; import from `quantizer`, `calibrator`, `granularity`, `schemes`
    - Define `@dataclass PTQLayer` with fields: `W_q`, `scales`, `zero_points`, `b`, `bits`, `scheme`, `granularity`
    - Implement `ptq_quantize_layer(W, b, calibration_inputs, bits, scheme, granularity, group_size=64) -> PTQLayer`
      - Raise `ValueError("at least 1 calibration sample is required")` if `calibration_inputs` is empty
      - Run `calibrate_min_max` on `calibration_inputs` to obtain `(scale, zp)`
      - Dispatch to `quantize_per_tensor`, `quantize_per_channel`, or `quantize_per_group` based on `granularity`
      - Return `PTQLayer` with quantized weights and metadata
    - Implement `ptq_infer(layer: PTQLayer, x: np.ndarray) -> np.ndarray`
      - Dequantize `W_q` using stored `scales` and `zero_points`
      - Compute `dequantize(W_q) @ x.T + layer.b`; output shape must be `[out, batch_size]`
    - Add demo cell: create a random `[16, 32]` layer, PTQ-quantize it, run inference, print output error
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [ ]* 9.2 Write unit tests for `ptq.py` in `tests/test_ptq.py`
    - Test `ptq_infer` output shape is `[out, batch_size]` for various input shapes
    - Test `ptq_quantize_layer` raises `ValueError` with fewer than 1 calibration input
    - Test that `error_per_tensor >= error_per_channel >= error_per_group` with 10+ calibration samples
    - Each test in its own `# %%` cell

- [ ] 10. Checkpoint — Ensure all tests pass for `calibrator.py`, `granularity.py`, and `ptq.py`
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 11. Implement `quantization/qat.py` (Jupyter-style, PyTorch)
  - [ ] 11.1 Write `qat.py` FakeQuantize and STE training loop
    - Create with header cell; import `torch`, `torch.nn as nn`, `numpy` in `# %%` cell; import `q_min_max` from `quantizer` (or mirror the logic in torch ops)
    - Implement `FakeQuantize(torch.autograd.Function)`:
      - `forward(ctx, x, scale, zero_point, bits, signed=True)`: quantize then dequantize; save `(x, q_min_val, q_max_val)` in `ctx`; return float tensor same shape as `x`
      - `backward(ctx, grad_output)`: STE — pass gradient unchanged where `q_min*scale <= x <= q_max*scale`, zero elsewhere; return `(grad_input, None, None, None, None)`
    - Implement `FakeQuantizeLayer(nn.Module)` wrapping `FakeQuantize.apply` with fixed `scale`, `zero_point`, `bits`
    - Implement a `train_qat(layer, dataset, n_steps=100, lr=1e-3) -> list[float]` helper that runs SGD for `n_steps` on MSE loss and returns per-step loss values
    - Add demo cell: train a `FakeQuantizeLayer`-wrapped linear layer for 100 steps on a fixed dataset; plot loss vs step (print to stdout in cell output)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [ ]* 11.2 Write property tests for `qat.py` in `tests/test_qat.py`
    - **Property 7: FakeQuantize Shape Invariant** — `FakeQuantize.apply(x, ...).shape == x.shape` for all tensor shapes
    - **Validates: Requirements 7.1, 7.6, 9.5**
    - **Property 8: STE Gradient** — gradient equals `1.0` inside `[q_min*s, q_max*s]`, `0.0` outside
    - **Validates: Requirements 7.2, 7.3, 7.4**
    - Each property in its own `# %%` cell; use `torch.autograd.gradcheck` or manual gradient checking

- [ ] 12. Implement `quantization/gptq.py` (Jupyter-style)
  - [ ] 12.1 Write `gptq.py` Hessian computation and column-wise update
    - Create with header cell; import `numpy` in `# %%` cell; import `quantize`, `q_min_max` from `quantizer`; import `QuantizationScheme` from `schemes`
    - Implement `compute_hessian(X: np.ndarray) -> np.ndarray` — `H = 2 * X.T @ X` where `X` is `[n_samples, in_features]`
    - Implement `gptq_quantize(W, H, bits, scheme, damp_factor=0.01) -> np.ndarray`:
      - Apply damping: `lambda = damp_factor * mean(diag(H))`; `H_damped = H + lambda * I`
      - Loop exactly `in_features` iterations; for each column `i`: if `H_damped[i,i] < 1e-8` skip; quantize column `i`; compute `q_error = W_q_col - W_original_col`; update columns `j > i`: `W[:, j] -= (q_error / H_damped[i,i]) * H_damped[i,j]`
      - Output shape must equal `W.shape`
    - Add demo cell: random `[8, 16]` weight matrix, compute Hessian from 20 calibration inputs, GPTQ-quantize, print MAE vs naive PTQ
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [ ]* 12.2 Write property and unit tests for `gptq.py` in `tests/test_gptq.py`
    - **Property 9: GPTQ Output Shape and Quality** — output shape equals `W.shape`; GPTQ MAE < naive per-tensor PTQ MAE for `W >= [4,4]` with 10+ calibration samples
    - **Validates: Requirements 8.4, 8.5**
    - Unit test: `compute_hessian` output matches `2 * X.T @ X` for a small known `X`
    - Unit test: verify exactly `in_features` loop iterations execute (mock inner quantize call)
    - Unit test: verify column is silently skipped when `H_damped[i,i] < 1e-8`

- [ ] 13. Checkpoint — Ensure all tests pass for `qat.py` and `gptq.py`
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 14. Build Streamlit dashboard shell (`ui/app.py`) — plain `.py`
  - [ ] 14.1 Create `ui/app.py` with sidebar navigation and session-state routing
    - Plain Streamlit file (no `# %%` cells); launched with `streamlit run ui/app.py`
    - Import all 15 panel modules from `ui/panels/`
    - Define `PANELS` dict mapping panel name → `render` function
    - Implement `main()`: `st.set_page_config(layout="wide", ...)`, initialize `st.session_state["active_panel"]` and `st.session_state["visited"]`
    - Sidebar: render nav buttons with `✅` prefix for visited panels; include "Concept Map" button
    - Main area: dispatch to `PANELS[active]()` or `render_concept_map()`
    - `render_concept_map()`: Plotly network graph with 15 panel nodes grouped into 5 topic clusters (Why Quantize, Equation, Mapping & Calibration, Granularity, Algorithms)
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6_
  - [ ] 14.2 Create stub `render()` functions for all 15 panel files
    - Create each file under `ui/panels/` as plain `.py`: `fp32_panel.py`, `pipeline_panel.py`, `distribution_panel.py`, `benefits_panel.py`, `rounding_clipping_panel.py`, `schemes_panel.py`, `calibration_panel.py`, `granularity_panel.py`, `dynamic_static_panel.py`, `activation_panel.py`, `mixed_precision_panel.py`, `formula_stepper_panel.py`, `ptq_panel.py`, `qat_panel.py`, `gptq_panel.py`
    - Each file: `import streamlit as st` + `def render(): st.header("Panel Title"); st.write("Coming soon...")` stub
    - This ensures the app launches without import errors before panels are filled in
    - _Requirements: 19.1, 19.2_


- [ ] 15. Implement UI panels: FP32, Pipeline, Distribution, Benefits (plain `.py`)
  - [ ] 15.1 Implement `ui/panels/fp32_panel.py`
    - `st.header("FP32 Format")`; float text input → call `decompose_fp32`
    - Render 32 colored squares (1 sign, 8 exponent, 23 mantissa) using Plotly or `st.markdown`
    - Show INT8 binary representation of quantized value side-by-side
    - Tooltip/legend explaining sign/exponent/mantissa fields
    - _Requirements: 10.1, 10.2, 10.3, 10.6_
  - [ ] 15.2 Implement `ui/panels/pipeline_panel.py`
    - `st.header("Quantization Equation")`; inputs: float value `r`, scale `s`, zero-point `z`, bits selector, scheme selector
    - Display `q_min`/`q_max` updating from bit/scheme selectors
    - Show two-step formula with live substitution: Step 1 `x = round(r/s) + z`, Step 2 `q = clamp(x, q_min, q_max)`
    - Highlight clamped values in red
    - Embed Formula Stepper (forward + reverse, 4+2 steps); show round-trip error `|r - dequantize(quantize(r))|`
    - _Requirements: 2.8, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_
  - [ ] 15.3 Implement `ui/panels/distribution_panel.py`
    - `st.header("Weight Distribution")`; user-configurable Gaussian mean/std; bit-width and scheme selectors
    - Left panel: float32 histogram with Gaussian curve overlay and quantization grid vertical lines
    - Right panel: INT histogram after quantization; summary stats for both distributions
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  - [ ] 15.4 Implement `ui/panels/benefits_panel.py`
    - `st.header("Why Quantize?")`; model preset selector (GPT-2, BERT-base, Llama-7B, Llama-70B, custom)
    - Comparison table: model size in GB for float32/float16/int8/int4 and compression ratios
    - Bar chart comparing all four formats; "Benefits" summary panel
    - Llama-70B reference values: float32 = 260 GB, int8 = 70 GB
    - _Requirements: 1.7, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_


- [ ] 16. Implement UI panels: Rounding/Clipping, Schemes, Calibration, Granularity (plain `.py`)
  - [ ] 16.1 Implement `ui/panels/rounding_clipping_panel.py`
    - `st.header("Rounding vs Clipping")`; range slider for `[r_min, r_max]`; number line with clipped values in red, integer grid inside range
    - Live metrics: Mean Absolute Clipping Error, Mean Absolute Rounding Error, Total Quantization Error (must equal sum of the two)
    - Plot of Total Error vs Range Width
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_
  - [ ] 16.2 Implement `ui/panels/schemes_panel.py`
    - `st.header("Mapping Schemes")`; tensor slider for adjustable input distribution
    - Side-by-side number line: symmetric vs asymmetric, showing scale and zero-point on each
    - _Requirements: 3.8_
  - [ ] 16.3 Implement `ui/panels/calibration_panel.py`
    - `st.header("Calibration Methods")`; load/generate calibration tensors
    - Three side-by-side Plotly histograms — one per calibration method — with the selected range overlaid on each
    - _Requirements: 4.8_
  - [ ] 16.4 Implement `ui/panels/granularity_panel.py`
    - `st.header("Quantization Granularity")`; weight tensor heatmap
    - Highlight per-tensor, per-channel, per-group scale regions in different colors using Plotly heatmap annotations
    - _Requirements: 5.9_

- [ ] 17. Implement UI panels: Dynamic/Static, Activation, Mixed Precision, Formula Stepper (plain `.py`)
  - [ ] 17.1 Implement `ui/panels/dynamic_static_panel.py`
    - `st.header("Dynamic vs Static")`; two side-by-side panels; input sequence controls
    - Dynamic: different `(scale, zp)` per input tensor; Static: single `(scale, zp)` from calibration
    - Per-tensor error display for both modes; "Distribution Shift" button introducing out-of-distribution input
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_
  - [ ] 17.2 Implement `ui/panels/activation_panel.py`
    - `st.header("Activation Quantization")`; toggle switches for "Quantize Weights" and "Quantize Activations"
    - Three-way output comparison: float32 baseline vs weight-only vs full quantization; MAE for each
    - Compute Mode labels: "Float32 MAC", "Float32 MAC (dequantized weights)", "INT8 MAC"
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  - [ ] 17.3 Implement `ui/panels/mixed_precision_panel.py`
    - `st.header("Mixed Precision")`; 3–5 linear layers each with bit-width selector (float32/int8/int4/int2)
    - Total memory in MB; per-layer MAE bar chart colored by bit-width; Model Error Score (sum of per-layer errors weighted by layer size)
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_
  - [ ] 17.4 Implement `ui/panels/formula_stepper_panel.py`
    - `st.header("Formula Stepper")`; inputs: `r`, `s`, `z`, bits, scheme; Next/Prev buttons
    - Forward: 4 steps with animated value substitution; Reverse: 2 dequantize steps; Round-trip error indicator
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_


- [ ] 18. Implement UI panels: PTQ, QAT, GPTQ algorithm demos (plain `.py`)
  - [ ] 18.1 Implement `ui/panels/ptq_panel.py`
    - `st.header("Post-Training Quantization")`; layer config controls (shape, bits, scheme) and calibration sample count
    - PTQ pipeline diagram; live error comparison table across per-tensor / per-channel / per-group
    - _Requirements: 6.7_
  - [ ] 18.2 Implement `ui/panels/qat_panel.py`
    - `st.header("Quantization-Aware Training")`; training run button; Plotly line chart showing QAT loss vs PTQ baseline across 100 steps
    - _Requirements: 7.7_
  - [ ] 18.3 Implement `ui/panels/gptq_panel.py`
    - `st.header("GPTQ Algorithm")`; weight matrix heatmap animated column-by-column using Plotly; each column lights up as it is processed
    - _Requirements: 8.9_

- [ ] 19. Checkpoint — Verify Streamlit dashboard launches and all 15 panels render
  - Ensure all tests pass, ask the user if questions arise.
  - Verify `streamlit run ui/app.py` starts without import errors (run it manually in terminal)
  - Each panel must render without raising an exception under a mock Streamlit context


- [ ] 20. Final wiring and integration
  - [ ] 20.1 Replace all panel stubs with full implementations and wire `ui/app.py` imports
    - Confirm every panel in `PANELS` dict imports from the correct panel file
    - Remove stub `render()` bodies added in task 14.2
    - Verify all `st.error(str(e))` catch blocks are in place for `ValueError` propagation from core modules
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5_
  - [ ] 20.2 Complete `quantization/__init__.py` public API exports
    - Export from `quantizer`: `quantize`, `dequantize`, `q_min_max`
    - Export from `schemes`: `QuantizationScheme`, `compute_scale_zero_point`, `compare_schemes`, `format_scheme_comparison`
    - Export from `calibrator`: `CalibrationMethod`, `calibrate_min_max`, `calibrate_percentile`, `calibrate_entropy`
    - Export from `granularity`: `GranularityMode`, `quantize_per_tensor`, `quantize_per_channel`, `quantize_per_group`, `format_granularity_comparison`
    - Export from `ptq`: `ptq_quantize_layer`, `ptq_infer`, `PTQLayer`
    - Export from `gptq`: `compute_hessian`, `gptq_quantize`
    - Export from `fp32_format`: `decompose_fp32`, `reconstruct_fp32`
    - Export from `benefits`: `memory_footprint`, `compression_ratio`, `mac_count`, `model_memory_table`
    - _Requirements: all_
  - [ ]* 20.3 Write integration tests for QAT convergence
    - `tests/test_qat.py`: 100-step QAT training on a fixed dataset; assert final QAT MAE < PTQ MAE on same dataset
    - _Requirements: 7.5_

- [ ] 21. Final checkpoint — Full test suite passes
  - Run `pytest quantization/tests/ --cov=quantization` and confirm all tests pass
  - Ensure all tests pass, ask the user if questions arise.


---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP pass. All property-based tests and unit tests carry `*`.
- Every source file under `quantization/` (except `ui/` files) MUST be structured with `# %%` cell markers so they run interactively in VS Code's Jupyter window.
- Streamlit UI files (`ui/app.py` and `ui/panels/*.py`) are plain `.py` files — no cell markers — because Streamlit runs them as scripts, not interactive notebooks.
- The dependency order mirrors the module graph: `benefits` and `fp32_format` have no internal deps → `schemes` → `quantizer` → `calibrator` + `granularity` → `ptq` → `qat` + `gptq` → UI.
- Property tests use `@settings(max_examples=100)` and the Hypothesis strategies from the design document (e.g., `in_range_tensor`, `non_uniform_weight_tensor`).
- Each property test is annotated with a comment `# Feature: ai-model-quantization, Property N: <title>`.
- Checkpoints (tasks 6, 10, 13, 19, 21) are not implementation tasks — they are validation gates before the next module group.

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.2", "5.1"] },
    { "id": 3, "tasks": ["5.2", "7.1", "8.1"] },
    { "id": 4, "tasks": ["7.2", "8.2", "9.1"] },
    { "id": 5, "tasks": ["9.2", "11.1", "12.1"] },
    { "id": 6, "tasks": ["11.2", "12.2", "14.1"] },
    { "id": 7, "tasks": ["14.2"] },
    { "id": 8, "tasks": ["15.1", "15.2", "15.3", "15.4"] },
    { "id": 9, "tasks": ["16.1", "16.2", "16.3", "16.4"] },
    { "id": 10, "tasks": ["17.1", "17.2", "17.3", "17.4"] },
    { "id": 11, "tasks": ["18.1", "18.2", "18.3"] },
    { "id": 12, "tasks": ["20.1", "20.2"] },
    { "id": 13, "tasks": ["20.3"] }
  ]
}
```
