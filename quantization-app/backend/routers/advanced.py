"""
Advanced quantization API endpoints - FP8, GPTQ, SmoothQuant, benchmarks, and decision guide
"""
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from typing import List

from models.schemas import (
    FP8ConvertRequest,
    FP8ConvertResponse,
    FP8FormatInfo,
    GPTQSimulateRequest,
    GPTQSimulateResponse,
    GPTQStep,
    SmoothQuantRequest,
    SmoothQuantResponse,
    RangeInfo,
    DecisionGuideRequest,
    DecisionGuideResponse,
    AlternativeMethod,
    BenchmarkMethodsResponse,
    BenchmarkMethodEntry,
    LiveBenchmarkRequest,
    LiveBenchmarkResponse,
    LiveBenchmarkResult,
    HardwareComparisonResponse,
    GPUInfo,
)
from services.quantizer import (
    quantize_symmetric_int8,
    quantize_asymmetric_int8,
    quantize_bfloat16,
)

router = APIRouter()


# --- FP8 Simulation Helpers ---


def simulate_fp8_e4m3(value: float) -> float:
    """Simulate FP8 E4M3 quantization for a single value."""
    max_val = 448.0
    clamped = float(np.clip(value, -max_val, max_val))
    if clamped == 0.0:
        return 0.0
    sign = np.sign(clamped)
    abs_val = abs(clamped)
    # Find exponent
    exp = np.floor(np.log2(abs_val))
    # Mantissa precision: 3 mantissa bits = 8 steps
    mantissa_steps = 2**3  # 8 steps
    # Round mantissa to nearest representable value
    normalized = abs_val / (2**exp)  # 1.xxx form
    quantized_mantissa = np.round(normalized * mantissa_steps) / mantissa_steps
    result = float(sign * quantized_mantissa * (2**exp))
    return float(np.clip(result, -max_val, max_val))


def simulate_fp8_e5m2(value: float) -> float:
    """Simulate FP8 E5M2 quantization for a single value."""
    max_val = 57344.0
    clamped = float(np.clip(value, -max_val, max_val))
    if clamped == 0.0:
        return 0.0
    sign = np.sign(clamped)
    abs_val = abs(clamped)
    # Find exponent
    exp = np.floor(np.log2(abs_val))
    # Mantissa precision: 2 mantissa bits = 4 steps
    mantissa_steps = 2**2  # 4 steps
    # Round mantissa to nearest representable value
    normalized = abs_val / (2**exp)  # 1.xx form
    quantized_mantissa = np.round(normalized * mantissa_steps) / mantissa_steps
    result = float(sign * quantized_mantissa * (2**exp))
    return float(np.clip(result, -max_val, max_val))


# --- Endpoints ---


@router.post("/fp8/convert", response_model=FP8ConvertResponse)
async def fp8_convert(request: FP8ConvertRequest):
    """
    Simulate FP8 quantization (E4M3 or E5M2 format).
    Clamps values to format range and reduces precision to simulate 8-bit float.
    """
    try:
        if request.format not in ("e4m3", "e5m2"):
            raise ValueError("format must be 'e4m3' or 'e5m2'")

        if len(request.weights) == 0:
            raise ValueError("weights must not be empty")

        original = request.weights
        quantized = []

        if request.format == "e4m3":
            for v in original:
                quantized.append(simulate_fp8_e4m3(v))
            format_info = FP8FormatInfo(
                name="E4M3",
                exponent_bits=4,
                mantissa_bits=3,
                max_value=448.0,
                min_value=-448.0,
            )
        else:
            for v in original:
                quantized.append(simulate_fp8_e5m2(v))
            format_info = FP8FormatInfo(
                name="E5M2",
                exponent_bits=5,
                mantissa_bits=2,
                max_value=57344.0,
                min_value=-57344.0,
            )

        errors = [float(o - q) for o, q in zip(original, quantized)]
        squared_errors = [(o - q) ** 2 for o, q in zip(original, quantized)]
        mse = float(np.mean(squared_errors)) if squared_errors else 0.0
        rss = float(np.sum(squared_errors)) if squared_errors else 0.0

        return FP8ConvertResponse(
            original=original,
            quantized=quantized,
            errors=errors,
            mse=mse,
            rss=rss,
            format_info=format_info,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gptq/simulate", response_model=GPTQSimulateResponse)
async def gptq_simulate(request: GPTQSimulateRequest):
    """
    Simulate GPTQ column-wise quantization with Hessian compensation.
    Demonstrates the core GPTQ algorithm on a small matrix.
    """
    try:
        W = np.array(request.weights, dtype=np.float64)
        if W.ndim != 2:
            raise ValueError("weights must be a 2D matrix")
        if W.shape[0] == 0 or W.shape[1] == 0:
            raise ValueError("weights matrix must not be empty")

        bits = request.bits
        n_rows, n_cols = W.shape
        W_orig = W.copy()
        W_q = np.zeros_like(W)

        # Compute pseudo-Hessian: H = W^T @ W + damping * I
        H = W.T @ W
        damping = 0.01 * np.mean(np.diag(H)) + 1e-6
        H += damping * np.eye(n_cols)

        # Invert H
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        # Quantization range
        q_max = (2 ** (bits - 1)) - 1
        q_min = -(2 ** (bits - 1))

        steps = []

        for col in range(n_cols):
            # Get current column values
            col_values = W[:, col].copy()
            original_values = col_values.tolist()

            # Quantize this column
            col_max = np.max(np.abs(col_values))
            if col_max == 0:
                scale = 1.0
            else:
                scale = col_max / q_max

            quantized_col = np.clip(np.round(col_values / scale), q_min, q_max)
            dequantized_col = quantized_col * scale
            W_q[:, col] = dequantized_col

            # Compute quantization error for this column
            col_error = col_values - dequantized_col
            col_mse = float(np.mean(col_error**2))

            # Compensate remaining columns using Hessian inverse
            compensation_applied = False
            if col < n_cols - 1:
                # GPTQ compensation: distribute error to remaining columns
                h_diag = H_inv[col, col]
                if h_diag > 1e-10:
                    for remaining_col in range(col + 1, n_cols):
                        h_ratio = H_inv[col, remaining_col] / h_diag
                        W[:, remaining_col] += col_error * h_ratio
                    compensation_applied = True

            steps.append(GPTQStep(
                column=col,
                original_values=[float(x) for x in original_values],
                quantized_values=[float(x) for x in dequantized_col],
                error=col_mse,
                compensation_applied=compensation_applied,
            ))

        # Compute total MSE
        total_error = W_orig - W_q
        total_mse = float(np.mean(total_error**2))

        # Compression ratio: original is float32 (32 bits), quantized is `bits` bits
        compression_ratio = 32.0 / bits

        return GPTQSimulateResponse(
            steps=steps,
            original_matrix=[[float(x) for x in row] for row in W_orig],
            quantized_matrix=[[float(x) for x in row] for row in W_q],
            total_mse=total_mse,
            compression_ratio=compression_ratio,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smoothquant/simulate", response_model=SmoothQuantResponse)
async def smoothquant_simulate(request: SmoothQuantRequest):
    """
    Simulate SmoothQuant transformation.
    Migrates quantization difficulty from activations to weights using per-channel scaling.
    """
    try:
        weights = np.array(request.weights, dtype=np.float64)
        activations = np.array(request.activations, dtype=np.float64)

        if weights.ndim != 2 or activations.ndim != 2:
            raise ValueError("weights and activations must be 2D matrices")
        if weights.shape[0] == 0 or activations.shape[0] == 0:
            raise ValueError("matrices must not be empty")
        if weights.shape[1] != activations.shape[1]:
            raise ValueError(
                "weights and activations must have the same number of columns "
                f"(got {weights.shape[1]} and {activations.shape[1]})"
            )

        alpha = request.alpha
        n_channels = weights.shape[1]

        # Compute per-channel scales
        # s_act = max(|activations|, dim=0) per column
        s_act = np.max(np.abs(activations), axis=0)
        # s_wt = max(|weights|, dim=0) per column
        s_wt = np.max(np.abs(weights), axis=0)

        # Avoid division by zero
        s_act = np.where(s_act == 0, 1e-8, s_act)
        s_wt = np.where(s_wt == 0, 1e-8, s_wt)

        # Smooth factor: s = s_act^alpha / s_wt^(1-alpha)
        smooth_factors = (s_act ** alpha) / (s_wt ** (1 - alpha))

        # Avoid zero/inf in smooth factors
        smooth_factors = np.where(smooth_factors == 0, 1e-8, smooth_factors)
        smooth_factors = np.where(np.isinf(smooth_factors), 1.0, smooth_factors)

        # Transform: new_weights = weights * diag(s), new_activations = activations / diag(s)
        smoothed_weights = weights * smooth_factors[np.newaxis, :]
        smoothed_activations = activations / smooth_factors[np.newaxis, :]

        return SmoothQuantResponse(
            original_weights=[[float(x) for x in row] for row in weights],
            smoothed_weights=[[float(x) for x in row] for row in smoothed_weights],
            original_activations=[[float(x) for x in row] for row in activations],
            smoothed_activations=[[float(x) for x in row] for row in smoothed_activations],
            smooth_factors=[float(x) for x in smooth_factors],
            alpha=alpha,
            weight_range_before=RangeInfo(
                min=float(np.min(weights)),
                max=float(np.max(weights)),
            ),
            weight_range_after=RangeInfo(
                min=float(np.min(smoothed_weights)),
                max=float(np.max(smoothed_weights)),
            ),
            activation_range_before=RangeInfo(
                min=float(np.min(activations)),
                max=float(np.max(activations)),
            ),
            activation_range_after=RangeInfo(
                min=float(np.min(smoothed_activations)),
                max=float(np.max(smoothed_activations)),
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision-guide", response_model=DecisionGuideResponse)
async def decision_guide(request: DecisionGuideRequest):
    """
    Recommend a quantization method based on model size, hardware, and constraints.
    """
    try:
        model_size = request.model_size_billions
        hardware = request.hardware
        tolerance = request.accuracy_tolerance
        has_cal_data = request.has_calibration_data
        use_case = request.use_case
        latency = request.latency_budget_ms

        warnings: List[str] = []
        reasoning: List[str] = []
        alternatives: List[AlternativeMethod] = []

        # Decision logic
        recommended_method = ""
        recommended_bits = 16
        recommended_format = ""
        expected_compression = ""
        expected_accuracy_loss = ""

        # Rule: small model + no tolerance -> don't quantize
        if model_size < 1.0 and tolerance == "none":
            recommended_method = "No quantization"
            recommended_bits = 16
            recommended_format = "FP16"
            expected_compression = "1x (no compression)"
            expected_accuracy_loss = "None"
            reasoning.append("Model is small (<1B params) and no accuracy loss is tolerated.")
            reasoning.append("FP16 provides sufficient memory savings without quantization artifacts.")
            alternatives.append(AlternativeMethod(
                method="BFloat16",
                pros=["Same range as FP32", "Good for training"],
                cons=["Slightly less precise than FP16 for small values"],
            ))

        # Rule: training use case -> QAT with BFloat16
        elif use_case == "training" or use_case == "both":
            recommended_method = "QAT (Quantization-Aware Training)"
            recommended_bits = 16
            recommended_format = "BFloat16 with INT8 forward pass"
            expected_compression = "2x"
            expected_accuracy_loss = "< 0.1 perplexity points (recoverable through training)"
            reasoning.append("Training use case requires gradient-compatible formats.")
            reasoning.append("QAT allows the model to learn to compensate for quantization noise.")
            reasoning.append("BFloat16 maintains float32 dynamic range during training.")
            alternatives.append(AlternativeMethod(
                method="Mixed-precision (FP16/FP32)",
                pros=["No quantization noise", "Well-supported by frameworks"],
                cons=["Only 2x compression", "No INT8 inference benefit"],
            ))

        # Rule: edge/cpu -> GGUF q4_K_M
        elif hardware in ("cpu", "edge"):
            recommended_method = "GGUF"
            recommended_bits = 4
            recommended_format = "GGUF Q4_K_M (4-bit with k-quant mixed precision)"
            expected_compression = "~4x"
            expected_accuracy_loss = "< 1.0 perplexity points"
            reasoning.append("CPU/edge hardware benefits most from GGUF format.")
            reasoning.append("Q4_K_M provides good balance of quality and speed on CPU.")
            reasoning.append("llama.cpp ecosystem provides optimized CPU inference kernels.")
            alternatives.append(AlternativeMethod(
                method="GGUF Q5_K_M",
                pros=["Better accuracy", "Still fast on CPU"],
                cons=["Larger model size", "Only ~3.2x compression"],
            ))
            alternatives.append(AlternativeMethod(
                method="GGUF Q3_K_M",
                pros=["Smaller model size", "Fits in less RAM"],
                cons=["Noticeable accuracy degradation", "May affect coherence"],
            ))

        # Rule: nvidia_gpu + aggressive -> GPTQ 3-bit
        elif hardware == "nvidia_gpu" and tolerance == "aggressive":
            recommended_method = "GPTQ"
            recommended_bits = 3
            recommended_format = "INT3 per-group (group_size=128)"
            expected_compression = "~10x"
            expected_accuracy_loss = "1-2 perplexity points"
            reasoning.append("Aggressive tolerance allows 3-bit quantization.")
            reasoning.append("GPTQ with Hessian compensation minimizes accuracy loss at extreme compression.")
            reasoning.append("Group-size=128 maintains reasonable accuracy at 3-bit.")
            if not has_cal_data:
                warnings.append("GPTQ requires calibration data for best results. Consider HQQ as alternative.")
            alternatives.append(AlternativeMethod(
                method="AWQ 4-bit",
                pros=["Better accuracy", "Faster inference kernels"],
                cons=["Less compression (8x vs 10x)"],
            ))

        # Rule: nvidia_gpu + large model + moderate tolerance -> AWQ 4-bit
        elif hardware == "nvidia_gpu" and model_size > 13 and tolerance == "moderate":
            recommended_method = "AWQ"
            recommended_bits = 4
            recommended_format = "INT4 per-group (group_size=128)"
            expected_compression = "4x"
            expected_accuracy_loss = "< 0.5 perplexity points"
            reasoning.append("Large model (>13B) on NVIDIA GPU with moderate tolerance.")
            reasoning.append("AWQ preserves salient weights, ideal for large models.")
            reasoning.append("4-bit per-group quantization with group_size=128 balances speed and accuracy.")
            alternatives.append(AlternativeMethod(
                method="GPTQ 4-bit",
                pros=["Well-established", "Good tooling"],
                cons=["Slightly slower inference than AWQ", "Requires more calibration data"],
            ))

        # Rule: nvidia_gpu + minimal tolerance -> FP8 or INT8 PTQ
        elif hardware == "nvidia_gpu" and tolerance == "minimal":
            recommended_method = "FP8 (E4M3) or INT8 PTQ"
            recommended_bits = 8
            recommended_format = "FP8 E4M3 (preferred on Hopper+) or INT8 symmetric"
            expected_compression = "2x"
            expected_accuracy_loss = "< 0.1 perplexity points"
            reasoning.append("Minimal tolerance requires high-fidelity quantization.")
            reasoning.append("8-bit formats provide near-lossless compression on NVIDIA GPUs.")
            reasoning.append("FP8 E4M3 is preferred on H100+ (Hopper architecture).")
            alternatives.append(AlternativeMethod(
                method="INT8 (symmetric)",
                pros=["Widely supported", "Well-understood"],
                cons=["Slightly less flexible than FP8 for outliers"],
            ))

        # Rule: no calibration data -> HQQ or dynamic quantization
        elif not has_cal_data:
            recommended_method = "HQQ (Half-Quadratic Quantization)"
            recommended_bits = 4
            recommended_format = "INT4 with HQQ optimization (no calibration needed)"
            expected_compression = "4x"
            expected_accuracy_loss = "< 1.0 perplexity points"
            reasoning.append("No calibration data available.")
            reasoning.append("HQQ does not require calibration data, using half-quadratic optimization instead.")
            reasoning.append("Achieves competitive quality with GPTQ/AWQ without calibration overhead.")
            alternatives.append(AlternativeMethod(
                method="Dynamic quantization (PyTorch)",
                pros=["No calibration needed", "Easy to implement"],
                cons=["Higher overhead per-inference", "Less compression"],
            ))

        # Default: AWQ 4-bit for GPU, GGUF for others
        else:
            if hardware in ("nvidia_gpu", "amd_gpu"):
                recommended_method = "AWQ"
                recommended_bits = 4
                recommended_format = "INT4 per-group (group_size=128)"
                expected_compression = "4x"
                expected_accuracy_loss = "< 0.5 perplexity points"
                reasoning.append("GPU target with standard requirements.")
                reasoning.append("AWQ 4-bit provides excellent speed/quality tradeoff on GPUs.")
            else:
                recommended_method = "GGUF"
                recommended_bits = 4
                recommended_format = "GGUF Q4_K_M"
                expected_compression = "~4x"
                expected_accuracy_loss = "< 1.0 perplexity points"
                reasoning.append("Default recommendation for non-GPU hardware.")
            alternatives.append(AlternativeMethod(
                method="GPTQ 4-bit",
                pros=["Well-established", "Good ecosystem support"],
                cons=["Requires calibration data", "Slightly slower than AWQ"],
            ))

        # Additional warnings
        if model_size > 65 and recommended_bits <= 4:
            warnings.append("Very large models (>65B) may need careful per-layer bit allocation.")
        if latency is not None and latency < 10 and recommended_bits > 4:
            warnings.append("Very tight latency budget may require more aggressive quantization.")

        return DecisionGuideResponse(
            recommended_method=recommended_method,
            recommended_bits=recommended_bits,
            recommended_format=recommended_format,
            expected_compression=expected_compression,
            expected_accuracy_loss=expected_accuracy_loss,
            reasoning=reasoning,
            alternatives=alternatives,
            warnings=warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks/methods", response_model=BenchmarkMethodsResponse)
async def benchmark_methods():
    """
    Return pre-computed benchmark data for various quantization methods on LLaMA-2-7B.
    """
    methods = [
        BenchmarkMethodEntry(name="FP16 (baseline)", bits=16, perplexity=5.47, model_size_gb=13.5, tokens_per_sec=45, memory_gb=14.2),
        BenchmarkMethodEntry(name="INT8 (symmetric)", bits=8, perplexity=5.49, model_size_gb=6.7, tokens_per_sec=72, memory_gb=7.8),
        BenchmarkMethodEntry(name="GPTQ 4-bit", bits=4, perplexity=5.63, model_size_gb=3.9, tokens_per_sec=95, memory_gb=5.2),
        BenchmarkMethodEntry(name="AWQ 4-bit", bits=4, perplexity=5.60, model_size_gb=3.9, tokens_per_sec=98, memory_gb=5.1),
        BenchmarkMethodEntry(name="GGUF Q4_K_M", bits=4.5, perplexity=5.58, model_size_gb=4.1, tokens_per_sec=35, memory_gb=4.8),
        BenchmarkMethodEntry(name="GPTQ 3-bit", bits=3, perplexity=5.88, model_size_gb=3.0, tokens_per_sec=105, memory_gb=4.1),
        BenchmarkMethodEntry(name="GPTQ 2-bit", bits=2, perplexity=8.20, model_size_gb=2.1, tokens_per_sec=110, memory_gb=3.2),
        BenchmarkMethodEntry(name="BFloat16", bits=16, perplexity=5.47, model_size_gb=13.5, tokens_per_sec=48, memory_gb=14.2),
        BenchmarkMethodEntry(name="FP8 E4M3", bits=8, perplexity=5.48, model_size_gb=6.7, tokens_per_sec=85, memory_gb=7.5),
    ]

    return BenchmarkMethodsResponse(
        model="LLaMA-2-7B",
        baseline_perplexity=5.47,
        methods=methods,
    )


@router.post("/benchmarks/live", response_model=LiveBenchmarkResponse)
async def benchmark_live(request: LiveBenchmarkRequest):
    """
    Run live quantization benchmarks on random weights.
    Times each method and returns real MSE/timing data.
    """
    try:
        num_weights = request.num_weights
        methods = request.methods

        # Generate random weights (simulate a normal distribution like real model weights)
        np.random.seed(42)
        weights = np.random.randn(num_weights).astype(np.float32) * 0.02

        results = []

        for method in methods:
            start_time = time.perf_counter()

            if method == "symmetric":
                result = quantize_symmetric_int8(weights, bits=8)
                mse = result["mse"]
                compression_ratio = 4.0  # float32 -> int8
            elif method == "asymmetric":
                result = quantize_asymmetric_int8(weights, bits=8)
                mse = result["mse"]
                compression_ratio = 4.0
            elif method == "bfloat16":
                result = quantize_bfloat16(weights)
                mse = result["mse"]
                compression_ratio = 2.0  # float32 -> bfloat16
            elif method == "symmetric_int4":
                result = quantize_symmetric_int8(weights, bits=4)
                mse = result["mse"]
                compression_ratio = 8.0  # float32 -> int4
            elif method == "asymmetric_int4":
                result = quantize_asymmetric_int8(weights, bits=4)
                mse = result["mse"]
                compression_ratio = 8.0
            else:
                # Unknown method, skip
                continue

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            results.append(LiveBenchmarkResult(
                method=method,
                mse=float(mse),
                time_ms=round(elapsed_ms, 4),
                compression_ratio=compression_ratio,
            ))

        if not results:
            raise ValueError("No valid methods provided. Use: symmetric, asymmetric, bfloat16, symmetric_int4, asymmetric_int4")

        return LiveBenchmarkResponse(
            num_weights=num_weights,
            results=results,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hardware/comparison", response_model=HardwareComparisonResponse)
async def hardware_comparison():
    """
    Return GPU architecture comparison data for quantization workloads.
    """
    gpus = [
        GPUInfo(name="T4 (Turing)", fp32_tflops=8.1, fp16_tflops=65, int8_tops=130, fp8_tops=None, memory_gb=16, year=2018),
        GPUInfo(name="A100 (Ampere)", fp32_tflops=19.5, fp16_tflops=312, int8_tops=624, fp8_tops=None, memory_gb=80, year=2020),
        GPUInfo(name="H100 (Hopper)", fp32_tflops=67, fp16_tflops=990, int8_tops=1979, fp8_tops=3958, memory_gb=80, year=2022),
        GPUInfo(name="RTX 4090", fp32_tflops=82.6, fp16_tflops=165, int8_tops=330, fp8_tops=661, memory_gb=24, year=2022),
        GPUInfo(name="MI300X (AMD)", fp32_tflops=81.7, fp16_tflops=163.4, int8_tops=326.8, fp8_tops=653.7, memory_gb=192, year=2023),
    ]

    return HardwareComparisonResponse(gpus=gpus)
