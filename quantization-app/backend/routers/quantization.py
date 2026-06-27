"""
Quantization API endpoints
"""
from fastapi import APIRouter, HTTPException
from models.schemas import (
    QuantizeRequest,
    QuantizeResponse,
    BFloat16Response,
    StepsResponse,
    ErrorDistributionRequest,
    ErrorDistributionResponse,
)
from services.quantizer import (
    quantize_symmetric_int8,
    quantize_asymmetric_int8,
    quantize_clipped_int8,
    quantize_bfloat16,
    quantize_symmetric_steps,
    compute_error_distribution,
)
import numpy as np

router = APIRouter()


@router.post("/quantize/symmetric", response_model=QuantizeResponse)
async def quantize_symmetric(request: QuantizeRequest):
    """
    Symmetric INT8 quantization endpoint
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = quantize_symmetric_int8(weights, request.bits)
        return QuantizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quantize/asymmetric", response_model=QuantizeResponse)
async def quantize_asymmetric(request: QuantizeRequest):
    """
    Asymmetric INT8 quantization endpoint
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = quantize_asymmetric_int8(weights, request.bits)
        return QuantizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quantize/clipped", response_model=QuantizeResponse)
async def quantize_clipped(request: QuantizeRequest):
    """
    Clipped symmetric quantization endpoint
    """
    try:
        if request.clip_min is None or request.clip_max is None:
            raise ValueError("clip_min and clip_max are required for clipped quantization")

        weights = np.array(request.weights, dtype=np.float32)
        result = quantize_clipped_int8(
            weights,
            request.clip_min,
            request.clip_max,
            request.bits
        )
        return QuantizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quantize/bfloat16", response_model=BFloat16Response)
async def quantize_bf16(request: QuantizeRequest):
    """
    BFloat16 quantization endpoint.
    Casts float32 weights to bfloat16 and computes error statistics.
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = quantize_bfloat16(weights)
        return BFloat16Response(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quantize/steps", response_model=StepsResponse)
async def quantize_steps(request: QuantizeRequest):
    """
    Step-by-step symmetric INT8 quantization.
    Returns intermediate values at each step with reversibility info.
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = quantize_symmetric_steps(weights)
        return StepsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quantize/error-distribution", response_model=ErrorDistributionResponse)
async def error_distribution(request: ErrorDistributionRequest):
    """
    Compute detailed error histogram for a given quantization method.
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = compute_error_distribution(
            weights,
            method=request.method,
            bits=request.bits,
            clip_min=request.clip_min,
            clip_max=request.clip_max,
        )
        return ErrorDistributionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
