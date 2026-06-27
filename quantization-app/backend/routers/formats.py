"""
Format and memory comparison API endpoints
"""
from fastapi import APIRouter, HTTPException
from models.schemas import (
    MemoryRequest,
    MemoryResponse,
    FormatsResponse,
)
from services.quantizer import compute_memory_comparison, get_format_info

router = APIRouter()


@router.post("/quantize/memory", response_model=MemoryResponse)
async def memory_comparison(request: MemoryRequest):
    """
    Compare memory usage across different quantization formats.

    Takes num_elements and returns memory usage for float32, float16,
    bfloat16, int8, and int4 formats.
    """
    try:
        result = compute_memory_comparison(request.num_elements)
        return MemoryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/formats", response_model=FormatsResponse)
async def format_info():
    """
    Return information about number formats used in quantization.

    Includes bit layout, range, and description for float32, float16,
    bfloat16, int8, and int4.
    """
    try:
        result = get_format_info()
        return FormatsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
