"""
Weights loading and distribution API endpoints
"""
from fastapi import APIRouter, HTTPException
from models.schemas import (
    ResNetWeightsResponse,
    WeightDistributionResponse,
    QuantizeRequest,
    OutlierRequest,
    OutlierResponse,
)
from services.resnet_loader import load_resnet50_fc_weights, get_sample_weights
from services.quantizer import compute_weight_distribution, compute_outlier_analysis
import numpy as np

router = APIRouter()


@router.get("/weights/resnet50", response_model=ResNetWeightsResponse)
async def get_resnet50_weights(sample: int = None):
    """
    Load ResNet50 FC layer weights

    Args:
        sample: Optional - number of samples to return (for faster loading)
    """
    try:
        if sample and sample > 0:
            weights = get_sample_weights(sample)
            return ResNetWeightsResponse(
                weights=weights.tolist(),
                shape=[len(weights)],
                num_elements=len(weights),
                dtype=str(weights.dtype),
                statistics={
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights)),
                    "mean": float(np.mean(weights)),
                    "std": float(np.std(weights)),
                    "median": float(np.median(weights)),
                }
            )
        else:
            result = load_resnet50_fc_weights()
            return ResNetWeightsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weights/distribution", response_model=WeightDistributionResponse)
async def get_weight_distribution(request: QuantizeRequest):
    """
    Compute weight distribution (histogram + cumulative)

    Args:
        request: QuantizeRequest with weights array
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = compute_weight_distribution(weights, num_bins=100)
        return WeightDistributionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/weights/outliers", response_model=OutlierResponse)
async def get_outliers(request: OutlierRequest):
    """
    Analyze outlier weights beyond a given threshold.

    Returns negative weights below -threshold, positive weights above +threshold,
    percentage of weights that are outliers, and histograms of each tail.
    """
    try:
        weights = np.array(request.weights, dtype=np.float32)
        result = compute_outlier_analysis(weights, threshold=request.threshold)
        return OutlierResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
