"""
ResNet50 weights loader service
"""
import numpy as np
import torch
import torchvision.models.resnet as resnet
from typing import Dict


def load_resnet50_fc_weights() -> Dict:
    """
    Load ResNet50 FC layer weights

    Returns:
        Dictionary with weights and metadata
    """
    # Download pre-trained ResNet50
    model = resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT)

    # Extract FC layer weights
    fc_weights = model.fc.weight.data.numpy().astype(np.float32)
    fc_weights_flat = fc_weights.flatten()

    # Compute statistics
    stats = {
        "min": float(np.min(fc_weights_flat)),
        "max": float(np.max(fc_weights_flat)),
        "mean": float(np.mean(fc_weights_flat)),
        "std": float(np.std(fc_weights_flat)),
        "median": float(np.median(fc_weights_flat)),
        "num_elements": int(fc_weights_flat.size),
        "shape": list(fc_weights.shape),
        "dtype": str(fc_weights.dtype),
        "memory_mb": float(fc_weights_flat.nbytes / (1024 * 1024)),
    }

    return {
        "weights": fc_weights_flat.tolist(),
        "shape": list(fc_weights.shape),
        "num_elements": int(fc_weights_flat.size),
        "dtype": str(fc_weights.dtype),
        "statistics": stats,
    }


def get_sample_weights(num_samples: int = 10000) -> np.ndarray:
    """
    Get a sample of weights for faster testing

    Args:
        num_samples: Number of samples to return

    Returns:
        Numpy array of sampled weights
    """
    weights_data = load_resnet50_fc_weights()
    weights = np.array(weights_data["weights"], dtype=np.float32)

    if num_samples >= len(weights):
        return weights

    # Random sampling
    indices = np.random.choice(len(weights), size=num_samples, replace=False)
    return weights[indices]
