"""
Device detection and management for GPU/CPU execution.
Auto-detects CUDA availability and provides a consistent device interface.
"""
import torch
from typing import Optional

_device: Optional[torch.device] = None


def get_device() -> torch.device:
    """Return the best available device (CUDA GPU if available, else CPU)."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def get_device_info() -> dict:
    """Return device information for diagnostics."""
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)),
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024)),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024)),
            "cuda_version": torch.version.cuda,
            "num_gpus": torch.cuda.device_count(),
        })
    return info


def move_to_device(tensor_or_model, device: Optional[torch.device] = None):
    """Move a tensor or model to the target device."""
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)
