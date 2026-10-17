from pathlib import Path
import torch


def detect_device(preferred: str = "auto") -> str:
    """Resolve compute device without hardcoding CUDA."""
    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_device_cache(device: str) -> None:
    """Flush accelerator memory when available."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()
