"""
app.models.mambavision — Official MambaVision integration (PyTorch).

This package intentionally uses the official MambaVision implementation and
official pretrained weights (ImageNet) via Hugging Face model artifacts.
"""

from app.models.mambavision.config import MambaVisionHFConfig
from app.models.mambavision.factory import build_mambavision_model
from app.models.mambavision.predictor import TorchImageClassifier

__all__ = [
    "MambaVisionHFConfig",
    "TorchImageClassifier",
    "build_mambavision_model",
]

