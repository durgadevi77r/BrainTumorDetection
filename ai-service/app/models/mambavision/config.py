"""
config.py — Official MambaVision configuration for this AI service.

The AI service historically used TensorFlow/Keras. This migration integrates
the official MambaVision implementation (PyTorch) and the official pretrained
weights distributed as Hugging Face model artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MambaVisionHFConfig:
    """
    Configuration for loading MambaVision via Hugging Face transformers.

    Notes
    -----
    - `repo_id` points to the official NVIDIA Hugging Face model repo.
    - `trust_remote_code=True` is required because the model uses custom code.
    - This config is for image classification; feature extraction is out of scope.
    """

    repo_id: str = "nvidia/MambaVision-T-1K"
    trust_remote_code: bool = True

    image_size: Tuple[int, int] = (224, 224)
    input_channels: int = 3

    num_classes: int = 4

    device_preference: str = "auto"

    @property
    def input_shape_nhwc(self) -> Tuple[int, int, int]:
        return (self.image_size[0], self.image_size[1], self.input_channels)

    def to_dict(self) -> Dict:
        return asdict(self)

