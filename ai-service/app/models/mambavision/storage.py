"""
storage.py — Model artifact IO for official MambaVision (Hugging Face format).

This AI service stores model artifacts under `saved_models/<model_name>/`.
For MambaVision, we store a Hugging Face `save_pretrained(...)` directory so
the official model code can be reloaded via `from_pretrained(local_dir, ...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import AutoModelForImageClassification

from app.core.config import settings


def model_dir(model_name: str) -> Path:
    return settings.saved_models_dir / model_name.lower()


def is_hf_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "config.json").is_file()


def save_hf_model(model: AutoModelForImageClassification, *, model_name: str) -> Path:
    path = model_dir(model_name)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def load_hf_model_dir(*, model_name: str) -> Optional[Path]:
    path = model_dir(model_name)
    return path if is_hf_model_dir(path) else None

