"""
tests/test_imports.py — Smoke tests verifying all modules import cleanly.

If any import fails here the service cannot start, so this is the fastest
signal that a dependency is missing or a module has a syntax error.
"""

from __future__ import annotations

import importlib

import pytest

THIRD_PARTY = [
    "fastapi",
    "uvicorn",
    "multipart",
    "torch",
    "torchvision",
    "transformers",
    # "mambavision" is not a separate pip package — it ships as the local
    # app.models.mambavision sub-package and is tested via APP_MODULES below.
    "cv2",
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "PIL",
    "dotenv",
    "pydantic",
    "pydantic_settings",
    "loguru",
]

APP_MODULES = [
    # Core
    "app.core.config",
    "app.core.logging",
    # Application
    "app.main",
    "app.api.routes",
    # Preprocessing
    "app.preprocessing.config",
    "app.preprocessing.transforms",
    "app.preprocessing.augmentation",
    "app.preprocessing.preprocess",
    "app.preprocessing",
    # Models — MambaVision sub-package
    "app.models.mambavision.config",
    "app.models.mambavision.factory",
    "app.models.mambavision.predictor",
    "app.models.mambavision.storage",
    # Models — main pipeline
    "app.models.architectures",
    "app.models.train",
    "app.models.predict",
    "app.models.evaluate",
    "app.models.load_model",
    "app.models.save_model",
    # Training package
    "training.config",
    "training.callbacks",
    "training.checkpoints",
    "training.experiment",
    "training.trainer",
    # Utils
    "app.utils.gradcam",
]


@pytest.mark.parametrize("module_name", THIRD_PARTY)
def test_third_party_import(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", APP_MODULES)
def test_app_module_import(module_name: str) -> None:
    importlib.import_module(module_name)
