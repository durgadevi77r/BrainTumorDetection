"""
load_model.py — PyTorch model loading with an in-memory cache.

The cache is keyed by model_name so each model is loaded only once
per process. Call clear_model_cache() between tests or when hot-reloading.

Usage
-----
    from app.models.load_model import load_keras_model
    model = load_keras_model("efficientnet")   # cached after first call
    preds = model.predict(tensor)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging import logger

# ── In-memory model cache ─────────────────────────────────────────────────────
# Populated lazily; persists for the lifetime of the process / worker.
_model_cache: Dict[str, Any] = {}


def _resolve_model_path(model_name: str) -> Path:
    """
    Locate saved weights for *model_name* inside ``settings.saved_models_dir``.

    Search order
    ------------
    1. ``saved_models/<model_name>/`` — Hugging Face model directory (config.json)

    Raises
    ------
    FileNotFoundError
        When no artefact can be found for the given model name.
    """
    base: Path = settings.saved_models_dir

    model_dir = base / model_name
    if model_dir.is_dir() and (model_dir / "config.json").is_file():
        return model_dir

    raise FileNotFoundError(
        f"No saved model found for '{model_name}' in {base}. "
        "Train the model first via POST /api/v1/train."
    )


def load_keras_model(model_name: Optional[str] = None) -> Any:
    """
    Load (and cache) an image classifier from the ``saved_models`` directory.

    Parameters
    ----------
    model_name : str | None
        Defaults to ``settings.active_model`` when *None*.

    Returns
    -------
    TorchImageClassifier
        Adapter exposing ``predict(np.ndarray) -> np.ndarray``.

    Raises
    ------
    FileNotFoundError
        When no saved weights exist for the requested model.
    RuntimeError
        When the model cannot be deserialised.
    """
    name = (model_name or settings.active_model).lower()

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if name in _model_cache:
        logger.debug(f"Model cache hit for '{name}'")
        return _model_cache[name]

    # ── Resolve path ──────────────────────────────────────────────────────────
    model_path = _resolve_model_path(name)
    logger.info(f"Loading model '{name}' from {model_path} …")

    # ── Load ──────────────────────────────────────────────────────────────────
    # Lazy imports: torch and transformers are heavy and not available in
    # lightweight test environments — only import when actually loading a model.
    from transformers import AutoModelForImageClassification  # noqa: PLC0415
    from app.models.mambavision.config import MambaVisionHFConfig  # noqa: PLC0415
    from app.models.mambavision.predictor import TorchImageClassifier  # noqa: PLC0415

    hf_cfg = MambaVisionHFConfig()
    try:
        model = AutoModelForImageClassification.from_pretrained(
            str(model_path),
            trust_remote_code=hf_cfg.trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{name}' from {model_path}: {exc}"
        ) from exc

    wrapped = TorchImageClassifier(model)

    _model_cache[name] = wrapped
    total_params = wrapped.count_params()
    logger.info(
        f"Model '{name}' loaded and cached | params={total_params:,}"
    )
    return wrapped


def get_model_info(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Return the metadata written by ``save_keras_model()`` as a dict.

    Parameters
    ----------
    model_name : str | None
        Defaults to ``settings.active_model``.

    Returns
    -------
    dict
        Contents of ``model_info.json``, or an empty dict if the file is
        absent (e.g. model was saved by an earlier version of this code).
    """
    name = (model_name or settings.active_model).lower()
    info_path = settings.saved_models_dir / name / "model_info.json"

    if not info_path.exists():
        logger.debug(f"No model_info.json found for '{name}' at {info_path}")
        return {}

    try:
        with open(info_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"Could not read model_info.json for '{name}': {exc}")
        return {}


def is_model_available(model_name: Optional[str] = None) -> bool:
    """
    Return True if saved weights exist for *model_name* (no loading).

    Parameters
    ----------
    model_name : str | None
        Defaults to ``settings.active_model``.
    """
    name = (model_name or settings.active_model).lower()
    try:
        _resolve_model_path(name)
        return True
    except FileNotFoundError:
        return False


def clear_model_cache(model_name: Optional[str] = None) -> None:
    """
    Evict one or all models from the in-memory cache.

    Parameters
    ----------
    model_name : str | None
        When given, removes only that architecture; when *None* clears all.
    """
    if model_name:
        name = model_name.lower()
        if name in _model_cache:
            del _model_cache[name]
            logger.info(f"Evicted '{name}' from model cache.")
    else:
        _model_cache.clear()
        logger.info("Model cache cleared.")
