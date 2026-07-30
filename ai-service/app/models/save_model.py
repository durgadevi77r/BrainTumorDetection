"""
save_model.py — PyTorch model persistence with companion metadata.

Two storage formats are used depending on architecture:

MambaVision (Hugging Face model)
---------------------------------
    saved_models/<model_name>/
        config.json             ← HF model config (written by save_pretrained)
        model.safetensors       ← weights in safetensors format
        model_info.json         ← training metadata snapshot

All other architectures (cnn / vgg16 / resnet50 / efficientnet)
----------------------------------------------------------------
    saved_models/<model_name>/
        weights.pt              ← torch.save(state_dict)
        model_info.json         ← training metadata snapshot

The ``model_info.json`` file is read back by ``load_model.get_model_info()``
and surfaced through the /evaluate and /health endpoints.

Usage
-----
    from app.models.save_model import save_keras_model
    paths = save_keras_model(model, "mambavision", metadata={"val_accuracy": 0.97})

The function is deliberately named ``save_keras_model`` to keep the call
sites in train.py and trainer.py unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

from app.core.config import settings
from app.core.logging import logger


def _is_hf_model(model: nn.Module) -> bool:
    """Return True when *model* is a Hugging Face transformers model."""
    try:
        from transformers import PreTrainedModel
        return isinstance(model, PreTrainedModel)
    except ImportError:
        return False


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─── Public save entry point ──────────────────────────────────────────────────

def save_keras_model(
    model: nn.Module,
    model_name: str,
    *,
    output_dir: Optional[str | Path] = None,
    save_format: str = "auto",          # "auto" | "hf" | "pt"
    also_save_h5: bool = False,         # legacy kwarg — silently ignored
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Persist a trained PyTorch model to disk with a metadata JSON sidecar.

    Parameters
    ----------
    model : nn.Module
        The trained model to save.
    model_name : str
        Sub-directory name under ``saved_models/`` (e.g. ``"mambavision"``).
    output_dir : str | Path | None
        Override the default ``settings.saved_models_dir``.
    save_format : str
        ``"auto"`` — uses HF save_pretrained for HF models, state_dict for
                      all others.
        ``"hf"``   — force Hugging Face save_pretrained (requires HF model).
        ``"pt"``   — force torch.save(state_dict).
    also_save_h5 : bool
        Legacy Keras kwarg — accepted and silently ignored.
    metadata : dict | None
        Arbitrary key/value pairs to include in ``model_info.json``.

    Returns
    -------
    dict
        {
            "model_dir":    str,   # absolute path to the model directory
            "model_path":   str,   # absolute path to the primary artefact
            "h5_path":      str,   # always "" (legacy compat key)
            "info_path":    str,   # absolute path to model_info.json
            "format":       str,   # "hf" or "pt"
        }

    Raises
    ------
    ValueError
        If *save_format* is not one of "auto" | "hf" | "pt".
    RuntimeError
        If the model cannot be serialised.
    """
    if save_format not in {"auto", "hf", "pt"}:
        raise ValueError(f"save_format must be 'auto', 'hf', or 'pt', got '{save_format}'")

    name      = model_name.lower()
    base_dir  = Path(output_dir) if output_dir else settings.saved_models_dir
    model_dir = base_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Resolve actual format
    use_hf = (save_format == "hf") or (save_format == "auto" and _is_hf_model(model))

    if use_hf:
        fmt, model_path = _save_hf(model, model_dir, name)
    else:
        fmt, model_path = _save_pt(model, model_dir, name)

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    info_path = model_dir / "model_info.json"
    model_info: Dict[str, Any] = {
        "model_name":    name,
        "save_format":   fmt,
        "input_shape":   list(settings.input_shape),
        "num_classes":   settings.num_classes,
        "class_names":   settings.classes,
        "total_params":  _count_params(model),
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "model_path":    str(model_path),
        "h5_path":       "",
    }

    if metadata:
        model_info.update(metadata)

    try:
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(model_info, fh, indent=2)
        logger.info(f"model_info.json written → {info_path}")
    except Exception as exc:
        logger.warning(f"Could not write model_info.json: {exc}")

    return {
        "model_dir":  str(model_dir),
        "model_path": str(model_path),
        "h5_path":    "",
        "info_path":  str(info_path),
        "format":     fmt,
    }


def _save_hf(
    model: nn.Module,
    model_dir: Path,
    name: str,
) -> tuple[str, Path]:
    """Save using Hugging Face ``save_pretrained``."""
    try:
        model.save_pretrained(str(model_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save HF model '{name}' via save_pretrained: {exc}"
        ) from exc

    # Primary artefact is either model.safetensors or pytorch_model.bin
    model_path = model_dir / "model.safetensors"
    if not model_path.exists():
        model_path = model_dir / "pytorch_model.bin"
    logger.info(f"Model '{name}' saved via save_pretrained → {model_dir}")
    return "hf", model_path


def _save_pt(
    model: nn.Module,
    model_dir: Path,
    name: str,
) -> tuple[str, Path]:
    """Save using ``torch.save(state_dict)``."""
    model_path = model_dir / "weights.pt"
    try:
        torch.save(model.state_dict(), str(model_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save model '{name}' state_dict: {exc}"
        ) from exc
    logger.info(f"Model '{name}' state_dict saved → {model_path}")
    return "pt", model_path


# ─── Checkpoint callback replacement ──────────────────────────────────────────

def save_best_checkpoint(
    model: nn.Module,
    model_name: str,
    *,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Save the current model state_dict as the best checkpoint.

    Called by the training loop whenever a new best val_accuracy is reached.
    Replaces the old Keras ``ModelCheckpoint`` callback.

    Writes to::
        saved_models/<model_name>/checkpoints/best_weights.pt

    Parameters
    ----------
    model : nn.Module
        Model whose weights to checkpoint.
    model_name : str
        Sub-directory key under ``saved_models/``.
    output_dir : str | Path | None
        Override the default ``settings.saved_models_dir``.

    Returns
    -------
    Path
        Absolute path to the written checkpoint file.
    """
    base_dir       = Path(output_dir) if output_dir else settings.saved_models_dir
    checkpoint_dir = base_dir / model_name.lower() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_weights.pt"

    torch.save(model.state_dict(), str(checkpoint_path))
    logger.debug(f"Best checkpoint saved → {checkpoint_path}")
    return checkpoint_path


def load_best_checkpoint(
    model: nn.Module,
    model_name: str,
    *,
    output_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> bool:
    """
    Load the best checkpoint weights into *model* in-place.

    Parameters
    ----------
    model : nn.Module
    model_name : str
    output_dir : str | Path | None
    device : torch.device | None
        Map location for loading weights.

    Returns
    -------
    bool
        True when checkpoint was loaded; False when no file exists.
    """
    base_dir        = Path(output_dir) if output_dir else settings.saved_models_dir
    checkpoint_path = base_dir / model_name.lower() / "checkpoints" / "best_weights.pt"

    if not checkpoint_path.exists():
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        return False

    map_location = device or torch.device("cpu")
    try:
        state = torch.load(str(checkpoint_path), map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        logger.info(f"Best checkpoint loaded from {checkpoint_path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {exc}")
        return False


# ─── Legacy alias kept for any direct callers ─────────────────────────────────
# ``save_best_checkpoint_callback`` was a Keras ModelCheckpoint factory.
# Callers in trainer.py now call save_best_checkpoint() directly inside
# the training loop, but we keep this name to avoid NameError in any
# code that imported it.

def save_best_checkpoint_callback(
    model_name: str,
    *,
    monitor: str = "val_accuracy",
    output_dir: Optional[str | Path] = None,
) -> None:
    """
    Legacy stub — previously returned a ``tf.keras.callbacks.ModelCheckpoint``.

    Now a no-op: the training loop in ``train.py`` calls
    ``save_best_checkpoint(model, model_name)`` directly instead of
    relying on a Keras callback.  Kept here so that any remaining import
    of this name does not raise ``ImportError``.
    """
    logger.debug(
        f"save_best_checkpoint_callback: no-op stub called for '{model_name}' "
        "(Keras callbacks are removed — use save_best_checkpoint() directly)"
    )
    return None
