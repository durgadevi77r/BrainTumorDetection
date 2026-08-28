"""
load_model.py — PyTorch model loading with an in-memory cache.

Two on-disk formats are supported, matching the two formats written by
``save_model.py``:

HF format (MambaVision and any ``save_format="hf"`` model)
-----------------------------------------------------------
    saved_models/<model_name>/
        config.json             ← Hugging Face model config
        model.safetensors       ← weights (or pytorch_model.bin)
        model_info.json         ← training metadata

PT format (cnn / vgg16 / resnet50 / efficientnet)
--------------------------------------------------
    saved_models/<model_name>/
        weights.pt              ← torch.save(state_dict)
        model_info.json         ← training metadata

The cache is keyed by model_name so each model is loaded only once
per process. Call clear_model_cache() between tests or when hot-reloading.

Usage
-----
    from app.models.load_model import load_model
    model = load_model("mambavision")   # HF format — cached after first call
    model = load_model("resnet50")      # PT  format — cached after first call
    preds = model.predict(tensor)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from app.core.config import settings
from app.core.logging import logger

# ── In-memory model cache ─────────────────────────────────────────────────────
# Populated lazily; persists for the lifetime of the process / worker.
_model_cache: Dict[str, Any] = {}

# Sentinel type for the resolved format
_Format = Literal["hf", "pt"]


def _resolve_model_path(model_name: str) -> Tuple[Path, _Format]:
    """
    Locate saved weights for *model_name* inside ``settings.saved_models_dir``
    and determine which on-disk format was used by ``save_model.py``.

    Search order
    ------------
    1. ``saved_models/<model_name>/config.json``              → HF format
    2. ``saved_models/<model_name>/weights.pt``               → PT format
    3. ``saved_models/<model_name>/checkpoints/best_weights.pt``  → PT fallback
       (used when training completed but save_model() was not called, e.g.
        an interrupted run that only wrote a best-checkpoint file)
    4. ``saved_models/<model_name>/checkpoints/<exp_id>/best_weights.pt``
       (experiment-scoped checkpoint written by the Trainer)

    Returns
    -------
    (path, format)
        *path* is the model directory for HF, or the ``weights.pt``/
        ``best_weights.pt`` file for PT.
        *format* is ``"hf"`` or ``"pt"``.

    Raises
    ------
    FileNotFoundError
        When no artefact can be found for the given model name.
    """
    base: Path = settings.saved_models_dir
    model_dir = base / model_name

    # ── HF format: directory containing config.json ───────────────────────────
    if model_dir.is_dir() and (model_dir / "config.json").is_file():
        logger.debug(f"_resolve_model_path: '{model_name}' → HF format at {model_dir}")
        return model_dir, "hf"

    # ── PT format: primary weights.pt ────────────────────────────────────────
    weights_file = model_dir / "weights.pt"
    if model_dir.is_dir() and weights_file.is_file():
        logger.debug(f"_resolve_model_path: '{model_name}' → PT format at {weights_file}")
        return weights_file, "pt"

    # ── PT fallback: checkpoints/best_weights.pt ──────────────────────────────
    # Handles the case where training completed but save_model() was never
    # called (e.g. an interrupted run, or efficientnet whose .keras file is
    # not loadable by the PyTorch loader).
    best_ckpt = model_dir / "checkpoints" / "best_weights.pt"
    if model_dir.is_dir() and best_ckpt.is_file():
        logger.warning(
            f"_resolve_model_path: '{model_name}' — no primary weights found; "
            f"falling back to checkpoint at {best_ckpt}"
        )
        return best_ckpt, "pt"

    # ── PT fallback: experiment-scoped checkpoints/<exp_id>/best_weights.pt ──
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.is_dir():
        # Pick the most recently modified experiment checkpoint
        exp_ckpts = sorted(
            [
                p / "best_weights.pt"
                for p in ckpt_dir.iterdir()
                if p.is_dir() and (p / "best_weights.pt").is_file()
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if exp_ckpts:
            chosen = exp_ckpts[0]
            logger.warning(
                f"_resolve_model_path: '{model_name}' — using experiment checkpoint "
                f"at {chosen}"
            )
            return chosen, "pt"

    raise FileNotFoundError(
        f"No saved model found for '{model_name}' in {base}. "
        "Expected 'config.json' (HF format), 'weights.pt' (PT format), "
        "or 'checkpoints/best_weights.pt'. "
        "Train the model first via POST /api/v1/train."
    )


def _load_hf_model(name: str, model_path: Path) -> Any:
    """
    Load a Hugging Face / MambaVision model from a local directory.

    The directory must contain ``config.json`` (and typically
    ``model.safetensors`` or ``pytorch_model.bin``).

    Loading strategy
    ----------------
    ``save_pretrained`` persists the HF model config as-is.  For MambaVision
    the upstream config stores ``num_labels=1000`` (ImageNet labels) regardless
    of how many classes were used during training, so calling
    ``from_pretrained(local_dir)`` naively would create a 1000-class head and
    then fail with a size-mismatch when loading the saved 4-class weights.

    We therefore:
    1. Build a fresh 4-class skeleton via ``build_mambavision_model`` (same
       architecture used during training, no pretrained weights).
    2. Load the saved ``model.safetensors`` / ``pytorch_model.bin`` as a plain
       state_dict into that skeleton.

    This is safe because both the skeleton and the saved file have identical
    architectures — the only difference from the vanilla HF model is the
    head size.

    Returns a ``TorchImageClassifier`` wrapping the loaded model.
    """
    import torch                                                            # noqa: PLC0415
    from safetensors.torch import load_file as load_safetensors            # noqa: PLC0415
    from app.models.mambavision.factory import build_mambavision_model     # noqa: PLC0415
    from app.models.mambavision.config import MambaVisionHFConfig          # noqa: PLC0415
    from app.models.mambavision.predictor import TorchImageClassifier      # noqa: PLC0415

    hf_cfg = MambaVisionHFConfig()

    # ── Step 1: build the correct 4-class skeleton (no pretrained weights) ──
    try:
        model = build_mambavision_model(cfg=hf_cfg, pretrained=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build MambaVision skeleton for '{name}': {exc}"
        ) from exc

    # Unfreeze all params so load_state_dict works without strict=False issues
    for p in model.parameters():
        p.requires_grad_(True)

    # ── Step 2: locate the weights file ──────────────────────────────────────
    safetensors_path = model_path / "model.safetensors"
    bin_path         = model_path / "pytorch_model.bin"

    try:
        if safetensors_path.is_file():
            state_dict = load_safetensors(str(safetensors_path), device="cpu")
        elif bin_path.is_file():
            state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor pytorch_model.bin found in {model_path}"
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"_load_hf_model '{name}': missing keys: {missing}")
        if unexpected:
            logger.warning(f"_load_hf_model '{name}': unexpected keys: {unexpected}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF model '{name}' from {model_path}: {exc}"
        ) from exc

    logger.info(f"HF model '{name}' weights loaded from {model_path}")
    return TorchImageClassifier(model)


def _load_pt_model(name: str, weights_path: Path) -> Any:
    """
    Reconstruct an architecture from the registry and load a ``state_dict``
    from ``weights.pt``.

    Uses ``build_model(name)`` so the architecture is guaranteed to match
    the one used during training (and saved by ``save_model.py``).

    Returns a ``TorchImageClassifier`` wrapping the reconstructed model.
    """
    import torch                                                       # noqa: PLC0415
    from app.models.architectures import build_model                   # noqa: PLC0415
    from app.models.mambavision.predictor import TorchImageClassifier  # noqa: PLC0415

    # Reconstruct the bare nn.Module (random weights at this point)
    try:
        nn_model = build_model(name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to reconstruct architecture for '{name}': {exc}"
        ) from exc

    # Load state dict — map to CPU first so GPU-saved weights work everywhere
    try:
        state = torch.load(
            str(weights_path),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        nn_model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load state_dict for '{name}' from {weights_path}: {exc}"
        ) from exc

    logger.debug(f"State dict loaded into '{name}' architecture")
    return TorchImageClassifier(nn_model)


def load_model(model_name: Optional[str] = None) -> Any:
    """
    Load (and cache) an image classifier from the ``saved_models`` directory.

    Automatically detects whether the saved artefact is in HF format
    (``config.json`` present) or PT format (``weights.pt`` present) and
    dispatches to the appropriate loader.

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

    # ── Resolve path and format ───────────────────────────────────────────────
    model_path, fmt = _resolve_model_path(name)
    logger.info(f"Loading model '{name}' ({fmt} format) from {model_path} …")

    # ── Dispatch to format-specific loader ────────────────────────────────────
    if fmt == "hf":
        wrapped = _load_hf_model(name, model_path)
    else:
        wrapped = _load_pt_model(name, model_path)

    # ── Cache and return ──────────────────────────────────────────────────────
    _model_cache[name] = wrapped
    total_params = wrapped.count_params()
    logger.info(
        f"Model '{name}' loaded and cached | fmt={fmt} params={total_params:,}"
    )
    return wrapped


def get_model_info(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Return the metadata written by ``save_model()`` as a dict.

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

    Checks for both HF format (config.json) and PT format (weights.pt).

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


# ─── Backward-compatible alias ────────────────────────────────────────────────
# Renamed from load_keras_model as part of the TensorFlow removal (Module 9).
# Kept here so that any code still importing the old name does not break.
load_keras_model = load_model
