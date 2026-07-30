"""
evaluation/loader.py — Checkpoint and model loading for the evaluation pipeline.

Discovers and loads the best available weights for a given architecture,
handling every checkpoint format written by this project:

  1. **Full training checkpoint** (written by ``training.callbacks._BestCheckpointSaver``)::

         {
           "model_state":     OrderedDict,
           "optimizer_state": ...,
           "scheduler_state": ...,
           "epoch":           int,
           "val_loss":        float,
           "val_accuracy":    float,
           "architecture":    str,
           "experiment_id":   str,
           "saved_at":        str,
         }

  2. **Legacy state-dict** (plain ``OrderedDict`` at top level, written by
     ``app.models.save_model.save_best_checkpoint``).

  3. **Hugging Face save_pretrained** directory (``config.json`` +
     ``model.safetensors`` / ``pytorch_model.bin``).  Loaded via
     ``AutoModelForImageClassification.from_pretrained()``.

Search order (most specific → most generic):

    a. ``saved_models/<arch>/checkpoints/<experiment_id>/best_weights.pt``
       (newest experiment first — uses ``training.checkpoints.list_checkpoints``)
    b. ``saved_models/<arch>/checkpoints/best_weights.pt``
       (written by ``app.models.save_model.save_best_checkpoint``)
    c. ``saved_models/<arch>/``
       (Hugging Face directory with ``config.json``)

Usage
-----
    from evaluation.loader import load_eval_model, find_best_checkpoint

    model, meta = load_eval_model("mambavision")
    # meta keys: source, path, epoch, val_loss, val_accuracy, experiment_id

    path = find_best_checkpoint("mambavision")   # raises FileNotFoundError if none
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from app.core.config import settings
from app.core.logging import logger


# ─── Checkpoint search ────────────────────────────────────────────────────────

def find_best_checkpoint(
    architecture: str,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Locate the best available checkpoint for *architecture*.

    Search order
    ------------
    1. Newest experiment checkpoint written by ``training.callbacks``
       (``saved_models/<arch>/checkpoints/<exp_id>/best_weights.pt``).
    2. Legacy checkpoint written by ``app.models.save_model``
       (``saved_models/<arch>/checkpoints/best_weights.pt``).
    3. Hugging Face ``save_pretrained`` directory
       (``saved_models/<arch>/config.json``).

    Parameters
    ----------
    architecture : str
        Model architecture key (e.g. ``"mambavision"``).
    output_dir : Path | None
        Override ``settings.saved_models_dir``.

    Returns
    -------
    tuple[Path, dict]
        ``(checkpoint_path_or_dir, metadata)``

        ``metadata`` always contains a ``"source"`` key:
        ``"training_full"`` | ``"legacy_state_dict"`` | ``"hf_pretrained"``.

    Raises
    ------
    FileNotFoundError
        When no checkpoint of any kind is found.
    """
    base = output_dir or settings.saved_models_dir
    arch = architecture.lower()

    # ── 1. Newest experiment checkpoint ──────────────────────────────────────
    exp_ckpt_dir = base / arch / "checkpoints"
    if exp_ckpt_dir.is_dir():
        # Collect experiment sub-directories (each has its own best_weights.pt)
        candidates = []
        for exp_dir in exp_ckpt_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            weights = exp_dir / "best_weights.pt"
            if not weights.exists():
                continue
            info_file = exp_dir / "checkpoint_info.json"
            saved_at  = ""
            meta: Dict[str, Any] = {}
            if info_file.exists():
                try:
                    with open(info_file, encoding="utf-8") as fh:
                        meta = json.load(fh)
                    saved_at = meta.get("saved_at", "")
                except Exception:
                    pass
            candidates.append((saved_at, weights, meta))

        if candidates:
            # Newest first
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_path, raw_meta = candidates[0]
            metadata = {
                "source":        "training_full",
                "path":          str(best_path),
                "experiment_id": raw_meta.get("experiment_id", ""),
                "epoch":         raw_meta.get("epoch"),
                "val_loss":      raw_meta.get("metrics", {}).get("val_loss"),
                "val_accuracy":  raw_meta.get("metrics", {}).get("val_accuracy"),
                "saved_at":      raw_meta.get("saved_at", ""),
            }
            logger.info(
                f"[EvalLoader] Found experiment checkpoint: {best_path} "
                f"(exp={metadata['experiment_id']})"
            )
            return best_path, metadata

        # ── 2. Legacy flat checkpoint ─────────────────────────────────────────
        legacy = exp_ckpt_dir / "best_weights.pt"
        if legacy.exists():
            metadata = {
                "source":       "legacy_state_dict",
                "path":         str(legacy),
                "experiment_id": "",
                "epoch":        None,
                "val_loss":     None,
                "val_accuracy": None,
                "saved_at":     "",
            }
            logger.info(f"[EvalLoader] Found legacy checkpoint: {legacy}")
            return legacy, metadata

    # ── 3. HF save_pretrained directory ──────────────────────────────────────
    hf_dir = base / arch
    if hf_dir.is_dir() and (hf_dir / "config.json").exists():
        metadata = {
            "source":       "hf_pretrained",
            "path":         str(hf_dir),
            "experiment_id": "",
            "epoch":        None,
            "val_loss":     None,
            "val_accuracy": None,
            "saved_at":     "",
        }
        # Try to read val_accuracy from model_info.json
        info_path = hf_dir / "model_info.json"
        if info_path.exists():
            try:
                with open(info_path, encoding="utf-8") as fh:
                    model_info = json.load(fh)
                metadata["val_accuracy"] = model_info.get("final_val_accuracy")
                metadata["saved_at"]     = model_info.get("saved_at", "")
            except Exception:
                pass
        logger.info(f"[EvalLoader] Found HF model directory: {hf_dir}")
        return hf_dir, metadata

    raise FileNotFoundError(
        f"No checkpoint found for architecture '{arch}' in {base}. "
        "Train the model first via the training pipeline."
    )


# ─── Weight loading ───────────────────────────────────────────────────────────

def _load_pt_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a ``.pt`` file into *model*.

    Handles both the full-checkpoint dict format and a plain state dict.

    Returns
    -------
    dict
        Metadata extracted from the checkpoint (empty dict for plain state dicts).
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # Full checkpoint (training.callbacks format)
        model.load_state_dict(ckpt["model_state"])
        meta = {k: v for k, v in ckpt.items()
                if k not in ("model_state", "optimizer_state", "scheduler_state")}
        logger.info(
            f"[EvalLoader] Full checkpoint loaded from {path} "
            f"(epoch={meta.get('epoch')} val_loss={meta.get('val_loss')})"
        )
        return meta
    else:
        # Plain state dict
        model.load_state_dict(ckpt)
        logger.info(f"[EvalLoader] State-dict checkpoint loaded from {path}")
        return {}


# ─── Public entry point ───────────────────────────────────────────────────────

def load_eval_model(
    architecture: str,
    *,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    num_classes: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load the best available checkpoint for *architecture* and return a model
    ready for evaluation.

    The model is returned in ``eval()`` mode on *device*.

    Parameters
    ----------
    architecture : str
        Architecture key: ``"mambavision"`` | ``"cnn"`` | ``"vgg16"`` |
        ``"resnet50"`` | ``"efficientnet"``.
    output_dir : Path | None
        Override ``settings.saved_models_dir``.
    device : torch.device | None
        Target device. Defaults to CUDA when available, else CPU.
    num_classes : int | None
        Number of output classes. Defaults to ``settings.num_classes``.

    Returns
    -------
    tuple[nn.Module, dict]
        ``(model, checkpoint_meta)``

        ``checkpoint_meta`` contains at least:
        ``source``, ``path``, ``epoch``, ``val_loss``, ``val_accuracy``,
        ``experiment_id``, ``saved_at``.

    Raises
    ------
    FileNotFoundError
        When no checkpoint is found.
    RuntimeError
        When the checkpoint cannot be loaded.
    """
    from app.models.architectures import build_model  # lazy import

    arch      = architecture.lower()
    n_classes = num_classes or settings.num_classes
    _device   = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    ckpt_path, search_meta = find_best_checkpoint(arch, output_dir=output_dir)
    source = search_meta.get("source", "unknown")

    try:
        if source == "hf_pretrained":
            # Load straight from the HF directory — no skeleton model needed
            from transformers import AutoModelForImageClassification  # noqa: PLC0415
            from app.models.mambavision.config import MambaVisionHFConfig  # noqa: PLC0415

            hf_cfg = MambaVisionHFConfig(num_classes=n_classes)
            model = AutoModelForImageClassification.from_pretrained(
                str(ckpt_path),
                trust_remote_code=hf_cfg.trust_remote_code,
            )
            ckpt_meta: Dict[str, Any] = {}
        else:
            # Build the skeleton and load weights into it
            model = build_model(arch, num_classes=n_classes)
            ckpt_meta = _load_pt_checkpoint(model, Path(ckpt_path), _device)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint for '{arch}' from {ckpt_path}: {exc}"
        ) from exc

    model.to(_device)
    model.eval()

    # Merge search metadata with checkpoint metadata (checkpoint takes precedence
    # for the keys it provides, but we keep the "source" key from search_meta)
    merged_meta: Dict[str, Any] = {**search_meta, **ckpt_meta}
    merged_meta["source"] = source   # always keep search-level source tag

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[EvalLoader] Model ready | arch={arch} source={source} "
        f"device={_device} params={total_params:,}"
    )
    return model, merged_meta
