"""
training/checkpoints.py — PyTorch checkpoint lifecycle helpers.

Provides functions for saving, loading, listing, and deleting model
checkpoints independently of the main training loop.

Directory layout written by this module
----------------------------------------
    <output_dir>/
        <architecture>/
            checkpoints/
                <experiment_id>/
                    best_weights.pt           ← torch state_dict
                    checkpoint_info.json      ← metadata snapshot

Usage
-----
    from training.checkpoints import save_checkpoint_info, load_best_weights
    from training.config import TrainingConfig
    import torch.nn as nn

    cfg = TrainingConfig(architecture="mambavision")
    save_checkpoint_info(cfg, "exp-001", metrics={"val_accuracy": 0.97})
    success = load_best_weights(model, cfg, "exp-001")
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from app.core.logging import logger
from training.config import TrainingConfig


# ─── Path helpers ──────────────────────────────────────────────────────────────

def checkpoint_dir(cfg: TrainingConfig, experiment_id: str) -> Path:
    """Return the directory holding checkpoints for one experiment."""
    return cfg.resolved_output_dir / cfg.architecture / "checkpoints" / experiment_id


def best_weights_path(cfg: TrainingConfig, experiment_id: str) -> Path:
    """Return the absolute path to ``best_weights.pt``."""
    return checkpoint_dir(cfg, experiment_id) / "best_weights.pt"


def checkpoint_info_path(cfg: TrainingConfig, experiment_id: str) -> Path:
    """Return the absolute path to ``checkpoint_info.json``."""
    return checkpoint_dir(cfg, experiment_id) / "checkpoint_info.json"


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_checkpoint_info(
    cfg: TrainingConfig,
    experiment_id: str,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    phase: int = 1,
) -> Path:
    """
    Write a ``checkpoint_info.json`` sidecar next to ``best_weights.pt``.

    Parameters
    ----------
    cfg : TrainingConfig
    experiment_id : str
    metrics : dict | None
        Metric snapshot at the best epoch.
    epoch : int | None
        The best epoch number (0-indexed).
    phase : int
        Training phase that produced this checkpoint.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    ckpt_dir = checkpoint_dir(cfg, experiment_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    info_path = checkpoint_info_path(cfg, experiment_id)

    info: Dict[str, Any] = {
        "experiment_id":  experiment_id,
        "architecture":   cfg.architecture,
        "phase":          phase,
        "epoch":          epoch,
        "metrics":        metrics or {},
        "weights_path":   str(best_weights_path(cfg, experiment_id)),
        "saved_at":       datetime.now(timezone.utc).isoformat(),
        "config_summary": {
            "learning_rate":  cfg.learning_rate,
            "batch_size":     cfg.batch_size,
            "image_size":     cfg.image_size,
            "fine_tune":      cfg.fine_tune,
        },
    }

    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    logger.info(f"Checkpoint info saved → {info_path}")
    return info_path


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_best_weights(
    model: nn.Module,
    cfg: TrainingConfig,
    experiment_id: str,
    *,
    device: Optional[torch.device] = None,
) -> bool:
    """
    Load ``best_weights.pt`` into *model* in-place.

    Handles both checkpoint formats:
    - **Full checkpoint** (written by ``_BestCheckpointSaver``): a dict
      containing ``"model_state"`` plus optimizer/scheduler state and
      metadata.  Only the model weights are restored here.
    - **Legacy format**: a plain ``state_dict`` at the top level.

    Parameters
    ----------
    model : nn.Module
        A model whose architecture matches the checkpoint.
    cfg : TrainingConfig
    experiment_id : str
    device : torch.device | None
        Map location for loading weights.  Defaults to CPU.

    Returns
    -------
    bool
        True when weights were loaded; False when no checkpoint exists.
    """
    path = best_weights_path(cfg, experiment_id)
    if not path.exists():
        logger.warning(f"No checkpoint found at {path}")
        return False

    map_loc = device or torch.device("cpu")
    try:
        # weights_only=False is required for full checkpoints that contain
        # non-tensor values (strings, dicts, etc.)
        ckpt = torch.load(str(path), map_location=map_loc, weights_only=False)
    except Exception as exc:
        logger.error(f"Failed to load checkpoint from {path}: {exc}")
        return False

    try:
        # Full checkpoint format: {"model_state": OrderedDict, ...}
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            # Legacy: plain state_dict at top level
            model.load_state_dict(ckpt)
        logger.info(f"Best weights loaded from {path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to restore model weights from {path}: {exc}")
        return False


def load_checkpoint_info(
    cfg: TrainingConfig,
    experiment_id: str,
) -> Optional[Dict[str, Any]]:
    """Return ``checkpoint_info.json`` for one experiment, or None."""
    path = checkpoint_info_path(cfg, experiment_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"Could not read checkpoint info at {path}: {exc}")
        return None


# ─── List / delete ────────────────────────────────────────────────────────────

def list_checkpoints(cfg: TrainingConfig) -> List[Dict[str, Any]]:
    """
    List all experiment checkpoints for one architecture.

    Returns a list of checkpoint info dicts sorted by ``saved_at``
    descending (newest first).
    """
    arch_dir = cfg.resolved_output_dir / cfg.architecture / "checkpoints"
    if not arch_dir.exists():
        return []

    results = []
    for exp_dir in arch_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        info_file    = exp_dir / "checkpoint_info.json"
        weights_file = exp_dir / "best_weights.pt"
        entry: Dict[str, Any] = {"experiment_id": exp_dir.name}
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as fh:
                    entry.update(json.load(fh))
            except Exception:
                pass
        entry["weights_exist"] = weights_file.exists()
        results.append(entry)

    results.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    return results


def delete_checkpoint(
    cfg: TrainingConfig,
    experiment_id: str,
    *,
    confirm: bool = False,
) -> bool:
    """
    Delete the checkpoint directory for *experiment_id*.

    Parameters
    ----------
    confirm : bool
        Must be True to execute (safety guard).

    Returns
    -------
    bool
        True when deletion succeeded.
    """
    if not confirm:
        logger.warning("delete_checkpoint() called without confirm=True — no action taken.")
        return False

    path = checkpoint_dir(cfg, experiment_id)
    if not path.exists():
        logger.warning(f"Checkpoint directory does not exist: {path}")
        return False

    try:
        shutil.rmtree(path)
        logger.info(f"Checkpoint deleted: {path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to delete checkpoint at {path}: {exc}")
        return False
