"""
training/callbacks.py — PyTorch training utilities (replaces Keras callbacks).

Provides ``build_callbacks()`` which returns a ``CallbackBundle`` — a
lightweight container the Trainer passes into its training loop.

Behaviours reproduced
---------------------
1. BestCheckpoint     — saves best-epoch weights when val_loss improves.
2. EarlyStopping      — stops when val_loss has not improved for ``patience``
                        epochs.
3. ReduceLROnPlateau  — wraps ``torch.optim.lr_scheduler.ReduceLROnPlateau``.
4. CSVLogger          — appends one row per epoch to a CSV file.

Usage
-----
    from training.config import TrainingConfig
    from training.callbacks import build_callbacks

    cfg     = TrainingConfig(architecture="mambavision")
    bundle  = build_callbacks(cfg, experiment_id="exp-001")

    # Inside the training loop:
    stop = bundle.on_epoch_end(
        epoch=epoch,
        val_loss=val_loss,
        metrics={"loss": tl, "accuracy": ta, "val_loss": vl, "val_accuracy": va},
        model=model,
        optimizer=optimizer,
    )
    if stop:
        break
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from app.core.config import settings
from app.core.logging import logger
from training.config import TrainingConfig


# ─── Individual callback helpers ──────────────────────────────────────────────

class _BestCheckpointSaver:
    """Saves model state_dict when val_loss reaches a new minimum."""

    def __init__(self, checkpoint_path: Path, min_delta: float = 1e-4) -> None:
        self.path       = checkpoint_path
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Parameters
        ----------
        val_loss : float
        model : nn.Module

        Returns
        -------
        bool
            True if a new best was recorded and weights were saved.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), str(self.path))
            logger.debug(f"Best checkpoint saved → {self.path} (val_loss={val_loss:.6f})")
            return True
        return False


class _EarlyStopper:
    """Triggers when val_loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0

    def step(self, val_loss: float) -> bool:
        """
        Returns
        -------
        bool
            True when training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                f"EarlyStopping: no improvement for {self.patience} epochs "
                f"(best={self.best_loss:.6f})"
            )
            return True
        return False


class _CSVLogger:
    """Appends one row per epoch to a CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self.path    = csv_path
        self._header = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, epoch: int, metrics: Dict[str, Any]) -> None:
        row = {"epoch": epoch, **metrics}
        write_header = not self.path.exists() or not self._header
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header = True
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                              for k, v in row.items()})


# ─── Path helpers ──────────────────────────────────────────────────────────────

def _checkpoint_path(cfg: TrainingConfig, experiment_id: str) -> Path:
    """Return the path where best-epoch weights are written."""
    base     = cfg.resolved_output_dir
    ckpt_dir = base / cfg.architecture / "checkpoints" / experiment_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / "best_weights.pt"


def _csv_log_path(cfg: TrainingConfig, experiment_id: str, phase: int) -> Path:
    log_dir = settings.log_dir / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{experiment_id}_phase{phase}.csv"


# ─── CallbackBundle ───────────────────────────────────────────────────────────

class CallbackBundle:
    """
    Container that orchestrates all callbacks for one training phase.

    Parameters
    ----------
    checkpoint_saver : _BestCheckpointSaver
    early_stopper : _EarlyStopper
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
    csv_logger : _CSVLogger | None
    """

    def __init__(
        self,
        checkpoint_saver: _BestCheckpointSaver,
        early_stopper: _EarlyStopper,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        csv_logger: Optional[_CSVLogger] = None,
    ) -> None:
        self._ckpt    = checkpoint_saver
        self._es      = early_stopper
        self._sched   = scheduler
        self._csv     = csv_logger

    def on_epoch_end(
        self,
        *,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, Any],
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> bool:
        """
        Call at the end of every epoch.

        Returns
        -------
        bool
            True when training should stop (early stopping triggered).
        """
        # 1. Checkpoint
        self._ckpt.step(val_loss, model)

        # 2. LR scheduler
        self._sched.step(val_loss)

        # 3. CSV log
        if self._csv is not None:
            lr = optimizer.param_groups[0]["lr"]
            self._csv.log(epoch, {**metrics, "lr": lr})

        # 4. Early stopping (checked last so the checkpoint is always saved)
        return self._es.step(val_loss)

    @property
    def best_checkpoint_path(self) -> Path:
        return self._ckpt.path


# ─── Public factory ───────────────────────────────────────────────────────────

def build_callbacks(
    cfg: TrainingConfig,
    experiment_id: str,
    *,
    phase: int = 1,
    optimizer: Optional[optim.Optimizer] = None,
) -> CallbackBundle:
    """
    Build a ``CallbackBundle`` for one training phase.

    Parameters
    ----------
    cfg : TrainingConfig
        Training configuration.
    experiment_id : str
        Unique experiment identifier used to namespace files.
    phase : int
        Training phase (1 = head training, 2 = fine-tuning).
        Phase 2 uses a tighter EarlyStopping patience
        (``cfg.early_stopping_patience // 2 + 3``).
    optimizer : optim.Optimizer | None
        Passed to ReduceLROnPlateau at bundle construction time.
        Can also be passed at ``on_epoch_end()`` call time.

    Returns
    -------
    CallbackBundle
    """
    ckpt_path = _checkpoint_path(cfg, experiment_id)

    checkpoint_saver = _BestCheckpointSaver(ckpt_path)

    es_patience = (
        cfg.early_stopping_patience
        if phase == 1
        else max(cfg.early_stopping_patience // 2, 3)
    )
    early_stopper = _EarlyStopper(patience=es_patience)

    # Dummy optimizer if none provided — scheduler will be replaced when
    # on_epoch_end is called with the real optimizer.
    _opt = optimizer or optim.SGD([torch.zeros(1, requires_grad=True)], lr=cfg.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        _opt,
        mode="min",
        factor=cfg.reduce_lr_factor,
        patience=cfg.reduce_lr_patience,
        min_lr=cfg.reduce_lr_min,
    )

    csv_logger: Optional[_CSVLogger] = None
    if cfg.csv_log:
        csv_logger = _CSVLogger(_csv_log_path(cfg, experiment_id, phase))

    logger.info(
        f"CallbackBundle built | experiment={experiment_id} phase={phase} "
        f"es_patience={es_patience} ckpt={ckpt_path}"
    )
    return CallbackBundle(checkpoint_saver, early_stopper, scheduler, csv_logger)


def get_best_checkpoint_path(
    cfg: TrainingConfig,
    experiment_id: str,
) -> Optional[Path]:
    """
    Return the path to the best checkpoint if it exists, else None.
    """
    p = _checkpoint_path(cfg, experiment_id)
    return p if p.exists() else None
