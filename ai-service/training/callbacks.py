"""
training/callbacks.py — PyTorch training utilities (replaces Keras callbacks).

Provides ``build_callbacks()`` which returns a ``CallbackBundle`` — a
lightweight container the Trainer passes into its training loop.

Behaviours provided
-------------------
1. BestCheckpoint     — saves a *full* checkpoint (model weights, optimiser
                        state, scheduler state, metadata) when val_loss
                        improves.
2. EarlyStopping      — stops when val_loss has not improved for ``patience``
                        epochs.
3. Scheduler step     — supports both ReduceLROnPlateau and CosineAnnealingLR.
4. CSVLogger          — appends one row per epoch to a CSV file.

Full checkpoint format
-----------------------
    {
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "epoch":            int,
        "val_loss":         float,
        "val_accuracy":     float,
        "architecture":     str,
        "experiment_id":    str,
        "saved_at":         ISO-8601 str,
    }

Usage
-----
    from training.config import TrainingConfig
    from training.callbacks import build_callbacks

    cfg     = TrainingConfig(architecture="mambavision")
    bundle  = build_callbacks(cfg, experiment_id="exp-001", optimizer=opt)

    # Inside the training loop:
    stop = bundle.on_epoch_end(
        epoch=epoch,
        val_loss=val_loss,
        metrics={"loss": tl, "accuracy": ta, "val_loss": vl, "val_accuracy": va},
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if stop:
        break
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from app.core.config import settings
from app.core.logging import logger
from training.config import TrainingConfig


# ─── Scheduler type alias ─────────────────────────────────────────────────────
_Scheduler = Union[
    lr_sched.ReduceLROnPlateau,
    lr_sched.CosineAnnealingLR,
]


# ─── Individual callback helpers ──────────────────────────────────────────────

class _BestCheckpointSaver:
    """
    Saves a full training checkpoint when val_loss reaches a new minimum.

    The checkpoint contains model weights, optimiser state, scheduler state,
    and training metadata so training can be resumed from any best epoch.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        min_delta: float = 1e-4,
        architecture: str = "",
        experiment_id: str = "",
    ) -> None:
        self.path          = checkpoint_path
        self.min_delta     = min_delta
        self.best_loss     = float("inf")
        self.architecture  = architecture
        self.experiment_id = experiment_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def step(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: _Scheduler,
        *,
        epoch: int,
        val_accuracy: float = 0.0,
    ) -> bool:
        """
        Save a full checkpoint when val_loss improves.

        Returns
        -------
        bool
            True when a new best was recorded and a checkpoint was written.
        """
        if val_loss >= self.best_loss - self.min_delta:
            return False

        self.best_loss = val_loss
        checkpoint = {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,
            "val_loss":        val_loss,
            "val_accuracy":    val_accuracy,
            "architecture":    self.architecture,
            "experiment_id":   self.experiment_id,
            "saved_at":        datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, str(self.path))
        logger.debug(
            f"Best checkpoint saved → {self.path} "
            f"(epoch={epoch} val_loss={val_loss:.6f} val_acc={val_accuracy:.4f})"
        )
        return True


class _EarlyStopper:
    """Triggers when val_loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0

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
            writer.writerow(
                {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()}
            )


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _checkpoint_path(cfg: TrainingConfig, experiment_id: str) -> Path:
    """Return the path where the best full checkpoint is written."""
    base     = cfg.resolved_output_dir
    ckpt_dir = base / cfg.architecture / "checkpoints" / experiment_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / "best_weights.pt"


def _csv_log_path(cfg: TrainingConfig, experiment_id: str, phase: int) -> Path:
    log_dir = settings.log_dir / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{experiment_id}_phase{phase}.csv"


# ─── Scheduler factory ────────────────────────────────────────────────────────

def build_scheduler(
    cfg: TrainingConfig,
    optimizer: optim.Optimizer,
    *,
    phase: int = 1,
    max_epochs: Optional[int] = None,
) -> _Scheduler:
    """
    Build the LR scheduler requested by ``cfg.scheduler``.

    Parameters
    ----------
    cfg : TrainingConfig
    optimizer : optim.Optimizer
    phase : int
        Phase number (1 or 2). Phase-2 uses tighter ReduceLROnPlateau patience.
    max_epochs : int | None
        Override CosineAnnealingLR T_max (defaults to ``cfg.effective_cosine_t_max``).

    Returns
    -------
    ReduceLROnPlateau | CosineAnnealingLR
    """
    if cfg.scheduler == "cosine":
        t_max = max_epochs or cfg.effective_cosine_t_max
        sched = lr_sched.CosineAnnealingLR(
            optimizer,
            T_max=max(t_max, 1),
            eta_min=cfg.cosine_eta_min,
        )
        logger.info(
            f"CosineAnnealingLR | T_max={t_max} eta_min={cfg.cosine_eta_min:.2e}"
        )
        return sched

    # Default: ReduceLROnPlateau
    patience = (
        cfg.reduce_lr_patience
        if phase == 1
        else max(cfg.reduce_lr_patience // 2, 2)
    )
    sched = lr_sched.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.reduce_lr_factor,
        patience=patience,
        min_lr=cfg.reduce_lr_min,
    )
    logger.info(
        f"ReduceLROnPlateau | factor={cfg.reduce_lr_factor} "
        f"patience={patience} min_lr={cfg.reduce_lr_min:.2e}"
    )
    return sched


# ─── CallbackBundle ───────────────────────────────────────────────────────────

class CallbackBundle:
    """
    Container that orchestrates all callbacks for one training phase.

    Parameters
    ----------
    checkpoint_saver : _BestCheckpointSaver
    early_stopper : _EarlyStopper
    scheduler : ReduceLROnPlateau | CosineAnnealingLR
    csv_logger : _CSVLogger | None
    """

    def __init__(
        self,
        checkpoint_saver: _BestCheckpointSaver,
        early_stopper: _EarlyStopper,
        scheduler: _Scheduler,
        csv_logger: Optional[_CSVLogger] = None,
    ) -> None:
        self._ckpt  = checkpoint_saver
        self._es    = early_stopper
        self._sched = scheduler
        self._csv   = csv_logger

    def on_epoch_end(
        self,
        *,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, Any],
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_Scheduler] = None,
    ) -> bool:
        """
        Call at the end of every epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-indexed).
        val_loss : float
            Validation loss for this epoch.
        metrics : dict
            Full metric dict (loss, accuracy, val_loss, val_accuracy).
        model : nn.Module
        optimizer : optim.Optimizer
        scheduler : _Scheduler | None
            If provided, overrides the bundle's internal scheduler for the
            step call. This lets the Trainer pass the authoritative scheduler
            instance when it creates one externally.

        Returns
        -------
        bool
            True when training should stop (early stopping triggered).
        """
        val_accuracy = float(metrics.get("val_accuracy", 0.0))
        active_sched = scheduler if scheduler is not None else self._sched

        # 1. Full checkpoint (model + optimizer + scheduler state)
        self._ckpt.step(
            val_loss, model, optimizer, active_sched,
            epoch=epoch, val_accuracy=val_accuracy,
        )

        # 2. LR scheduler step
        if isinstance(active_sched, lr_sched.ReduceLROnPlateau):
            active_sched.step(val_loss)
        else:
            # CosineAnnealingLR and other epoch-based schedulers
            active_sched.step()

        # 3. CSV log
        if self._csv is not None:
            lr = optimizer.param_groups[0]["lr"]
            self._csv.log(epoch, {**metrics, "lr": lr})

        # 4. Early stopping (after checkpoint so the best epoch is always saved)
        return self._es.step(val_loss)

    @property
    def best_checkpoint_path(self) -> Path:
        return self._ckpt.path

    @property
    def scheduler(self) -> _Scheduler:
        """Return the bundle's internal scheduler instance."""
        return self._sched


# ─── Checkpoint load helper ───────────────────────────────────────────────────

def load_best_checkpoint_full(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[_Scheduler] = None,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a full checkpoint written by ``_BestCheckpointSaver``.

    Restores model weights (required) and optionally optimizer/scheduler
    states. Silently skips optimizer/scheduler restore when they are None.

    Parameters
    ----------
    path : Path
        Path to the ``best_weights.pt`` checkpoint file.
    model : nn.Module
        Model to load weights into (in-place).
    optimizer : optim.Optimizer | None
        Optimizer to restore state into (optional).
    scheduler : _Scheduler | None
        Scheduler to restore state into (optional).
    device : torch.device | None
        Map location for weight loading.

    Returns
    -------
    dict
        The metadata portion of the checkpoint (epoch, val_loss, etc.).

    Raises
    ------
    FileNotFoundError
        When the checkpoint file does not exist.
    RuntimeError
        When the checkpoint cannot be loaded.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_loc = device or torch.device("cpu")
    try:
        ckpt = torch.load(str(path), map_location=map_loc, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc

    # Restore model weights (always required)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        # Legacy format — plain state_dict at top level
        model.load_state_dict(ckpt)

    # Optionally restore optimizer and scheduler
    if optimizer is not None and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as exc:
            logger.warning(f"Could not restore optimizer state: {exc}")

    if scheduler is not None and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as exc:
            logger.warning(f"Could not restore scheduler state: {exc}")

    metadata = {k: v for k, v in ckpt.items()
                if k not in ("model_state", "optimizer_state", "scheduler_state")}
    logger.info(
        f"Checkpoint loaded from {path} "
        f"(epoch={metadata.get('epoch')}, val_loss={metadata.get('val_loss')})"
    )
    return metadata


# ─── Public factory ───────────────────────────────────────────────────────────

def build_callbacks(
    cfg: TrainingConfig,
    experiment_id: str,
    *,
    phase: int = 1,
    optimizer: Optional[optim.Optimizer] = None,
    max_epochs: Optional[int] = None,
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
        Phase 2 uses tighter EarlyStopping and ReduceLROnPlateau patience.
    optimizer : optim.Optimizer | None
        Required for scheduler construction. A dummy SGD is used when None
        (the scheduler will be replaced by the Trainer's authoritative one).
    max_epochs : int | None
        Override T_max for CosineAnnealingLR. Defaults to cfg.epochs.

    Returns
    -------
    CallbackBundle
    """
    ckpt_path = _checkpoint_path(cfg, experiment_id)

    checkpoint_saver = _BestCheckpointSaver(
        ckpt_path,
        architecture=cfg.architecture,
        experiment_id=experiment_id,
    )

    es_patience = (
        cfg.early_stopping_patience
        if phase == 1
        else max(cfg.early_stopping_patience // 2, 3)
    )
    early_stopper = _EarlyStopper(patience=es_patience)

    _opt = optimizer or optim.SGD(
        [torch.zeros(1, requires_grad=True)], lr=cfg.learning_rate
    )
    scheduler = build_scheduler(
        cfg, _opt, phase=phase, max_epochs=max_epochs
    )

    csv_logger: Optional[_CSVLogger] = None
    if cfg.csv_log:
        csv_logger = _CSVLogger(_csv_log_path(cfg, experiment_id, phase))

    logger.info(
        f"CallbackBundle built | experiment={experiment_id} phase={phase} "
        f"scheduler={cfg.scheduler} es_patience={es_patience} ckpt={ckpt_path}"
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
