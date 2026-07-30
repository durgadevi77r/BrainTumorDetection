"""
tests/test_training_callbacks.py — Unit tests for training.callbacks.

Tests
-----
- build_callbacks returns a CallbackBundle.
- CallbackBundle.on_epoch_end saves a checkpoint when val_loss improves.
- CallbackBundle.on_epoch_end returns False before patience is exceeded.
- CallbackBundle.on_epoch_end returns True (stop) after patience epochs.
- Phase 2 uses tighter early-stopping patience than Phase 1.
- CSVLogger creates a file when csv_log=True.
- No CSV file created when csv_log=False.
- get_best_checkpoint_path returns None when file does not exist.
- get_best_checkpoint_path returns path when file exists.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import pytest

from training.callbacks import (
    CallbackBundle,
    build_callbacks,
    get_best_checkpoint_path,
)
from training.config import TrainingConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cfg(tmp_path: Path, **kwargs) -> TrainingConfig:
    defaults = dict(output_dir=str(tmp_path), csv_log=True)
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def _tiny_model() -> nn.Module:
    """Return a minimal PyTorch model for checkpoint save/load tests."""
    return nn.Sequential(nn.Linear(4, 4))


def _make_optimizer(model: nn.Module, lr: float = 1e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)


# ─── build_callbacks ──────────────────────────────────────────────────────────

class TestBuildCallbacks:
    def test_returns_callback_bundle(self, tmp_path):
        cfg    = _cfg(tmp_path)
        bundle = build_callbacks(cfg, "exp-001")
        assert isinstance(bundle, CallbackBundle)

    def test_phase2_has_tighter_early_stopping(self, tmp_path):
        cfg    = _cfg(tmp_path, early_stopping_patience=10)
        b1     = build_callbacks(cfg, "exp-002", phase=1)
        b2     = build_callbacks(cfg, "exp-002", phase=2)
        # Phase-2 stopper should fire sooner
        assert b2._es.patience < b1._es.patience

    def test_csv_logger_present_when_enabled(self, tmp_path):
        cfg    = _cfg(tmp_path, csv_log=True)
        bundle = build_callbacks(cfg, "exp-003")
        assert bundle._csv is not None

    def test_csv_logger_absent_when_disabled(self, tmp_path):
        cfg    = _cfg(tmp_path, csv_log=False)
        bundle = build_callbacks(cfg, "exp-004")
        assert bundle._csv is None

    def test_best_checkpoint_path_is_inside_output_dir(self, tmp_path):
        cfg    = _cfg(tmp_path)
        bundle = build_callbacks(cfg, "exp-005")
        assert str(tmp_path) in str(bundle.best_checkpoint_path)


# ─── CallbackBundle.on_epoch_end ──────────────────────────────────────────────

class TestCallbackBundleOnEpochEnd:
    def _bundle(self, tmp_path: Path, patience: int = 3) -> tuple[CallbackBundle, nn.Module]:
        cfg   = _cfg(tmp_path, early_stopping_patience=patience, csv_log=True)
        model = _tiny_model()
        opt   = _make_optimizer(model)
        b     = build_callbacks(cfg, "test-bundle", phase=1, optimizer=opt)
        return b, model

    def test_returns_false_before_patience_exceeded(self, tmp_path):
        bundle, model = self._bundle(tmp_path, patience=5)
        opt = _make_optimizer(model)
        # Feed non-improving val_loss for patience-1 epochs — should not stop
        for i in range(4):
            stop = bundle.on_epoch_end(
                epoch=i + 1,
                val_loss=1.0,          # no improvement
                metrics={"loss": 1.0, "accuracy": 0.5,
                         "val_loss": 1.0, "val_accuracy": 0.5},
                model=model,
                optimizer=opt,
            )
        assert stop is False

    def test_returns_true_after_patience_exceeded(self, tmp_path):
        bundle, model = self._bundle(tmp_path, patience=3)
        opt = _make_optimizer(model)
        stop = False
        for i in range(4):
            stop = bundle.on_epoch_end(
                epoch=i + 1,
                val_loss=1.0,
                metrics={"loss": 1.0, "accuracy": 0.5,
                         "val_loss": 1.0, "val_accuracy": 0.5},
                model=model,
                optimizer=opt,
            )
        assert stop is True

    def test_checkpoint_saved_on_improvement(self, tmp_path):
        bundle, model = self._bundle(tmp_path)
        opt = _make_optimizer(model)
        bundle.on_epoch_end(
            epoch=1,
            val_loss=0.5,          # improvement over inf
            metrics={"loss": 0.5, "accuracy": 0.8,
                     "val_loss": 0.5, "val_accuracy": 0.8},
            model=model,
            optimizer=opt,
        )
        assert bundle.best_checkpoint_path.exists()

    def test_csv_file_created(self, tmp_path):
        bundle, model = self._bundle(tmp_path)
        opt = _make_optimizer(model)
        bundle.on_epoch_end(
            epoch=1,
            val_loss=0.5,
            metrics={"loss": 0.5, "accuracy": 0.8,
                     "val_loss": 0.5, "val_accuracy": 0.8},
            model=model,
            optimizer=opt,
        )
        assert bundle._csv is not None
        assert bundle._csv.path.exists()


# ─── get_best_checkpoint_path ──────────────────────────────────────────────────

class TestGetBestCheckpointPath:
    def test_returns_none_when_missing(self, tmp_path):
        cfg    = _cfg(tmp_path)
        result = get_best_checkpoint_path(cfg, "nonexistent-exp")
        assert result is None

    def test_returns_path_when_file_exists(self, tmp_path):
        cfg     = _cfg(tmp_path)
        exp_id  = "existing-exp-001"
        ckpt_dir = tmp_path / cfg.architecture / "checkpoints" / exp_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        weights_file = ckpt_dir / "best_weights.pt"
        torch.save({"w": torch.zeros(1)}, str(weights_file))

        result = get_best_checkpoint_path(cfg, exp_id)
        assert result is not None
        assert result.exists()
