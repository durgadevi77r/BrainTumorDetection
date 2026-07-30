"""
tests/test_training_trainer.py — Unit tests for training.trainer.Trainer.

All tests that would actually train a neural network use mocks so the
suite runs in milliseconds without a GPU or real dataset.

Tests
-----
- Trainer.__init__ creates an Experiment with status "created".
- Trainer.run() calls the correct sequence of internal methods.
- Trainer.run() returns the expected result keys.
- Trainer.run() marks the experiment as "completed" on success.
- Trainer.run() marks the experiment as "failed" and re-raises on error.
- Trainer._build_generators raises FileNotFoundError for missing dataset.
- train() convenience wrapper passes args to TrainingConfig correctly.
- CLI arg parser accepts all documented flags.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from training.config import TrainingConfig
from training.trainer import Trainer, train, _build_arg_parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg(tmp_path: Path, arch: str = "cnn") -> TrainingConfig:
    """Return a minimal config pointing to tmp_path for output and dataset."""
    return TrainingConfig(
        architecture=arch,
        epochs=1,
        batch_size=4,
        fine_tune=False,
        output_dir=str(tmp_path / "output"),
        dataset_dir=str(tmp_path / "data"),
    )


def _fake_model() -> nn.Module:
    """Return a tiny PyTorch model (no GPU / real weights needed)."""
    return nn.Sequential(nn.Flatten(), nn.Linear(4 * 4, 4))


def _fake_history(n_epochs: int = 2) -> dict:
    return {
        "loss":         [0.5] * n_epochs,
        "accuracy":     [0.6] * n_epochs,
        "val_loss":     [0.4] * n_epochs,
        "val_accuracy": [0.7] * n_epochs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainerInit:
    def test_creates_experiment_on_init(self, tmp_path):
        cfg     = _make_cfg(tmp_path)
        trainer = Trainer(cfg, experiments_dir=tmp_path / "exps")
        assert trainer.experiment is not None
        assert trainer.experiment.status == "created"

    def test_experiment_id_matches_architecture(self, tmp_path):
        cfg     = _make_cfg(tmp_path, arch="cnn")
        trainer = Trainer(cfg, experiments_dir=tmp_path / "exps")
        assert "cnn" in trainer.experiment_id

    def test_experiment_saved_to_disk_on_init(self, tmp_path):
        cfg      = _make_cfg(tmp_path)
        exps_dir = tmp_path / "exps"
        trainer  = Trainer(cfg, experiments_dir=exps_dir)
        exp_file = exps_dir / trainer.experiment_id / "experiment.json"
        assert exp_file.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer.run() — mocked end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainerRun:
    def _patch_trainer(self, tmp_path: Path, arch: str = "cnn"):
        """Return a Trainer with all expensive methods stubbed out."""
        cfg      = _make_cfg(tmp_path, arch=arch)
        exps_dir = tmp_path / "exps"
        trainer  = Trainer(cfg, experiments_dir=exps_dir)

        fake_paths = {
            "model_dir":  str(tmp_path / "output" / arch),
            "model_path": str(tmp_path / "output" / arch / "weights.pt"),
            "h5_path":    "",
            "info_path":  str(tmp_path / "output" / arch / "model_info.json"),
            "format":     "pt",
        }
        fake_metrics = {
            "accuracy": 0.9, "f1": 0.88,
            "precision": 0.89, "recall": 0.87,
            "auc_roc": 0.97,
            "confusion_matrix": [[1, 0], [0, 1]],
            "per_class": {},
            "num_samples": 8,
            "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
            "model_info": {},
        }

        trainer.model = _fake_model()

        # Mock all heavy internal steps
        train_loader = MagicMock()
        val_loader   = MagicMock()
        train_loader.dataset = MagicMock()
        train_loader.dataset.__len__ = MagicMock(return_value=8)
        val_loader.dataset   = MagicMock()
        val_loader.dataset.__len__ = MagicMock(return_value=4)

        trainer._build_generators = MagicMock(return_value=(train_loader, val_loader))
        trainer._build_model      = MagicMock(side_effect=lambda: None)
        trainer._train_phase1     = MagicMock(return_value=_fake_history())
        trainer._train_phase2     = MagicMock(return_value={})
        trainer._save_final_model = MagicMock(return_value=fake_paths)
        trainer._evaluate         = MagicMock(return_value=fake_metrics)

        return trainer, fake_paths, fake_metrics

    def test_run_returns_dict(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        result = trainer.run()
        assert isinstance(result, dict)

    def test_run_result_has_expected_keys(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        result = trainer.run()
        expected_keys = {
            "experiment_id", "architecture", "epochs_phase1",
            "epochs_phase2", "best_val_accuracy", "final_val_loss",
            "training_duration_s", "eval_metrics", "model_paths", "status",
        }
        assert expected_keys.issubset(result.keys())

    def test_run_marks_experiment_completed(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        trainer.run()
        assert trainer.experiment.status == "completed"

    def test_run_calls_build_generators(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        trainer.run()
        trainer._build_generators.assert_called_once()

    def test_run_calls_build_model(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        trainer.run()
        trainer._build_model.assert_called_once()

    def test_run_calls_save_final_model(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        trainer.run()
        trainer._save_final_model.assert_called_once()

    def test_run_calls_evaluate(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        trainer.run()
        trainer._evaluate.assert_called_once()

    def test_run_marks_failed_on_exception(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        trainer._build_generators.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            trainer.run()
        assert trainer.experiment.status == "failed"
        assert "RuntimeError" in trainer.experiment.error

    def test_run_result_architecture_matches_config(self, tmp_path):
        trainer, _, _ = self._patch_trainer(tmp_path)
        result = trainer.run()
        assert result["architecture"] == "cnn"


# ─────────────────────────────────────────────────────────────────────────────
# _build_generators
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGenerators:
    def test_raises_when_dataset_dir_missing(self, tmp_path):
        cfg     = _make_cfg(tmp_path)
        trainer = Trainer(cfg, experiments_dir=tmp_path / "exps")
        with pytest.raises(FileNotFoundError, match="Dataset directory"):
            trainer._build_generators()


# ─────────────────────────────────────────────────────────────────────────────
# train() convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainConvenienceWrapper:
    def test_train_calls_trainer_run(self, tmp_path):
        fake_result = {"experiment_id": "x", "status": "completed"}
        with patch("training.trainer.Trainer") as MockTrainer:
            instance = MagicMock()
            instance.run.return_value = fake_result
            MockTrainer.return_value  = instance

            result = train(
                architecture="cnn",
                epochs=1,
                batch_size=4,
                experiments_dir=tmp_path / "exps",
            )

        assert MockTrainer.called
        instance.run.assert_called_once()
        assert result == fake_result

    def test_train_passes_architecture(self, tmp_path):
        fake_result = {"experiment_id": "x", "status": "completed"}
        with patch("training.trainer.Trainer") as MockTrainer:
            instance = MagicMock()
            instance.run.return_value = fake_result
            MockTrainer.return_value  = instance

            train(architecture="resnet50", epochs=2, experiments_dir=tmp_path)

        cfg_arg: TrainingConfig = MockTrainer.call_args[0][0]
        assert cfg_arg.architecture == "resnet50"


# ─────────────────────────────────────────────────────────────────────────────
# CLI arg parser
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIArgParser:
    def test_defaults(self):
        parser = _build_arg_parser()
        args   = parser.parse_args([])
        assert args.architecture == "mambavision"
        assert args.epochs       == 30
        assert args.batch_size   == 32

    def test_architecture_override(self):
        parser = _build_arg_parser()
        args   = parser.parse_args(["--architecture", "resnet50"])
        assert args.architecture == "resnet50"

    def test_epochs_override(self):
        parser = _build_arg_parser()
        args   = parser.parse_args(["--epochs", "10"])
        assert args.epochs == 10

    def test_no_fine_tune_flag(self):
        parser = _build_arg_parser()
        args   = parser.parse_args(["--no-fine-tune"])
        assert args.fine_tune is False

    def test_seed_override(self):
        parser = _build_arg_parser()
        args   = parser.parse_args(["--seed", "123"])
        assert args.seed == 123

    def test_invalid_architecture_rejected(self):
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--architecture", "densenet"])
