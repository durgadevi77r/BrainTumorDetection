"""
tests/test_evaluation.py — Unit tests for the evaluation package.

Coverage
--------
TestEvaluationPackageImports   evaluation/__init__.py — all public symbols importable
TestFindBestCheckpoint         evaluation/loader.py   — search order, error on missing
TestLoadEvalModel              evaluation/loader.py   — PT, full-ckpt, HF formats
TestArtifactWriter             evaluation/artifacts.py — all 6 artifact files created
TestEvaluationPipelineMetrics  evaluation/evaluator.py — _compute_metrics correctness
TestEvaluationPipelineRun      evaluation/evaluator.py — run() end-to-end (mocked)
TestEvaluateCLIParser          evaluation/evaluator.py — CLI arg parser
TestEvaluateModelCompat        app/models/evaluate.py  — backward-compat + output_dir
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ─── Shared fixtures / helpers ────────────────────────────────────────────────

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
N_CLASSES   = len(CLASS_NAMES)


def _make_perfect_arrays(n: int = 40):
    """Return (y_true, y_pred, y_probs) with 100 % accuracy, equal classes."""
    y_true  = np.repeat(np.arange(N_CLASSES), n // N_CLASSES).astype(np.int64)
    y_pred  = y_true.copy()
    y_probs = np.eye(N_CLASSES, dtype=np.float32)[y_true]
    return y_true, y_pred, y_probs


def _make_noisy_arrays(n: int = 80, seed: int = 0):
    """Return arrays with realistic random predictions (~30 % noise)."""
    rng    = np.random.default_rng(seed)
    y_true = np.repeat(np.arange(N_CLASSES), n // N_CLASSES).astype(np.int64)
    y_pred = y_true.copy()
    noise_idx = rng.choice(len(y_true), size=len(y_true) // 3, replace=False)
    y_pred[noise_idx] = rng.integers(0, N_CLASSES, size=len(noise_idx))
    y_probs = np.eye(N_CLASSES, dtype=np.float32)[y_pred]
    y_probs += rng.uniform(0, 0.05, y_probs.shape).astype(np.float32)
    y_probs /= y_probs.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_probs


def _tiny_model() -> nn.Module:
    """4×4 RGB input → N_CLASSES logits — minimal, no HF dependency."""
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, N_CLASSES))


def _make_fake_dataset(root: Path, n_per_class: int = 4) -> Path:
    """Create a minimal class-folder image directory using PIL."""
    from PIL import Image
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.full((8, 8, 3), i * 30, dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(cls_dir / f"img_{i:04d}.png")
    return root


# ═════════════════════════════════════════════════════════════════════════════
# 1. Package imports
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluationPackageImports:
    """All public symbols re-exported from evaluation/__init__.py must be importable."""

    def test_import_evaluation_pipeline(self):
        from evaluation import EvaluationPipeline
        assert callable(EvaluationPipeline)

    def test_import_evaluate_function(self):
        from evaluation import evaluate
        assert callable(evaluate)

    def test_import_artifact_writer(self):
        from evaluation import ArtifactWriter
        assert callable(ArtifactWriter)

    def test_import_find_best_checkpoint(self):
        from evaluation import find_best_checkpoint
        assert callable(find_best_checkpoint)

    def test_import_load_eval_model(self):
        from evaluation import load_eval_model
        assert callable(load_eval_model)

    def test_all_dunder(self):
        import evaluation
        for name in evaluation.__all__:
            assert hasattr(evaluation, name), f"evaluation.__all__ lists '{name}' but it is not exported"


# ═════════════════════════════════════════════════════════════════════════════
# 2. find_best_checkpoint — search order and error path
# ═════════════════════════════════════════════════════════════════════════════

class TestFindBestCheckpoint:
    """evaluation/loader.py::find_best_checkpoint"""

    def test_raises_when_no_checkpoint(self, tmp_path):
        from evaluation.loader import find_best_checkpoint
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            find_best_checkpoint("mambavision", output_dir=tmp_path)

    def test_finds_hf_directory(self, tmp_path):
        """A saved_models/<arch>/ dir with config.json is found as 'hf_pretrained'."""
        from evaluation.loader import find_best_checkpoint
        arch_dir = tmp_path / "mambavision"
        arch_dir.mkdir()
        (arch_dir / "config.json").write_text("{}")
        path, meta = find_best_checkpoint("mambavision", output_dir=tmp_path)
        assert meta["source"] == "hf_pretrained"
        assert Path(path) == arch_dir

    def test_finds_legacy_checkpoint(self, tmp_path):
        """A best_weights.pt at checkpoints/ root is found as 'legacy_state_dict'."""
        from evaluation.loader import find_best_checkpoint
        ckpt_dir = tmp_path / "mambavision" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        weights = ckpt_dir / "best_weights.pt"
        torch.save({}, str(weights))
        path, meta = find_best_checkpoint("mambavision", output_dir=tmp_path)
        assert meta["source"] == "legacy_state_dict"
        assert Path(path) == weights

    def test_finds_experiment_checkpoint(self, tmp_path):
        """Experiment sub-directories are preferred over the legacy flat checkpoint."""
        from evaluation.loader import find_best_checkpoint
        exp_dir = tmp_path / "mambavision" / "checkpoints" / "exp_001"
        exp_dir.mkdir(parents=True)
        weights = exp_dir / "best_weights.pt"
        torch.save({}, str(weights))
        info = {
            "experiment_id": "exp_001",
            "saved_at": "2025-01-01T00:00:00",
            "metrics": {"val_loss": 0.1, "val_accuracy": 0.95},
        }
        (exp_dir / "checkpoint_info.json").write_text(json.dumps(info))
        path, meta = find_best_checkpoint("mambavision", output_dir=tmp_path)
        assert meta["source"] == "training_full"
        assert meta["experiment_id"] == "exp_001"
        assert Path(path) == weights

    def test_experiment_preferred_over_legacy(self, tmp_path):
        """When both experiment and legacy checkpoints exist, experiment wins."""
        from evaluation.loader import find_best_checkpoint
        # Legacy flat checkpoint
        ckpt_dir = tmp_path / "arch" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        torch.save({}, str(ckpt_dir / "best_weights.pt"))
        # Newer experiment checkpoint
        exp_dir = ckpt_dir / "exp_A"
        exp_dir.mkdir()
        torch.save({}, str(exp_dir / "best_weights.pt"))
        (exp_dir / "checkpoint_info.json").write_text(
            json.dumps({"experiment_id": "exp_A", "saved_at": "2025-06-01T00:00:00"})
        )
        _, meta = find_best_checkpoint("arch", output_dir=tmp_path)
        assert meta["source"] == "training_full"

    def test_hf_reads_model_info_val_accuracy(self, tmp_path):
        """When model_info.json exists next to config.json, val_accuracy is extracted."""
        from evaluation.loader import find_best_checkpoint
        arch_dir = tmp_path / "efficientnet"
        arch_dir.mkdir()
        (arch_dir / "config.json").write_text("{}")
        (arch_dir / "model_info.json").write_text(
            json.dumps({"final_val_accuracy": 0.97, "saved_at": "2025-03-01T00:00:00"})
        )
        _, meta = find_best_checkpoint("efficientnet", output_dir=tmp_path)
        assert meta["val_accuracy"] == pytest.approx(0.97)
        assert meta["saved_at"] == "2025-03-01T00:00:00"


# ═════════════════════════════════════════════════════════════════════════════
# 3. load_eval_model — checkpoint format dispatch
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadEvalModel:
    """evaluation/loader.py::load_eval_model — loads weights, returns eval-mode model."""

    def _save_legacy_checkpoint(self, model: nn.Module, path: Path) -> None:
        """Write a plain state-dict .pt file (legacy format)."""
        torch.save(model.state_dict(), str(path))

    def _save_full_checkpoint(self, model: nn.Module, path: Path) -> None:
        """Write a full training-checkpoint dict."""
        torch.save(
            {
                "model_state":     model.state_dict(),
                "optimizer_state": {},
                "scheduler_state": {},
                "epoch":           5,
                "val_loss":        0.12,
                "val_accuracy":    0.93,
                "architecture":    "cnn",
                "experiment_id":   "exp_test",
                "saved_at":        "2025-01-01T00:00:00",
            },
            str(path),
        )

    def test_loads_legacy_state_dict(self, tmp_path):
        """Plain state_dict .pt is loaded into a skeleton model."""
        from evaluation.loader import load_eval_model

        model  = _tiny_model()
        ckpt   = tmp_path / "cnn" / "checkpoints" / "best_weights.pt"
        ckpt.parent.mkdir(parents=True)
        self._save_legacy_checkpoint(model, ckpt)

        # build_model is a lazy import inside load_eval_model — patch at its source
        with patch("app.models.architectures.build_model", return_value=_tiny_model()):
            loaded, meta = load_eval_model("cnn", output_dir=tmp_path,
                                           device=torch.device("cpu"))

        assert meta["source"] == "legacy_state_dict"
        assert loaded.training is False   # must be in eval mode

    def test_loads_full_checkpoint_extracts_meta(self, tmp_path):
        """Full training checkpoint: model_state loaded, extra keys surfaced in meta."""
        from evaluation.loader import load_eval_model

        model  = _tiny_model()
        ckpt   = tmp_path / "cnn" / "checkpoints" / "best_weights.pt"
        ckpt.parent.mkdir(parents=True)
        self._save_full_checkpoint(model, ckpt)

        with patch("app.models.architectures.build_model", return_value=_tiny_model()):
            loaded, meta = load_eval_model("cnn", output_dir=tmp_path,
                                           device=torch.device("cpu"))

        assert meta["epoch"] == 5
        assert meta["val_accuracy"] == pytest.approx(0.93)
        assert loaded.training is False

    def test_raises_when_no_checkpoint(self, tmp_path):
        from evaluation.loader import load_eval_model
        with pytest.raises(FileNotFoundError):
            load_eval_model("resnet50", output_dir=tmp_path)

    def test_model_moved_to_device(self, tmp_path):
        """Returned model parameters must live on the requested device."""
        from evaluation.loader import load_eval_model

        model = _tiny_model()
        ckpt  = tmp_path / "cnn" / "checkpoints" / "best_weights.pt"
        ckpt.parent.mkdir(parents=True)
        self._save_legacy_checkpoint(model, ckpt)

        cpu = torch.device("cpu")
        with patch("app.models.architectures.build_model", return_value=_tiny_model()):
            loaded, _ = load_eval_model("cnn", output_dir=tmp_path, device=cpu)

        for p in loaded.parameters():
            assert p.device.type == "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# 4. ArtifactWriter — all six files are created with correct content
# ═════════════════════════════════════════════════════════════════════════════

class TestArtifactWriter:
    """evaluation/artifacts.py::ArtifactWriter"""

    @pytest.fixture()
    def writer(self, tmp_path) -> "ArtifactWriter":
        from evaluation.artifacts import ArtifactWriter
        return ArtifactWriter(
            output_dir=tmp_path / "eval_out",
            class_names=CLASS_NAMES,
            architecture="test_arch",
        )

    @pytest.fixture()
    def arrays(self):
        return _make_perfect_arrays(n=40)

    @pytest.fixture()
    def dummy_metrics(self, arrays):
        y_true, y_pred, y_probs = arrays
        return {
            "accuracy":           1.0,
            "precision_macro":    1.0,
            "precision_weighted": 1.0,
            "recall_macro":       1.0,
            "recall_weighted":    1.0,
            "f1_macro":           1.0,
            "f1_weighted":        1.0,
            "auc_roc":            1.0,
            "confusion_matrix":   np.eye(N_CLASSES, dtype=int).tolist(),
            "per_class":          {c: {"precision": 1.0, "recall": 1.0,
                                       "f1": 1.0, "support": 10}
                                   for c in CLASS_NAMES},
            "per_class_accuracy": {c: 1.0 for c in CLASS_NAMES},
            "num_samples":        40,
            "class_names":        CLASS_NAMES,
            "duration_s":         0.5,
        }

    def test_write_all_returns_six_paths(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        assert len(paths) == 6

    def test_all_artifact_files_exist(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        for name, p in paths.items():
            assert Path(p).exists(), f"Artifact '{name}' was not written to disk"

    def test_confusion_matrix_png_created(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        assert "confusion_matrix_png" in paths
        p = Path(paths["confusion_matrix_png"])
        assert p.suffix == ".png"
        assert p.stat().st_size > 0

    def test_roc_curve_png_created(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        assert "roc_curve_png" in paths
        p = Path(paths["roc_curve_png"])
        assert p.suffix == ".png"
        assert p.stat().st_size > 0

    def test_classification_report_json_valid(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        json_path = Path(paths["classification_report_json"])
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        for cls in CLASS_NAMES:
            assert cls in data
            assert "precision" in data[cls]
            assert "recall" in data[cls]
            assert "f1-score" in data[cls]

    def test_classification_report_txt_contains_class_names(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        txt = Path(paths["classification_report_txt"]).read_text()
        for cls in CLASS_NAMES:
            assert cls in txt

    def test_metrics_summary_json_has_required_keys(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
            dataset_dir="/data/test",
            checkpoint_meta={"source": "hf_pretrained", "epoch": None},
        )
        data = json.loads(Path(paths["metrics_summary_json"]).read_text())
        for key in ("architecture", "class_names", "num_samples",
                    "dataset_dir", "evaluated_at", "checkpoint_meta", "metrics"):
            assert key in data, f"metrics_summary.json missing key '{key}'"
        assert data["architecture"] == "test_arch"
        assert data["dataset_dir"]  == "/data/test"

    def test_evaluation_log_contains_scalar_metrics(self, writer, arrays, dummy_metrics):
        y_true, y_pred, y_probs = arrays
        paths = writer.write_all(
            metrics=dummy_metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        log_text = Path(paths["evaluation_log"]).read_text()
        for phrase in ("accuracy", "precision", "recall", "F1", "AUC-ROC"):
            assert phrase in log_text, f"evaluation.log missing '{phrase}'"

    def test_output_dir_created_if_absent(self, tmp_path):
        """ArtifactWriter creates the output directory automatically."""
        from evaluation.artifacts import ArtifactWriter
        new_dir = tmp_path / "deep" / "nested" / "dir"
        assert not new_dir.exists()
        ArtifactWriter(output_dir=new_dir, class_names=CLASS_NAMES)
        assert new_dir.exists()

    def test_noisy_arrays_do_not_raise(self, writer):
        """Non-perfect predictions should not raise during artifact generation."""
        y_true, y_pred, y_probs = _make_noisy_arrays(n=80)
        metrics = {
            "accuracy": 0.7, "precision_macro": 0.7, "precision_weighted": 0.7,
            "recall_macro": 0.7, "recall_weighted": 0.7,
            "f1_macro": 0.7, "f1_weighted": 0.7, "auc_roc": 0.9,
            "confusion_matrix": [[0]*N_CLASSES]*N_CLASSES,
            "per_class": {c: {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 20}
                          for c in CLASS_NAMES},
            "per_class_accuracy": {c: 0.7 for c in CLASS_NAMES},
            "num_samples": 80, "class_names": CLASS_NAMES, "duration_s": 1.0,
        }
        paths = writer.write_all(
            metrics=metrics, y_true=y_true,
            y_pred=y_pred, y_probs=y_probs,
        )
        assert len(paths) == 6


# ═════════════════════════════════════════════════════════════════════════════
# 5. EvaluationPipeline._compute_metrics — metric correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluationPipelineMetrics:
    """evaluation/evaluator.py::EvaluationPipeline._compute_metrics"""

    @pytest.fixture()
    def pipeline(self):
        from evaluation.evaluator import EvaluationPipeline
        return EvaluationPipeline(architecture="mambavision")

    def test_perfect_predictions_accuracy_one(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_perfect_predictions_f1_one(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert m["f1_macro"]    == pytest.approx(1.0)
        assert m["f1_weighted"] == pytest.approx(1.0)

    def test_perfect_predictions_auc_one(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert m["auc_roc"] == pytest.approx(1.0)

    def test_per_class_keys_present(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert set(m["per_class"].keys()) == set(CLASS_NAMES)
        for cls in CLASS_NAMES:
            for field in ("precision", "recall", "f1", "support"):
                assert field in m["per_class"][cls]

    def test_per_class_accuracy_keys_present(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert set(m["per_class_accuracy"].keys()) == set(CLASS_NAMES)

    def test_confusion_matrix_shape(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        cm = m["confusion_matrix"]
        assert len(cm) == N_CLASSES
        assert all(len(row) == N_CLASSES for row in cm)

    def test_num_samples_correct(self, pipeline):
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert m["num_samples"] == 40

    def test_noisy_accuracy_below_one(self, pipeline):
        y_true, y_pred, y_probs = _make_noisy_arrays(n=80)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert 0.0 < m["accuracy"] < 1.0

    def test_legacy_compat_keys_present(self, pipeline):
        """Keys 'precision', 'recall', 'f1' must exist for backward compat."""
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        for key in ("precision", "recall", "f1"):
            assert key in m, f"Legacy key '{key}' missing from metrics dict"

    def test_macro_equals_weighted_for_balanced_classes(self, pipeline):
        """With perfectly balanced classes macro and weighted averages must agree."""
        y_true, y_pred, y_probs = _make_perfect_arrays(n=40)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        assert m["precision_macro"] == pytest.approx(m["precision_weighted"], abs=1e-4)
        assert m["recall_macro"]    == pytest.approx(m["recall_weighted"],    abs=1e-4)
        assert m["f1_macro"]        == pytest.approx(m["f1_weighted"],        abs=1e-4)

    def test_all_scalar_metrics_in_range(self, pipeline):
        """Every scalar metric must be a float in [0, 1]."""
        y_true, y_pred, y_probs = _make_noisy_arrays(n=80)
        m = pipeline._compute_metrics(y_true, y_pred, y_probs)
        scalars = [
            "accuracy", "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted",
            "f1_macro", "f1_weighted", "auc_roc",
        ]
        for key in scalars:
            val = m[key]
            assert isinstance(val, float), f"'{key}' is not a float"
            assert 0.0 <= val <= 1.0,      f"'{key}' = {val} is out of [0, 1]"


# ═════════════════════════════════════════════════════════════════════════════
# 6. EvaluationPipeline.run() — end-to-end with mocked I/O
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluationPipelineRun:
    """evaluation/evaluator.py::EvaluationPipeline.run() — mocked end-to-end."""

    def _make_fake_loader(self, n_per_class: int = 4):
        """
        Build a real DataLoader from a tiny in-memory fake dataset so the
        pipeline's inference loop runs without touching disk.
        """
        import tempfile, os
        from torch.utils.data import DataLoader, TensorDataset

        n_total = N_CLASSES * n_per_class
        images  = torch.zeros(n_total, 3, 8, 8)          # tiny 8×8 RGB
        labels  = torch.repeat_interleave(
            torch.arange(N_CLASSES), n_per_class
        )
        ds = TensorDataset(images, labels)
        # Attach class_to_idx so the pipeline can build the canonical map
        ds.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        ds.classes      = CLASS_NAMES
        return DataLoader(ds, batch_size=8, shuffle=False)

    def _fake_model(self):
        """Tiny model that returns a namespace with .logits."""
        class _FakeOut:
            def __init__(self, logits): self.logits = logits
        class _M(nn.Module):
            def forward(self, x):
                b = x.shape[0]
                # Always predict class 0 — deterministic
                logits = torch.zeros(b, N_CLASSES)
                logits[:, 0] = 10.0
                return _FakeOut(logits)
        return _M()

    def test_run_returns_required_keys(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        pipeline = EvaluationPipeline(
            architecture="mambavision",
            output_dir=tmp_path / "out",
        )
        loader = self._make_fake_loader()
        model  = self._fake_model()

        with (
            patch.object(pipeline, "_load_model",   return_value=(model, {})),
            patch.object(pipeline, "_build_loader", return_value=loader),
        ):
            result = pipeline.run(save_artifacts=False)

        for key in ("architecture", "dataset_dir", "num_samples",
                    "class_names", "checkpoint_meta", "metrics",
                    "artifact_paths", "duration_s"):
            assert key in result, f"run() result missing key '{key}'"

    def test_run_metrics_sub_dict_has_required_keys(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        pipeline = EvaluationPipeline(
            architecture="mambavision",
            output_dir=tmp_path / "out",
        )
        loader = self._make_fake_loader()
        model  = self._fake_model()

        with (
            patch.object(pipeline, "_load_model",   return_value=(model, {})),
            patch.object(pipeline, "_build_loader", return_value=loader),
        ):
            result = pipeline.run(save_artifacts=False)

        m = result["metrics"]
        for key in ("accuracy", "f1_macro", "auc_roc", "confusion_matrix",
                    "per_class", "num_samples"):
            assert key in m, f"metrics dict missing key '{key}'"

    def test_run_save_artifacts_true_writes_files(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        out = tmp_path / "artifacts"
        pipeline = EvaluationPipeline(
            architecture="mambavision",
            output_dir=out,
        )
        loader = self._make_fake_loader(n_per_class=10)
        model  = self._fake_model()

        with (
            patch.object(pipeline, "_load_model",   return_value=(model, {})),
            patch.object(pipeline, "_build_loader", return_value=loader),
        ):
            result = pipeline.run(save_artifacts=True)

        assert len(result["artifact_paths"]) > 0
        for name, path in result["artifact_paths"].items():
            assert Path(path).exists(), f"Artifact '{name}' not written: {path}"

    def test_run_save_artifacts_false_no_files(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        out = tmp_path / "no_artifacts"
        pipeline = EvaluationPipeline(
            architecture="mambavision",
            output_dir=out,
        )
        loader = self._make_fake_loader()
        model  = self._fake_model()

        with (
            patch.object(pipeline, "_load_model",   return_value=(model, {})),
            patch.object(pipeline, "_build_loader", return_value=loader),
        ):
            result = pipeline.run(save_artifacts=False)

        assert result["artifact_paths"] == {}

    def test_run_num_samples_matches_loader(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        pipeline = EvaluationPipeline(architecture="mambavision",
                                       output_dir=tmp_path / "out")
        loader = self._make_fake_loader(n_per_class=5)   # 4 × 5 = 20 samples
        model  = self._fake_model()

        with (
            patch.object(pipeline, "_load_model",   return_value=(model, {})),
            patch.object(pipeline, "_build_loader", return_value=loader),
        ):
            result = pipeline.run(save_artifacts=False)

        assert result["num_samples"] == 20
        assert result["metrics"]["num_samples"] == 20

    def test_run_class_names_match_settings(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        pipeline = EvaluationPipeline(architecture="mambavision",
                                       output_dir=tmp_path / "out")
        loader = self._make_fake_loader()
        model  = self._fake_model()

        with (
            patch.object(pipeline, "_load_model",   return_value=(model, {})),
            patch.object(pipeline, "_build_loader", return_value=loader),
        ):
            result = pipeline.run(save_artifacts=False)

        assert result["class_names"] == CLASS_NAMES

    def test_build_loader_raises_on_missing_dir(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        pipeline = EvaluationPipeline(
            architecture="mambavision",
            dataset_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(FileNotFoundError):
            pipeline._build_loader()

    def test_build_loader_raises_on_empty_dir(self, tmp_path):
        from evaluation.evaluator import EvaluationPipeline
        empty = tmp_path / "empty"
        empty.mkdir()
        # Create class sub-dirs but with NO images
        for c in CLASS_NAMES:
            (empty / c).mkdir()
        pipeline = EvaluationPipeline(
            architecture="mambavision",
            dataset_dir=empty,
            output_dir=tmp_path / "out",
        )
        with pytest.raises(ValueError, match="No images found"):
            pipeline._build_loader()


# ═════════════════════════════════════════════════════════════════════════════
# 7. CLI arg parser
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateCLIParser:
    """evaluation/evaluator.py::_build_arg_parser — argument parsing."""

    @pytest.fixture()
    def parser(self):
        from evaluation.evaluator import _build_arg_parser
        return _build_arg_parser()

    def test_defaults(self, parser):
        args = parser.parse_args([])
        assert args.architecture   == "mambavision"
        assert args.batch_size     == 32
        assert args.num_workers    == 0
        assert args.save_artifacts is True
        assert args.use_amp        is True
        assert args.dataset_dir    is None
        assert args.output_dir     is None

    def test_architecture_override(self, parser):
        args = parser.parse_args(["--architecture", "resnet50"])
        assert args.architecture == "resnet50"

    def test_short_flag_architecture(self, parser):
        args = parser.parse_args(["-a", "vgg16"])
        assert args.architecture == "vgg16"

    def test_batch_size(self, parser):
        args = parser.parse_args(["--batch-size", "64"])
        assert args.batch_size == 64

    def test_no_artifacts_flag(self, parser):
        args = parser.parse_args(["--no-artifacts"])
        assert args.save_artifacts is False

    def test_no_amp_flag(self, parser):
        args = parser.parse_args(["--no-amp"])
        assert args.use_amp is False

    def test_dataset_dir(self, parser):
        args = parser.parse_args(["--dataset-dir", "/data/test"])
        assert args.dataset_dir == "/data/test"

    def test_output_dir(self, parser):
        args = parser.parse_args(["--output-dir", "/out/eval"])
        assert args.output_dir == "/out/eval"

    def test_invalid_architecture_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--architecture", "unknown_arch"])


# ═════════════════════════════════════════════════════════════════════════════
# 8. app/models/evaluate.py — backward-compat + output_dir delegation
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateModelCompat:
    """app/models/evaluate.py::evaluate_model — API-compat mode + full-pipeline mode."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lightweight_metrics(self, model_name: str = "mambavision") -> dict:
        """Minimal dict that evaluate_model returns in lightweight mode."""
        return {
            "model_name":       model_name,
            "accuracy":         0.95,
            "precision":        0.95,
            "recall":           0.95,
            "f1":               0.95,
            "auc_roc":          0.99,
            "confusion_matrix": [[10, 0, 0, 0]] * N_CLASSES,
            "per_class":        {c: {"precision": 0.95, "recall": 0.95,
                                     "f1": 0.95, "support": 10}
                                 for c in CLASS_NAMES},
            "num_samples":      40,
            "class_names":      CLASS_NAMES,
            "model_info":       {},
        }

    def _pipeline_result(self, model_name: str = "mambavision") -> dict:
        """
        Mimics what _run_full_pipeline returns — including the top-level flattened
        API-compat keys that evaluate_model adds via setdefault() after delegation.
        """
        metrics = {
            "accuracy":           0.97,
            "precision_macro":    0.97,
            "precision_weighted": 0.97,
            "recall_macro":       0.97,
            "recall_weighted":    0.97,
            "f1_macro":           0.97,
            "f1_weighted":        0.97,
            "auc_roc":            0.99,
            "precision":          0.97,
            "recall":             0.97,
            "f1":                 0.97,
            "confusion_matrix":   [[10, 0, 0, 0]] * N_CLASSES,
            "per_class":          {c: {"precision": 0.97, "recall": 0.97,
                                       "f1": 0.97, "support": 10}
                                   for c in CLASS_NAMES},
            "per_class_accuracy": {c: 0.97 for c in CLASS_NAMES},
            "num_samples":        40,
            "class_names":        CLASS_NAMES,
            "duration_s":         1.0,
        }
        # _run_full_pipeline includes the setdefault flattening, so these
        # top-level keys are already present on the returned dict.
        return {
            "architecture":    model_name,
            "dataset_dir":     "/data/test",
            "num_samples":     40,
            "class_names":     CLASS_NAMES,
            "checkpoint_meta": {"source": "hf_pretrained"},
            "metrics":         metrics,
            "artifact_paths":  {"confusion_matrix_png": "/out/cm.png"},
            "duration_s":      1.0,
            # Flattened API-compat keys (added by _run_full_pipeline)
            "model_name":      model_name,
            "model_info":      {},
            "accuracy":        0.97,
            "precision":       0.97,
            "recall":          0.97,
            "f1":              0.97,
            "auc_roc":         0.99,
            "confusion_matrix": [[10, 0, 0, 0]] * N_CLASSES,
            "per_class":       {c: {"precision": 0.97, "recall": 0.97,
                                    "f1": 0.97, "support": 10}
                                for c in CLASS_NAMES},
        }

    # ── Lightweight mode (no output_dir) ──────────────────────────────────────

    def test_lightweight_mode_returns_api_keys(self):
        """With no output_dir, evaluate_model returns the seven API-expected keys."""
        from app.models.evaluate import evaluate_model

        expected = self._lightweight_metrics()

        with (
            patch("app.models.evaluate.load_keras_model") as mock_load,
            patch("app.models.evaluate.build_test_generator") as mock_gen,
            patch("app.models.evaluate.get_model_info", return_value={}),
        ):
            # Build a tiny model that outputs softmax-compatible logits
            model      = _tiny_model()
            wrapped    = MagicMock()
            wrapped.device = torch.device("cpu")
            wrapped.model  = model

            # fake DataLoader
            from torch.utils.data import TensorDataset, DataLoader
            n = N_CLASSES * 4
            ds = TensorDataset(
                torch.zeros(n, 3, 4, 4),
                torch.repeat_interleave(torch.arange(N_CLASSES), 4),
            )
            ds.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
            loader = DataLoader(ds, batch_size=8, shuffle=False)

            mock_load.return_value = wrapped
            mock_gen.return_value  = loader

            result = evaluate_model("mambavision", batch_size=8)

        for key in ("accuracy", "precision", "recall", "f1",
                    "auc_roc", "confusion_matrix", "per_class",
                    "num_samples", "class_names", "model_name"):
            assert key in result, f"evaluate_model() result missing key '{key}'"

    def test_lightweight_accuracy_in_range(self):
        from app.models.evaluate import evaluate_model

        with (
            patch("app.models.evaluate.load_keras_model") as mock_load,
            patch("app.models.evaluate.build_test_generator") as mock_gen,
            patch("app.models.evaluate.get_model_info", return_value={}),
        ):
            model   = _tiny_model()
            wrapped = MagicMock()
            wrapped.device = torch.device("cpu")
            wrapped.model  = model

            from torch.utils.data import TensorDataset, DataLoader
            n  = N_CLASSES * 4
            ds = TensorDataset(
                torch.zeros(n, 3, 4, 4),
                torch.repeat_interleave(torch.arange(N_CLASSES), 4),
            )
            ds.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
            loader = DataLoader(ds, batch_size=8, shuffle=False)

            mock_load.return_value = wrapped
            mock_gen.return_value  = loader

            result = evaluate_model("mambavision", batch_size=8)

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_lightweight_empty_dataset_raises_value_error(self):
        from app.models.evaluate import evaluate_model

        with (
            patch("app.models.evaluate.load_keras_model") as mock_load,
            patch("app.models.evaluate.build_test_generator") as mock_gen,
        ):
            wrapped = MagicMock()
            wrapped.device = torch.device("cpu")

            from torch.utils.data import TensorDataset, DataLoader
            ds = TensorDataset(torch.zeros(0, 3, 4, 4), torch.zeros(0, dtype=torch.long))
            loader = DataLoader(ds, batch_size=8)

            mock_load.return_value = wrapped
            mock_gen.return_value  = loader

            with pytest.raises(ValueError, match="No images found"):
                evaluate_model("mambavision")

    # ── Full pipeline mode (output_dir provided) ──────────────────────────────

    def test_output_dir_delegates_to_pipeline(self, tmp_path):
        """When output_dir is set, evaluate_model must call EvaluationPipeline."""
        from app.models.evaluate import evaluate_model

        pipeline_result = self._pipeline_result()

        with (
            patch("app.models.evaluate._run_full_pipeline",
                  return_value=pipeline_result) as mock_pipe,
            patch("app.models.evaluate.get_model_info", return_value={}),
        ):
            result = evaluate_model("mambavision", output_dir=str(tmp_path / "out"))

        mock_pipe.assert_called_once()

    def test_output_dir_result_has_api_keys(self, tmp_path):
        """Full-pipeline result must still expose the seven API-compat top-level keys."""
        from app.models.evaluate import evaluate_model

        pipeline_result = self._pipeline_result()

        with (
            patch("app.models.evaluate._run_full_pipeline",
                  return_value=pipeline_result),
            patch("app.models.evaluate.get_model_info", return_value={}),
        ):
            result = evaluate_model("mambavision", output_dir=str(tmp_path / "out"))

        for key in ("accuracy", "precision", "recall", "f1",
                    "auc_roc", "confusion_matrix", "per_class", "num_samples"):
            assert key in result, f"Full-pipeline result missing API key '{key}'"

    def test_model_name_defaults_to_active_model(self):
        """When model_name is None, falls back to settings.active_model."""
        from app.models.evaluate import evaluate_model
        from app.core.config import settings

        with (
            patch("app.models.evaluate.load_keras_model") as mock_load,
            patch("app.models.evaluate.build_test_generator") as mock_gen,
            patch("app.models.evaluate.get_model_info", return_value={}),
        ):
            wrapped = MagicMock()
            wrapped.device = torch.device("cpu")
            model = _tiny_model()
            wrapped.model = model

            from torch.utils.data import TensorDataset, DataLoader
            n  = N_CLASSES * 4
            ds = TensorDataset(
                torch.zeros(n, 3, 4, 4),
                torch.repeat_interleave(torch.arange(N_CLASSES), 4),
            )
            ds.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
            loader = DataLoader(ds, batch_size=8, shuffle=False)

            mock_load.return_value = wrapped
            mock_gen.return_value  = loader

            result = evaluate_model(model_name=None, batch_size=8)

        assert result["model_name"] == settings.active_model

    def test_full_pipeline_exposes_artifact_paths(self, tmp_path):
        """Full-pipeline mode must surface artifact_paths from the sub-pipeline."""
        from app.models.evaluate import evaluate_model

        pipeline_result = self._pipeline_result()
        pipeline_result["artifact_paths"] = {
            "confusion_matrix_png": str(tmp_path / "cm.png"),
            "roc_curve_png":        str(tmp_path / "roc.png"),
        }

        with (
            patch("app.models.evaluate._run_full_pipeline",
                  return_value=pipeline_result),
            patch("app.models.evaluate.get_model_info", return_value={}),
        ):
            result = evaluate_model("mambavision", output_dir=str(tmp_path / "out"))

        assert "artifact_paths" in result
        assert "confusion_matrix_png" in result["artifact_paths"]

    def test_full_pipeline_saves_artifacts_flag(self, tmp_path):
        """save_artifacts=False propagates to _run_full_pipeline."""
        from app.models.evaluate import evaluate_model

        pipeline_result = self._pipeline_result()

        with patch("app.models.evaluate._run_full_pipeline",
                   return_value=pipeline_result) as mock_pipe:
            evaluate_model(
                "mambavision",
                output_dir=str(tmp_path / "out"),
                save_artifacts=False,
            )

        _, kwargs = mock_pipe.call_args
        assert kwargs.get("save_artifacts") is False
