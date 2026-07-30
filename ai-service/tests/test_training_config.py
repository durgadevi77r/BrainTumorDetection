"""
tests/test_training_config.py — Unit tests for training.config.TrainingConfig.

Tests
-----
- Default construction reads sensible values.
- Architecture and optimiser validation.
- Class-weight map conversion.
- Effective fine-tune LR calculation.
- to_dict() → from_dict() round-trip.
- from_json() / save_json() persistence.
- from_settings() reads the app singleton.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from training.config import (
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_OPTIMISERS,
    TrainingConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConfigDefaults:
    def test_default_architecture(self):
        cfg = TrainingConfig()
        assert cfg.architecture == "mambavision"

    def test_default_epochs(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 30

    def test_default_batch_size(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 32

    def test_default_learning_rate(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == pytest.approx(1e-4)

    def test_default_seed(self):
        assert TrainingConfig().seed == 42

    def test_default_class_weights_none(self):
        assert TrainingConfig().class_weights is None

    def test_default_fine_tune_enabled(self):
        assert TrainingConfig().fine_tune is True

    def test_default_csv_log_enabled(self):
        assert TrainingConfig().csv_log is True


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConfigValidation:
    @pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
    def test_valid_architecture(self, arch):
        cfg = TrainingConfig(architecture=arch)
        assert cfg.architecture == arch

    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="architecture"):
            TrainingConfig(architecture="densenet")

    def test_architecture_normalised_to_lower(self):
        cfg = TrainingConfig(architecture="ResNet50")
        assert cfg.architecture == "resnet50"

    @pytest.mark.parametrize("opt", SUPPORTED_OPTIMISERS)
    def test_valid_optimiser(self, opt):
        cfg = TrainingConfig(optimiser=opt)
        assert cfg.optimiser == opt

    def test_invalid_optimiser_raises(self):
        with pytest.raises(ValueError, match="optimiser"):
            TrainingConfig(optimiser="lion")

    def test_negative_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_dropout_rate_ge_1_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            TrainingConfig(dropout_rate=1.0)

    def test_dropout_rate_negative_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            TrainingConfig(dropout_rate=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Computed properties
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConfigProperties:
    def test_effective_fine_tune_lr_default(self):
        cfg = TrainingConfig(learning_rate=1e-4)
        assert cfg.effective_fine_tune_lr == pytest.approx(1e-5)

    def test_effective_fine_tune_lr_explicit(self):
        cfg = TrainingConfig(learning_rate=1e-4, fine_tune_lr=5e-6)
        assert cfg.effective_fine_tune_lr == pytest.approx(5e-6)

    def test_class_weight_map_none_when_unset(self):
        cfg = TrainingConfig()
        assert cfg.class_weight_map is None

    def test_class_weight_map_converts_names_to_indices(self):
        cfg = TrainingConfig(
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            class_weights={"glioma": 1.5, "notumor": 0.8},
        )
        wmap = cfg.class_weight_map
        assert wmap is not None
        assert wmap[0] == pytest.approx(1.5)   # glioma → index 0
        assert wmap[2] == pytest.approx(0.8)   # notumor → index 2

    def test_resolved_dataset_dir_uses_settings_when_none(self):
        from app.core.config import settings
        cfg = TrainingConfig(dataset_dir=None)
        assert cfg.resolved_dataset_dir == settings.dataset_processed_dir

    def test_resolved_dataset_dir_uses_override(self, tmp_path):
        cfg = TrainingConfig(dataset_dir=str(tmp_path))
        assert cfg.resolved_dataset_dir == tmp_path

    def test_resolved_output_dir_uses_settings_when_none(self):
        from app.core.config import settings
        cfg = TrainingConfig(output_dir=None)
        assert cfg.resolved_output_dir == settings.saved_models_dir


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConfigSerialisation:
    def test_to_dict_returns_dict(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["architecture"] == "mambavision"

    def test_from_dict_round_trip(self):
        cfg1 = TrainingConfig(architecture="vgg16", epochs=10, batch_size=16)
        d    = cfg1.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.architecture == cfg1.architecture
        assert cfg2.epochs       == cfg1.epochs
        assert cfg2.batch_size   == cfg1.batch_size

    def test_from_dict_ignores_unknown_keys(self):
        d = TrainingConfig().to_dict()
        d["unknown_future_field"] = "ignored"
        cfg = TrainingConfig.from_dict(d)
        assert cfg.architecture == "mambavision"

    def test_save_json_and_from_json(self, tmp_path):
        cfg1 = TrainingConfig(architecture="resnet50", epochs=5)
        path = tmp_path / "cfg.json"
        cfg1.save_json(path)

        assert path.exists()

        cfg2 = TrainingConfig.from_json(path)
        assert cfg2.architecture == "resnet50"
        assert cfg2.epochs       == 5

    def test_save_json_content_is_valid_json(self, tmp_path):
        cfg  = TrainingConfig(architecture="cnn", epochs=3)
        path = tmp_path / "cfg.json"
        cfg.save_json(path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["architecture"] == "cnn"
        assert data["epochs"]       == 3

    def test_from_settings_classmethod(self):
        cfg = TrainingConfig.from_settings()
        from app.core.config import settings
        assert cfg.image_size   == settings.image_size
        assert cfg.num_classes  == settings.num_classes
        assert cfg.class_names  == settings.classes
