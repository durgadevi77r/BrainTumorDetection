"""
tests/test_training_checkpoints.py — Unit tests for training.checkpoints.

Tests
-----
- save_checkpoint_info writes a valid JSON sidecar.
- load_checkpoint_info reads the sidecar back.
- load_best_weights returns False when no file exists.
- load_best_weights returns True after writing a real weight file.
- list_checkpoints returns sorted entries.
- delete_checkpoint removes the directory when confirm=True.
- delete_checkpoint is a no-op when confirm=False.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training.checkpoints import (
    best_weights_path,
    checkpoint_dir,
    checkpoint_info_path,
    delete_checkpoint,
    list_checkpoints,
    load_best_weights,
    load_checkpoint_info,
    save_checkpoint_info,
)
from training.config import TrainingConfig


def _cfg(tmp_path: Path, arch: str = "cnn") -> TrainingConfig:
    return TrainingConfig(architecture=arch, output_dir=str(tmp_path))


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4))


class TestSaveCheckpointInfo:
    def test_creates_json_file(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-save-001"
        path   = save_checkpoint_info(cfg, exp_id, metrics={"val_accuracy": 0.9})
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_contains_expected_keys(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-save-002"
        save_checkpoint_info(cfg, exp_id, metrics={"val_accuracy": 0.85}, epoch=5, phase=1)
        with open(checkpoint_info_path(cfg, exp_id)) as fh:
            data = json.load(fh)
        assert data["experiment_id"]          == exp_id
        assert data["architecture"]           == "cnn"
        assert data["phase"]                  == 1
        assert data["epoch"]                  == 5
        assert data["metrics"]["val_accuracy"] == pytest.approx(0.85)

    def test_config_summary_included(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-save-003"
        save_checkpoint_info(cfg, exp_id)
        with open(checkpoint_info_path(cfg, exp_id)) as fh:
            data = json.load(fh)
        assert "config_summary" in data
        assert "learning_rate" in data["config_summary"]

    def test_weights_path_points_to_pt_file(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-save-004"
        save_checkpoint_info(cfg, exp_id)
        with open(checkpoint_info_path(cfg, exp_id)) as fh:
            data = json.load(fh)
        assert data["weights_path"].endswith(".pt")


class TestLoadCheckpointInfo:
    def test_returns_none_when_missing(self, tmp_path):
        cfg    = _cfg(tmp_path)
        result = load_checkpoint_info(cfg, "nonexistent")
        assert result is None

    def test_returns_dict_after_save(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-load-001"
        save_checkpoint_info(cfg, exp_id, metrics={"val_accuracy": 0.91})
        info   = load_checkpoint_info(cfg, exp_id)
        assert isinstance(info, dict)
        assert info["experiment_id"] == exp_id


class TestLoadBestWeights:
    def test_returns_false_when_no_file(self, tmp_path):
        cfg    = _cfg(tmp_path)
        model  = _tiny_model()
        result = load_best_weights(model, cfg, "no-such-exp")
        assert result is False

    def test_returns_true_after_saving_weights(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-weights-001"
        model  = _tiny_model()

        # Save the state_dict as a proper checkpoint
        wpath = best_weights_path(cfg, exp_id)
        wpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(wpath))

        result = load_best_weights(model, cfg, exp_id)
        assert result is True

    def test_loaded_weights_match_saved(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-weights-002"

        # Create model, set known weights, save
        model1 = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            model1[0].weight.fill_(3.14)

        wpath = best_weights_path(cfg, exp_id)
        wpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model1.state_dict(), str(wpath))

        # Load into a fresh model and verify
        model2 = nn.Sequential(nn.Linear(4, 4, bias=False))
        load_best_weights(model2, cfg, exp_id)
        assert torch.allclose(model2[0].weight, model1[0].weight)


class TestListCheckpoints:
    def test_returns_empty_list_when_no_checkpoints(self, tmp_path):
        cfg    = _cfg(tmp_path)
        result = list_checkpoints(cfg)
        assert result == []

    def test_returns_entries_after_save(self, tmp_path):
        cfg = _cfg(tmp_path)
        for exp_id in ["exp-list-001", "exp-list-002"]:
            save_checkpoint_info(cfg, exp_id, metrics={"val_accuracy": 0.8})
        entries = list_checkpoints(cfg)
        assert len(entries) == 2

    def test_entries_sorted_newest_first(self, tmp_path):
        import time
        cfg = _cfg(tmp_path)
        save_checkpoint_info(cfg, "exp-older", metrics={"val_accuracy": 0.7})
        time.sleep(0.05)
        save_checkpoint_info(cfg, "exp-newer", metrics={"val_accuracy": 0.9})
        entries = list_checkpoints(cfg)
        assert entries[0]["experiment_id"] == "exp-newer"


class TestDeleteCheckpoint:
    def test_no_action_without_confirm(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-del-001"
        save_checkpoint_info(cfg, exp_id)
        ckpt_d = checkpoint_dir(cfg, exp_id)

        result = delete_checkpoint(cfg, exp_id, confirm=False)
        assert result is False
        assert ckpt_d.exists()

    def test_deletes_directory_with_confirm(self, tmp_path):
        cfg    = _cfg(tmp_path)
        exp_id = "exp-del-002"
        save_checkpoint_info(cfg, exp_id)
        ckpt_d = checkpoint_dir(cfg, exp_id)
        assert ckpt_d.exists()

        result = delete_checkpoint(cfg, exp_id, confirm=True)
        assert result is True
        assert not ckpt_d.exists()
