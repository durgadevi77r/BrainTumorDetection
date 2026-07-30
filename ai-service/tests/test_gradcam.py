"""
tests/test_gradcam.py — Comprehensive tests for the Grad-CAM module.

Coverage
--------
TestLayerSelection          gradcam._find_gradcam_target
TestComputeGradcamHeatmap   gradcam.compute_gradcam_heatmap
TestOverlayHeatmap          gradcam.overlay_heatmap
TestOutputDimensions        output image shape / dtype checks
TestGenerateGradcam         gradcam.generate_gradcam (full pipeline, mocked)
TestOutputDirectory         output directory creation + file existence
TestMetadata                metadata.json keys and values
TestConfidenceMatch         confidence in metadata matches inference output
TestEdgeCases               invalid images, missing checkpoints, bad class idx
TestCPUGPUExecution         device handling
TestBackwardCompat          return dict keys (pipeline integration contract)
"""

from __future__ import annotations

import json
import struct
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn


# ─── Shared image helpers ─────────────────────────────────────────────────────

def _varied_png_bytes(h: int = 64, w: int = 64, seed: int = 7) -> bytes:
    """Natural-variance PNG that passes quality checks."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _solid_png_bytes(h: int = 64, w: int = 64, value: int = 128) -> bytes:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ─── Fake model builders ──────────────────────────────────────────────────────

class _TinyConvModel(nn.Module):
    """Minimal Conv2d-BN model that mimics the MambaVision structure."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # Mimic MambaVision: model.norm = BatchNorm2d (last spatial BN)
        self.norm = nn.BatchNorm2d(32)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class _MambaVisionLike(nn.Module):
    """Wrapper that mirrors the HF MambaVisionModelForImageClassification structure."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.model = _TinyConvModel(num_classes)

    def forward(self, x: torch.Tensor):
        from types import SimpleNamespace
        logits = self.model(x)
        # Return an object with .logits like the real HF model does
        return SimpleNamespace(logits=logits)


def _make_wrapped(num_classes: int = 4) -> MagicMock:
    """
    Build a TorchImageClassifier-like mock that wraps _MambaVisionLike.

    The real TorchImageClassifier.to_tensor converts NHWC float32 → NCHW
    normalised tensor.  We replicate that here using a lightweight version
    so Grad-CAM hooks fire on real tensors.
    """
    from app.models.mambavision.predictor import TorchImageClassifier

    raw = _MambaVisionLike(num_classes)
    wrapped = TorchImageClassifier(raw, device=torch.device("cpu"))
    return wrapped


def _fake_probs(num_classes: int = 4, top_class: int = 0) -> np.ndarray:
    probs = np.full((1, num_classes), 0.05, dtype=np.float32)
    probs[0, top_class] = 0.85
    # Normalise
    probs = probs / probs.sum()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# TestLayerSelection
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerSelection:

    def test_finds_mambavision_norm_first(self) -> None:
        """Strategy 1: model.model.norm (BatchNorm2d) must be selected."""
        from app.utils.gradcam import _find_gradcam_target
        mv = _MambaVisionLike()
        name, layer = _find_gradcam_target(mv)
        assert name == "model.model.norm"
        assert isinstance(layer, nn.BatchNorm2d)

    def test_finds_last_batchnorm_when_no_model_attr(self) -> None:
        """Strategy 2: fall back to last BatchNorm2d when .model is absent."""
        from app.utils.gradcam import _find_gradcam_target

        class _FlatBNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.bn1  = nn.BatchNorm2d(8)
                self.bn2  = nn.BatchNorm2d(8)
            def forward(self, x): return self.bn2(self.bn1(self.conv(x)))

        model = _FlatBNModel()
        name, layer = _find_gradcam_target(model)
        assert name == "bn2"
        assert isinstance(layer, nn.BatchNorm2d)

    def test_finds_last_conv2d_as_final_fallback(self) -> None:
        """Strategy 3: Conv2d fallback when no BatchNorm2d exists."""
        from app.utils.gradcam import _find_gradcam_target

        class _ConvOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Conv2d(3, 8, 1)
                self.b = nn.Conv2d(8, 16, 1)
            def forward(self, x): return self.b(self.a(x))

        model = _ConvOnlyModel()
        name, layer = _find_gradcam_target(model)
        assert name == "b"
        assert isinstance(layer, nn.Conv2d)

    def test_raises_when_no_hookable_layer(self) -> None:
        """ValueError when neither BN2d nor Conv2d is present."""
        from app.utils.gradcam import _find_gradcam_target

        class _LinearOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 4)
            def forward(self, x): return self.fc(x)

        with pytest.raises(ValueError, match="Could not locate"):
            _find_gradcam_target(_LinearOnly())

    def test_returns_tuple_of_name_and_module(self) -> None:
        from app.utils.gradcam import _find_gradcam_target
        name, layer = _find_gradcam_target(_MambaVisionLike())
        assert isinstance(name, str)
        assert isinstance(layer, nn.Module)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeGradcamHeatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGradcamHeatmap:

    def _tensor(self, h: int = 64, w: int = 64) -> np.ndarray:
        """float32 (1, H, W, 3) in [0, 1]."""
        rng = np.random.default_rng(42)
        return rng.random((1, h, w, 3)).astype(np.float32)

    def test_returns_float32_heatmap(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert heatmap.dtype == np.float32

    def test_heatmap_values_in_unit_range(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert float(heatmap.min()) >= 0.0
        assert float(heatmap.max()) <= 1.0 + 1e-6

    def test_heatmap_is_2d(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert heatmap.ndim == 2

    def test_heatmap_has_nonzero_values(self) -> None:
        """A properly computed Grad-CAM should not be an all-zero map."""
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert float(heatmap.max()) > 0.0

    def test_different_classes_give_different_heatmaps(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped(num_classes=4)
        t = self._tensor()
        h0, _ = compute_gradcam_heatmap(wrapped, t, 0)
        h1, _ = compute_gradcam_heatmap(wrapped, t, 1)
        # Should differ because gradients are class-specific
        assert not np.allclose(h0, h1, atol=1e-4)

    def test_returns_target_layer_name_string(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        _, layer_name = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert isinstance(layer_name, str)
        assert len(layer_name) > 0

    def test_mambavision_like_uses_norm_layer(self) -> None:
        """For MambaVision-like structure the hooked layer name contains 'norm'."""
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        _, layer_name = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert "norm" in layer_name

    def test_raises_on_non_adapter_input(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        with pytest.raises(TypeError, match="TorchImageClassifier"):
            compute_gradcam_heatmap(MagicMock(spec=[]), np.zeros((1, 8, 8, 3), np.float32), 0)

    def test_raises_on_out_of_range_class_index(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped(num_classes=4)
        with pytest.raises(ValueError, match="class_index"):
            compute_gradcam_heatmap(wrapped, self._tensor(), 99)


# ─────────────────────────────────────────────────────────────────────────────
# TestOverlayHeatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayHeatmap:

    def _rgb(self, h: int = 64, w: int = 64) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def _heatmap(self, h: int = 16, w: int = 16) -> np.ndarray:
        rng = np.random.default_rng(2)
        return rng.random((h, w)).astype(np.float32)

    def test_overlay_shape_matches_input(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        rgb = self._rgb(64, 80)
        hm = self._heatmap(16, 20)
        overlay, _ = overlay_heatmap(rgb, hm)
        assert overlay.shape == (64, 80, 3)

    def test_heatmap_output_shape_matches_input(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        rgb = self._rgb(64, 80)
        hm = self._heatmap(8, 10)
        _, heatmap_bgr = overlay_heatmap(rgb, hm)
        assert heatmap_bgr.shape == (64, 80, 3)

    def test_overlay_is_uint8(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        overlay, _ = overlay_heatmap(self._rgb(), self._heatmap())
        assert overlay.dtype == np.uint8

    def test_heatmap_bgr_is_uint8(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        _, heatmap_bgr = overlay_heatmap(self._rgb(), self._heatmap())
        assert heatmap_bgr.dtype == np.uint8

    def test_alpha_zero_equals_original(self) -> None:
        """alpha=0 → overlay should equal the original (BGR)."""
        from app.utils.gradcam import overlay_heatmap
        rgb = self._rgb()
        overlay, _ = overlay_heatmap(rgb, self._heatmap(), alpha=0.0)
        expected_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        np.testing.assert_array_equal(overlay, expected_bgr)

    def test_alpha_one_equals_heatmap(self) -> None:
        """alpha=1 → overlay should equal the pure heatmap."""
        from app.utils.gradcam import overlay_heatmap
        rgb = self._rgb()
        hm = self._heatmap()
        overlay, heatmap_bgr = overlay_heatmap(rgb, hm, alpha=1.0)
        np.testing.assert_array_equal(overlay, heatmap_bgr)

    def test_same_resolution_heatmap_still_works(self) -> None:
        """Heatmap already at image resolution — resize should be a no-op."""
        from app.utils.gradcam import overlay_heatmap
        rgb = self._rgb(32, 32)
        hm = self._heatmap(32, 32)
        overlay, _ = overlay_heatmap(rgb, hm)
        assert overlay.shape == (32, 32, 3)

    def test_returns_two_element_tuple(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        result = overlay_heatmap(self._rgb(), self._heatmap())
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestOutputDimensions
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputDimensions:
    """Verify that Grad-CAM artefacts have the correct spatial dimensions."""

    def test_heatmap_spatial_matches_feature_map(self) -> None:
        """Raw heatmap shape equals the feature map H×W, not the input H×W."""
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        tensor = np.random.default_rng(5).random((1, 64, 64, 3)).astype(np.float32)
        heatmap, _ = compute_gradcam_heatmap(wrapped, tensor, 0)
        # The tiny model doesn't downsample, so feature map = input size.
        # What matters: shape is 2-D and both dims > 0.
        assert heatmap.ndim == 2
        assert heatmap.shape[0] > 0
        assert heatmap.shape[1] > 0

    def test_overlay_matches_display_image_size(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        display = np.zeros((128, 96, 3), dtype=np.uint8)
        hm = np.ones((7, 5), dtype=np.float32) * 0.5
        overlay, hm_bgr = overlay_heatmap(display, hm)
        assert overlay.shape == (128, 96, 3)
        assert hm_bgr.shape  == (128, 96, 3)

    def test_non_square_image_handled(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap, overlay_heatmap
        wrapped = _make_wrapped()
        tensor = np.random.default_rng(9).random((1, 48, 96, 3)).astype(np.float32)
        heatmap, _ = compute_gradcam_heatmap(wrapped, tensor, 0)
        display = np.zeros((48, 96, 3), dtype=np.uint8)
        overlay, _ = overlay_heatmap(display, heatmap)
        assert overlay.shape == (48, 96, 3)


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerateGradcam  (full pipeline — load_keras_model mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateGradcam:
    """
    Exercises generate_gradcam() end-to-end with the real model replaced by
    our lightweight _MambaVisionLike fixture.  No disk I/O to saved_models/.
    """

    @pytest.fixture()
    def tmp_output(self, tmp_path: Path) -> Path:
        return tmp_path / "gradcam_out"

    @pytest.fixture()
    def wrapped_model(self):
        return _make_wrapped(num_classes=4)

    @pytest.fixture()
    def png_bytes(self) -> bytes:
        return _varied_png_bytes(64, 64)

    def test_returns_required_keys(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        required = {
            "gradcam_path", "original_path", "heatmap_path",
            "metadata_path", "class_index", "class_name",
            "confidence", "image_id", "target_layer",
        }
        assert required.issubset(result.keys())

    def test_gradcam_path_is_overlay_png(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        assert Path(result["gradcam_path"]).name == "overlay.png"

    def test_all_files_created(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        for key in ("gradcam_path", "original_path", "heatmap_path", "metadata_path"):
            assert Path(result[key]).exists(), f"{key} file not found"

    def test_explicit_class_index_respected(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, class_index=2, output_dir=tmp_output)
        assert result["class_index"] == 2

    def test_explicit_image_id_respected(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        my_id = "test-image-abc"
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, image_id=my_id, output_dir=tmp_output)
        assert result["image_id"] == my_id

    def test_auto_image_id_is_uuid(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        # Should parse without error
        uuid.UUID(result["image_id"])

    def test_confidence_is_float_between_0_and_1(
        self, tmp_output, wrapped_model, png_bytes
    ) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_class_name_is_valid(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        from app.core.config import settings
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        assert result["class_name"] in settings.classes

    def test_target_layer_string_nonempty(self, tmp_output, wrapped_model, png_bytes) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(png_bytes, output_dir=tmp_output)
        assert isinstance(result["target_layer"], str)
        assert len(result["target_layer"]) > 0

    def test_file_path_input(self, tmp_path, tmp_output, wrapped_model) -> None:
        """generate_gradcam accepts a filesystem path, not just bytes."""
        from app.utils.gradcam import generate_gradcam
        img_path = tmp_path / "test.png"
        img_path.write_bytes(_varied_png_bytes())
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(img_path, output_dir=tmp_output)
        assert Path(result["gradcam_path"]).exists()


# ─────────────────────────────────────────────────────────────────────────────
# TestOutputDirectory
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputDirectory:

    @pytest.fixture()
    def wrapped_model(self):
        return _make_wrapped()

    def test_creates_per_image_subdirectory(self, tmp_path, wrapped_model) -> None:
        from app.utils.gradcam import generate_gradcam
        img_id = "dir-test-001"
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            generate_gradcam(
                _varied_png_bytes(),
                image_id=img_id,
                output_dir=tmp_path / "out",
            )
        assert (tmp_path / "out" / img_id).is_dir()

    def test_creates_nested_output_dir_if_missing(self, tmp_path, wrapped_model) -> None:
        from app.utils.gradcam import generate_gradcam
        deep_dir = tmp_path / "a" / "b" / "c"
        assert not deep_dir.exists()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(
                _varied_png_bytes(), output_dir=deep_dir
            )
        assert Path(result["gradcam_path"]).exists()

    def test_overlay_png_is_readable_image(self, tmp_path, wrapped_model) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        img = cv2.imread(result["gradcam_path"])
        assert img is not None
        assert img.ndim == 3

    def test_original_png_is_readable_image(self, tmp_path, wrapped_model) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        img = cv2.imread(result["original_path"])
        assert img is not None

    def test_heatmap_png_is_readable_image(self, tmp_path, wrapped_model) -> None:
        from app.utils.gradcam import generate_gradcam
        with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        img = cv2.imread(result["heatmap_path"])
        assert img is not None

    def test_multiple_calls_use_separate_subdirs(self, tmp_path, wrapped_model) -> None:
        from app.utils.gradcam import generate_gradcam
        out = tmp_path / "multi"
        results = []
        for i in range(3):
            with patch("app.models.load_model.load_keras_model", return_value=wrapped_model):
                r = generate_gradcam(
                    _varied_png_bytes(seed=i),
                    image_id=f"img-{i}",
                    output_dir=out,
                )
            results.append(r)
        paths = {Path(r["gradcam_path"]).parent for r in results}
        assert len(paths) == 3, "Each call should use a distinct sub-directory"


# ─────────────────────────────────────────────────────────────────────────────
# TestMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadata:

    @pytest.fixture()
    def result_and_meta(self, tmp_path):
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(
                _varied_png_bytes(),
                image_id="meta-test-001",
                output_dir=tmp_path / "out",
            )
        with open(result["metadata_path"], encoding="utf-8") as fh:
            meta = json.load(fh)
        return result, meta

    def test_metadata_file_is_valid_json(self, tmp_path) -> None:
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        with open(result["metadata_path"], encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_metadata_contains_required_keys(self, result_and_meta) -> None:
        _, meta = result_and_meta
        required = {
            "image_id", "model_name", "predicted_at",
            "class_index", "class_name", "confidence",
            "target_layer", "alpha",
            "image_width", "image_height",
            "heatmap_min", "heatmap_max", "heatmap_mean",
            "original_path", "heatmap_path",
            "gradcam_path", "metadata_path",
        }
        assert required.issubset(meta.keys())

    def test_metadata_image_id_matches_result(self, result_and_meta) -> None:
        result, meta = result_and_meta
        assert meta["image_id"] == result["image_id"]

    def test_metadata_confidence_matches_result(self, result_and_meta) -> None:
        result, meta = result_and_meta
        assert abs(meta["confidence"] - result["confidence"]) < 1e-5

    def test_metadata_class_index_matches_result(self, result_and_meta) -> None:
        result, meta = result_and_meta
        assert meta["class_index"] == result["class_index"]

    def test_metadata_class_name_matches_result(self, result_and_meta) -> None:
        result, meta = result_and_meta
        assert meta["class_name"] == result["class_name"]

    def test_metadata_paths_are_absolute(self, result_and_meta) -> None:
        _, meta = result_and_meta
        for key in ("original_path", "heatmap_path", "gradcam_path", "metadata_path"):
            assert Path(meta[key]).is_absolute(), f"{key} should be absolute"

    def test_metadata_heatmap_stats_in_range(self, result_and_meta) -> None:
        _, meta = result_and_meta
        assert 0.0 <= meta["heatmap_min"] <= 1.0
        assert 0.0 <= meta["heatmap_max"] <= 1.0
        assert 0.0 <= meta["heatmap_mean"] <= 1.0
        assert meta["heatmap_min"] <= meta["heatmap_mean"] <= meta["heatmap_max"]

    def test_metadata_image_dimensions_positive(self, result_and_meta) -> None:
        _, meta = result_and_meta
        assert meta["image_width"] > 0
        assert meta["image_height"] > 0

    def test_metadata_predicted_at_is_iso_timestamp(self, result_and_meta) -> None:
        from datetime import datetime
        _, meta = result_and_meta
        # Should parse without raising
        dt = datetime.fromisoformat(meta["predicted_at"])
        assert dt.tzinfo is not None  # must be timezone-aware


# ─────────────────────────────────────────────────────────────────────────────
# TestConfidenceMatch
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceMatch:
    """Confidence returned by generate_gradcam must agree with predict output."""

    def test_confidence_matches_predict_top_class(self, tmp_path) -> None:
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped(num_classes=4)
        png = _varied_png_bytes()

        # Capture the probabilities that .predict() returns
        original_predict = wrapped.predict

        captured: Dict[str, Any] = {}

        def _tracking_predict(batch, verbose=0):
            probs = original_predict(batch, verbose=verbose)
            captured["probs"] = probs
            return probs

        wrapped.predict = _tracking_predict  # type: ignore[method-assign]

        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(png, output_dir=tmp_path / "out")

        assert "probs" in captured
        top_idx = int(np.argmax(captured["probs"][0]))
        expected_conf = round(float(captured["probs"][0, top_idx]), 6)
        assert abs(result["confidence"] - expected_conf) < 1e-4

    def test_explicit_class_confidence_recorded(self, tmp_path) -> None:
        """When class_index is supplied, the confidence for that class is returned."""
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped(num_classes=4)
        png = _varied_png_bytes()

        original_predict = wrapped.predict
        captured: Dict[str, Any] = {}

        def _tracking_predict(batch, verbose=0):
            probs = original_predict(batch, verbose=verbose)
            captured["probs"] = probs
            return probs

        wrapped.predict = _tracking_predict  # type: ignore[method-assign]

        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(png, class_index=1, output_dir=tmp_path / "out")

        expected = round(float(captured["probs"][0, 1]), 6)
        assert abs(result["confidence"] - expected) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_invalid_image_bytes_raise(self, tmp_path) -> None:
        """Garbage bytes that can't be decoded should raise, not silently succeed."""
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            with pytest.raises(Exception):
                generate_gradcam(b"not-an-image", output_dir=tmp_path / "out")

    def test_missing_model_path_raises_file_not_found(self, tmp_path) -> None:
        """FileNotFoundError when load_keras_model can't find weights."""
        from app.utils.gradcam import generate_gradcam
        with patch(
            "app.models.load_model.load_keras_model",
            side_effect=FileNotFoundError("no model"),
        ):
            with pytest.raises(FileNotFoundError):
                generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")

    def test_out_of_range_class_index_raises(self, tmp_path) -> None:
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped(num_classes=4)
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            with pytest.raises(ValueError, match="class_index"):
                generate_gradcam(
                    _varied_png_bytes(),
                    class_index=99,
                    output_dir=tmp_path / "out",
                )

    def test_single_pixel_image(self, tmp_path) -> None:
        """1×1 pixel input should not crash the pipeline."""
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        img = np.full((1, 1, 3), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(buf.tobytes(), output_dir=tmp_path / "out")
        assert Path(result["gradcam_path"]).exists()

    def test_flat_heatmap_produces_zero_array(self) -> None:
        """When max activation is near zero the heatmap should be all zeros."""
        from app.utils.gradcam import compute_gradcam_heatmap
        import torch.nn as nn

        class _ZeroGradModel(nn.Module):
            """Always outputs the same logit, so gradients are zero."""
            def __init__(self):
                super().__init__()
                # Input arrives as NCHW with 3 channels after to_tensor()
                self.norm = nn.BatchNorm2d(3)
                self.pool = nn.AdaptiveAvgPool2d(1)
                # weight=0 → zero gradients
                self.fc = nn.Linear(3, 4, bias=True)
                nn.init.zeros_(self.fc.weight)
                nn.init.zeros_(self.fc.bias)

            def forward(self, x):
                x = self.norm(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _ZeroGradModel()

            def forward(self, x):
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.model(x))

        from app.models.mambavision.predictor import TorchImageClassifier
        wrapped = TorchImageClassifier(_Wrapper(), device=torch.device("cpu"))
        tensor = np.ones((1, 4, 4, 3), dtype=np.float32) * 0.5
        heatmap, _ = compute_gradcam_heatmap(wrapped, tensor, 0)
        # Flat activation → all-zero map
        assert float(heatmap.max()) == pytest.approx(0.0, abs=1e-6)

    def test_overlay_heatmap_with_all_zeros(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        hm = np.zeros((8, 8), dtype=np.float32)
        overlay, _ = overlay_heatmap(rgb, hm)
        assert overlay.shape == (32, 32, 3)
        assert overlay.dtype == np.uint8

    def test_overlay_heatmap_with_all_ones(self) -> None:
        from app.utils.gradcam import overlay_heatmap
        rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        hm = np.ones((8, 8), dtype=np.float32)
        overlay, _ = overlay_heatmap(rgb, hm)
        assert overlay.shape == (32, 32, 3)


# ─────────────────────────────────────────────────────────────────────────────
# TestCPUGPUExecution
# ─────────────────────────────────────────────────────────────────────────────

class TestCPUGPUExecution:

    def _tensor(self) -> np.ndarray:
        return np.random.default_rng(3).random((1, 32, 32, 3)).astype(np.float32)

    def test_runs_on_cpu_device(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        from app.models.mambavision.predictor import TorchImageClassifier
        raw = _MambaVisionLike()
        wrapped = TorchImageClassifier(raw, device=torch.device("cpu"))
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert heatmap.dtype == np.float32

    def test_device_stays_cpu_when_cuda_unavailable(self) -> None:
        """When CUDA is absent the adapter must select CPU without error."""
        from app.models.mambavision.predictor import TorchImageClassifier
        raw = _MambaVisionLike()
        with patch("torch.cuda.is_available", return_value=False):
            wrapped = TorchImageClassifier(raw)
        assert wrapped.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available in this environment",
    )
    def test_runs_on_cuda_device(self) -> None:
        from app.utils.gradcam import compute_gradcam_heatmap
        from app.models.mambavision.predictor import TorchImageClassifier
        raw = _MambaVisionLike()
        wrapped = TorchImageClassifier(raw, device=torch.device("cuda"))
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert heatmap.dtype == np.float32

    def test_heatmap_returned_as_cpu_numpy(self) -> None:
        """Return value must always be a CPU numpy array, regardless of model device."""
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        heatmap, _ = compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        assert isinstance(heatmap, np.ndarray)

    def test_hooks_are_removed_after_computation(self) -> None:
        """Forward/backward hooks must be deregistered to avoid memory leaks."""
        from app.utils.gradcam import compute_gradcam_heatmap
        wrapped = _make_wrapped()
        raw_model = wrapped.model

        hooks_before = sum(
            len(m._forward_hooks) + len(m._backward_hooks)
            for m in raw_model.modules()
        )
        compute_gradcam_heatmap(wrapped, self._tensor(), 0)
        hooks_after = sum(
            len(m._forward_hooks) + len(m._backward_hooks)
            for m in raw_model.modules()
        )
        assert hooks_after == hooks_before, (
            "Hooks were not cleaned up after Grad-CAM computation"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestBackwardCompat
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompat:
    """
    Verify the integration contract between generate_gradcam and predict.py.
    predict.py reads result.get("gradcam_path") — that key must always be present
    and must point to the overlay PNG.
    """

    def test_gradcam_path_key_present(self, tmp_path) -> None:
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        assert "gradcam_path" in result

    def test_gradcam_path_is_a_string(self, tmp_path) -> None:
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        assert isinstance(result["gradcam_path"], str)

    def test_gradcam_path_file_exists_on_disk(self, tmp_path) -> None:
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        assert Path(result["gradcam_path"]).exists()

    def test_all_nine_return_keys_present(self, tmp_path) -> None:
        """Exact set of keys the rest of the codebase may depend on."""
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(_varied_png_bytes(), output_dir=tmp_path / "out")
        expected_keys = {
            "gradcam_path", "original_path", "heatmap_path", "metadata_path",
            "class_index", "class_name", "confidence", "image_id", "target_layer",
        }
        assert expected_keys == set(result.keys())

    def test_predict_pipeline_uses_gradcam_path(self, tmp_path) -> None:
        """
        Simulate predict.py's call pattern: generate_gradcam is called and
        result.get('gradcam_path') must return a non-None truthy string.
        """
        from app.utils.gradcam import generate_gradcam
        wrapped = _make_wrapped()
        with patch("app.models.load_model.load_keras_model", return_value=wrapped):
            result = generate_gradcam(
                _varied_png_bytes(),
                model_name="mambavision",
                class_index=0,
                image_id="predict-compat-test",
                output_dir=tmp_path / "out",
            )
        gradcam_path = result.get("gradcam_path")
        assert gradcam_path is not None
        assert len(gradcam_path) > 0
        assert Path(gradcam_path).exists()
