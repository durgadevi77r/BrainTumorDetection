"""
tests/test_preprocessing.py — Comprehensive tests for the preprocessing module.

Coverage
--------
TestPreprocessConfig         config.py  — dataclass defaults, to_dict/from_dict,
                                          from_settings, IMAGENET constants
TestTransformsIO             transforms.py — load_image_bgr, encode_image_png/base64
TestTransformsDenoise        transforms.py — median filter, gaussian blur
TestTransformsContrast       transforms.py — CLAHE, histogram equalisation
TestTransformsSpatial        transforms.py — resize, pad_to_square
TestTransformsColour         transforms.py — bgr_to_rgb, rgb_to_bgr
TestTransformsNormalisation  transforms.py — normalize_image, rescale_only,
                                              denormalize_image, round-trip
TestTransformsPipeline       transforms.py — apply_spatial_pipeline with cfg
TestQualityChecks            quality.py   — every check (pass + fail paths),
                                            batch validation, summary/to_dict
TestAugmentationConfig       augmentation.py — dataclass, to_dict
TestAugmentationGenerators   augmentation.py — build_train/eval_datagen,
                                               build_data_generators_from_split
TestAugmentationApply        augmentation.py — apply_augmentation, n_samples
TestPipelineInference        preprocess.py  — preprocess_for_inference
TestPipelineGradcam          preprocess.py  — preprocess_for_gradcam
TestPipelinePreview          preprocess.py  — preprocess_for_preview
TestPipelineShims            preprocess.py  — backward-compat shims
TestPreprocessingAPIRoutes   routes.py      — POST /preprocess/quality-check
                                              POST /preprocess/preview
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _make_bgr(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)


def _make_rgb(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:
    bgr = _make_bgr(h, w, seed)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _png_bytes(h: int = 64, w: int = 64, value: int = 128) -> bytes:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _varied_png_bytes(h: int = 64, w: int = 64) -> bytes:
    """PNG with natural variation — passes all quality checks."""
    rng = np.random.default_rng(7)
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# config.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessConfig:
    def test_defaults_match_imagenet_constants(self) -> None:
        from app.preprocessing.config import PreprocessConfig, IMAGENET_MEAN, IMAGENET_STD
        cfg = PreprocessConfig()
        assert cfg.norm_mean == IMAGENET_MEAN
        assert cfg.norm_std  == IMAGENET_STD

    def test_default_image_size(self) -> None:
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig()
        assert cfg.image_size == 224

    def test_to_dict_is_json_serialisable(self) -> None:
        import json
        from app.preprocessing.config import PreprocessConfig
        d = PreprocessConfig().to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_contains_all_fields(self) -> None:
        from app.preprocessing.config import PreprocessConfig
        d = PreprocessConfig().to_dict()
        for key in ("image_size", "apply_denoise", "apply_clahe",
                    "clahe_clip_limit", "normalise", "norm_mean", "norm_std",
                    "min_width", "min_height", "max_file_size_bytes",
                    "min_mean_intensity", "max_mean_intensity",
                    "min_laplacian_variance"):
            assert key in d, f"Missing key: {key}"

    def test_from_dict_round_trip(self) -> None:
        from app.preprocessing.config import PreprocessConfig
        original = PreprocessConfig(image_size=299, clahe_clip_limit=3.5)
        restored = PreprocessConfig.from_dict(original.to_dict())
        assert restored.image_size       == 299
        assert restored.clahe_clip_limit == 3.5
        assert restored.norm_mean        == original.norm_mean

    def test_from_dict_converts_list_to_tuple(self) -> None:
        from app.preprocessing.config import PreprocessConfig
        d = PreprocessConfig().to_dict()
        assert isinstance(d["norm_mean"], list)
        cfg = PreprocessConfig.from_dict(d)
        assert isinstance(cfg.norm_mean, tuple)

    def test_from_settings_returns_config(self) -> None:
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig.from_settings()
        assert isinstance(cfg, PreprocessConfig)
        assert cfg.image_size > 0

    def test_default_config_singleton_is_preprocess_config(self) -> None:
        from app.preprocessing.config import DEFAULT_CONFIG, PreprocessConfig
        assert isinstance(DEFAULT_CONFIG, PreprocessConfig)

    def test_override_does_not_mutate_default(self) -> None:
        from app.preprocessing.config import DEFAULT_CONFIG, PreprocessConfig
        original_size = DEFAULT_CONFIG.image_size
        _ = PreprocessConfig(image_size=512)
        assert DEFAULT_CONFIG.image_size == original_size


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — I/O
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsIO:
    def test_load_image_bgr_from_bytes(self) -> None:
        from app.preprocessing.transforms import load_image_bgr
        img = load_image_bgr(_png_bytes())
        assert img.dtype == np.uint8
        assert img.ndim  == 3
        assert img.shape[2] == 3

    def test_load_image_bgr_from_path(self, tmp_path: Path) -> None:
        from app.preprocessing.transforms import load_image_bgr
        p = tmp_path / "test.png"
        p.write_bytes(_png_bytes())
        img = load_image_bgr(p)
        assert img.shape[2] == 3

    def test_load_invalid_bytes_raises_value_error(self) -> None:
        from app.preprocessing.transforms import load_image_bgr
        with pytest.raises(ValueError, match="decode"):
            load_image_bgr(b"not an image")

    def test_load_missing_path_raises_value_error(self, tmp_path: Path) -> None:
        from app.preprocessing.transforms import load_image_bgr
        with pytest.raises(ValueError, match="Failed to load"):
            load_image_bgr(tmp_path / "nonexistent.png")

    def test_encode_image_png_returns_bytes(self) -> None:
        from app.preprocessing.transforms import encode_image_png
        rgb = _make_rgb()
        out = encode_image_png(rgb)
        assert isinstance(out, bytes)
        assert len(out) > 0

    def test_encode_image_png_is_valid_png(self) -> None:
        from app.preprocessing.transforms import encode_image_png
        rgb = _make_rgb()
        png = encode_image_png(rgb)
        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_image_base64_is_ascii_string(self) -> None:
        from app.preprocessing.transforms import encode_image_base64
        b64 = encode_image_base64(_make_rgb())
        assert isinstance(b64, str)
        base64.b64decode(b64)  # must not raise

    def test_encode_decode_round_trip(self) -> None:
        from app.preprocessing.transforms import encode_image_png, load_image_bgr, bgr_to_rgb
        rgb_orig = _make_rgb(32, 32)
        png_bytes = encode_image_png(rgb_orig)
        bgr_back  = load_image_bgr(png_bytes)
        rgb_back  = bgr_to_rgb(bgr_back)
        assert rgb_orig.shape == rgb_back.shape
        assert np.allclose(rgb_orig, rgb_back, atol=2)


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — Denoising
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsDenoise:
    def test_median_filter_preserves_shape(self) -> None:
        from app.preprocessing.transforms import apply_median_filter
        img = _make_bgr()
        assert apply_median_filter(img).shape == img.shape

    def test_median_filter_preserves_dtype(self) -> None:
        from app.preprocessing.transforms import apply_median_filter
        img = _make_bgr()
        assert apply_median_filter(img).dtype == np.uint8

    def test_median_filter_kernel_5(self) -> None:
        from app.preprocessing.transforms import apply_median_filter
        img = _make_bgr(32, 32)
        out = apply_median_filter(img, kernel_size=5)
        assert out.shape == img.shape

    def test_median_filter_even_kernel_raises(self) -> None:
        from app.preprocessing.transforms import apply_median_filter
        with pytest.raises(ValueError, match="odd"):
            apply_median_filter(_make_bgr(), kernel_size=4)

    def test_median_filter_reduces_salt_pepper_noise(self) -> None:
        from app.preprocessing.transforms import apply_median_filter
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        noisy = img.copy()
        noisy[5, 5] = 0    # salt pixel
        noisy[6, 6] = 255  # pepper pixel
        out = apply_median_filter(noisy, kernel_size=3)
        # Noise pixels should be smoothed towards neighbourhood median
        assert abs(int(out[5, 5, 0]) - 128) < abs(int(noisy[5, 5, 0]) - 128)

    def test_gaussian_blur_preserves_shape(self) -> None:
        from app.preprocessing.transforms import apply_gaussian_blur
        img = _make_bgr()
        assert apply_gaussian_blur(img).shape == img.shape

    def test_gaussian_blur_even_kernel_raises(self) -> None:
        from app.preprocessing.transforms import apply_gaussian_blur
        with pytest.raises(ValueError, match="odd"):
            apply_gaussian_blur(_make_bgr(), kernel_size=4)


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — Contrast
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsContrast:
    def test_clahe_preserves_shape(self) -> None:
        from app.preprocessing.transforms import apply_clahe
        img = _make_bgr()
        assert apply_clahe(img).shape == img.shape

    def test_clahe_preserves_dtype(self) -> None:
        from app.preprocessing.transforms import apply_clahe
        assert apply_clahe(_make_bgr()).dtype == np.uint8

    def test_clahe_values_in_range(self) -> None:
        from app.preprocessing.transforms import apply_clahe
        out = apply_clahe(_make_bgr())
        assert out.min() >= 0 and out.max() <= 255

    def test_clahe_custom_clip_limit(self) -> None:
        from app.preprocessing.transforms import apply_clahe
        out = apply_clahe(_make_bgr(), clip_limit=4.0, tile_grid_size=(4, 4))
        assert out.shape == (64, 64, 3)

    def test_histogram_equalisation_shape(self) -> None:
        from app.preprocessing.transforms import apply_histogram_equalisation
        out = apply_histogram_equalisation(_make_bgr())
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — Spatial
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsSpatial:
    @pytest.mark.parametrize("size", [32, 64, 128, 224])
    def test_resize_produces_correct_shape(self, size: int) -> None:
        from app.preprocessing.transforms import resize_image
        out = resize_image(_make_bgr(100, 80), size)
        assert out.shape == (size, size, 3)

    def test_resize_preserves_dtype(self) -> None:
        from app.preprocessing.transforms import resize_image
        assert resize_image(_make_bgr(), 64).dtype == np.uint8

    def test_resize_zero_raises(self) -> None:
        from app.preprocessing.transforms import resize_image
        with pytest.raises(ValueError, match="positive"):
            resize_image(_make_bgr(), 0)

    def test_pad_to_square_already_square(self) -> None:
        from app.preprocessing.transforms import pad_to_square
        img = _make_bgr(64, 64)
        out = pad_to_square(img)
        assert out.shape == (64, 64, 3)
        assert np.array_equal(out, img)

    def test_pad_to_square_wide_image(self) -> None:
        from app.preprocessing.transforms import pad_to_square
        img = _make_bgr(40, 80)
        out = pad_to_square(img)
        assert out.shape[0] == out.shape[1] == 80

    def test_pad_to_square_tall_image(self) -> None:
        from app.preprocessing.transforms import pad_to_square
        img = _make_bgr(80, 40)
        out = pad_to_square(img)
        assert out.shape[0] == out.shape[1] == 80

    def test_pad_to_square_content_centred(self) -> None:
        from app.preprocessing.transforms import pad_to_square
        img = np.full((20, 40, 3), 200, dtype=np.uint8)
        out = pad_to_square(img, fill_value=0)
        # Centre rows should be non-zero; far-edge rows should be zero
        assert out[0, 0, 0] == 0
        assert out[10, 10, 0] == 200


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — Colour conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsColour:
    def test_bgr_to_rgb_swaps_channels(self) -> None:
        from app.preprocessing.transforms import bgr_to_rgb
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10  # B channel
        bgr[:, :, 2] = 30  # R channel
        rgb = bgr_to_rgb(bgr)
        assert rgb[0, 0, 0] == 30  # R is now first
        assert rgb[0, 0, 2] == 10  # B is now last

    def test_rgb_to_bgr_inverts_bgr_to_rgb(self) -> None:
        from app.preprocessing.transforms import bgr_to_rgb, rgb_to_bgr
        bgr = _make_bgr()
        assert np.array_equal(bgr, rgb_to_bgr(bgr_to_rgb(bgr)))

    def test_colour_conversion_preserves_shape(self) -> None:
        from app.preprocessing.transforms import bgr_to_rgb
        img = _make_bgr(32, 48)
        assert bgr_to_rgb(img).shape == (32, 48, 3)


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsNormalisation:
    def test_normalize_returns_float32(self) -> None:
        from app.preprocessing.transforms import normalize_image
        assert normalize_image(_make_rgb()).dtype == np.float32

    def test_normalize_shape_unchanged(self) -> None:
        from app.preprocessing.transforms import normalize_image
        img = _make_rgb(48, 48)
        assert normalize_image(img).shape == (48, 48, 3)

    def test_normalize_zero_image_is_negative(self) -> None:
        from app.preprocessing.transforms import normalize_image
        # black image → (0 - mean) / std → negative
        out = normalize_image(np.zeros((8, 8, 3), dtype=np.uint8))
        assert np.all(out < 0)

    def test_normalize_all_values_are_finite(self) -> None:
        from app.preprocessing.transforms import normalize_image
        assert np.all(np.isfinite(normalize_image(_make_rgb())))

    def test_rescale_only_range(self) -> None:
        from app.preprocessing.transforms import rescale_only
        out = rescale_only(_make_rgb())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_rescale_only_dtype(self) -> None:
        from app.preprocessing.transforms import rescale_only
        assert rescale_only(_make_rgb()).dtype == np.float32

    def test_denormalize_round_trip(self) -> None:
        from app.preprocessing.transforms import normalize_image, denormalize_image
        rgb = _make_rgb(32, 32)
        normalised  = normalize_image(rgb)
        recovered   = denormalize_image(normalised)
        # Allow ±2 LSB rounding from float32 ↔ uint8 conversion
        assert np.allclose(rgb.astype(np.float32), recovered.astype(np.float32), atol=2)

    def test_denormalize_returns_uint8(self) -> None:
        from app.preprocessing.transforms import normalize_image, denormalize_image
        assert denormalize_image(normalize_image(_make_rgb())).dtype == np.uint8

    def test_normalize_custom_mean_std(self) -> None:
        from app.preprocessing.transforms import normalize_image
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))
        # 128/255 ≈ 0.502 → 0.502 - 0.5 = 0.002 (very close to 0)
        assert np.allclose(out, 0.0, atol=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# transforms.py — apply_spatial_pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformsPipeline:
    def test_output_shape_matches_config(self) -> None:
        from app.preprocessing.transforms import apply_spatial_pipeline
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(image_size=64)
        out = apply_spatial_pipeline(_make_bgr(120, 90), cfg)
        assert out.shape == (64, 64, 3)

    def test_output_is_uint8(self) -> None:
        from app.preprocessing.transforms import apply_spatial_pipeline
        from app.preprocessing.config import PreprocessConfig
        out = apply_spatial_pipeline(_make_bgr(), PreprocessConfig(image_size=32))
        assert out.dtype == np.uint8

    def test_denoise_disabled(self) -> None:
        from app.preprocessing.transforms import apply_spatial_pipeline
        from app.preprocessing.config import PreprocessConfig
        # Should not raise with denoise disabled
        out = apply_spatial_pipeline(_make_bgr(), PreprocessConfig(apply_denoise=False))
        assert out.shape[0] == 224

    def test_clahe_disabled(self) -> None:
        from app.preprocessing.transforms import apply_spatial_pipeline
        from app.preprocessing.config import PreprocessConfig
        out = apply_spatial_pipeline(_make_bgr(), PreprocessConfig(apply_clahe=False))
        assert out.shape[0] == 224

    def test_all_disabled_is_just_resize(self) -> None:
        from app.preprocessing.transforms import apply_spatial_pipeline
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(image_size=48, apply_denoise=False, apply_clahe=False)
        out = apply_spatial_pipeline(_make_bgr(100, 100), cfg)
        assert out.shape == (48, 48, 3)


# ─────────────────────────────────────────────────────────────────────────────
# quality.py
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityChecks:
    """Each test isolates one quality dimension so failures are unambiguous."""

    # ── QualityCheck / ImageQualityReport dataclasses ────────────────────────

    def test_quality_check_to_dict_keys(self) -> None:
        from app.preprocessing.quality import QualityCheck
        d = QualityCheck(name="test", passed=True, value=1.0, message="ok").to_dict()
        assert {"name", "passed", "value", "threshold", "message"} <= d.keys()

    def test_report_to_dict_keys(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        r = validate_image_quality(_varied_png_bytes())
        d = r.to_dict()
        assert {"is_valid", "image_width", "image_height",
                "file_size_bytes", "checks", "warnings", "errors"} <= d.keys()

    def test_report_summary_returns_string(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        r = validate_image_quality(_varied_png_bytes())
        assert isinstance(r.summary(), str)
        assert len(r.summary()) > 0

    # ── Decode failure ────────────────────────────────────────────────────────

    def test_invalid_bytes_is_invalid(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        r = validate_image_quality(b"not an image")
        assert r.is_valid is False
        assert any("Decode" in e or "decode" in e for e in r.errors)

    # ── File-size check ───────────────────────────────────────────────────────

    def test_oversized_file_fails(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(max_file_size_bytes=10)  # 10 bytes — tiny
        r = validate_image_quality(_varied_png_bytes(), cfg)
        file_check = next(c for c in r.checks if c.name == "file_size")
        assert file_check.passed is False

    def test_no_file_size_limit(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(max_file_size_bytes=0)
        r = validate_image_quality(_varied_png_bytes(), cfg)
        file_check = next(c for c in r.checks if c.name == "file_size")
        assert file_check.passed is True

    # ── Dimension check ───────────────────────────────────────────────────────

    def test_image_below_min_size_fails(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        tiny = _png_bytes(h=4, w=4)
        cfg = PreprocessConfig(min_width=32, min_height=32)
        r = validate_image_quality(tiny, cfg)
        dim_check = next(c for c in r.checks if c.name == "dimensions")
        assert dim_check.passed is False
        assert r.is_valid is False

    def test_image_at_min_size_passes(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(min_width=32, min_height=32,
                               min_laplacian_variance=0.0)
        r = validate_image_quality(_varied_png_bytes(32, 32), cfg)
        dim_check = next(c for c in r.checks if c.name == "dimensions")
        assert dim_check.passed is True

    # ── Channel check ─────────────────────────────────────────────────────────

    def test_normal_colour_image_passes_channel_check(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        r = validate_image_quality(_varied_png_bytes())
        ch = next(c for c in r.checks if c.name == "channels")
        assert ch.passed is True

    # ── Mean intensity check ──────────────────────────────────────────────────

    def test_near_black_image_fails_intensity(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        black = _png_bytes(value=0)
        cfg = PreprocessConfig(min_mean_intensity=5.0)
        r = validate_image_quality(black, cfg)
        mi = next(c for c in r.checks if c.name == "mean_intensity")
        assert mi.passed is False
        assert "too dark" in mi.message

    def test_near_white_image_fails_intensity(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        white = _png_bytes(value=255)
        cfg = PreprocessConfig(max_mean_intensity=250.0)
        r = validate_image_quality(white, cfg)
        mi = next(c for c in r.checks if c.name == "mean_intensity")
        assert mi.passed is False
        assert "too bright" in mi.message

    def test_normal_image_passes_intensity(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        r = validate_image_quality(_varied_png_bytes())
        mi = next(c for c in r.checks if c.name == "mean_intensity")
        assert mi.passed is True

    # ── Sharpness check (blur → warning, not error) ───────────────────────────

    def test_blurry_image_produces_warning_not_error(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(min_laplacian_variance=1e9)  # impossibly high
        r = validate_image_quality(_varied_png_bytes(), cfg)
        sharpness = next(c for c in r.checks if c.name == "sharpness")
        assert sharpness.passed is False
        # Sharpness failure goes to warnings, not errors — image still usable
        assert any("laplacian" in w.lower() or "blur" in w.lower()
                   for w in r.warnings)
        # is_valid can still be True (no errors from sharpness alone)

    def test_sharp_image_passes_sharpness(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(min_laplacian_variance=0.0)
        r = validate_image_quality(_varied_png_bytes(), cfg)
        sharpness = next(c for c in r.checks if c.name == "sharpness")
        assert sharpness.passed is True

    # ── Pixel variance check ──────────────────────────────────────────────────

    def test_uniform_image_fails_variance(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        flat = _png_bytes(value=128)  # completely uniform
        r = validate_image_quality(flat)
        var = next(c for c in r.checks if c.name == "pixel_variance")
        assert var.passed is False

    def test_varied_image_passes_variance(self) -> None:
        from app.preprocessing.quality import validate_image_quality
        r = validate_image_quality(_varied_png_bytes())
        var = next(c for c in r.checks if c.name == "pixel_variance")
        assert var.passed is True

    # ── Batch validation ──────────────────────────────────────────────────────

    def test_batch_returns_dict_keyed_by_path(self, tmp_path: Path) -> None:
        from app.preprocessing.quality import validate_batch_quality
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        p1.write_bytes(_varied_png_bytes())
        p2.write_bytes(_varied_png_bytes())
        results = validate_batch_quality([p1, p2])
        assert str(p1) in results and str(p2) in results

    def test_batch_fail_fast_raises_on_first_failure(self, tmp_path: Path) -> None:
        from app.preprocessing.quality import validate_batch_quality
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a png")
        with pytest.raises(ValueError):
            validate_batch_quality([bad], fail_fast=True)


# ─────────────────────────────────────────────────────────────────────────────
# augmentation.py
# ─────────────────────────────────────────────────────────────────────────────

class TestAugmentationConfig:
    def test_defaults_are_mri_safe(self) -> None:
        from app.preprocessing.augmentation import AugmentationConfig
        cfg = AugmentationConfig()
        assert cfg.vertical_flip is False   # brain orientation must be preserved
        assert cfg.horizontal_flip is True  # left-right symmetry is valid

    def test_to_dict_returns_dict(self) -> None:
        from app.preprocessing.augmentation import AugmentationConfig
        d = AugmentationConfig().to_dict()
        assert isinstance(d, dict)
        assert "rotation_range" in d

    def test_brightness_range_in_to_dict(self) -> None:
        from app.preprocessing.augmentation import AugmentationConfig
        d = AugmentationConfig(brightness_range=(0.8, 1.2)).to_dict()
        assert d["brightness_range"] == [0.8, 1.2]

    def test_brightness_none_in_to_dict(self) -> None:
        from app.preprocessing.augmentation import AugmentationConfig
        d = AugmentationConfig(brightness_range=None).to_dict()
        assert d["brightness_range"] is None

    def test_default_aug_config_singleton(self) -> None:
        from app.preprocessing.augmentation import DEFAULT_AUG_CONFIG, AugmentationConfig
        assert isinstance(DEFAULT_AUG_CONFIG, AugmentationConfig)


class TestAugmentationGenerators:
    def test_build_train_datagen_returns_idg(self) -> None:
        from torchvision import transforms as T
        from app.preprocessing.augmentation import build_train_datagen
        gen = build_train_datagen()
        assert isinstance(gen, T.Compose)

    def test_build_eval_datagen_returns_idg(self) -> None:
        from torchvision import transforms as T
        from app.preprocessing.augmentation import build_eval_datagen
        gen = build_eval_datagen()
        assert isinstance(gen, T.Compose)

    def test_train_datagen_has_augmentation(self) -> None:
        from torchvision import transforms as T
        from app.preprocessing.augmentation import build_train_datagen
        gen = build_train_datagen()
        # Training transform has more steps than eval (RandomAffine / flips)
        assert len(gen.transforms) > 2   # Resize + ToTensor + Normalize + at least one augment

    def test_eval_datagen_no_augmentation(self) -> None:
        from torchvision import transforms as T
        from app.preprocessing.augmentation import build_eval_datagen
        gen = build_eval_datagen()
        # Eval transform is minimal: Resize + ToTensor + Normalize only (3 steps)
        assert len(gen.transforms) == 3
        assert isinstance(gen.transforms[0], T.Resize)

    def test_build_data_generators_from_split(self, tmp_path: Path) -> None:
        from app.preprocessing.augmentation import build_data_generators_from_split
        CLASSES = ["a", "b"]
        for split in ("train_dir", "val_dir"):
            for cls in CLASSES:
                d = tmp_path / split / cls
                d.mkdir(parents=True)
                (d / "img.png").write_bytes(_varied_png_bytes(32, 32))
        train_loader, val_loader = build_data_generators_from_split(
            tmp_path / "train_dir",
            tmp_path / "val_dir",
            image_size=32,
            batch_size=2,
        )
        # PyTorch DataLoader — use len(dataset) not .samples
        assert len(train_loader.dataset) == 2
        assert len(val_loader.dataset)   == 2

    def test_build_data_generators_missing_train_raises(self, tmp_path: Path) -> None:
        from app.preprocessing.augmentation import build_data_generators_from_split
        with pytest.raises(FileNotFoundError, match="Training directory"):
            build_data_generators_from_split(
                tmp_path / "nonexistent_train",
                tmp_path / "nonexistent_val",
            )


class TestAugmentationApply:
    def test_returns_list_of_correct_length(self) -> None:
        from app.preprocessing.augmentation import apply_augmentation
        rgb = _make_rgb(64, 64)
        variants = apply_augmentation(rgb, n_samples=3)
        assert len(variants) == 3

    def test_single_sample_default(self) -> None:
        from app.preprocessing.augmentation import apply_augmentation
        out = apply_augmentation(_make_rgb())
        assert len(out) == 1

    def test_output_is_uint8_rgb(self) -> None:
        from app.preprocessing.augmentation import apply_augmentation
        out = apply_augmentation(_make_rgb(64, 64))[0]
        assert out.dtype == np.uint8
        assert out.ndim  == 3
        assert out.shape[2] == 3

    def test_output_shape_matches_input(self) -> None:
        from app.preprocessing.augmentation import apply_augmentation
        rgb = _make_rgb(48, 48)
        # apply_augmentation resizes to image_size; pass matching size to preserve shape
        out = apply_augmentation(rgb, n_samples=2, image_size=48)[0]
        assert out.shape == (48, 48, 3)

    def test_same_seed_same_output(self) -> None:
        from app.preprocessing.augmentation import apply_augmentation
        rgb = _make_rgb(32, 32, seed=1)
        a = apply_augmentation(rgb, seed=99, n_samples=1)[0]
        b = apply_augmentation(rgb, seed=99, n_samples=1)[0]
        assert np.array_equal(a, b)

    def test_vertical_flip_not_applied(self) -> None:
        """Augmentation config has vertical_flip=False — validate it's honoured."""
        from app.preprocessing.augmentation import AugmentationConfig, apply_augmentation
        cfg = AugmentationConfig(
            rotation_range=0, width_shift_range=0, height_shift_range=0,
            shear_range=0, zoom_range=0, horizontal_flip=False,
            vertical_flip=False, brightness_range=None,
        )
        rgb = _make_rgb(32, 32, seed=5)
        # apply_augmentation resizes to image_size; pass matching size to preserve shape
        out = apply_augmentation(rgb, aug_cfg=cfg, seed=0, n_samples=1, image_size=32)[0]
        # With all transforms disabled the output should be very close to input
        assert np.allclose(rgb.astype(float), out.astype(float), atol=2)


# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py — context-aware pipeline functions
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineInference:
    def test_returns_float32(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        out = preprocess_for_inference(_varied_png_bytes())
        assert out.dtype == np.float32

    def test_returns_batch_tensor_by_default(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        out = preprocess_for_inference(_varied_png_bytes())
        assert out.ndim == 4
        assert out.shape[0] == 1

    def test_no_batch_dim_when_expand_false(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        out = preprocess_for_inference(_varied_png_bytes(), expand_dims=False)
        assert out.ndim == 3

    def test_spatial_dimensions_match_config(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(image_size=48)
        out = preprocess_for_inference(_varied_png_bytes(), cfg=cfg)
        assert out.shape == (1, 48, 48, 3)

    def test_values_are_finite(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        out = preprocess_for_inference(_varied_png_bytes())
        assert np.all(np.isfinite(out))

    def test_normalise_false_stays_in_0_1(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(normalise=False)
        out = preprocess_for_inference(_varied_png_bytes(), cfg=cfg)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_invalid_bytes_raises_value_error(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        with pytest.raises(ValueError):
            preprocess_for_inference(b"garbage")

    def test_from_file_path(self, tmp_path: Path) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        p = tmp_path / "scan.png"
        p.write_bytes(_varied_png_bytes())
        out = preprocess_for_inference(p)
        assert out.shape[0] == 1

    def test_denoise_disabled(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(apply_denoise=False, image_size=32)
        out = preprocess_for_inference(_varied_png_bytes(), cfg=cfg)
        assert out.shape == (1, 32, 32, 3)

    def test_clahe_disabled(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_inference
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(apply_clahe=False, image_size=32)
        out = preprocess_for_inference(_varied_png_bytes(), cfg=cfg)
        assert out.shape == (1, 32, 32, 3)


class TestPipelineGradcam:
    def test_returns_two_element_tuple(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_gradcam
        result = preprocess_for_gradcam(_varied_png_bytes())
        assert isinstance(result, tuple) and len(result) == 2

    def test_tensor_is_float32_batch(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_gradcam
        tensor, _ = preprocess_for_gradcam(_varied_png_bytes())
        assert tensor.dtype == np.float32
        assert tensor.shape[0] == 1

    def test_display_is_uint8_rgb(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_gradcam
        _, display = preprocess_for_gradcam(_varied_png_bytes())
        assert display.dtype == np.uint8
        assert display.ndim  == 3
        assert display.shape[2] == 3

    def test_tensor_and_display_same_spatial_size(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_gradcam
        from app.preprocessing.config import PreprocessConfig
        cfg = PreprocessConfig(image_size=48)
        tensor, display = preprocess_for_gradcam(_varied_png_bytes(), cfg=cfg)
        assert tensor.shape[1:3] == (48, 48)
        assert display.shape[:2] == (48, 48)

    def test_display_not_normalised(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_gradcam
        _, display = preprocess_for_gradcam(_varied_png_bytes())
        # uint8 display image must be in [0, 255]
        assert display.max() <= 255
        assert display.min() >= 0

    def test_tensor_values_finite(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_gradcam
        tensor, _ = preprocess_for_gradcam(_varied_png_bytes())
        assert np.all(np.isfinite(tensor))


class TestPipelinePreview:
    def test_returns_uint8(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_preview
        out = preprocess_for_preview(_varied_png_bytes())
        assert out.dtype == np.uint8

    def test_returns_3d_rgb(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_preview
        out = preprocess_for_preview(_varied_png_bytes())
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_spatial_size_from_config(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_preview
        from app.preprocessing.config import PreprocessConfig
        out = preprocess_for_preview(_varied_png_bytes(), cfg=PreprocessConfig(image_size=56))
        assert out.shape == (56, 56, 3)

    def test_values_in_uint8_range(self) -> None:
        from app.preprocessing.preprocess import preprocess_for_preview
        out = preprocess_for_preview(_varied_png_bytes())
        assert out.min() >= 0 and out.max() <= 255


# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py — backward-compatible shims
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineShims:
    """Ensure old callers (predict.py, gradcam.py, train.py) still work."""

    def test_preprocess_image_returns_float32_batch(self) -> None:
        from app.preprocessing.preprocess import preprocess_image
        out = preprocess_image(_varied_png_bytes(), target_size=32)
        assert out.dtype == np.float32 and out.shape == (1, 32, 32, 3)

    def test_preprocess_image_expand_dims_false(self) -> None:
        from app.preprocessing.preprocess import preprocess_image
        out = preprocess_image(_varied_png_bytes(), target_size=32, expand_dims=False)
        assert out.ndim == 3

    def test_preprocess_image_apply_denoise_false(self) -> None:
        from app.preprocessing.preprocess import preprocess_image
        out = preprocess_image(_varied_png_bytes(), target_size=32, apply_denoise=False)
        assert out.shape == (1, 32, 32, 3)

    def test_preprocess_image_apply_contrast_false(self) -> None:
        from app.preprocessing.preprocess import preprocess_image
        out = preprocess_image(_varied_png_bytes(), target_size=32, apply_contrast=False)
        assert out.shape == (1, 32, 32, 3)

    def test_preprocess_image_for_gradcam_tuple(self) -> None:
        from app.preprocessing.preprocess import preprocess_image_for_gradcam
        tensor, display = preprocess_image_for_gradcam(_varied_png_bytes(), target_size=32)
        assert tensor.shape == (1, 32, 32, 3)
        assert display.shape == (32, 32, 3)
        assert display.dtype == np.uint8

    def test_low_level_apply_median_filter_shim(self) -> None:
        from app.preprocessing.preprocess import apply_median_filter
        out = apply_median_filter(_make_bgr())
        assert out.shape == (64, 64, 3)

    def test_low_level_apply_clahe_shim(self) -> None:
        from app.preprocessing.preprocess import apply_clahe
        out = apply_clahe(_make_bgr())
        assert out.shape == (64, 64, 3)

    def test_low_level_resize_image_shim(self) -> None:
        from app.preprocessing.preprocess import resize_image
        out = resize_image(_make_bgr(), 48)
        assert out.shape == (48, 48, 3)

    def test_low_level_normalize_image_shim(self) -> None:
        from app.preprocessing.preprocess import normalize_image
        out = normalize_image(_make_rgb())
        assert out.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# __init__.py — package-level re-exports
# ─────────────────────────────────────────────────────────────────────────────

class TestPackageInit:
    def test_all_pipeline_symbols_importable(self) -> None:
        from app.preprocessing import (  # noqa: F401
            preprocess_for_inference, preprocess_for_gradcam,
            preprocess_for_preview, build_generators, build_test_generator,
        )

    def test_all_config_symbols_importable(self) -> None:
        from app.preprocessing import (  # noqa: F401
            PreprocessConfig, DEFAULT_CONFIG, IMAGENET_MEAN, IMAGENET_STD,
        )

    def test_all_quality_symbols_importable(self) -> None:
        from app.preprocessing import (  # noqa: F401
            validate_image_quality, validate_batch_quality,
            ImageQualityReport, QualityCheck,
        )

    def test_all_augmentation_symbols_importable(self) -> None:
        from app.preprocessing import (  # noqa: F401
            AugmentationConfig, DEFAULT_AUG_CONFIG,
            build_train_datagen, build_eval_datagen, apply_augmentation,
        )

    def test_all_transform_symbols_importable(self) -> None:
        from app.preprocessing import (  # noqa: F401
            load_image_bgr, encode_image_png, encode_image_base64,
            apply_median_filter, apply_gaussian_blur, apply_clahe,
            resize_image, pad_to_square, bgr_to_rgb, rgb_to_bgr,
            normalize_image, rescale_only, denormalize_image,
            apply_spatial_pipeline,
        )

    def test_shim_symbols_importable(self) -> None:
        from app.preprocessing import (  # noqa: F401
            preprocess_image, preprocess_image_for_gradcam,
            build_data_generators,
        )

    def test_all_exports_in_all_list(self) -> None:
        import app.preprocessing as pkg
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists '{name}' but it is not importable"


# ─────────────────────────────────────────────────────────────────────────────
# routes.py — POST /preprocess/quality-check
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIQualityCheck:
    URL = "/api/v1/preprocess/quality-check"

    def test_valid_image_returns_200(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200

    def test_response_has_required_keys(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        body = resp.json()
        assert "success" in body
        assert "data"    in body
        assert "message" in body

    def test_data_contains_quality_fields(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        d = resp.json()["data"]
        for key in ("is_valid", "checks", "image_width",
                    "image_height", "file_size_bytes"):
            assert key in d, f"Missing key: {key}"

    def test_checks_list_has_all_six_checks(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        names = {c["name"] for c in resp.json()["data"]["checks"]}
        assert {"file_size", "dimensions", "channels",
                "mean_intensity", "sharpness", "pixel_variance"} <= names

    def test_unsupported_content_type_returns_400(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.gif", b"GIF89a", "image/gif")},
        )
        assert resp.status_code == 400

    def test_empty_file_returns_400(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_jpeg_content_type_accepted(self) -> None:
        ok, buf = cv2.imencode(".jpg", _make_bgr())
        assert ok
        resp = client.post(
            self.URL,
            files={"image": ("scan.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_uniform_image_success_false(self) -> None:
        """Uniform image fails pixel_variance — success must be False."""
        resp = client.post(
            self.URL,
            files={"image": ("flat.png", _png_bytes(value=128), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_near_black_image_fails(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("black.png", _png_bytes(value=0), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_near_white_image_fails(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("white.png", _png_bytes(value=255), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_invalid_bytes_returns_200_with_failure(self) -> None:
        """Corrupt upload must return 200 with is_valid=False, not a 5xx."""
        resp = client.post(
            self.URL,
            files={"image": ("corrupt.png", b"not an image", "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["is_valid"] is False

    def test_image_size_form_field_accepted(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"image_size": "64"},
        )
        assert resp.status_code == 200

    def test_apply_denoise_form_field_accepted(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"apply_denoise": "false"},
        )
        assert resp.status_code == 200

    def test_dimensions_in_response(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(64, 64), "image/png")},
        )
        d = resp.json()["data"]
        assert d["image_width"]  == 64
        assert d["image_height"] == 64

    def test_file_size_bytes_in_response(self) -> None:
        png = _varied_png_bytes()
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", png, "image/png")},
        )
        assert resp.json()["data"]["file_size_bytes"] == len(png)


# ─────────────────────────────────────────────────────────────────────────────
# routes.py — POST /preprocess/preview
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIPreview:
    URL = "/api/v1/preprocess/preview"

    def test_valid_image_returns_200(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200

    def test_response_has_required_keys(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        body = resp.json()
        assert "success" in body
        assert "data"    in body
        assert "message" in body

    def test_data_contains_required_keys(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        d = resp.json()["data"]
        for key in ("quality", "config", "preprocessed_b64", "augmented_b64"):
            assert key in d, f"Missing key: {key}"

    def test_preprocessed_b64_is_valid_base64_png(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        b64 = resp.json()["data"]["preprocessed_b64"]
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_augmented_b64_list_length(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"n_augmented": "3"},
        )
        augmented = resp.json()["data"]["augmented_b64"]
        assert len(augmented) == 3

    def test_augmented_each_entry_is_valid_base64(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"n_augmented": "2"},
        )
        for b64 in resp.json()["data"]["augmented_b64"]:
            base64.b64decode(b64)  # must not raise

    def test_include_augmented_false_returns_empty_list(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"include_augmented": "false"},
        )
        assert resp.json()["data"]["augmented_b64"] == []

    def test_config_in_response_reflects_image_size(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"image_size": "64"},
        )
        cfg = resp.json()["data"]["config"]
        assert cfg["image_size"] == 64

    def test_quality_section_in_response(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        quality = resp.json()["data"]["quality"]
        assert "is_valid"  in quality
        assert "checks"    in quality

    def test_unsupported_content_type_returns_400(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.gif", b"GIF89a", "image/gif")},
        )
        assert resp.status_code == 400

    def test_empty_file_returns_400(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_invalid_image_bytes_returns_400(self) -> None:
        """Corrupt bytes fail quality check → 400, not 200."""
        resp = client.post(
            self.URL,
            files={"image": ("corrupt.png", b"not an image", "image/png")},
        )
        assert resp.status_code == 400

    def test_uniform_image_returns_400(self) -> None:
        """Uniform image fails pixel_variance → quality invalid → 400."""
        resp = client.post(
            self.URL,
            files={"image": ("flat.png", _png_bytes(value=128), "image/png")},
        )
        assert resp.status_code == 400

    def test_jpeg_accepted(self) -> None:
        ok, buf = cv2.imencode(".jpg", _make_bgr())
        assert ok
        resp = client.post(
            self.URL,
            files={"image": ("scan.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_success_true_in_response(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        assert resp.json()["success"] is True

    def test_message_contains_image_size(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        assert "224" in resp.json()["message"]

    def test_apply_clahe_false_accepted(self) -> None:
        resp = client.post(
            self.URL,
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"apply_clahe": "false"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["config"]["apply_clahe"] is False
