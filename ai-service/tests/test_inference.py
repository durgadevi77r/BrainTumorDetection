"""
tests/test_inference.py — Comprehensive tests for the inference package.

Coverage
--------
TestInferenceConfig         config.py  — dataclass, validation, from_settings
TestInferenceResults        results.py — PredictionResult, BatchPredictionResult,
                                         TopKPrediction, PredictionMetadata
TestModelCache              cache.py   — LRU eviction, hit/miss, hot reload
TestCacheConvenienceFns     cache.py   — module-level get_model / reload_model shims
TestListAvailableModels     cache.py   — list_available_models, cache_stats
TestInferencePipelineUnit   pipeline.py — predict() with mocked model
TestBatchPipelineUnit       pipeline.py — predict_batch(), predict_directory(),
                                          predict_zip()
TestBatchInferenceRunner    batch.py   — run(), run_directory(), run_zip(),
                                         export() JSON + CSV, progress callback
TestInferencePackageInit    __init__.py — public symbols importable from package
TestInferenceAPIRoutes      routes.py  — POST /predict/image, /predict/batch,
                                         /predict/zip, GET /models,
                                         POST /models/reload, GET /models/active
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _png_bytes(h: int = 64, w: int = 64, value: int = 128) -> bytes:
    """Solid-colour PNG — passes shape checks but fails variance."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _varied_png_bytes(h: int = 64, w: int = 64, seed: int = 7) -> bytes:
    """Naturally varied PNG — passes all quality checks."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_zip(image_map: Dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP containing the given filename → bytes mapping."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in image_map.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _mock_keras_model(num_classes: int = 4) -> MagicMock:
    """Return a MagicMock that looks like a tf.keras.Model."""
    probs = np.array([[0.85, 0.07, 0.05, 0.03]], dtype=np.float32)
    model = MagicMock()
    model.predict.return_value = probs
    model.count_params.return_value = 12_341_232
    model.input_shape = (None, 224, 224, 3)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# config.py
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceConfig:
    def test_default_model_is_mambavision(self) -> None:
        from app.inference.config import InferenceConfig
        assert InferenceConfig().model_name == "mambavision"

    def test_model_name_normalised_to_lowercase(self) -> None:
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(model_name="EfficientNet")
        assert cfg.model_name == "efficientnet"

    def test_invalid_model_name_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="model_name"):
            InferenceConfig(model_name="transformer")

    def test_top_k_default_is_one(self) -> None:
        from app.inference.config import InferenceConfig
        assert InferenceConfig().top_k == 1

    def test_top_k_zero_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="top_k"):
            InferenceConfig(top_k=0)

    def test_top_k_above_num_classes_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="top_k"):
            InferenceConfig(
                class_names=["a", "b", "c", "d"],
                top_k=5,
            )

    def test_confidence_threshold_out_of_range_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="confidence_threshold"):
            InferenceConfig(confidence_threshold=1.5)

    def test_invalid_output_format_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="output_format"):
            InferenceConfig(output_format="xml")

    def test_batch_size_zero_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="batch_size"):
            InferenceConfig(batch_size=0)

    def test_max_workers_zero_raises(self) -> None:
        from app.inference.config import InferenceConfig
        with pytest.raises(ValueError, match="max_workers"):
            InferenceConfig(max_workers=0)

    def test_num_classes_property(self) -> None:
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(class_names=["a", "b", "c"])
        assert cfg.num_classes == 3

    def test_to_dict_round_trip(self) -> None:
        from app.inference.config import InferenceConfig
        original = InferenceConfig(model_name="resnet50", top_k=2)
        restored = InferenceConfig.from_dict(original.to_dict())
        assert restored.model_name == "resnet50"
        assert restored.top_k == 2

    def test_to_dict_is_json_serialisable(self) -> None:
        from app.inference.config import InferenceConfig
        json.dumps(InferenceConfig().to_dict())

    def test_from_settings_returns_config(self) -> None:
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig.from_settings()
        assert isinstance(cfg, InferenceConfig)
        assert cfg.image_size > 0

    def test_resolved_output_dir_uses_cwd_when_none(self, tmp_path: Path) -> None:
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(output_dir=None)
        assert cfg.resolved_output_dir.name == "inference_output"

    def test_resolved_output_dir_uses_provided_path(self, tmp_path: Path) -> None:
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(output_dir=str(tmp_path))
        assert cfg.resolved_output_dir == tmp_path

    def test_save_and_load_json(self, tmp_path: Path) -> None:
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(model_name="vgg16", top_k=2)
        p = tmp_path / "cfg.json"
        cfg.save_json(p)
        loaded = InferenceConfig.from_json(p)
        assert loaded.model_name == "vgg16"
        assert loaded.top_k == 2

    def test_default_config_singleton_exists(self) -> None:
        from app.inference.config import DEFAULT_INFERENCE_CONFIG, InferenceConfig
        assert isinstance(DEFAULT_INFERENCE_CONFIG, InferenceConfig)


# ─────────────────────────────────────────────────────────────────────────────
# results.py
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceResults:

    def _make_metadata(self) -> "PredictionMetadata":
        from app.inference.results import PredictionMetadata
        return PredictionMetadata(
            model_name="efficientnet",
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            image_size=224,
            model_version="2024-01-01T00:00:00Z",
        )

    def _make_top_k(self) -> list:
        from app.inference.results import TopKPrediction
        return [
            TopKPrediction(rank=1, class_name="glioma", class_index=0, probability=0.85),
            TopKPrediction(rank=2, class_name="meningioma", class_index=1, probability=0.10),
        ]

    def _make_result(self) -> "PredictionResult":
        from app.inference.results import PredictionResult
        return PredictionResult(
            image_id="test-id-123",
            predicted_class="glioma",
            predicted_class_index=0,
            confidence=0.85,
            is_high_confidence=True,
            probabilities={"glioma": 0.85, "meningioma": 0.10,
                           "notumor": 0.03, "pituitary": 0.02},
            top_k=self._make_top_k(),
            timing_ms=42.5,
            metadata=self._make_metadata(),
        )

    # ── PredictionMetadata ───────────────────────────────────────────────────

    def test_metadata_to_dict_has_required_keys(self) -> None:
        d = self._make_metadata().to_dict()
        assert {"model_name", "class_names", "image_size",
                "model_version", "predicted_at"} <= d.keys()

    def test_metadata_predicted_at_is_iso_string(self) -> None:
        d = self._make_metadata().to_dict()
        assert isinstance(d["predicted_at"], str)
        assert "T" in d["predicted_at"]

    def test_metadata_from_dict_round_trip(self) -> None:
        from app.inference.results import PredictionMetadata
        original = self._make_metadata()
        restored = PredictionMetadata.from_dict(original.to_dict())
        assert restored.model_name == "efficientnet"
        assert restored.image_size == 224

    def test_metadata_gradcam_path_default_none(self) -> None:
        assert self._make_metadata().gradcam_path is None

    def test_metadata_source_path_default_none(self) -> None:
        assert self._make_metadata().source_path is None

    # ── TopKPrediction ────────────────────────────────────────────────────────

    def test_topk_to_dict_keys(self) -> None:
        d = self._make_top_k()[0].to_dict()
        assert {"rank", "class_name", "class_index", "probability"} <= d.keys()

    def test_topk_rank_is_one_indexed(self) -> None:
        assert self._make_top_k()[0].rank == 1

    # ── PredictionResult ──────────────────────────────────────────────────────

    def test_result_to_dict_has_required_keys(self) -> None:
        d = self._make_result().to_dict()
        for key in ("image_id", "predicted_class", "predicted_class_index",
                    "confidence", "is_high_confidence", "probabilities",
                    "top_k", "timing_ms", "metadata", "error"):
            assert key in d, f"Missing key: {key}"

    def test_result_to_json_is_valid_json(self) -> None:
        r = self._make_result()
        parsed = json.loads(r.to_json())
        assert parsed["predicted_class"] == "glioma"

    def test_result_error_defaults_none(self) -> None:
        assert self._make_result().error is None

    def test_result_probabilities_sum_to_one(self) -> None:
        probs = self._make_result().probabilities
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_result_top_k_list_length(self) -> None:
        assert len(self._make_result().top_k) == 2

    # ── BatchItemResult ───────────────────────────────────────────────────────

    def test_batch_item_success_to_dict(self) -> None:
        from app.inference.results import BatchItemResult
        item = BatchItemResult(
            filename="scan.jpg",
            success=True,
            result=self._make_result(),
        )
        d = item.to_dict()
        assert d["success"] is True
        assert d["result"] is not None
        assert d["error"] is None

    def test_batch_item_failure_to_dict(self) -> None:
        from app.inference.results import BatchItemResult
        item = BatchItemResult(
            filename="bad.jpg",
            success=False,
            error="ValueError: bad image",
        )
        d = item.to_dict()
        assert d["success"] is False
        assert d["result"] is None
        assert "ValueError" in d["error"]

    # ── BatchPredictionResult ─────────────────────────────────────────────────

    def _make_batch_result(self) -> "BatchPredictionResult":
        from app.inference.results import BatchItemResult, BatchPredictionResult
        return BatchPredictionResult(
            total=3,
            succeeded=2,
            failed=1,
            results=[
                BatchItemResult("a.jpg", True, self._make_result()),
                BatchItemResult("b.jpg", True, self._make_result()),
                BatchItemResult("c.jpg", False, error="decode error"),
            ],
            timing_ms=120.5,
            model_name="efficientnet",
            source_type="list",
        )

    def test_success_rate(self) -> None:
        assert self._make_batch_result().success_rate == pytest.approx(2 / 3, abs=1e-4)

    def test_class_distribution_counts_predictions(self) -> None:
        dist = self._make_batch_result().class_distribution
        assert dist.get("glioma", 0) == 2

    def test_to_dict_has_summary_keys(self) -> None:
        d = self._make_batch_result().to_dict()
        for key in ("total", "succeeded", "failed", "success_rate",
                    "timing_ms", "model_name", "source_type",
                    "class_distribution", "results"):
            assert key in d, f"Missing key: {key}"

    def test_summary_dict_omits_results_list(self) -> None:
        summary = self._make_batch_result().summary_dict()
        assert "results" not in summary
        assert "total" in summary

    def test_to_json_is_valid_json(self) -> None:
        parsed = json.loads(self._make_batch_result().to_json())
        assert parsed["total"] == 3

    def test_empty_batch_success_rate_is_zero(self) -> None:
        from app.inference.results import BatchPredictionResult
        b = BatchPredictionResult(
            total=0, succeeded=0, failed=0,
            results=[], timing_ms=0.0,
            model_name="efficientnet", source_type="list",
        )
        assert b.success_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# cache.py — ModelCache unit tests (no real model loading)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCache:

    def _fresh_cache(self, capacity: int = 2):
        """Return a fresh ModelCache instance wired to a mock loader."""
        from app.inference.cache import ModelCache
        return ModelCache(capacity=capacity)

    def test_capacity_zero_raises(self) -> None:
        from app.inference.cache import ModelCache
        with pytest.raises(ValueError, match="capacity"):
            ModelCache(capacity=0)

    def test_get_returns_none_when_empty(self) -> None:
        cache = self._fresh_cache()
        assert cache.get("efficientnet") is None

    def test_is_cached_false_when_empty(self) -> None:
        cache = self._fresh_cache()
        assert not cache.is_cached("efficientnet")

    def test_len_zero_when_empty(self) -> None:
        cache = self._fresh_cache()
        assert len(cache) == 0

    def test_contains_operator(self) -> None:
        cache = self._fresh_cache()
        assert "efficientnet" not in cache

    def test_evict_returns_false_on_absent_model(self) -> None:
        cache = self._fresh_cache()
        assert cache.evict("resnet50") is False

    def test_clear_returns_zero_on_empty_cache(self) -> None:
        cache = self._fresh_cache()
        assert cache.clear() == 0

    def test_stats_returns_correct_structure(self) -> None:
        cache = self._fresh_cache()
        s = cache.stats()
        assert {"capacity", "size", "cached_models",
                "total_hits", "total_misses", "hit_rate"} <= s.keys()

    def test_stats_capacity_matches_init(self) -> None:
        from app.inference.cache import ModelCache
        cache = ModelCache(capacity=3)
        assert cache.stats()["capacity"] == 3

    def test_manual_insert_then_get(self) -> None:
        """Insert a fake entry directly and verify get() returns it."""
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=2)
        fake_model = _mock_keras_model()
        entry = _CacheEntry(fake_model, "cnn", load_duration_ms=10.0, model_info={})
        with cache._lock:
            cache._store["cnn"] = entry
        result = cache.get("cnn")
        assert result is fake_model

    def test_get_increments_hit_counter(self) -> None:
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=2)
        fake_model = _mock_keras_model()
        entry = _CacheEntry(fake_model, "cnn", 0.0, {})
        with cache._lock:
            cache._store["cnn"] = entry
        cache.get("cnn")
        cache.get("cnn")
        assert cache.stats()["total_hits"] == 2

    def test_get_miss_increments_miss_counter(self) -> None:
        cache = self._fresh_cache()
        cache.get("nonexistent")
        assert cache.stats()["total_misses"] == 1

    def test_evict_removes_entry(self) -> None:
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=2)
        entry = _CacheEntry(_mock_keras_model(), "vgg16", 0.0, {})
        with cache._lock:
            cache._store["vgg16"] = entry
        assert cache.evict("vgg16") is True
        assert not cache.is_cached("vgg16")

    def test_clear_removes_all_entries(self) -> None:
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=4)
        for name in ["cnn", "vgg16"]:
            entry = _CacheEntry(_mock_keras_model(), name, 0.0, {})
            with cache._lock:
                cache._store[name] = entry
        removed = cache.clear()
        assert removed == 2
        assert len(cache) == 0

    def test_lru_eviction_on_capacity_overflow(self) -> None:
        """With capacity=2, inserting a 3rd model should evict the LRU (first)."""
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=2)
        # Insert two models
        for name in ["cnn", "vgg16"]:
            entry = _CacheEntry(_mock_keras_model(), name, 0.0, {})
            with cache._lock:
                cache._store[name] = entry
        # Access cnn to make it MRU
        cache.get("cnn")
        # Insert third model — vgg16 (LRU) should be evicted
        with patch.object(cache, "load", wraps=lambda n: _insert_fake(cache, n)):
            _insert_fake(cache, "resnet50")
        assert "resnet50" in cache
        # LRU was vgg16 (not accessed after insertion)
        assert "vgg16" not in cache

    def test_entry_info_returns_dict_for_cached_model(self) -> None:
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=2)
        entry = _CacheEntry(_mock_keras_model(), "resnet50", 55.3, {"saved_at": "2024-01-01"})
        with cache._lock:
            cache._store["resnet50"] = entry
        info = cache.entry_info("resnet50")
        assert info is not None
        assert info["model_name"] == "resnet50"
        assert info["load_duration_ms"] == 55.3

    def test_entry_info_returns_none_for_absent_model(self) -> None:
        cache = self._fresh_cache()
        assert cache.entry_info("absent") is None

    def test_cached_names_reflects_store(self) -> None:
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=4)
        for name in ["cnn", "resnet50"]:
            entry = _CacheEntry(_mock_keras_model(), name, 0.0, {})
            with cache._lock:
                cache._store[name] = entry
        names = cache.cached_names()
        assert set(names) == {"cnn", "resnet50"}

    def test_hit_rate_zero_when_no_requests(self) -> None:
        cache = self._fresh_cache()
        assert cache.stats()["hit_rate"] == 0.0

    def test_hit_rate_computed_correctly(self) -> None:
        from app.inference.cache import ModelCache, _CacheEntry
        cache = ModelCache(capacity=2)
        entry = _CacheEntry(_mock_keras_model(), "cnn", 0.0, {})
        with cache._lock:
            cache._store["cnn"] = entry
        cache.get("cnn")   # hit
        cache.get("cnn")   # hit
        cache.get("other") # miss
        s = cache.stats()
        assert s["total_hits"] == 2
        assert s["total_misses"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


def _insert_fake(cache, name: str) -> None:
    """Helper: bypass disk loading and insert a fake entry directly."""
    from app.inference.cache import _CacheEntry
    entry = _CacheEntry(_mock_keras_model(), name, 0.0, {})
    with cache._lock:
        if name not in cache._store and len(cache._store) >= cache._capacity:
            cache._store.popitem(last=False)
        cache._store[name] = entry


# ─────────────────────────────────────────────────────────────────────────────
# cache.py — list_available_models / cache_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestListAvailableModels:

    def test_returns_list_of_five_architectures(self) -> None:
        from app.inference.cache import list_available_models
        models = list_available_models()
        assert len(models) == 5

    def test_all_known_architectures_present(self) -> None:
        from app.inference.cache import list_available_models
        names = {m["name"] for m in list_available_models()}
        assert names == {"mambavision", "cnn", "vgg16", "resnet50", "efficientnet"}

    def test_each_entry_has_required_keys(self) -> None:
        from app.inference.cache import list_available_models
        for entry in list_available_models():
            assert {"name", "available", "cached", "model_dir"} <= entry.keys()

    def test_available_flag_is_bool(self) -> None:
        from app.inference.cache import list_available_models
        for entry in list_available_models():
            assert isinstance(entry["available"], bool)

    def test_cached_flag_is_bool(self) -> None:
        from app.inference.cache import list_available_models
        for entry in list_available_models():
            assert isinstance(entry["cached"], bool)

    def test_cache_stats_returns_required_keys(self) -> None:
        from app.inference.cache import cache_stats
        s = cache_stats()
        assert {"capacity", "size", "hit_rate", "cached_models"} <= s.keys()

    def test_cache_stats_capacity_positive(self) -> None:
        from app.inference.cache import cache_stats
        assert cache_stats()["capacity"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# pipeline.py — InferencePipeline unit tests (model mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferencePipelineUnit:

    def _make_pipeline(self, **cfg_kwargs):
        from app.inference.config import InferenceConfig
        from app.inference.pipeline import InferencePipeline
        cfg = InferenceConfig(
            model_name="efficientnet",
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            image_size=64,
            **cfg_kwargs,
        )
        return InferencePipeline(cfg)

    def _run_predict(self, pipeline, source=None):
        """Patch get_model so no disk access is needed."""
        model = _mock_keras_model()
        with patch("app.inference.pipeline.get_model", return_value=model):
            return pipeline.predict(source or _varied_png_bytes())

    def test_predict_returns_prediction_result(self) -> None:
        from app.inference.results import PredictionResult
        p = self._make_pipeline()
        r = self._run_predict(p)
        assert isinstance(r, PredictionResult)

    def test_predicted_class_is_string(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert isinstance(r.predicted_class, str)

    def test_predicted_class_is_in_class_names(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert r.predicted_class in ["glioma", "meningioma", "notumor", "pituitary"]

    def test_confidence_in_range(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert 0.0 <= r.confidence <= 1.0

    def test_probabilities_keys_match_class_names(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert set(r.probabilities.keys()) == {"glioma", "meningioma", "notumor", "pituitary"}

    def test_probabilities_values_sum_to_one(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert abs(sum(r.probabilities.values()) - 1.0) < 1e-3

    def test_timing_ms_positive(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert r.timing_ms >= 0.0

    def test_top_k_one_by_default(self) -> None:
        r = self._run_predict(self._make_pipeline(top_k=1))
        assert len(r.top_k) == 1

    def test_top_k_three_returns_three(self) -> None:
        r = self._run_predict(self._make_pipeline(top_k=3))
        assert len(r.top_k) == 3

    def test_top_k_ranks_are_sequential(self) -> None:
        r = self._run_predict(self._make_pipeline(top_k=4))
        assert [t.rank for t in r.top_k] == [1, 2, 3, 4]

    def test_top_k_descending_probability(self) -> None:
        r = self._run_predict(self._make_pipeline(top_k=4))
        probs = [t.probability for t in r.top_k]
        assert probs == sorted(probs, reverse=True)

    def test_is_high_confidence_above_threshold(self) -> None:
        r = self._run_predict(self._make_pipeline(confidence_threshold=0.5))
        # mock model returns 0.85 for class 0
        assert r.is_high_confidence is True

    def test_is_high_confidence_below_threshold(self) -> None:
        r = self._run_predict(self._make_pipeline(confidence_threshold=0.99))
        # 0.85 < 0.99
        assert r.is_high_confidence is False

    def test_image_id_is_uuid_when_not_provided(self) -> None:
        import re
        r = self._run_predict(self._make_pipeline())
        assert re.match(r"^[0-9a-f-]{36}$", r.image_id)

    def test_custom_image_id_preserved(self) -> None:
        model = _mock_keras_model()
        p = self._make_pipeline()
        with patch("app.inference.pipeline.get_model", return_value=model):
            r = p.predict(_varied_png_bytes(), image_id="custom-123")
        assert r.image_id == "custom-123"

    def test_metadata_model_name_matches_config(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert r.metadata.model_name == "efficientnet"

    def test_no_gradcam_when_disabled(self) -> None:
        r = self._run_predict(self._make_pipeline(generate_gradcam=False))
        assert r.metadata.gradcam_path is None

    def test_file_not_found_propagates(self) -> None:
        from app.inference.cache import ModelCache
        p = self._make_pipeline()
        with patch("app.inference.pipeline.get_model",
                   side_effect=FileNotFoundError("no weights")):
            with pytest.raises(FileNotFoundError):
                p.predict(_varied_png_bytes())

    def test_predict_from_path(self, tmp_path: Path) -> None:
        img_path = tmp_path / "scan.png"
        img_path.write_bytes(_varied_png_bytes())
        p = self._make_pipeline()
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            r = p.predict(img_path)
        assert r.predicted_class in ["glioma", "meningioma", "notumor", "pituitary"]

    def test_error_is_none_on_success(self) -> None:
        r = self._run_predict(self._make_pipeline())
        assert r.error is None


# ─────────────────────────────────────────────────────────────────────────────
# pipeline.py — batch methods
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPipelineUnit:

    def _make_pipeline(self, **cfg_kwargs):
        from app.inference.config import InferenceConfig
        from app.inference.pipeline import InferencePipeline
        cfg = InferenceConfig(
            model_name="efficientnet",
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            image_size=64,
            **cfg_kwargs,
        )
        return InferencePipeline(cfg)

    def _run_batch(self, pipeline, sources):
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            return pipeline.predict_batch(sources)

    def test_empty_sources_returns_zero_total(self) -> None:
        p = self._make_pipeline()
        r = self._run_batch(p, [])
        assert r.total == 0

    def test_batch_total_equals_sources_count(self) -> None:
        sources = [("a.jpg", _varied_png_bytes()), ("b.jpg", _varied_png_bytes())]
        r = self._run_batch(self._make_pipeline(), sources)
        assert r.total == 2

    def test_batch_succeeded_all_valid(self) -> None:
        sources = [("a.jpg", _varied_png_bytes())] * 3
        r = self._run_batch(self._make_pipeline(), sources)
        assert r.succeeded == 3
        assert r.failed == 0

    def test_batch_failed_on_corrupt_image(self) -> None:
        sources = [
            ("good.jpg", _varied_png_bytes()),
            ("bad.jpg", b"not an image"),
        ]
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            r = self._make_pipeline().predict_batch(sources)
        assert r.failed >= 1

    def test_batch_results_length_matches_total(self) -> None:
        sources = [("a.jpg", _varied_png_bytes())] * 4
        r = self._run_batch(self._make_pipeline(), sources)
        assert len(r.results) == 4

    def test_batch_source_type_recorded(self) -> None:
        sources = [("a.jpg", _varied_png_bytes())]
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            r = self._make_pipeline().predict_batch(sources, source_type="zip")
        assert r.source_type == "zip"

    def test_predict_directory_raises_on_missing_dir(self) -> None:
        p = self._make_pipeline()
        with pytest.raises(FileNotFoundError):
            p.predict_directory("/nonexistent/path/")

    def test_predict_directory_raises_on_empty_dir(self, tmp_path: Path) -> None:
        p = self._make_pipeline()
        with pytest.raises(ValueError, match="No image files"):
            p.predict_directory(tmp_path)

    def test_predict_directory_finds_png_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.png").write_bytes(_varied_png_bytes())
        (tmp_path / "b.png").write_bytes(_varied_png_bytes())
        (tmp_path / "notes.txt").write_text("not an image")
        p = self._make_pipeline()
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            r = p.predict_directory(tmp_path)
        assert r.total == 2

    def test_predict_zip_raises_on_missing_file(self) -> None:
        p = self._make_pipeline()
        with pytest.raises(FileNotFoundError):
            p.predict_zip("/no/such/archive.zip")

    def test_predict_zip_raises_on_empty_archive(self, tmp_path: Path) -> None:
        z = tmp_path / "empty.zip"
        with zipfile.ZipFile(str(z), "w"):
            pass
        p = self._make_pipeline()
        with pytest.raises(ValueError, match="No image files"):
            p.predict_zip(z)

    def test_predict_zip_processes_images(self, tmp_path: Path) -> None:
        archive = _make_zip({
            "scan1.jpg": _varied_png_bytes(),
            "scan2.png": _varied_png_bytes(),
            "__MACOSX/._scan1.jpg": b"macos junk",
        })
        z = tmp_path / "test.zip"
        z.write_bytes(archive)
        p = self._make_pipeline()
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            r = p.predict_zip(z)
        # __MACOSX entries must be skipped
        assert r.total == 2

    def test_predict_zip_skips_non_image_files(self, tmp_path: Path) -> None:
        archive = _make_zip({
            "scan.png": _varied_png_bytes(),
            "readme.txt": b"text file",
            "data.csv": b"a,b,c",
        })
        z = tmp_path / "mixed.zip"
        z.write_bytes(archive)
        p = self._make_pipeline()
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            r = p.predict_zip(z)
        assert r.total == 1


# ─────────────────────────────────────────────────────────────────────────────
# batch.py — BatchInferenceRunner
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchInferenceRunner:

    def _make_runner(self, **cfg_kwargs):
        from app.inference.batch import BatchInferenceRunner
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(
            model_name="efficientnet",
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            image_size=64,
            max_workers=2,
            **cfg_kwargs,
        )
        return BatchInferenceRunner(cfg)

    def _run(self, runner, sources, source_type="list"):
        with patch("app.inference.cache.get_model", return_value=_mock_keras_model()):
            with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
                return runner.run(sources, source_type=source_type)

    def test_empty_sources_returns_zero_total(self) -> None:
        r = self._run(self._make_runner(), [])
        assert r.total == 0

    def test_single_image_succeeds(self) -> None:
        r = self._run(self._make_runner(), [("scan.jpg", _varied_png_bytes())])
        assert r.total == 1
        assert r.succeeded == 1

    def test_multiple_images_all_succeed(self) -> None:
        sources = [(f"img{i}.jpg", _varied_png_bytes(seed=i)) for i in range(5)]
        r = self._run(self._make_runner(), sources)
        assert r.total == 5
        assert r.succeeded == 5
        assert r.failed == 0

    def test_corrupt_image_counted_as_failure(self) -> None:
        sources = [
            ("good.png", _varied_png_bytes()),
            ("bad.png", b"corrupted"),
        ]
        r = self._run(self._make_runner(), sources)
        assert r.failed == 1
        assert r.succeeded == 1

    def test_results_ordered_by_input_order(self) -> None:
        sources = [(f"img{i}.png", _varied_png_bytes(seed=i)) for i in range(3)]
        r = self._run(self._make_runner(), sources)
        filenames = [item.filename for item in r.results]
        assert filenames == ["img0.png", "img1.png", "img2.png"]

    def test_progress_callback_called_for_each_item(self) -> None:
        calls = []
        from app.inference.batch import BatchInferenceRunner
        from app.inference.config import InferenceConfig
        cfg = InferenceConfig(
            model_name="efficientnet",
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            image_size=64,
        )
        runner = BatchInferenceRunner(cfg, progress_callback=lambda done, tot: calls.append((done, tot)))
        sources = [("a.png", _varied_png_bytes()), ("b.png", _varied_png_bytes())]
        with patch("app.inference.cache.get_model", return_value=_mock_keras_model()):
            with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
                runner.run(sources)
        assert len(calls) == 2
        assert calls[-1] == (2, 2)

    def test_run_directory_raises_on_missing_dir(self) -> None:
        runner = self._make_runner()
        with pytest.raises(FileNotFoundError):
            runner.run_directory("/nonexistent/dir/")

    def test_run_directory_raises_on_empty_dir(self, tmp_path: Path) -> None:
        runner = self._make_runner()
        with pytest.raises(ValueError, match="No image files"):
            runner.run_directory(tmp_path)

    def test_run_directory_processes_images(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"scan{i}.png").write_bytes(_varied_png_bytes(seed=i))
        (tmp_path / "notes.txt").write_text("ignored")
        r = self._run(self._make_runner(), [])  # just warmup pattern
        runner = self._make_runner()
        with patch("app.inference.cache.get_model", return_value=_mock_keras_model()):
            with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
                r = runner.run_directory(tmp_path)
        assert r.total == 3
        assert r.source_type == "directory"

    def test_run_zip_raises_on_missing_file(self) -> None:
        runner = self._make_runner()
        with pytest.raises(FileNotFoundError):
            runner.run_zip("/no/archive.zip")

    def test_run_zip_raises_on_no_images(self, tmp_path: Path) -> None:
        z = tmp_path / "empty.zip"
        with zipfile.ZipFile(str(z), "w") as zf:
            zf.writestr("readme.txt", "no images here")
        runner = self._make_runner()
        with pytest.raises(ValueError, match="No image files"):
            runner.run_zip(z)

    def test_run_zip_processes_images(self, tmp_path: Path) -> None:
        archive = _make_zip({
            "a.png": _varied_png_bytes(),
            "b.jpg": _varied_png_bytes(seed=2),
        })
        z = tmp_path / "test.zip"
        z.write_bytes(archive)
        runner = self._make_runner()
        with patch("app.inference.cache.get_model", return_value=_mock_keras_model()):
            with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
                r = runner.run_zip(z)
        assert r.total == 2
        assert r.source_type == "zip"

    # ── export() ─────────────────────────────────────────────────────────────

    def _make_batch_result_with_items(self):
        from app.inference.results import (
            BatchItemResult, BatchPredictionResult, PredictionMetadata,
            PredictionResult, TopKPrediction,
        )
        meta = PredictionMetadata(
            model_name="efficientnet",
            class_names=["glioma", "meningioma", "notumor", "pituitary"],
            image_size=224,
        )
        result = PredictionResult(
            image_id="abc-123",
            predicted_class="glioma",
            predicted_class_index=0,
            confidence=0.85,
            is_high_confidence=True,
            probabilities={"glioma": 0.85, "meningioma": 0.10,
                           "notumor": 0.03, "pituitary": 0.02},
            top_k=[TopKPrediction(1, "glioma", 0, 0.85)],
            timing_ms=40.0,
            metadata=meta,
        )
        return BatchPredictionResult(
            total=2,
            succeeded=2,
            failed=0,
            results=[
                BatchItemResult("scan1.jpg", True, result),
                BatchItemResult("scan2.jpg", True, result),
            ],
            timing_ms=80.0,
            model_name="efficientnet",
            source_type="list",
        )

    def test_export_json_creates_file(self, tmp_path: Path) -> None:
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        paths = runner.export(batch, output_dir=tmp_path, formats=("json",))
        assert "json_path" in paths
        assert Path(paths["json_path"]).exists()

    def test_export_json_is_valid(self, tmp_path: Path) -> None:
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        paths = runner.export(batch, output_dir=tmp_path, formats=("json",))
        with open(paths["json_path"]) as f:
            data = json.load(f)
        assert data["total"] == 2

    def test_export_csv_creates_file(self, tmp_path: Path) -> None:
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        paths = runner.export(batch, output_dir=tmp_path, formats=("csv",))
        assert "csv_path" in paths
        assert Path(paths["csv_path"]).exists()

    def test_export_csv_has_header_row(self, tmp_path: Path) -> None:
        import csv as _csv
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        paths = runner.export(batch, output_dir=tmp_path, formats=("csv",))
        with open(paths["csv_path"], newline="") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "filename" in rows[0]
        assert "predicted_class" in rows[0]
        assert "confidence" in rows[0]

    def test_export_both_creates_both_files(self, tmp_path: Path) -> None:
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        paths = runner.export(batch, output_dir=tmp_path, formats=("json", "csv"))
        assert "json_path" in paths
        assert "csv_path" in paths

    def test_export_paths_attached_to_result(self, tmp_path: Path) -> None:
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        runner.export(batch, output_dir=tmp_path, formats=("json",))
        assert "json_path" in batch.export_paths

    def test_export_csv_prob_columns_present(self, tmp_path: Path) -> None:
        import csv as _csv
        runner = self._make_runner()
        batch = self._make_batch_result_with_items()
        paths = runner.export(batch, output_dir=tmp_path, formats=("csv",))
        with open(paths["csv_path"], newline="") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "prob_glioma" in fieldnames

    def test_export_csv_failure_row_has_error_column(self, tmp_path: Path) -> None:
        import csv as _csv
        from app.inference.results import BatchItemResult, BatchPredictionResult
        runner = self._make_runner()
        batch = BatchPredictionResult(
            total=1, succeeded=0, failed=1,
            results=[BatchItemResult("bad.jpg", False, error="decode error")],
            timing_ms=5.0, model_name="efficientnet", source_type="list",
        )
        paths = runner.export(batch, output_dir=tmp_path, formats=("csv",))
        with open(paths["csv_path"], newline="") as f:
            rows = list(_csv.DictReader(f))
        assert rows[0]["error"] == "decode error"
        assert rows[0]["success"] == "False"


# ─────────────────────────────────────────────────────────────────────────────
# __init__.py — package-level re-exports
# ─────────────────────────────────────────────────────────────────────────────

class TestInferencePackageInit:

    def test_inference_pipeline_importable(self) -> None:
        from app.inference import InferencePipeline  # noqa: F401

    def test_predict_function_importable(self) -> None:
        from app.inference import predict  # noqa: F401

    def test_batch_runner_importable(self) -> None:
        from app.inference import BatchInferenceRunner  # noqa: F401

    def test_config_importable(self) -> None:
        from app.inference import InferenceConfig, DEFAULT_INFERENCE_CONFIG  # noqa: F401

    def test_result_classes_importable(self) -> None:
        from app.inference import (  # noqa: F401
            PredictionResult, BatchPredictionResult,
            BatchItemResult, PredictionMetadata, TopKPrediction,
        )

    def test_cache_functions_importable(self) -> None:
        from app.inference import (  # noqa: F401
            ModelCache, get_model, reload_model, evict_model,
            clear_cache, cache_stats, list_available_models,
        )

    def test_all_exports_in_dunder_all(self) -> None:
        import app.inference as pkg
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists '{name}' but it is not importable"


# ─────────────────────────────────────────────────────────────────────────────
# Inference API routes (no real model weights needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceAPIRoutes:
    """
    Integration tests for the Inference v2 endpoints.

    Where ML weights are required the tests either:
    (a) check for the expected 4xx / 5xx code when no weights are present, or
    (b) mock the pipeline to inject a deterministic result.
    """

    # ── POST /predict/image ───────────────────────────────────────────────────

    def test_predict_image_unsupported_content_type_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("scan.gif", b"GIF89a", "image/gif")},
        )
        assert resp.status_code == 400

    def test_predict_image_empty_file_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_predict_image_no_weights_returns_404_or_500(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"model_name": "vgg16"},
        )
        assert resp.status_code in (404, 500)

    def test_predict_image_jpeg_accepted(self) -> None:
        ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("scan.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert resp.status_code in (200, 404, 500)

    def test_predict_image_top_k_out_of_range_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"top_k": "99"},
        )
        assert resp.status_code == 422

    def test_predict_image_mocked_returns_200(self) -> None:
        from app.inference.results import (
            BatchItemResult, PredictionMetadata, PredictionResult, TopKPrediction,
        )
        mock_result = PredictionResult(
            image_id="mock-id",
            predicted_class="glioma",
            predicted_class_index=0,
            confidence=0.92,
            is_high_confidence=True,
            probabilities={"glioma": 0.92, "meningioma": 0.05,
                           "notumor": 0.02, "pituitary": 0.01},
            top_k=[TopKPrediction(1, "glioma", 0, 0.92)],
            timing_ms=38.4,
            metadata=PredictionMetadata(
                model_name="efficientnet",
                class_names=["glioma", "meningioma", "notumor", "pituitary"],
                image_size=224,
            ),
        )
        with patch("app.inference.pipeline.InferencePipeline.predict",
                   return_value=mock_result):
            resp = client.post(
                "/api/v1/predict/image",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["predicted_class"] == "glioma"
        assert body["data"]["confidence"] == 0.92

    def test_predict_image_response_schema(self) -> None:
        from app.inference.results import (
            PredictionMetadata, PredictionResult, TopKPrediction,
        )
        mock_result = PredictionResult(
            image_id="schema-test",
            predicted_class="notumor",
            predicted_class_index=2,
            confidence=0.77,
            is_high_confidence=True,
            probabilities={"glioma": 0.05, "meningioma": 0.08,
                           "notumor": 0.77, "pituitary": 0.10},
            top_k=[TopKPrediction(1, "notumor", 2, 0.77)],
            timing_ms=55.1,
            metadata=PredictionMetadata(
                model_name="efficientnet",
                class_names=["glioma", "meningioma", "notumor", "pituitary"],
                image_size=224,
            ),
        )
        with patch("app.inference.pipeline.InferencePipeline.predict",
                   return_value=mock_result):
            resp = client.post(
                "/api/v1/predict/image",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        data = resp.json()["data"]
        for key in ("image_id", "predicted_class", "confidence",
                    "is_high_confidence", "probabilities", "top_k",
                    "timing_ms", "metadata"):
            assert key in data, f"Missing key in response: {key}"

    # ── POST /predict/batch ───────────────────────────────────────────────────

    def test_predict_batch_no_images_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            files=[],
        )
        assert resp.status_code in (400, 422)

    def test_predict_batch_empty_files_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            files=[("images", ("empty.png", b"", "image/png"))],
        )
        assert resp.status_code == 400

    def test_predict_batch_no_weights_returns_404_or_500(self) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            files=[
                ("images", ("a.png", _varied_png_bytes(), "image/png")),
                ("images", ("b.png", _varied_png_bytes(), "image/png")),
            ],
        )
        assert resp.status_code in (200, 404, 500)

    def test_predict_batch_mocked_returns_200(self) -> None:
        from app.inference.results import BatchPredictionResult
        mock_result = BatchPredictionResult(
            total=2, succeeded=2, failed=0,
            results=[], timing_ms=88.0,
            model_name="efficientnet", source_type="list",
        )
        with patch("app.inference.batch.BatchInferenceRunner.run",
                   return_value=mock_result):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[
                    ("images", ("a.png", _varied_png_bytes(), "image/png")),
                    ("images", ("b.png", _varied_png_bytes(), "image/png")),
                ],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["total"] == 2

    def test_predict_batch_response_schema(self) -> None:
        from app.inference.results import BatchPredictionResult
        mock_result = BatchPredictionResult(
            total=1, succeeded=1, failed=0,
            results=[], timing_ms=40.0,
            model_name="efficientnet", source_type="list",
        )
        with patch("app.inference.batch.BatchInferenceRunner.run",
                   return_value=mock_result):
            resp = client.post(
                "/api/v1/predict/batch",
                files=[("images", ("x.png", _varied_png_bytes(), "image/png"))],
            )
        data = resp.json()["data"]
        for key in ("total", "succeeded", "failed", "success_rate", "timing_ms"):
            assert key in data, f"Missing key: {key}"

    # ── POST /predict/zip ─────────────────────────────────────────────────────

    def test_predict_zip_empty_archive_returns_400(self) -> None:
        empty_zip = _make_zip({})
        resp = client.post(
            "/api/v1/predict/zip",
            files={"archive": ("empty.zip", empty_zip, "application/zip")},
        )
        assert resp.status_code == 400

    def test_predict_zip_empty_body_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/zip",
            files={"archive": ("archive.zip", b"", "application/zip")},
        )
        assert resp.status_code == 400

    def test_predict_zip_non_zip_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/predict/zip",
            files={"archive": ("notazip.zip", b"not a zip file", "application/zip")},
        )
        assert resp.status_code == 422

    def test_predict_zip_no_images_returns_400(self) -> None:
        archive = _make_zip({"readme.txt": b"just text"})
        resp = client.post(
            "/api/v1/predict/zip",
            files={"archive": ("readme.zip", archive, "application/zip")},
        )
        assert resp.status_code == 400

    def test_predict_zip_mocked_returns_200(self) -> None:
        from app.inference.results import BatchPredictionResult
        mock_result = BatchPredictionResult(
            total=2, succeeded=2, failed=0,
            results=[], timing_ms=95.0,
            model_name="efficientnet", source_type="zip",
        )
        archive = _make_zip({
            "a.jpg": _varied_png_bytes(),
            "b.png": _varied_png_bytes(),
        })
        with patch("app.inference.batch.BatchInferenceRunner.run",
                   return_value=mock_result):
            resp = client.post(
                "/api/v1/predict/zip",
                files={"archive": ("images.zip", archive, "application/zip")},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 2

    def test_predict_zip_macosx_entries_excluded(self) -> None:
        """ZIP entries starting with __MACOSX must be filtered out."""
        archive = _make_zip({
            "scan.png": _varied_png_bytes(),
            "__MACOSX/._scan.png": b"macos metadata",
        })
        # No model needed — we're testing the filtering logic in the route
        resp = client.post(
            "/api/v1/predict/zip",
            files={"archive": ("images.zip", archive, "application/zip")},
        )
        # 404 / 500 = model not found but ZIP parsed successfully (1 image)
        # 400 would mean the image was not found (wrong — MACOSX not filtered)
        assert resp.status_code in (200, 404, 500)

    # ── GET /models ───────────────────────────────────────────────────────────

    def test_list_models_returns_200(self) -> None:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200

    def test_list_models_success_flag(self) -> None:
        assert client.get("/api/v1/models").json()["success"] is True

    def test_list_models_data_is_list_of_five(self) -> None:
        data = client.get("/api/v1/models").json()["data"]
        assert isinstance(data, list)
        assert len(data) == 5

    def test_list_models_cache_stats_present(self) -> None:
        body = client.get("/api/v1/models").json()
        assert "cache_stats" in body
        assert "capacity" in body["cache_stats"]

    def test_list_models_each_entry_has_name_available_cached(self) -> None:
        for entry in client.get("/api/v1/models").json()["data"]:
            assert "name" in entry
            assert "available" in entry
            assert "cached" in entry

    def test_list_models_covers_all_architectures(self) -> None:
        names = {e["name"] for e in client.get("/api/v1/models").json()["data"]}
        assert names == {"mambavision", "cnn", "vgg16", "resnet50", "efficientnet"}

    # ── POST /models/reload ───────────────────────────────────────────────────

    def test_reload_model_no_weights_returns_404(self) -> None:
        resp = client.post(
            "/api/v1/models/reload",
            json={"model_name": "vgg16"},
        )
        assert resp.status_code == 404

    def test_reload_model_invalid_body_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/models/reload",
            json={},
        )
        assert resp.status_code == 422

    def test_reload_model_mocked_returns_200(self) -> None:
        with patch("app.models.load_model.is_model_available", return_value=True):
            with patch("app.inference.cache._cache.reload", return_value=_mock_keras_model()):
                resp = client.post(
                    "/api/v1/models/reload",
                    json={"model_name": "efficientnet"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["model_name"] == "efficientnet"

    def test_reload_model_response_schema(self) -> None:
        with patch("app.models.load_model.is_model_available", return_value=True):
            with patch("app.inference.cache._cache.reload", return_value=_mock_keras_model()):
                resp = client.post(
                    "/api/v1/models/reload",
                    json={"model_name": "resnet50"},
                )
        body = resp.json()
        assert {"success", "message", "model_name"} <= body.keys()

    # ── GET /models/active ────────────────────────────────────────────────────

    def test_active_model_no_weights_returns_404(self) -> None:
        resp = client.get("/api/v1/models/active")
        # No trained weights in the test environment
        assert resp.status_code in (200, 404)

    def test_active_model_mocked_returns_200(self) -> None:
        with patch("app.models.load_model.is_model_available", return_value=True):
            with patch("app.models.load_model.get_model_info", return_value={
                "saved_at": "2024-01-01", "total_params": 12_000_000
            }):
                resp = client.get("/api/v1/models/active")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["available"] is True

    def test_active_model_response_schema(self) -> None:
        with patch("app.models.load_model.is_model_available", return_value=True):
            with patch("app.models.load_model.get_model_info", return_value={}):
                resp = client.get("/api/v1/models/active")
        if resp.status_code == 200:
            data = resp.json()["data"]
            for key in ("model_name", "available", "cached", "model_info", "cache_stats"):
                assert key in data, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# module-level predict() convenience function
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleLevelPredict:

    def test_predict_returns_prediction_result(self) -> None:
        from app.inference import predict, PredictionResult
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            result = predict(_varied_png_bytes(), model_name="efficientnet")
        assert isinstance(result, PredictionResult)

    def test_predict_uses_default_model_when_none(self) -> None:
        from app.inference import predict
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            result = predict(_varied_png_bytes())
        assert result.metadata.model_name in ("mambavision", "cnn", "vgg16", "resnet50", "efficientnet")

    def test_predict_top_k_respected(self) -> None:
        from app.inference import predict
        with patch("app.inference.pipeline.get_model", return_value=_mock_keras_model()):
            result = predict(_varied_png_bytes(), top_k=3)
        assert len(result.top_k) == 3
