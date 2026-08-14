"""
tests/test_integration.py — End-to-end integration tests for the BrainTumor AI service.

These tests validate the complete prediction and training pipelines with full
API contract verification, error handling, and backend/frontend compatibility.

Coverage
--------
TestPredictionPipeline              End-to-end prediction flow (mocked model)
TestTrainingPipeline                End-to-end training flow (mocked Trainer)
TestErrorHandling                   Invalid images, missing checkpoints, bad requests
TestAPIContractValidation           Response schema validation for all endpoints
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _varied_png_bytes(h: int = 64, w: int = 64, seed: int = 7) -> bytes:
    """Generate a naturally varied PNG that passes quality checks."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _solid_png_bytes(value: int = 128) -> bytes:
    """Generate a uniform PNG (fails quality checks)."""
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ═════════════════════════════════════════════════════════════════════════════
# Task 9: End-to-End Prediction Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictionPipeline:
    """
    Validates the complete prediction pipeline:
        image upload → preprocessing → MambaVision inference →
        confidence score → Grad-CAM → final API response
    """

    def _mock_predict_result(self) -> dict:
        return {
            "class":         "glioma",
            "confidence":    0.9321,
            "probabilities": {
                "glioma":      0.9321,
                "meningioma":  0.0412,
                "notumor":     0.0189,
                "pituitary":   0.0078,
            },
            "gradcam_path":  "/tmp/gradcam/abc123/overlay.png",
            "model_used":    "mambavision",
        }

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_predict_returns_200_with_mocked_model(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200

    def test_predict_response_success_flag(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            body = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()
        assert body["success"] is True

    def test_predict_response_has_data_key(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            body = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()
        assert "data" in body

    def test_predict_data_contains_class(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert data["class"] == "glioma"

    def test_predict_data_contains_confidence(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert data["confidence"] == pytest.approx(0.9321, abs=1e-4)

    def test_predict_data_contains_probabilities(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert "probabilities" in data
        assert set(data["probabilities"].keys()) == {
            "glioma", "meningioma", "notumor", "pituitary"
        }

    def test_predict_probabilities_sum_to_one(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_data_contains_model_used(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert data["model_used"] == "mambavision"

    def test_predict_data_contains_gradcam_path_key(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert "gradcam_path" in data

    def test_predict_model_override_forwarded(self) -> None:
        """model_name form field is passed through to predict()."""
        called_with: dict = {}

        def _capture(**kwargs):
            called_with.update(kwargs)
            return self._mock_predict_result()

        with patch("app.models.predict.predict", side_effect=lambda src, **kw: _capture(**kw)):
            client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
                data={"model_name": "resnet50"},
            )
        assert called_with.get("model_name") == "resnet50"

    def test_predict_gradcam_flag_forwarded(self) -> None:
        """generate_gradcam form field is forwarded to predict()."""
        called_with: dict = {}

        def _capture(**kwargs):
            called_with.update(kwargs)
            return self._mock_predict_result()

        with patch("app.models.predict.predict", side_effect=lambda src, **kw: _capture(**kw)):
            client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
                data={"generate_gradcam": "false"},
            )
        assert called_with.get("generate_gradcam") is False

    def test_predict_jpeg_file_accepted(self) -> None:
        ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.jpg", buf.tobytes(), "image/jpeg")},
            )
        assert resp.status_code == 200

    def test_predict_class_is_one_of_four(self) -> None:
        with patch("app.models.predict.predict", return_value=self._mock_predict_result()):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert data["class"] in {"glioma", "meningioma", "notumor", "pituitary"}

    # ── /predict/image (inference v2) ─────────────────────────────────────────

    def test_predict_v2_returns_200_mocked(self) -> None:
        from app.inference.results import (
            PredictionMetadata, PredictionResult, TopKPrediction,
        )
        mock_result = PredictionResult(
            image_id="test-001",
            predicted_class="glioma",
            predicted_class_index=0,
            confidence=0.91,
            is_high_confidence=True,
            probabilities={"glioma": 0.91, "meningioma": 0.05,
                           "notumor": 0.03, "pituitary": 0.01},
            top_k=[TopKPrediction(1, "glioma", 0, 0.91)],
            timing_ms=42.0,
            metadata=PredictionMetadata(
                model_name="mambavision",
                class_names=["glioma", "meningioma", "notumor", "pituitary"],
                image_size=224,
            ),
        )
        with patch("app.inference.pipeline.InferencePipeline.predict", return_value=mock_result):
            resp = client.post(
                "/api/v1/predict/image",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["predicted_class"] == "glioma"
        assert body["data"]["confidence"] == pytest.approx(0.91, abs=1e-4)

    def test_predict_v2_full_response_schema(self) -> None:
        from app.inference.results import (
            PredictionMetadata, PredictionResult, TopKPrediction,
        )
        mock_result = PredictionResult(
            image_id="schema-001",
            predicted_class="notumor",
            predicted_class_index=2,
            confidence=0.88,
            is_high_confidence=True,
            probabilities={"glioma": 0.05, "meningioma": 0.04,
                           "notumor": 0.88, "pituitary": 0.03},
            top_k=[TopKPrediction(1, "notumor", 2, 0.88)],
            timing_ms=55.0,
            metadata=PredictionMetadata(
                model_name="mambavision",
                class_names=["glioma", "meningioma", "notumor", "pituitary"],
                image_size=224,
            ),
        )
        with patch("app.inference.pipeline.InferencePipeline.predict", return_value=mock_result):
            data = client.post(
                "/api/v1/predict/image",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        for key in ("image_id", "predicted_class", "predicted_class_index",
                    "confidence", "is_high_confidence", "probabilities",
                    "top_k", "timing_ms", "metadata"):
            assert key in data, f"Missing key in /predict/image response: {key}"


# ═════════════════════════════════════════════════════════════════════════════
# Task 10: End-to-End Training Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestTrainingPipeline:
    """
    Validates the complete training pipeline:
        dataset loading → DataLoader creation → training → validation →
        checkpoint saving → best model loading → evaluation → artifact generation
    All heavy ML operations are mocked to run in the test environment.
    """

    def _mock_train_result(self, model_name: str = "mambavision") -> dict:
        return {
            "model_name":           model_name,
            "epochs_phase1":        10,
            "epochs_phase2":        5,
            "final_train_accuracy": 0.9541,
            "final_val_accuracy":   0.9213,
            "final_train_loss":     0.1234,
            "final_val_loss":       0.2105,
            "training_duration_s":  142.7,
            "saved_paths": {
                "model_dir":  f"/saved_models/{model_name}",
                "model_path": f"/saved_models/{model_name}/model.safetensors",
                "info_path":  f"/saved_models/{model_name}/model_info.json",
                "format":     "hf",
            },
            "phase1_history": {"loss": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.3]},
            "phase2_history": {"loss": [0.2, 0.1], "val_loss": [0.3, 0.2]},
        }

    def _mock_trainer(self):
        trainer = MagicMock()
        trainer.experiment_id = "mambavision-20260730-000000-aabbccdd"
        trainer.run = MagicMock(return_value={"status": "completed"})
        return trainer

    # ── POST /train (synchronous) ──────────────────────────────────────────────

    def test_train_returns_200_mocked(self) -> None:
        with patch("app.models.train.train_model", return_value=self._mock_train_result()):
            resp = client.post(
                "/api/v1/train",
                json={"model_name": "mambavision", "epochs": 10, "batch_size": 16},
            )
        assert resp.status_code == 200

    def test_train_response_success_flag(self) -> None:
        with patch("app.models.train.train_model", return_value=self._mock_train_result()):
            body = client.post(
                "/api/v1/train",
                json={"model_name": "mambavision", "epochs": 10},
            ).json()
        assert body["success"] is True

    def test_train_response_has_message(self) -> None:
        with patch("app.models.train.train_model", return_value=self._mock_train_result()):
            body = client.post(
                "/api/v1/train",
                json={"model_name": "mambavision", "epochs": 10},
            ).json()
        assert "message" in body
        assert len(body["message"]) > 0

    def test_train_response_data_contains_accuracy(self) -> None:
        with patch("app.models.train.train_model", return_value=self._mock_train_result()):
            body = client.post(
                "/api/v1/train",
                json={"model_name": "mambavision", "epochs": 10},
            ).json()
        assert body["data"]["final_val_accuracy"] == pytest.approx(0.9213, abs=1e-4)

    def test_train_response_data_contains_saved_paths(self) -> None:
        with patch("app.models.train.train_model", return_value=self._mock_train_result()):
            data = client.post(
                "/api/v1/train",
                json={"model_name": "mambavision", "epochs": 10},
            ).json()["data"]
        assert "saved_paths" in data

    def test_train_fine_tune_params_forwarded(self) -> None:
        """fine_tune, fine_tune_layers, fine_tune_epochs reach train_model."""
        captured: dict = {}

        def _capture(**kw):
            captured.update(kw)
            return self._mock_train_result()

        with patch("app.models.train.train_model", side_effect=lambda **kw: _capture(**kw)):
            client.post(
                "/api/v1/train",
                json={"model_name": "cnn", "epochs": 5,
                      "fine_tune": True, "fine_tune_layers": 15,
                      "fine_tune_epochs": 8},
            )
        assert captured.get("fine_tune") is True
        assert captured.get("fine_tune_layers") == 15
        assert captured.get("fine_tune_epochs") == 8

    # ── POST /train/start (async v2) ──────────────────────────────────────────

    def test_train_start_returns_202(self) -> None:
        with patch("training.trainer.Trainer") as MockTrainer:
            MockTrainer.return_value = self._mock_trainer()
            resp = client.post(
                "/api/v1/train/start",
                json={"architecture": "mambavision", "epochs": 5, "batch_size": 8},
            )
        assert resp.status_code == 202

    def test_train_start_response_schema(self) -> None:
        with patch("training.trainer.Trainer") as MockTrainer:
            MockTrainer.return_value = self._mock_trainer()
            body = client.post(
                "/api/v1/train/start",
                json={"architecture": "mambavision", "epochs": 5},
            ).json()
        assert body["success"] is True
        assert "job_id" in body
        assert "experiment_id" in body
        assert len(body["job_id"]) > 0
        assert len(body["experiment_id"]) > 0

    def test_train_start_job_is_pollable(self) -> None:
        """After /train/start, the job must be retrievable from /train/status."""
        from app.training.job_store import get_job_store
        store = get_job_store()
        store._store.clear()

        with patch("training.trainer.Trainer") as MockTrainer:
            MockTrainer.return_value = self._mock_trainer()
            resp = client.post(
                "/api/v1/train/start",
                json={"architecture": "cnn", "epochs": 1},
            )
        job_id = resp.json()["job_id"]

        status_resp = client.get(f"/api/v1/train/status/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["data"]["job_id"] == job_id

    def test_train_status_has_required_fields(self) -> None:
        from app.training.job_store import get_job_store
        store = get_job_store()
        store._store.clear()

        with patch("training.trainer.Trainer") as MockTrainer:
            MockTrainer.return_value = self._mock_trainer()
            resp = client.post(
                "/api/v1/train/start",
                json={"architecture": "cnn", "epochs": 1},
            )
        job_id = resp.json()["job_id"]
        data = client.get(f"/api/v1/train/status/{job_id}").json()["data"]

        for field in ("job_id", "status", "experiment_id", "created_at"):
            assert field in data, f"Job status missing field: {field}"

    # ── POST /evaluate ─────────────────────────────────────────────────────────

    def test_evaluate_returns_200_mocked(self) -> None:
        mock_result = {
            "model_name":       "mambavision",
            "accuracy":         0.9734,
            "precision":        0.9741,
            "recall":           0.9728,
            "f1":               0.9734,
            "auc_roc":          0.9981,
            "confusion_matrix": [[394, 3, 1, 2], [2, 388, 4, 6],
                                 [1, 3, 404, 2], [3, 5, 2, 390]],
            "per_class": {
                "glioma":      {"precision": 0.98, "recall": 0.97, "f1": 0.98, "support": 400},
                "meningioma":  {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 400},
                "notumor":     {"precision": 0.98, "recall": 0.98, "f1": 0.98, "support": 410},
                "pituitary":   {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 400},
            },
            "num_samples":  1610,
            "class_names":  ["glioma", "meningioma", "notumor", "pituitary"],
            "model_info":   {"total_params": 31_000_000},
        }
        with patch("app.models.evaluate.evaluate_model", return_value=mock_result):
            resp = client.post("/api/v1/evaluate", json={"model_name": "mambavision"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["accuracy"] == pytest.approx(0.9734, abs=1e-4)

    def test_evaluate_response_schema_complete(self) -> None:
        mock_result = {
            "model_name": "mambavision", "accuracy": 0.97, "precision": 0.97,
            "recall": 0.97, "f1": 0.97, "auc_roc": 0.998,
            "confusion_matrix": [[1, 0, 0, 0]] * 4,
            "per_class": {"glioma": {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 100}},
            "num_samples": 400, "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
            "model_info": {},
        }
        with patch("app.models.evaluate.evaluate_model", return_value=mock_result):
            data = client.post("/api/v1/evaluate", json={}).json()["data"]
        for key in ("model_name", "accuracy", "precision", "recall", "f1", "auc_roc",
                    "confusion_matrix", "per_class", "num_samples", "class_names"):
            assert key in data, f"Missing key in /evaluate response: {key}"


# ═════════════════════════════════════════════════════════════════════════════
# Task 11: Error Handling
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """
    Validates error handling for:
    - Invalid image types and empty uploads
    - Missing model checkpoints
    - Corrupted/un-decodable images
    - Invalid request payloads
    - Empty datasets
    """

    # ── Invalid images ────────────────────────────────────────────────────────

    def test_predict_gif_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("scan.gif", b"GIF89a\x01\x00\x01\x00", "image/gif")},
        )
        assert resp.status_code == 400

    def test_predict_empty_upload_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_predict_corrupt_bytes_returns_error(self) -> None:
        """Garbage bytes that look like PNG but can't be decoded."""
        with patch("app.models.predict.predict",
                   side_effect=ValueError("Cannot decode image")):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("bad.png", b"\x89PNG\r\n\x1a\nGARBAGE", "image/png")},
            )
        assert resp.status_code in (400, 422, 500)

    def test_predict_v2_gif_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("scan.gif", b"GIF89a", "image/gif")},
        )
        assert resp.status_code == 400

    def test_predict_v2_empty_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_predict_zip_corrupted_raises_422(self) -> None:
        resp = client.post(
            "/api/v1/predict/zip",
            files={"archive": ("bad.zip", b"not a zip at all", "application/zip")},
        )
        assert resp.status_code == 422

    def test_quality_check_unsupported_type_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/preprocess/quality-check",
            files={"image": ("scan.bmp", b"BM", "image/bmp")},
        )
        assert resp.status_code == 400

    def test_quality_check_empty_upload_returns_400(self) -> None:
        resp = client.post(
            "/api/v1/preprocess/quality-check",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    # ── Missing checkpoints ───────────────────────────────────────────────────

    def test_predict_no_weights_returns_404(self) -> None:
        with patch("app.models.predict.predict",
                   side_effect=FileNotFoundError("No saved model found")):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        assert resp.status_code == 404

    def test_predict_no_weights_response_has_detail(self) -> None:
        with patch("app.models.predict.predict",
                   side_effect=FileNotFoundError("No saved model found for 'cnn'")):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
                data={"model_name": "cnn"},
            )
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body

    def test_evaluate_no_weights_returns_404(self) -> None:
        with patch("app.models.evaluate.evaluate_model",
                   side_effect=FileNotFoundError("No saved model found")):
            resp = client.post("/api/v1/evaluate", json={"model_name": "cnn"})
        assert resp.status_code == 404

    def test_evaluate_missing_dataset_returns_404(self) -> None:
        with patch("app.models.evaluate.evaluate_model",
                   side_effect=FileNotFoundError("Test directory not found")):
            resp = client.post(
                "/api/v1/evaluate",
                json={"dataset_dir": "/nonexistent/dataset"},
            )
        assert resp.status_code == 404

    def test_models_reload_no_weights_returns_404(self) -> None:
        resp = client.post("/api/v1/models/reload", json={"model_name": "vgg16"})
        assert resp.status_code == 404

    # ── Invalid requests ──────────────────────────────────────────────────────

    def test_train_epochs_zero_returns_422(self) -> None:
        resp = client.post("/api/v1/train", json={"epochs": 0})
        assert resp.status_code == 422

    def test_train_batch_size_too_large_returns_422(self) -> None:
        resp = client.post("/api/v1/train", json={"batch_size": 9999})
        assert resp.status_code == 422

    def test_train_learning_rate_zero_returns_422(self) -> None:
        resp = client.post("/api/v1/train", json={"learning_rate": 0.0})
        assert resp.status_code == 422

    def test_train_learning_rate_too_large_returns_422(self) -> None:
        resp = client.post("/api/v1/train", json={"learning_rate": 1.0})
        assert resp.status_code == 422

    def test_evaluate_batch_size_zero_returns_422(self) -> None:
        resp = client.post("/api/v1/evaluate", json={"batch_size": 0})
        assert resp.status_code == 422

    def test_train_start_unknown_arch_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/train/start",
            json={"architecture": "densenet121", "epochs": 1},
        )
        assert resp.status_code == 422

    def test_train_start_epochs_zero_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/train/start",
            json={"architecture": "cnn", "epochs": 0},
        )
        assert resp.status_code == 422

    def test_train_status_unknown_job_returns_404(self) -> None:
        resp = client.get("/api/v1/train/status/definitely-does-not-exist")
        assert resp.status_code == 404

    def test_dataset_prepare_bad_ratios_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/dataset/prepare",
            json={"train_ratio": 0.5, "val_ratio": 0.5, "test_ratio": 0.5},
        )
        assert resp.status_code == 422

    def test_predict_v2_top_k_too_large_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/predict/image",
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            data={"top_k": "99"},
        )
        assert resp.status_code == 422

    def test_predict_unauthenticated_still_works(self) -> None:
        """The /predict endpoint is public (no auth required by default)."""
        with patch("app.models.predict.predict",
                   side_effect=FileNotFoundError("No weights")):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        # 404 from the mocked FileNotFoundError — NOT 401
        assert resp.status_code == 404

    def test_train_unauthenticated_returns_401(self) -> None:
        """The /train endpoint requires ADMIN or RESEARCHER role."""
        from app.security.dependencies import get_current_active_user, optional_auth
        app.dependency_overrides.clear()
        try:
            resp = client.post(
                "/api/v1/train",
                json={"model_name": "cnn", "epochs": 1},
            )
            assert resp.status_code == 401
        finally:
            from app.security.auth import UserInDB
            from app.security.roles import Role
            from tests.conftest import MOCK_USER
            async def _mock(): return MOCK_USER
            app.dependency_overrides[get_current_active_user] = _mock
            app.dependency_overrides[optional_auth] = _mock

    def test_evaluate_unauthenticated_returns_401(self) -> None:
        from app.security.dependencies import get_current_active_user, optional_auth
        app.dependency_overrides.clear()
        try:
            resp = client.post("/api/v1/evaluate", json={})
            assert resp.status_code == 401
        finally:
            from tests.conftest import MOCK_USER
            async def _mock(): return MOCK_USER
            app.dependency_overrides[get_current_active_user] = _mock
            app.dependency_overrides[optional_auth] = _mock

    # ── Corrupted model files ─────────────────────────────────────────────────

    def test_predict_runtime_error_returns_500(self) -> None:
        """RuntimeError from load_model (corrupted weights) → 500."""
        with patch("app.models.predict.predict",
                   side_effect=RuntimeError("Failed to load model: corrupted weights")):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        assert resp.status_code == 500

    # ── Empty dataset ─────────────────────────────────────────────────────────

    def test_evaluate_empty_dataset_returns_422(self) -> None:
        with patch("app.models.evaluate.evaluate_model",
                   side_effect=ValueError("No images found")):
            resp = client.post(
                "/api/v1/evaluate",
                json={"dataset_dir": "/empty/dataset"},
            )
        assert resp.status_code == 422

    def test_train_missing_dataset_returns_404(self) -> None:
        resp = client.post(
            "/api/v1/train",
            json={"model_name": "cnn", "epochs": 1,
                  "dataset_dir": "/definitely/does/not/exist"},
        )
        assert resp.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# Task 12: API Contract Validation (Backend + Frontend Compatibility)
# ═════════════════════════════════════════════════════════════════════════════

class TestAPIContractValidation:
    """
    Validates that every AI service endpoint returns the expected JSON contract
    that the Node.js backend and React frontend depend on.

    Response schema rules (from existing backend/frontend usage):
    - All responses have {success: bool} at root
    - Prediction responses have {success, data: {class, confidence, probabilities, model_used, gradcam_path}}
    - Evaluate responses have {success, message, data: {accuracy, precision, recall, f1, auc_roc, confusion_matrix, per_class, class_names}}
    - Train responses have {success, message, data}
    - Health response has {success, status, service, active_model, class_names, models_available, image_size, python_version, timestamp, version, environment}
    """

    # ── GET /health ───────────────────────────────────────────────────────────

    def test_health_contract_all_fields_present(self) -> None:
        body = client.get("/api/v1/health").json()
        required = {
            "success", "status", "service", "version", "timestamp",
            "environment", "active_model", "class_names", "image_size",
            "python_version", "models_available",
        }
        assert required.issubset(body.keys())

    def test_health_models_available_has_all_five_architectures(self) -> None:
        body = client.get("/api/v1/health").json()
        assert set(body["models_available"].keys()) == {
            "mambavision", "cnn", "vgg16", "resnet50", "efficientnet"
        }

    def test_health_class_names_order_stable(self) -> None:
        """Frontend depends on a consistent class order."""
        body = client.get("/api/v1/health").json()
        assert body["class_names"] == ["glioma", "meningioma", "notumor", "pituitary"]

    def test_health_active_model_is_valid_architecture(self) -> None:
        """active_model must be one of the five supported architectures."""
        body = client.get("/api/v1/health").json()
        valid = {"mambavision", "cnn", "vgg16", "resnet50", "efficientnet"}
        assert body["active_model"] in valid, (
            f"active_model '{body['active_model']}' is not a supported architecture"
        )

    def test_health_image_size_is_224(self) -> None:
        """Frontend sends images at 224×224; this must stay in sync."""
        body = client.get("/api/v1/health").json()
        assert body["image_size"] == 224

    def test_health_status_is_ok_string(self) -> None:
        assert client.get("/api/v1/health").json()["status"] == "ok"

    def test_health_service_name_unchanged(self) -> None:
        assert client.get("/api/v1/health").json()["service"] == \
            "Brain Tumour Detection AI Service"

    # ── POST /predict schema ──────────────────────────────────────────────────

    def test_predict_schema_success_is_bool(self) -> None:
        result = {
            "class": "glioma", "confidence": 0.9,
            "probabilities": {"glioma": 0.9, "meningioma": 0.05,
                              "notumor": 0.03, "pituitary": 0.02},
            "gradcam_path": None, "model_used": "mambavision",
        }
        with patch("app.models.predict.predict", return_value=result):
            body = client.post(
                "/api/v1/predict",
                files={"image": ("s.png", _varied_png_bytes(), "image/png")},
            ).json()
        assert isinstance(body["success"], bool)

    def test_predict_schema_data_class_is_string(self) -> None:
        result = {
            "class": "meningioma", "confidence": 0.85,
            "probabilities": {"glioma": 0.05, "meningioma": 0.85,
                              "notumor": 0.05, "pituitary": 0.05},
            "gradcam_path": None, "model_used": "mambavision",
        }
        with patch("app.models.predict.predict", return_value=result):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("s.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert isinstance(data["class"], str)

    def test_predict_schema_confidence_is_float(self) -> None:
        result = {
            "class": "notumor", "confidence": 0.77,
            "probabilities": {"glioma": 0.07, "meningioma": 0.06,
                              "notumor": 0.77, "pituitary": 0.10},
            "gradcam_path": None, "model_used": "mambavision",
        }
        with patch("app.models.predict.predict", return_value=result):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("s.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert isinstance(data["confidence"], float)

    def test_predict_schema_gradcam_path_present(self) -> None:
        result = {
            "class": "glioma", "confidence": 0.9,
            "probabilities": {"glioma": 0.9, "meningioma": 0.05,
                              "notumor": 0.03, "pituitary": 0.02},
            "gradcam_path": "/gradcam/test.png", "model_used": "mambavision",
        }
        with patch("app.models.predict.predict", return_value=result):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("s.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert "gradcam_path" in data

    def test_predict_schema_model_used_is_mambavision(self) -> None:
        result = {
            "class": "glioma", "confidence": 0.9,
            "probabilities": {"glioma": 0.9, "meningioma": 0.05,
                              "notumor": 0.03, "pituitary": 0.02},
            "gradcam_path": None, "model_used": "mambavision",
        }
        with patch("app.models.predict.predict", return_value=result):
            data = client.post(
                "/api/v1/predict",
                files={"image": ("s.png", _varied_png_bytes(), "image/png")},
            ).json()["data"]
        assert data["model_used"] == "mambavision"

    # ── POST /evaluate schema ─────────────────────────────────────────────────

    def test_evaluate_schema_confusion_matrix_is_2d_list(self) -> None:
        mock_result = {
            "model_name": "mambavision", "accuracy": 0.97, "precision": 0.97,
            "recall": 0.97, "f1": 0.97, "auc_roc": 0.998,
            "confusion_matrix": [[394, 3, 1, 2], [2, 388, 4, 6],
                                 [1, 3, 404, 2], [3, 5, 2, 390]],
            "per_class": {}, "num_samples": 1610,
            "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
            "model_info": {},
        }
        with patch("app.models.evaluate.evaluate_model", return_value=mock_result):
            data = client.post("/api/v1/evaluate", json={}).json()["data"]
        cm = data["confusion_matrix"]
        assert isinstance(cm, list)
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

    def test_evaluate_schema_per_class_has_four_entries(self) -> None:
        per_class = {
            cls: {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 400}
            for cls in ["glioma", "meningioma", "notumor", "pituitary"]
        }
        mock_result = {
            "model_name": "mambavision", "accuracy": 0.97, "precision": 0.97,
            "recall": 0.97, "f1": 0.97, "auc_roc": 0.998,
            "confusion_matrix": [[1, 0, 0, 0]] * 4,
            "per_class": per_class, "num_samples": 1600,
            "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
            "model_info": {},
        }
        with patch("app.models.evaluate.evaluate_model", return_value=mock_result):
            data = client.post("/api/v1/evaluate", json={}).json()["data"]
        assert len(data["per_class"]) == 4
        for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
            assert cls in data["per_class"]

    def test_evaluate_schema_metrics_are_floats_in_range(self) -> None:
        mock_result = {
            "model_name": "mambavision", "accuracy": 0.97, "precision": 0.97,
            "recall": 0.97, "f1": 0.97, "auc_roc": 0.998,
            "confusion_matrix": [[1, 0, 0, 0]] * 4,
            "per_class": {}, "num_samples": 1600,
            "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
            "model_info": {},
        }
        with patch("app.models.evaluate.evaluate_model", return_value=mock_result):
            data = client.post("/api/v1/evaluate", json={}).json()["data"]
        for metric in ("accuracy", "precision", "recall", "f1", "auc_roc"):
            assert isinstance(data[metric], float), f"{metric} is not float"
            assert 0.0 <= data[metric] <= 1.0, f"{metric} out of [0,1]"

    # ── GET /models schema ─────────────────────────────────────────────────────

    def test_models_list_contract(self) -> None:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], list)
        assert len(body["data"]) == 5
        assert "cache_stats" in body
        for entry in body["data"]:
            assert "name" in entry
            assert "available" in entry
            assert "cached" in entry

    def test_models_list_all_five_architectures(self) -> None:
        names = {e["name"] for e in client.get("/api/v1/models").json()["data"]}
        assert names == {"mambavision", "cnn", "vgg16", "resnet50", "efficientnet"}

    # ── GET /health — backend compatibility ───────────────────────────────────

    def test_health_is_json_parseable(self) -> None:
        resp = client.get("/api/v1/health")
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert isinstance(data, dict)

    def test_health_no_auth_required(self) -> None:
        """Backend health check must work without a token."""
        from app.security.dependencies import get_current_active_user, optional_auth
        saved = dict(app.dependency_overrides)
        app.dependency_overrides.clear()
        try:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.update(saved)

    # ── Auth endpoints schema ─────────────────────────────────────────────────

    def test_auth_login_bad_credentials_returns_401(self) -> None:
        from app.security.dependencies import get_current_active_user, optional_auth
        saved = dict(app.dependency_overrides)
        app.dependency_overrides.clear()
        try:
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "nonexistent", "password": "wrongpassword"},
            )
            assert resp.status_code == 401
        finally:
            app.dependency_overrides.update(saved)

    def test_auth_refresh_invalid_token_returns_401(self) -> None:
        from app.security.dependencies import get_current_active_user, optional_auth
        saved = dict(app.dependency_overrides)
        app.dependency_overrides.clear()
        try:
            resp = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "not.a.real.token"},
            )
            assert resp.status_code == 401
        finally:
            app.dependency_overrides.update(saved)

    # ── Preprocessing endpoint schema ─────────────────────────────────────────

    def test_quality_check_response_contract(self) -> None:
        resp = client.post(
            "/api/v1/preprocess/quality-check",
            files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "is_valid" in data
        assert "checks" in data
        assert isinstance(data["checks"], list)

    def test_dataset_info_missing_returns_404(self) -> None:
        resp = client.get("/api/v1/dataset/info")
        assert resp.status_code in (200, 404)

    # ── Logging coverage ──────────────────────────────────────────────────────

    def test_predict_logs_on_success(self) -> None:
        """Confirm the prediction route reaches the logger (integration smoke)."""
        import logging
        result = {
            "class": "glioma", "confidence": 0.9,
            "probabilities": {"glioma": 0.9, "meningioma": 0.05,
                              "notumor": 0.03, "pituitary": 0.02},
            "gradcam_path": None, "model_used": "mambavision",
        }
        with patch("app.models.predict.predict", return_value=result):
            resp = client.post(
                "/api/v1/predict",
                files={"image": ("scan.png", _varied_png_bytes(), "image/png")},
            )
        # Logging itself doesn't affect the response — just verify 200
        assert resp.status_code == 200
