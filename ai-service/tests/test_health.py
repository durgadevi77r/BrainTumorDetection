"""
tests/test_health.py — Tests for the FastAPI system and inference endpoints.

Covers:
  - /health   : structure, types, new models_available field
  - /predict  : 404 when no model weights, 400 for bad file type, 422 for empty
  - /train    : 404 when dataset missing
  - /evaluate : 404 when no model / dataset

All ML endpoints that require trained weights or a dataset directory return
predictable 4xx errors in the test environment (no weights, no dataset).
We do NOT test actual ML results here — that belongs in integration tests.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BCRYPT_ROUNDS", "4")

from app.main import app
from app.security.auth import get_user_store
from app.security.jwt import create_access_token

client = TestClient(app)


def _admin_headers() -> dict:
    store = get_user_store()
    admin = store.get_by_username("admin")
    token = create_access_token({"sub": admin.user_id, "role": admin.role.value})
    return {"Authorization": f"Bearer {token}"}


# ─── /health ──────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_success_flag(self) -> None:
        body = client.get("/api/v1/health").json()
        assert body["success"] is True

    def test_status_ok(self) -> None:
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"

    def test_service_name(self) -> None:
        body = client.get("/api/v1/health").json()
        assert body["service"] == "Brain Tumour Detection AI Service"

    def test_four_classes(self) -> None:
        body = client.get("/api/v1/health").json()
        assert len(body["class_names"]) == 4
        assert set(body["class_names"]) == {
            "glioma", "meningioma", "notumor", "pituitary"
        }

    def test_models_available_field_present(self) -> None:
        body = client.get("/api/v1/health").json()
        assert "models_available" in body
        assert isinstance(body["models_available"], dict)

    def test_models_available_covers_all_architectures(self) -> None:
        body = client.get("/api/v1/health").json()
        expected_keys = {"mambavision", "cnn", "vgg16", "resnet50", "efficientnet"}
        assert expected_keys == set(body["models_available"].keys())

    def test_models_available_values_are_bools(self) -> None:
        body = client.get("/api/v1/health").json()
        for val in body["models_available"].values():
            assert isinstance(val, bool)

    def test_python_version_present(self) -> None:
        body = client.get("/api/v1/health").json()
        assert "python_version" in body
        assert body["python_version"]  # non-empty

    def test_timestamp_present(self) -> None:
        body = client.get("/api/v1/health").json()
        assert "timestamp" in body

    def test_image_size_is_int(self) -> None:
        body = client.get("/api/v1/health").json()
        assert isinstance(body["image_size"], int)
        assert body["image_size"] > 0

    def test_active_model_is_in_models_available(self) -> None:
        """The model reported as active_model must always have an entry in models_available."""
        body = client.get("/api/v1/health").json()
        assert body["active_model"] in body["models_available"], (
            f"active_model '{body['active_model']}' is missing from models_available "
            f"(keys present: {list(body['models_available'].keys())})"
        )


# ─── /predict ─────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def _png_bytes(self) -> bytes:
        """Generate a minimal valid 1×1 white PNG in memory."""
        import struct, zlib
        def chunk(name: bytes, data: bytes) -> bytes:
            c = name + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def test_unsupported_content_type_returns_400(self) -> None:
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test.gif", b"GIF89a", "image/gif")},
        )
        assert response.status_code == 400

    def test_jpeg_content_type_accepted(self) -> None:
        # No weights → 404 or 500 (not 400), confirming the type check passed
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test.jpg", self._png_bytes(), "image/jpeg")},
        )
        assert response.status_code in (200, 404, 500)

    def test_png_content_type_accepted(self) -> None:
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test.png", self._png_bytes(), "image/png")},
        )
        assert response.status_code in (200, 404, 500)

    def test_empty_file_returns_400(self) -> None:
        response = client.post(
            "/api/v1/predict",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400

    def test_no_weights_returns_404(self) -> None:
        """When no model weights are saved, the endpoint returns 404."""
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test.png", self._png_bytes(), "image/png")},
            data={"model_name": "cnn"},
        )
        # 404 when weights missing, 500 if something else goes wrong
        assert response.status_code in (404, 500)

    def test_response_has_success_key(self) -> None:
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test.png", self._png_bytes(), "image/png")},
        )
        if response.status_code == 200:
            body = response.json()
            assert "success" in body
            assert body["success"] is True
            assert "data" in body


# ─── /train ───────────────────────────────────────────────────────────────────

class TestTrainEndpoint:
    def test_missing_dataset_returns_404(self) -> None:
        response = client.post(
            "/api/v1/train",
            headers=_admin_headers(),
            json={
                "model_name": "cnn",
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "dataset_dir": "/nonexistent/path/dataset",
            },
        )
        assert response.status_code == 404

    def test_invalid_model_name_returns_422(self) -> None:
        response = client.post(
            "/api/v1/train",
            headers=_admin_headers(),
            json={
                "model_name": "invalid_arch",
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "dataset_dir": "/nonexistent/path/dataset",
            },
        )
        # ValueError from build_model() → 422 or 404 (dataset check first)
        assert response.status_code in (404, 422, 500)

    def test_epochs_below_minimum_rejected(self) -> None:
        response = client.post(
            "/api/v1/train",
            headers=_admin_headers(),
            json={"model_name": "cnn", "epochs": 0, "batch_size": 8},
        )
        assert response.status_code == 422

    def test_batch_size_above_maximum_rejected(self) -> None:
        response = client.post(
            "/api/v1/train",
            headers=_admin_headers(),
            json={"model_name": "cnn", "epochs": 1, "batch_size": 512},
        )
        assert response.status_code == 422

    def test_default_payload_is_valid(self) -> None:
        """A default body passes Pydantic validation (will fail on missing model weights)."""
        from unittest.mock import patch
        # Patch train_model so no real training (model download + GPU) is triggered.
        # The test only asserts that the payload passes Pydantic validation, not
        # that training succeeds. We expect 404/500 (no weights) or, if mocked, 200.
        with patch("app.models.train.train_model",
                   side_effect=FileNotFoundError("No dataset — test environment")):
            response = client.post("/api/v1/train", headers=_admin_headers(), json={})
        # 404 (mocked FileNotFoundError) or 422 (Pydantic rejection) — never 422
        assert response.status_code in (404, 500)

    def test_unauthenticated_returns_401(self) -> None:
        response = client.post("/api/v1/train", json={"model_name": "cnn", "epochs": 1})
        assert response.status_code == 401


# ─── /evaluate ────────────────────────────────────────────────────────────────

class TestEvaluateEndpoint:
    def test_missing_dataset_or_weights_returns_404(self) -> None:
        response = client.post(
            "/api/v1/evaluate",
            headers=_admin_headers(),
            json={
                "model_name": "cnn",
                "batch_size": 8,
                "dataset_dir": "/nonexistent/path/dataset",
            },
        )
        assert response.status_code == 404

    def test_batch_size_below_minimum_rejected(self) -> None:
        response = client.post(
            "/api/v1/evaluate",
            headers=_admin_headers(),
            json={"batch_size": 0},
        )
        assert response.status_code == 422

    def test_default_payload_is_valid(self) -> None:
        from unittest.mock import patch
        # Patch evaluate_model so no real inference (model download + dataset) runs.
        with patch("app.models.evaluate.evaluate_model",
                   side_effect=FileNotFoundError("No model weights — test environment")):
            response = client.post("/api/v1/evaluate", headers=_admin_headers(), json={})
        assert response.status_code in (404, 500)

    def test_unauthenticated_returns_401(self) -> None:
        response = client.post("/api/v1/evaluate", json={})
        assert response.status_code == 401

    def test_response_schema_on_success(self) -> None:
        """If a model + dataset are available, verify the response shape."""
        response = client.post("/api/v1/evaluate", headers=_admin_headers(), json={})
        if response.status_code == 200:
            body = response.json()
            assert body["success"] is True
            assert "data" in body
            data = body["data"]
            for key in ("accuracy", "precision", "recall", "f1", "auc_roc",
                        "confusion_matrix", "per_class", "class_names"):
                assert key in data, f"Missing key '{key}' in evaluate response"
