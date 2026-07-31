"""
routes.py — FastAPI route definitions for the AI service.

Endpoints
---------
GET  /health                              Liveness probe.
POST /predict                             Run inference on an MRI image.
POST /glcm                                Extract GLCM texture features from an image.
POST /train                               Train a model (synchronous, legacy).
POST /evaluate                            Evaluate a trained model.
GET  /dataset/info                        Return saved dataset metadata.
POST /dataset/validate                    Validate raw dataset structure.
POST /dataset/prepare                     Split and index the dataset.
POST /preprocess/quality-check            Image quality check.
POST /preprocess/preview                  Preprocessing preview.

Training v2 (async, experiment-tracked)
-----------------------------------------
POST /train/start                         Start an async training job.
GET  /train/status/{job_id}              Poll job status.
GET  /train/experiments                   List all experiment runs.
GET  /train/experiments/{experiment_id}  Get full experiment metadata.
"""


import io
import platform
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.security.audit import AuditEvent, log_audit
from app.security.auth import UserInDB
from app.security.dependencies import (
    get_current_active_user,
    optional_auth,
    require_roles,
)
from app.security.rate_limit import limiter, limits
from app.security.roles import Role

router = APIRouter()


# ─── Shared response / request schemas ───────────────────────────────────────

class HealthResponse(BaseModel):
    success: bool
    status: str
    service: str
    version: str
    timestamp: str
    environment: str
    active_model: str
    class_names: List[str]
    image_size: int
    python_version: str
    models_available: Dict[str, bool]


class PredictionResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class TrainRequest(BaseModel):
    model_name: str = Field(
        default="mambavision",
        description="Architecture to train: mambavision | cnn | vgg16 | resnet50 | efficientnet",
    )
    epochs: int          = Field(default=30,    ge=1,   le=500)
    batch_size: int      = Field(default=32,    ge=1,   le=256)
    learning_rate: float = Field(default=1e-4,  gt=0.0, lt=1.0)
    dataset_dir: Optional[str] = Field(
        default=None,
        description="Override the dataset directory path.",
    )
    fine_tune: bool       = Field(default=True,  description="Run Phase-2 fine-tuning after head training.")
    fine_tune_layers: int = Field(default=20, ge=1, le=200, description="Backbone layers to unfreeze in Phase 2.")
    fine_tune_epochs: int = Field(default=10, ge=1, le=200, description="Max additional epochs for Phase 2.")


class TrainResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class EvaluateRequest(BaseModel):
    model_name: Optional[str] = Field(
        default=None,
        description="Architecture to evaluate. Defaults to settings.active_model.",
    )
    batch_size: int = Field(default=32, ge=1, le=256)
    dataset_dir: Optional[str] = Field(
        default=None,
        description="Override the dataset directory for evaluation.",
    )


class EvaluateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# ─── GET /health ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """
    Returns 200 with server metadata and a per-model availability map.

    Used by the Node.js backend and Docker health-check to confirm the
    AI service is reachable and correctly configured.
    """
    from app.models.load_model import is_model_available

    supported = ["mambavision", "cnn", "vgg16", "resnet50", "efficientnet"]
    models_available = {m: is_model_available(m) for m in supported}

    return HealthResponse(
        success=True,
        status="ok",
        service="Brain Tumour Detection AI Service",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=settings.ai_service_env,
        active_model=settings.active_model,
        class_names=settings.classes,
        image_size=settings.image_size,
        python_version=platform.python_version(),
        models_available=models_available,
    )


# ─── POST /predict ────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run inference on an MRI image",
    tags=["Inference"],
)
@limiter.limit(limits.PREDICTION)
async def predict_endpoint(
    request: Request,
    image: UploadFile = File(..., description="MRI scan — JPEG or PNG"),
    model_name: Optional[str] = Form(
        default=None,
        description="Override the active model (cnn | vgg16 | resnet50 | efficientnet)",
    ),
    generate_gradcam: bool = Form(
        default=True,
        description="Generate a Grad-CAM heatmap overlay.",
    ),
    current_user: Optional[UserInDB] = Depends(optional_auth),
) -> PredictionResponse:
    """
    Accept an MRI image upload and return a tumour classification result.

    **Request** — multipart/form-data:
    - `image`           : JPEG or PNG file field
    - `model_name`      : optional string to override the active model
    - `generate_gradcam`: bool (default true)

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "class":         "glioma",
        "confidence":    0.94,
        "probabilities": {"glioma": 0.94, "meningioma": 0.03, ...},
        "gradcam_path":  "/path/to/gradcam/<image_id>.png",
        "model_used":    "mambavision"
      }
    }
    ```
    """
    # ── Validate content type ─────────────────────────────────────────────────
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{image.content_type}'. "
                "Only image/jpeg and image/png are accepted."
            ),
        )

    # ── Read bytes ────────────────────────────────────────────────────────────
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    image_id = str(uuid.uuid4())

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        from app.models.predict import predict
        result = predict(
            image_bytes,
            model_name=model_name,
            generate_gradcam=generate_gradcam,
            image_id=image_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Prediction failed for image_id={image_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    # Audit log for authenticated predictions
    if current_user is not None:
        log_audit(
            AuditEvent.PREDICT_SINGLE,
            username=current_user.username,
            user_id=current_user.user_id,
            endpoint="/predict",
            outcome="success",
            details={"image_id": image_id, "model": model_name or settings.active_model},
        )

    return PredictionResponse(success=True, data=result)


# ─── POST /train ──────────────────────────────────────────────────────────────

@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Train a model",
    tags=["Training"],
)
@limiter.limit(limits.TRAINING)
async def train_endpoint(
    request: Request,
    body: TrainRequest,
    current_user: UserInDB = Depends(require_roles(Role.ADMIN, Role.RESEARCHER)),
) -> TrainResponse:
    """
    Start a model training job (runs synchronously in this process).

    For long-running production jobs, wire this into a task queue
    (Celery / RQ) and return a job ID immediately.

    **Request body** (JSON):
    ```json
    {
      "model_name":       "efficientnet",
      "epochs":           30,
      "batch_size":       32,
      "learning_rate":    0.0001,
      "fine_tune":        true,
      "fine_tune_layers": 20,
      "fine_tune_epochs": 10
    }
    ```
    """
    logger.info(
        f"Training request | model={body.model_name} epochs={body.epochs} "
        f"batch={body.batch_size} lr={body.learning_rate}"
    )

    log_audit(
        AuditEvent.TRAIN_START,
        username=current_user.username,
        user_id=current_user.user_id,
        endpoint="/train",
        outcome="initiated",
        details={"model": body.model_name, "epochs": body.epochs},
    )

    try:
        from app.models.train import train_model
        result = train_model(
            model_name=body.model_name,
            epochs=body.epochs,
            batch_size=body.batch_size,
            learning_rate=body.learning_rate,
            dataset_dir=body.dataset_dir,
            fine_tune=body.fine_tune,
            fine_tune_layers=body.fine_tune_layers,
            fine_tune_epochs=body.fine_tune_epochs,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Training failed for model={body.model_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        )

    return TrainResponse(
        success=True,
        message=(
            f"Model '{body.model_name}' trained successfully "
            f"in {result.get('training_duration_s', 0):.1f}s. "
            f"Val accuracy: {result.get('final_val_accuracy', 0):.4f}"
        ),
        data=result,
    )


# ─── POST /evaluate ───────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a trained model",
    tags=["Training"],
)
async def evaluate_endpoint(
    body: EvaluateRequest,
    current_user: UserInDB = Depends(require_roles(Role.ADMIN, Role.RESEARCHER)),
) -> EvaluateResponse:
    """
    Run the evaluation loop on the dataset and return metrics.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "model_name":       "efficientnet",
        "accuracy":         0.973,
        "precision":        0.974,
        "recall":           0.972,
        "f1":               0.973,
        "auc_roc":          0.998,
        "confusion_matrix": [[...], ...],
        "per_class":        {"glioma": {"precision": 0.98, ...}, ...},
        "num_samples":      394,
        "class_names":      ["glioma", "meningioma", "notumor", "pituitary"]
      }
    }
    ```
    """
    model_name = body.model_name or settings.active_model
    logger.info(f"Evaluation request | model={model_name} batch={body.batch_size}")

    try:
        from app.models.evaluate import evaluate_model
        result = evaluate_model(
            model_name=model_name,
            dataset_dir=body.dataset_dir,
            batch_size=body.batch_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Evaluation failed for model={model_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        )

    return EvaluateResponse(
        success=True,
        message=(
            f"Evaluation complete for '{model_name}'. "
            f"Accuracy: {result.get('accuracy', 0):.4f} | "
            f"F1: {result.get('f1', 0):.4f} | "
            f"AUC-ROC: {result.get('auc_roc', 0):.4f}"
        ),
        data=result,
    )


# ─── Dataset schemas ──────────────────────────────────────────────────────────

class DatasetInfoResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""


class DatasetPrepareRequest(BaseModel):
    raw_dir: Optional[str] = Field(
        default=None,
        description="Source dataset root. Defaults to settings.dataset_raw_dir.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Destination for split dataset. Defaults to settings.dataset_processed_dir.",
    )
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_ratio:   float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio:  float = Field(default=0.15, gt=0.0, lt=1.0)
    seed:        int   = Field(default=42, ge=0)
    overwrite:   bool  = Field(default=False, description="Replace existing split.")
    full_stats:  bool  = Field(default=False, description="Compute pixel mean/std (slower).")


class DatasetValidateRequest(BaseModel):
    raw_dir: Optional[str] = Field(
        default=None,
        description="Dataset directory to validate. Defaults to settings.dataset_raw_dir.",
    )
    min_images_per_class: int = Field(default=10, ge=1)


# ─── GET /dataset/info ────────────────────────────────────────────────────────

@router.get(
    "/dataset/info",
    response_model=DatasetInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Return saved dataset metadata",
    tags=["Dataset"],
)
def dataset_info_endpoint(
    processed_dir: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_active_user),
) -> DatasetInfoResponse:
    """
    Return the ``dataset_info.json`` saved by the last ``/dataset/prepare`` call.

    Query params
    ------------
    - `processed_dir` : override the default ``settings.dataset_processed_dir``

    **Response** (when metadata exists):
    ```json
    {
      "success": true,
      "data": {
        "classes":          ["glioma", ...],
        "class_to_index":   {"glioma": 0, ...},
        "total_per_split":  {"train": 2184, "val": 467, "test": 467},
        "total_images":     3118,
        "imbalance_ratio":  1.02,
        "is_balanced":      true,
        ...
      }
    }
    ```

    Returns **404** when no metadata file has been created yet.
    """
    from app.dataset.metadata import load_dataset_info, dataset_info_exists
    from app.dataset.stats import compute_split_stats

    target = processed_dir or str(settings.dataset_processed_dir)

    if not dataset_info_exists(target):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No dataset_info.json found in '{target}'. "
                "Run POST /api/v1/dataset/prepare first."
            ),
        )

    info = load_dataset_info(target)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read dataset_info.json.",
        )

    # Append live split counts for freshness
    try:
        info["live_split_stats"] = compute_split_stats(target)
    except Exception:
        pass  # non-fatal — metadata is still returned

    return DatasetInfoResponse(
        success=True,
        data=info,
        message="Dataset metadata loaded successfully.",
    )


# ─── POST /dataset/validate ───────────────────────────────────────────────────

@router.post(
    "/dataset/validate",
    response_model=DatasetInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate raw dataset structure",
    tags=["Dataset"],
)
def dataset_validate_endpoint(
    body: DatasetValidateRequest,
    current_user: UserInDB = Depends(require_roles(Role.ADMIN, Role.RESEARCHER)),
) -> DatasetInfoResponse:
    """
    Validate the raw dataset directory and return a structured report.

    Does **not** modify any files — safe to call at any time.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "is_valid":        true,
        "classes_found":   ["glioma", "meningioma", "notumor", "pituitary"],
        "class_counts":    {"glioma": 1321, ...},
        "total_images":    3264,
        "errors":          [],
        "warnings":        []
      }
    }
    ```
    """
    from app.dataset.validator import validate_dataset

    raw_dir = body.raw_dir or str(settings.dataset_raw_dir)

    try:
        result = validate_dataset(
            raw_dir,
            min_images_per_class=body.min_images_per_class,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {exc}",
        )

    msg = "Dataset is valid." if result.is_valid else f"Dataset has {len(result.errors)} error(s)."
    return DatasetInfoResponse(
        success=result.is_valid,
        data=result.to_dict(),
        message=msg,
    )


# ─── POST /dataset/prepare ────────────────────────────────────────────────────

@router.post(
    "/dataset/prepare",
    response_model=DatasetInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate, split, and index the dataset",
    tags=["Dataset"],
)
def dataset_prepare_endpoint(
    body: DatasetPrepareRequest,
    current_user: UserInDB = Depends(require_roles(Role.ADMIN, Role.RESEARCHER)),
) -> DatasetInfoResponse:
    """
    Run the full dataset preparation pipeline:

    1. **Validate** raw directory structure.
    2. **Compute** per-class statistics and class weights.
    3. **Split** images into ``train/`` ``val/`` ``test/`` sub-directories.
    4. **Save** ``dataset_info.json`` with class index map and split counts.

    **Request body** (JSON, all fields optional):
    ```json
    {
      "train_ratio": 0.70,
      "val_ratio":   0.15,
      "test_ratio":  0.15,
      "seed":        42,
      "overwrite":   false,
      "full_stats":  false
    }
    ```

    Returns **422** when the split ratios do not sum to 1.0.
    Returns **409 Conflict** when the output directory already exists and
    ``overwrite`` is False.
    Returns **404** when the raw dataset directory is not found.
    """
    from app.dataset import prepare_dataset

    # Validate ratio sum before touching the filesystem
    ratio_sum = body.train_ratio + body.val_ratio + body.test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, "
                f"got {ratio_sum:.4f}."
            ),
        )

    logger.info(
        f"Dataset prepare request | "
        f"raw={body.raw_dir or 'default'} "
        f"out={body.output_dir or 'default'} "
        f"split={body.train_ratio}/{body.val_ratio}/{body.test_ratio} "
        f"seed={body.seed} overwrite={body.overwrite}"
    )

    log_audit(
        AuditEvent.DATASET_PREPARE,
        username=current_user.username,
        user_id=current_user.user_id,
        endpoint="/dataset/prepare",
        outcome="initiated",
        details={"train_ratio": body.train_ratio, "seed": body.seed},
    )

    try:
        result = prepare_dataset(
            raw_dir=body.raw_dir,
            output_dir=body.output_dir,
            train_ratio=body.train_ratio,
            val_ratio=body.val_ratio,
            test_ratio=body.test_ratio,
            seed=body.seed,
            overwrite=body.overwrite,
            full_stats=body.full_stats,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Dataset preparation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset preparation failed: {exc}",
        )

    split  = result["split"]
    totals = split["total_per_split"]
    return DatasetInfoResponse(
        success=True,
        data=result,
        message=(
            f"Dataset prepared in {result['duration_s']}s. "
            f"train={totals['train']} val={totals['val']} test={totals['test']}"
        ),
    )


# ─── Preprocessing schemas ────────────────────────────────────────────────────

class PreprocessQualityResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str = ""


class PreprocessPreviewResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str = ""


class PreprocessConfigOverride(BaseModel):
    """Optional per-request overrides for the preprocessing pipeline."""
    image_size:          Optional[int]   = Field(default=None, ge=16, le=1024)
    apply_denoise:       Optional[bool]  = None
    apply_clahe:         Optional[bool]  = None
    clahe_clip_limit:    Optional[float] = Field(default=None, gt=0.0, le=10.0)
    denoise_kernel_size: Optional[int]   = Field(default=None, ge=3, le=11)
    normalise:           Optional[bool]  = None


def _build_cfg(override: Optional[PreprocessConfigOverride]) -> "PreprocessConfig":
    """Merge a request override onto DEFAULT_CONFIG and return the result."""
    from app.preprocessing.config import DEFAULT_CONFIG, PreprocessConfig
    if override is None:
        return DEFAULT_CONFIG
    fields = {
        k: v for k, v in {
            "image_size":          override.image_size,
            "apply_denoise":       override.apply_denoise,
            "apply_clahe":         override.apply_clahe,
            "clahe_clip_limit":    override.clahe_clip_limit,
            "denoise_kernel_size": override.denoise_kernel_size,
            "normalise":           override.normalise,
        }.items()
        if v is not None
    }
    if not fields:
        return DEFAULT_CONFIG
    base = DEFAULT_CONFIG.to_dict()
    base.update(fields)
    return PreprocessConfig.from_dict(base)


# ─── POST /preprocess/quality-check ──────────────────────────────────────────

@router.post(
    "/preprocess/quality-check",
    response_model=PreprocessQualityResponse,
    status_code=status.HTTP_200_OK,
    summary="Run image quality checks before training or inference",
    tags=["Preprocessing"],
)
@limiter.limit(limits.PREDICTION)
async def preprocess_quality_check(
    request: Request,
    image: UploadFile = File(..., description="MRI scan — JPEG or PNG"),
    image_size:          Optional[int]   = Form(default=None),
    apply_denoise:       Optional[bool]  = Form(default=None),
    apply_clahe:         Optional[bool]  = Form(default=None),
    clahe_clip_limit:    Optional[float] = Form(default=None),
    denoise_kernel_size: Optional[int]   = Form(default=None),
    current_user: UserInDB = Depends(require_roles(
        Role.ADMIN, Role.RESEARCHER, Role.OPERATOR
    )),
) -> PreprocessQualityResponse:
    """
    Validate image quality without modifying or storing anything.

    Checks performed
    ----------------
    - **file_size**      — within the configured limit
    - **dimensions**     — at least ``min_width × min_height`` pixels
    - **channels**       — 1 or 3 channels (decodable as colour)
    - **mean_intensity** — not a blank / near-black / near-white image
    - **sharpness**      — Laplacian variance above blur threshold (warning)
    - **pixel_variance** — not a completely flat / uniform image

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "is_valid":        true,
        "image_width":     512,
        "image_height":    512,
        "file_size_bytes": 87432,
        "checks": [
          {"name": "file_size",      "passed": true,  "value": 87432, ...},
          {"name": "dimensions",     "passed": true,  "value": 512,   ...},
          {"name": "channels",       "passed": true,  "value": 3,     ...},
          {"name": "mean_intensity", "passed": true,  "value": 128.4, ...},
          {"name": "sharpness",      "passed": true,  "value": 312.1, ...},
          {"name": "pixel_variance", "passed": true,  "value": 64.3,  ...}
        ],
        "warnings": [],
        "errors":   []
      }
    }
    ```
    Returns **200** regardless of validity — check ``data.is_valid``.
    Returns **400** for unsupported content type or empty upload.
    """
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{image.content_type}'. Use image/jpeg or image/png.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    override = PreprocessConfigOverride(
        image_size=image_size,
        apply_denoise=apply_denoise,
        apply_clahe=apply_clahe,
        clahe_clip_limit=clahe_clip_limit,
        denoise_kernel_size=denoise_kernel_size,
    )
    cfg = _build_cfg(override)

    try:
        from app.preprocessing.quality import validate_image_quality
        report = validate_image_quality(image_bytes, cfg)
    except Exception as exc:
        logger.exception("Quality check failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality check error: {exc}",
        )

    msg = (
        "Image passed all quality checks."
        if report.is_valid
        else f"Image failed {len(report.errors)} quality check(s): "
             + "; ".join(report.errors[:3])
    )

    return PreprocessQualityResponse(
        success=report.is_valid,
        data=report.to_dict(),
        message=msg,
    )


# ─── POST /preprocess/preview ─────────────────────────────────────────────────

@router.post(
    "/preprocess/preview",
    response_model=PreprocessPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview the preprocessed image and augmentation variants",
    tags=["Preprocessing"],
)
@limiter.limit(limits.PREDICTION)
async def preprocess_preview(
    request: Request,
    image: UploadFile = File(..., description="MRI scan — JPEG or PNG"),
    image_size:          Optional[int]   = Form(default=None),
    apply_denoise:       Optional[bool]  = Form(default=None),
    apply_clahe:         Optional[bool]  = Form(default=None),
    clahe_clip_limit:    Optional[float] = Form(default=None),
    denoise_kernel_size: Optional[int]   = Form(default=None),
    include_augmented:   bool            = Form(default=True,
        description="Include augmentation preview variants (training mode)."),
    n_augmented:         int             = Form(default=4, ge=1, le=12,
        description="Number of augmented variants to generate."),
    current_user: UserInDB = Depends(require_roles(
        Role.ADMIN, Role.RESEARCHER, Role.OPERATOR
    )),
) -> PreprocessPreviewResponse:
    """
    Run the preprocessing pipeline on an uploaded image and return:

    1. **Quality report** — same checks as ``/preprocess/quality-check``.
    2. **Preprocessed image** — base64-encoded PNG after spatial transforms
       (denoise + CLAHE + resize).  No normalisation so it renders correctly
       in a browser ``<img>`` tag.
    3. **Augmentation variants** (optional) — ``n_augmented`` base64-encoded
       PNG images produced by the training augmentation stack, so the user can
       visually verify the augmentation config before training.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "quality":    { "is_valid": true, "checks": [...], ... },
        "config":     { "image_size": 224, "apply_denoise": true, ... },
        "preprocessed_b64": "<base64 PNG>",
        "augmented_b64":    ["<base64 PNG>", ...]
      }
    }
    ```
    Returns **400** for unsupported content type, empty upload, or failed
    quality check (invalid images are not preprocessed further).
    """
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{image.content_type}'. Use image/jpeg or image/png.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    override = PreprocessConfigOverride(
        image_size=image_size,
        apply_denoise=apply_denoise,
        apply_clahe=apply_clahe,
        clahe_clip_limit=clahe_clip_limit,
        denoise_kernel_size=denoise_kernel_size,
    )
    cfg = _build_cfg(override)

    try:
        from app.preprocessing.quality import validate_image_quality
        from app.preprocessing.preprocess import preprocess_for_preview
        from app.preprocessing.augmentation import apply_augmentation
        from app.preprocessing.transforms import encode_image_base64

        # ── 1. Quality check ──────────────────────────────────────────────────
        quality_report = validate_image_quality(image_bytes, cfg)
        if not quality_report.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Image failed quality checks and cannot be previewed: "
                    + "; ".join(quality_report.errors)
                ),
            )

        # ── 2. Preprocessed preview ───────────────────────────────────────────
        preview_rgb = preprocess_for_preview(image_bytes, cfg=cfg)
        preprocessed_b64 = encode_image_base64(preview_rgb)

        # ── 3. Augmentation variants ──────────────────────────────────────────
        augmented_b64: list[str] = []
        if include_augmented:
            variants = apply_augmentation(preview_rgb, n_samples=n_augmented)
            augmented_b64 = [encode_image_base64(v) for v in variants]

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Preview generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preview error: {exc}",
        )

    return PreprocessPreviewResponse(
        success=True,
        data={
            "quality":           quality_report.to_dict(),
            "config":            cfg.to_dict(),
            "preprocessed_b64":  preprocessed_b64,
            "augmented_b64":     augmented_b64,
        },
        message=(
            f"Preview generated. Image: {cfg.image_size}×{cfg.image_size} px. "
            f"Augmented variants: {len(augmented_b64)}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training v2 — async, experiment-tracked
# ─────────────────────────────────────────────────────────────────────────────
#
#   POST /train/start                       → kick off a background training job
#   GET  /train/status/{job_id}             → poll job status
#   GET  /train/experiments                 → list all experiment runs
#   GET  /train/experiments/{experiment_id} → get full experiment metadata
#
# These complement (but do not replace) the existing POST /train endpoint.
# ─────────────────────────────────────────────────────────────────────────────


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TrainStartRequest(BaseModel):
    architecture: str = Field(
        default="mambavision",
        description="Architecture to train: mambavision | cnn | vgg16 | resnet50 | efficientnet",
    )
    epochs: int          = Field(default=30, ge=1, le=500)
    batch_size: int      = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0.0, lt=1.0)
    dataset_dir: Optional[str] = Field(
        default=None,
        description="Processed dataset root (train/val/test inside). Defaults to settings.",
    )
    fine_tune: bool       = Field(default=True)
    fine_tune_layers: int = Field(default=20, ge=1, le=200)
    fine_tune_epochs: int = Field(default=10, ge=1, le=200)
    fine_tune_lr: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    class_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description='Per-class weight map e.g. {"glioma": 1.5, "notumor": 0.8}',
    )
    seed: int = Field(default=42, ge=0)


class TrainStartResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    experiment_id: str


class TrainStatusResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class ExperimentListResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    total: int


class ExperimentDetailResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


# ── POST /train/start ─────────────────────────────────────────────────────────

@router.post(
    "/train/start",
    response_model=TrainStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an async training job",
    tags=["Training v2"],
)
@limiter.limit(limits.TRAINING)
async def train_start_endpoint(
    request: Request,
    body: TrainStartRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(require_roles(Role.ADMIN, Role.RESEARCHER)),
) -> TrainStartResponse:
    """
    Kick off a model training job in the background and return immediately.

    The response includes a ``job_id`` for polling and an ``experiment_id``
    created synchronously before the background thread starts.

    **Request body** (JSON):
    ```json
    {
      "architecture":   "efficientnet",
      "epochs":         30,
      "batch_size":     32,
      "learning_rate":  0.0001,
      "fine_tune":      true,
      "fine_tune_layers": 20,
      "fine_tune_epochs": 10
    }
    ```

    **Response** (202 Accepted):
    ```json
    {
      "success":       true,
      "message":       "Training job queued.",
      "job_id":        "a3f2b1c0...",
      "experiment_id": "efficientnet-20240715-143022-ab12cd34"
    }
    ```
    Poll ``GET /train/status/{job_id}`` for progress.
    """
    from training.config import TrainingConfig
    from training.trainer import Trainer
    from app.training.job_store import get_job_store

    store = get_job_store()

    # Build the config and create the Trainer synchronously so the
    # experiment_id is available immediately in the response.
    try:
        cfg = TrainingConfig(
            architecture=body.architecture,
            epochs=body.epochs,
            batch_size=body.batch_size,
            learning_rate=body.learning_rate,
            dataset_dir=body.dataset_dir,
            fine_tune=body.fine_tune,
            fine_tune_layers=body.fine_tune_layers,
            fine_tune_epochs=body.fine_tune_epochs,
            fine_tune_lr=body.fine_tune_lr,
            class_weights=body.class_weights,
            seed=body.seed,
            image_size=settings.image_size,
            num_classes=settings.num_classes,
            class_names=settings.classes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    trainer = Trainer(cfg)
    job_id = store.create_job(cfg.to_dict())
    store.mark_running(job_id, experiment_id=trainer.experiment_id)

    def _run_job() -> None:
        try:
            result = trainer.run()
            store.mark_completed(job_id, result=result)
            logger.info(
                f"Background job completed | job_id={job_id} "
                f"experiment_id={trainer.experiment_id}"
            )
        except Exception as exc:
            store.mark_failed(job_id, error=str(exc))
            logger.error(
                f"Background job failed | job_id={job_id} error={exc}"
            )

    background_tasks.add_task(_run_job)

    logger.info(
        f"Training job started | job_id={job_id} "
        f"experiment_id={trainer.experiment_id} "
        f"arch={body.architecture}"
    )

    log_audit(
        AuditEvent.TRAIN_START,
        username=current_user.username,
        user_id=current_user.user_id,
        endpoint="/train/start",
        outcome="queued",
        details={"job_id": job_id, "arch": body.architecture, "epochs": body.epochs},
    )

    return TrainStartResponse(
        success=True,
        message=(
            f"Training job queued for '{body.architecture}'. "
            f"Poll GET /api/v1/train/status/{job_id} for progress."
        ),
        job_id=job_id,
        experiment_id=trainer.experiment_id,
    )


# ── GET /train/status/{job_id} ────────────────────────────────────────────────

@router.get(
    "/train/status/{job_id}",
    response_model=TrainStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll background training job status",
    tags=["Training v2"],
)
def train_status_endpoint(
    job_id: str,
    current_user: UserInDB = Depends(require_roles(
        Role.ADMIN, Role.RESEARCHER, Role.OPERATOR
    )),
) -> TrainStatusResponse:
    """
    Return the current status of a training job started via ``POST /train/start``.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "job_id":        "a3f2b1c0...",
        "status":        "running",
        "experiment_id": "efficientnet-20240715-143022-ab12cd34",
        "created_at":    "2024-07-15T14:30:22Z",
        "started_at":    "2024-07-15T14:30:23Z",
        "finished_at":   null,
        "result":        null,
        "error":         null
      }
    }
    ```

    ``status`` is one of: ``queued`` | ``running`` | ``completed`` | ``failed``.
    """
    from app.training.job_store import get_job_store

    store = get_job_store()
    job = store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return TrainStatusResponse(success=True, data=job)


# ── GET /train/experiments ────────────────────────────────────────────────────

@router.get(
    "/train/experiments",
    response_model=ExperimentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all training experiment runs",
    tags=["Training v2"],
)
def list_experiments_endpoint(
    architecture: Optional[str] = None,
    exp_status: Optional[str] = None,
    limit: int = 50,
    current_user: UserInDB = Depends(get_current_active_user),
) -> ExperimentListResponse:
    """
    Return a list of past training runs from the experiment registry.

    Query parameters
    ----------------
    - ``architecture`` : filter by architecture name (e.g. ``efficientnet``).
    - ``exp_status``   : filter by status (``created`` | ``running`` |
                          ``completed`` | ``failed`` | ``interrupted``).
    - ``limit``        : max entries (default 50).

    **Response**:
    ```json
    {
      "success": true,
      "total":   12,
      "data": [
        {
          "experiment_id":     "efficientnet-20240715-143022-ab12cd34",
          "architecture":      "efficientnet",
          "status":            "completed",
          "created_at":        "2024-07-15T14:30:22Z",
          "finished_at":       "2024-07-15T15:02:11Z",
          "duration_s":        1909.4,
          "epochs_trained":    38,
          "best_val_accuracy": 0.973,
          "notes":             ""
        },
        ...
      ]
    }
    ```
    """
    from training.experiment import ExperimentRegistry

    registry = ExperimentRegistry()
    experiments = registry.list_experiments(
        architecture=architecture,
        status=exp_status,
        limit=limit,
    )

    return ExperimentListResponse(
        success=True,
        data=experiments,
        total=len(experiments),
    )


# ── GET /train/experiments/{experiment_id} ────────────────────────────────────

@router.get(
    "/train/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get full metadata for one experiment run",
    tags=["Training v2"],
)
def get_experiment_endpoint(
    experiment_id: str,
    current_user: UserInDB = Depends(get_current_active_user),
) -> ExperimentDetailResponse:
    """
    Return the complete experiment record for *experiment_id*.

    Includes the full training config snapshot, per-epoch history for both
    phases, post-training evaluation metrics, and saved model paths.

    Returns **404** when the experiment is not found.
    """
    from training.experiment import ExperimentRegistry

    registry = ExperimentRegistry()
    data = registry.get(experiment_id)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )

    return ExperimentDetailResponse(success=True, data=data)


# ─────────────────────────────────────────────────────────────────────────────
# Inference v2 — production inference pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
#   POST /predict/image                 → single-image inference (InferencePipeline)
#   POST /predict/batch                 → multi-file upload batch prediction
#   POST /predict/zip                   → ZIP archive batch prediction
#   GET  /models                        → list all models (available + cached)
#   POST /models/reload                 → hot-reload a model from disk
#   GET  /models/active                 → details for the currently active model
#
# These complement (but do not replace) the existing POST /predict endpoint.
# ─────────────────────────────────────────────────────────────────────────────


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class InferencePredictRequest(BaseModel):
    model_name: Optional[str] = Field(
        default=None,
        description="Architecture to use. Defaults to settings.active_model.",
    )
    top_k: int = Field(default=1, ge=1, le=4)
    generate_gradcam: bool = Field(default=False)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class InferencePredictResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class BatchPredictResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class ModelListResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    cache_stats: Dict[str, Any]


class ModelReloadRequest(BaseModel):
    model_name: str = Field(
        description="Architecture to hot-reload: cnn | vgg16 | resnet50 | efficientnet"
    )


class ModelReloadResponse(BaseModel):
    success: bool
    message: str
    model_name: str


class ActiveModelResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


# ── POST /predict/image ───────────────────────────────────────────────────────

@router.post(
    "/predict/image",
    response_model=InferencePredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Single-image inference via InferencePipeline",
    tags=["Inference v2"],
)
@limiter.limit(limits.PREDICTION)
async def inference_predict_image(
    request: Request,
    image: UploadFile = File(..., description="MRI scan — JPEG or PNG"),
    model_name: Optional[str]  = Form(default=None),
    top_k: int                 = Form(default=1, ge=1, le=4),
    generate_gradcam: bool     = Form(default=False),
    confidence_threshold: float = Form(default=0.5, ge=0.0, le=1.0),
    current_user: Optional[UserInDB] = Depends(optional_auth),
) -> InferencePredictResponse:
    """
    Run single-image inference using the production ``InferencePipeline``.

    Returns top-K predictions, a full probability distribution, timing
    information, and an optional Grad-CAM heatmap path.

    **Form fields** (multipart/form-data):
    - ``image``               — JPEG or PNG file upload (required)
    - ``model_name``          — cnn | vgg16 | resnet50 | efficientnet
    - ``top_k``               — number of top predictions (1–4)
    - ``generate_gradcam``    — true/false
    - ``confidence_threshold`` — 0–1

    **Response** (200 OK):
    ```json
    {
      "success": true,
      "data": {
        "image_id":             "...",
        "predicted_class":      "glioma",
        "confidence":           0.9732,
        "is_high_confidence":   true,
        "probabilities":        {"glioma": 0.9732, ...},
        "top_k":                [{"rank": 1, "class_name": "glioma", ...}],
        "timing_ms":            42.1,
        "metadata": {
          "model_name":     "efficientnet",
          "model_version":  "2024-07-15T14:30:22Z",
          "gradcam_path":   "/abs/path/to/overlay.png"
        }
      }
    }
    ```
    """
    from app.inference.config import InferenceConfig
    from app.inference.pipeline import InferencePipeline

    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{image.content_type}'. Use image/jpeg or image/png.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    _model = (model_name or settings.active_model).lower()
    try:
        cfg = InferenceConfig(
            model_name=_model,
            top_k=top_k,
            generate_gradcam=generate_gradcam,
            confidence_threshold=confidence_threshold,
            class_names=settings.classes,
            image_size=settings.image_size,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    pipeline = InferencePipeline(cfg)
    try:
        result = pipeline.predict(image_bytes, source_path=image.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference predict/image failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    return InferencePredictResponse(success=True, data=result.to_dict())


# ── POST /predict/batch ───────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch inference on multiple uploaded images",
    tags=["Inference v2"],
)
@limiter.limit(limits.BATCH_PREDICTION)
async def inference_predict_batch(
    request: Request,
    images: List[UploadFile] = File(..., description="Multiple MRI scans"),
    model_name: Optional[str] = Form(default=None),
    top_k: int                = Form(default=1, ge=1, le=4),
    generate_gradcam: bool    = Form(default=False),
    current_user: Optional[UserInDB] = Depends(optional_auth),
) -> BatchPredictResponse:
    """
    Run batch inference on multiple uploaded image files.

    **Form fields** (multipart/form-data):
    - ``images``      — one or more JPEG/PNG files
    - ``model_name``  — architecture override
    - ``top_k``       — top-K predictions per image
    - ``generate_gradcam`` — generate heatmaps for each image

    **Response** (200 OK):
    ```json
    {
      "success": true,
      "data": {
        "total": 5, "succeeded": 5, "failed": 0,
        "success_rate": 1.0,
        "timing_ms": 312.4,
        "class_distribution": {"glioma": 3, "notumor": 2},
        "results": [...]
      }
    }
    ```
    Returns **400** when no images are provided.
    """
    from app.inference.config import InferenceConfig
    from app.inference.batch import BatchInferenceRunner

    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images provided.",
        )

    _model = (model_name or settings.active_model).lower()
    try:
        cfg = InferenceConfig(
            model_name=_model,
            top_k=top_k,
            generate_gradcam=generate_gradcam,
            class_names=settings.classes,
            image_size=settings.image_size,
            max_workers=min(len(images), 4),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Read all files up front (async I/O)
    sources = []
    for upload in images:
        data = await upload.read()
        if data:
            sources.append((upload.filename or "unknown", data))

    if not sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All uploaded files were empty.",
        )

    runner = BatchInferenceRunner(cfg)
    try:
        result = runner.run(sources, source_type="list")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {exc}",
        )

    return BatchPredictResponse(success=True, data=result.to_dict())


# ── POST /predict/zip ─────────────────────────────────────────────────────────

@router.post(
    "/predict/zip",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch inference on a ZIP archive of images",
    tags=["Inference v2"],
)
@limiter.limit(limits.BATCH_PREDICTION)
async def inference_predict_zip(
    request: Request,
    archive: UploadFile = File(..., description="ZIP archive containing MRI images"),
    model_name: Optional[str] = Form(default=None),
    top_k: int                = Form(default=1, ge=1, le=4),
    generate_gradcam: bool    = Form(default=False),
    current_user: Optional[UserInDB] = Depends(optional_auth),
) -> BatchPredictResponse:
    """
    Run batch inference on all images inside an uploaded ZIP archive.

    **Form fields** (multipart/form-data):
    - ``archive``     — ZIP file containing JPEG/PNG images
    - ``model_name``  — architecture override
    - ``top_k``       — top-K predictions per image
    - ``generate_gradcam`` — generate heatmaps

    Returns **400** when the archive contains no valid images.
    Returns **422** when the uploaded file is not a valid ZIP.
    """
    from app.inference.config import InferenceConfig
    from app.inference.batch import BatchInferenceRunner

    if archive.content_type not in (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ):
        # Be lenient — browsers may send octet-stream for ZIP
        pass

    archive_bytes = await archive.read()
    if not archive_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded archive is empty.",
        )

    # Validate it's actually a ZIP
    if not zipfile.is_zipfile(io.BytesIO(archive_bytes)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid ZIP archive.",
        )

    _model = (model_name or settings.active_model).lower()
    try:
        cfg = InferenceConfig(
            model_name=_model,
            top_k=top_k,
            generate_gradcam=generate_gradcam,
            class_names=settings.classes,
            image_size=settings.image_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Extract images from the in-memory ZIP
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    sources = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        for name in sorted(zf.namelist()):
            suffix = Path(name).suffix.lower()
            if suffix in image_extensions and not name.startswith("__MACOSX"):
                try:
                    sources.append((name, zf.read(name)))
                except Exception as exc:
                    logger.warning(f"Could not read '{name}' from ZIP: {exc}")

    if not sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid image files found in the ZIP archive.",
        )

    runner = BatchInferenceRunner(cfg)
    try:
        result = runner.run(sources, source_type="zip")
    except Exception as exc:
        logger.exception("ZIP inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ZIP inference error: {exc}",
        )

    return BatchPredictResponse(success=True, data=result.to_dict())


# ── GET /models ───────────────────────────────────────────────────────────────

@router.get(
    "/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all models with availability and cache status",
    tags=["Inference v2"],
)
def list_models_endpoint(
    current_user: UserInDB = Depends(get_current_active_user),
) -> ModelListResponse:
    """
    Scan the ``saved_models/`` directory and return per-architecture status.

    **Response** (200 OK):
    ```json
    {
      "success": true,
      "cache_stats": {"capacity": 4, "size": 1, "hit_rate": 0.8, ...},
      "data": [
        {
          "name": "efficientnet", "available": true, "cached": true,
          "model_version": "2024-07-15T14:30:22Z",
          "total_params": 12341232,
          "model_dir": "/abs/path/to/saved_models/efficientnet"
        },
        ...
      ]
    }
    ```
    """
    from app.inference.cache import list_available_models, cache_stats

    return ModelListResponse(
        success=True,
        data=list_available_models(),
        cache_stats=cache_stats(),
    )


# ── POST /models/reload ───────────────────────────────────────────────────────

@router.post(
    "/models/reload",
    response_model=ModelReloadResponse,
    status_code=status.HTTP_200_OK,
    summary="Hot-reload a model from disk into the cache",
    tags=["Inference v2"],
)
def reload_model_endpoint(
    body: ModelReloadRequest,
    request: Request,
    current_user: UserInDB = Depends(require_roles(Role.ADMIN, Role.OPERATOR)),
) -> ModelReloadResponse:
    """
    Evict *model_name* from the LRU cache and reload it fresh from disk.

    Use this after a training run completes to make the updated weights
    available without restarting the server.

    **Request body**:
    ```json
    {"model_name": "efficientnet"}
    ```

    Returns **404** when no saved weights exist for the requested model.
    """
    from app.inference.cache import reload_model
    from app.models.load_model import is_model_available

    name = body.model_name.lower()
    if not is_model_available(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No saved weights found for '{name}'. "
                "Train the model first via POST /api/v1/train/start."
            ),
        )

    try:
        reload_model(name)
    except Exception as exc:
        logger.exception(f"Hot reload failed for '{name}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reload failed: {exc}",
        )

    logger.info(f"Model '{name}' hot-reloaded via API.")
    log_audit(
        AuditEvent.MODEL_RELOAD,
        username=current_user.username,
        user_id=current_user.user_id,
        endpoint="/models/reload",
        outcome="success",
        details={"model_name": name},
    )
    return ModelReloadResponse(
        success=True,
        message=f"Model '{name}' reloaded successfully.",
        model_name=name,
    )


# ── GET /models/active ────────────────────────────────────────────────────────

@router.get(
    "/models/active",
    response_model=ActiveModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Return details for the currently active model",
    tags=["Inference v2"],
)
def active_model_endpoint(
    current_user: UserInDB = Depends(get_current_active_user),
) -> ActiveModelResponse:
    """
    Return metadata for the model currently set as ``settings.active_model``.

    Includes model_info.json content (accuracy, params, training config)
    and cache status.

    Returns **404** when no saved weights exist for the active model.
    """
    from app.inference.cache import cache_stats
    from app.models.load_model import get_model_info, is_model_available

    name = settings.active_model
    if not is_model_available(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No saved weights found for active model '{name}'. "
                "Train the model first via POST /api/v1/train/start."
            ),
        )

    model_info = get_model_info(name)
    stats      = cache_stats()
    cached     = name in stats.get("cached_models", [])

    return ActiveModelResponse(
        success=True,
        data={
            "model_name":    name,
            "available":     True,
            "cached":        cached,
            "model_info":    model_info,
            "cache_stats":   stats,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLCM Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────


class GLCMResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


@router.post(
    "/glcm",
    response_model=GLCMResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract GLCM texture features from an MRI image",
    tags=["Features"],
)
@limiter.limit(limits.PREDICTION)
async def glcm_endpoint(
    request: Request,
    image: UploadFile = File(..., description="MRI scan — JPEG or PNG"),
    current_user: Optional[UserInDB] = Depends(optional_auth),
) -> GLCMResponse:
    """
    Compute 7 GLCM texture features (Eq. 8–14) from the uploaded image.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "entropy":     5.123,
        "correlation": 0.894,
        "energy":      0.041,
        "contrast":    18.32,
        "mean":        3.24,
        "std_dev":     1.89,
        "variance":    3.58
      }
    }
    ```
    """
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{image.content_type}'. Only JPEG and PNG are accepted.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        from app.utils.glcm_features import extract_glcm_features
        features = extract_glcm_features(image_bytes)
    except Exception as exc:
        logger.exception("GLCM extraction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GLCM extraction failed: {exc}",
        )

    return GLCMResponse(success=True, data=features)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard / Metrics — Module 8
# ─────────────────────────────────────────────────────────────────────────────
#
#   GET /dashboard/overview    → composite snapshot of all metric domains
#   GET /dashboard/system      → CPU / RAM / disk / GPU / process metrics
#   GET /dashboard/inference   → prediction counts, latency, class distribution
#   GET /dashboard/training    → job counts, best accuracy, recent runs
#   GET /dashboard/history     → rolling time-series for any metric type
#
# ─────────────────────────────────────────────────────────────────────────────


class DashboardResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class DashboardHistoryResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


# ── GET /dashboard/overview ───────────────────────────────────────────────────

@router.get(
    "/dashboard/overview",
    response_model=DashboardResponse,
    status_code=status.HTTP_200_OK,
    summary="Composite dashboard snapshot",
    tags=["Dashboard"],
)
@limiter.limit(limits.DASHBOARD)
def dashboard_overview_endpoint(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
) -> DashboardResponse:
    """
    Return a single composite payload covering system, inference, training,
    and model-cache metrics plus threshold-based alerts.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "timestamp": "...",
        "system":    { "cpu_percent": 34.1, "ram_percent": 61.2, ... },
        "inference": { "total_predictions": 142, "avg_latency_ms": 38.4, ... },
        "training":  { "total_jobs": 7, "running_jobs": 0, ... },
        "models":    { "capacity": 4, "size": 1, "hit_rate": 0.83, ... },
        "alerts":    []
      }
    }
    ```
    """
    from app.metrics.dashboard import get_dashboard_overview
    from app.metrics.storage import get_metrics_store

    data = get_dashboard_overview()
    # Persist snapshot for history queries
    try:
        get_metrics_store().save_snapshot({"type": "overview", **data})
    except Exception:
        pass

    return DashboardResponse(success=True, data=data)


# ── GET /dashboard/system ─────────────────────────────────────────────────────

@router.get(
    "/dashboard/system",
    response_model=DashboardResponse,
    status_code=status.HTTP_200_OK,
    summary="System resource metrics",
    tags=["Dashboard"],
)
@limiter.limit(limits.DASHBOARD)
def dashboard_system_endpoint(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
) -> DashboardResponse:
    """
    Return a live snapshot of system resource metrics:
    CPU, RAM, disk, GPU (if present), uptime, and process stats.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "timestamp":           "...",
        "uptime_seconds":      3612.4,
        "cpu_percent":         34.1,
        "cpu_per_core":        [28.0, 40.2, ...],
        "ram_used_mb":         4201.3,
        "ram_percent":         61.2,
        "disk_percent":        42.7,
        "gpu_available":       false,
        "process_ram_mb":      412.8,
        ...
      }
    }
    ```
    """
    from app.metrics.system import get_system_metrics
    from app.metrics.storage import get_metrics_store

    data = get_system_metrics()
    try:
        get_metrics_store().save_snapshot({"type": "system", **data})
    except Exception:
        pass

    return DashboardResponse(success=True, data=data)


# ── GET /dashboard/inference ──────────────────────────────────────────────────

@router.get(
    "/dashboard/inference",
    response_model=DashboardResponse,
    status_code=status.HTTP_200_OK,
    summary="Inference / prediction metrics",
    tags=["Dashboard"],
)
@limiter.limit(limits.DASHBOARD)
def dashboard_inference_endpoint(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
) -> DashboardResponse:
    """
    Return aggregated inference statistics accumulated since service start.

    Includes prediction counts, success/failure rates, latency percentiles,
    confidence distribution histogram, class distribution, and recent
    predictions.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "total_predictions": 142,
        "succeeded": 138,
        "failed": 4,
        "success_rate": 0.9718,
        "avg_latency_ms": 38.4,
        "p95_latency_ms": 74.1,
        "confidence_distribution": {
          "buckets": ["<50%", "50–70%", ...],
          "counts":  [2, 5, ...]
        },
        "class_distribution": {"glioma": 67, "notumor": 44, ...},
        "top_classes": [{"class_name": "glioma", "count": 67}, ...],
        "batch_runs": 3,
        "recent_predictions": [...]
      }
    }
    ```
    """
    from app.metrics.inference import get_inference_metrics
    from app.metrics.storage import get_metrics_store

    data = get_inference_metrics()
    try:
        get_metrics_store().save_snapshot({"type": "inference", **data})
    except Exception:
        pass

    return DashboardResponse(success=True, data=data)


# ── GET /dashboard/training ───────────────────────────────────────────────────

@router.get(
    "/dashboard/training",
    response_model=DashboardResponse,
    status_code=status.HTTP_200_OK,
    summary="Training job and experiment metrics",
    tags=["Dashboard"],
)
@limiter.limit(limits.DASHBOARD)
def dashboard_training_endpoint(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
) -> DashboardResponse:
    """
    Return training job counts, best validation accuracy, architecture
    popularity, and a list of recent jobs / experiments.

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "total_jobs":        7,
        "running_jobs":      0,
        "completed_jobs":    6,
        "failed_jobs":       1,
        "best_val_accuracy": 0.9732,
        "architecture_counts": {"efficientnet": 4, "resnet50": 2, ...},
        "recent_jobs": [...],
        "total_experiments": 6,
        "recent_experiments": [...]
      }
    }
    ```
    """
    from app.metrics.training import get_training_metrics
    from app.metrics.storage import get_metrics_store

    data = get_training_metrics()
    try:
        get_metrics_store().save_snapshot({"type": "training", **data})
    except Exception:
        pass

    return DashboardResponse(success=True, data=data)


# ── GET /dashboard/history ────────────────────────────────────────────────────

@router.get(
    "/dashboard/history",
    response_model=DashboardHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Rolling metrics history",
    tags=["Dashboard"],
)
@limiter.limit(limits.DASHBOARD)
def dashboard_history_endpoint(
    request: Request,
    metric_type: str = "system",
    hours: int       = 24,
    current_user: UserInDB = Depends(get_current_active_user),
) -> DashboardHistoryResponse:
    """
    Return a time-series of metric snapshots from the rolling history.

    Query parameters
    ----------------
    - ``metric_type`` : ``system`` | ``inference`` | ``training`` | ``overview``
    - ``hours``       : lookback window (1–168; default 24)

    **Response**:
    ```json
    {
      "success": true,
      "data": {
        "metric_type": "system",
        "hours":       24,
        "count":       288,
        "data":        [{ "timestamp": "...", "cpu_percent": 34.1, ... }, ...]
      }
    }
    ```
    """
    valid_types = {"system", "inference", "training", "overview"}
    if metric_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"metric_type must be one of {sorted(valid_types)}",
        )

    hours = max(1, min(hours, 168))
    from app.metrics.dashboard import get_history_summary

    data = get_history_summary(metric_type=metric_type, hours=hours)
    return DashboardHistoryResponse(success=True, data=data)
