# API Reference

Brain Tumour Detection — complete REST API documentation for the AI Service.

**Base URL:** `http://localhost:8000/api/v1`  
**Interactive docs:** http://localhost:8000/docs (Swagger UI) | http://localhost:8000/redoc (ReDoc)

---

## Table of Contents

1. [Authentication](#authentication)
2. [System](#system)
3. [Prediction](#prediction)
4. [Training](#training)
5. [Dataset Management](#dataset-management)
6. [Preprocessing](#preprocessing)
7. [Metrics & Dashboard](#metrics--dashboard)
8. [Performance Monitoring](#performance-monitoring)
9. [Auth Endpoints](#auth-endpoints)
10. [Error Responses](#error-responses)
11. [Rate Limits](#rate-limits)

---

## Authentication

Most endpoints accept unauthenticated requests in development mode. In production, sensitive endpoints require a JWT bearer token.

### Obtaining a token

```
POST /api/v1/auth/login
```

Include the token in subsequent requests:
```
Authorization: Bearer <access_token>
```

Tokens expire after 30 minutes. Use `POST /api/v1/auth/refresh` to renew.

---

## System

### GET /health

Liveness probe — returns server metadata and per-model availability.

**Authentication:** None required  
**Rate limit:** None

**Response 200:**
```json
{
  "success": true,
  "status": "ok",
  "service": "Brain Tumour Detection AI Service",
  "version": "1.0.0",
  "timestamp": "2026-07-14T12:00:00Z",
  "environment": "production",
  "active_model": "mambavision",
  "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
  "image_size": 224,
  "python_version": "3.12.0",
  "models_available": {
    "mambavision": true,
    "cnn": false,
    "vgg16": false,
    "resnet50": false,
    "efficientnet": false
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/health
```

---

## Prediction

### POST /predict

Run inference on a single MRI image. Returns class probabilities and a Grad-CAM heatmap.

**Authentication:** Optional (controlled by `PREDICTION_AUTH_MODE` env var)  
**Rate limit:** 60 requests / minute  
**Content-Type:** `multipart/form-data`

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | Yes | MRI image file (JPEG, PNG, BMP, TIFF) |
| `model_name` | string | No | Override active model (`mambavision`, `cnn`, `vgg16`, `resnet50`, `efficientnet`) |
| `include_gradcam` | boolean | No | Include Grad-CAM base64 (default: `true`) |

**Response 200:**
```json
{
  "success": true,
  "data": {
    "prediction": "glioma",
    "confidence": 0.9423,
    "probabilities": {
      "glioma":     0.9423,
      "meningioma": 0.0312,
      "notumor":    0.0189,
      "pituitary":  0.0076
    },
    "model_used":   "mambavision",
    "image_size":   [224, 224],
    "inference_ms": 47.3,
    "gradcam_b64":  "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

**Error responses:**

| Status | Code | Reason |
|---|---|---|
| 400 | INVALID_IMAGE | File is not a valid image |
| 404 | MODEL_NOT_FOUND | Requested model has no trained weights |
| 422 | VALIDATION_ERROR | Missing or invalid fields |
| 429 | RATE_LIMITED | Too many requests |

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@brain_mri.jpg"
```

---

### POST /predict/batch

Run inference on multiple images. Accepts individual files or a ZIP archive.

**Authentication:** Optional  
**Rate limit:** 10 requests / minute  
**Content-Type:** `multipart/form-data`

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | binary[] | One of | Individual image files (up to 50) |
| `zip_file` | binary | One of | ZIP archive containing image files |
| `model_name` | string | No | Override active model |

**Response 200:**
```json
{
  "success": true,
  "data": {
    "total":     3,
    "completed": 3,
    "failed":    0,
    "elapsed_ms": 142.5,
    "results": [
      {
        "filename":   "scan_001.jpg",
        "prediction": "glioma",
        "confidence": 0.9423,
        "probabilities": { "glioma": 0.9423, "meningioma": 0.031, "notumor": 0.019, "pituitary": 0.008 },
        "inference_ms": 44.1,
        "error": null
      },
      {
        "filename":   "scan_002.jpg",
        "prediction": "notumor",
        "confidence": 0.9812,
        "probabilities": { "glioma": 0.003, "meningioma": 0.007, "notumor": 0.981, "pituitary": 0.009 },
        "inference_ms": 41.8,
        "error": null
      }
    ]
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -F "files=@scan1.jpg" \
  -F "files=@scan2.jpg"
```

---

## Training

### POST /train

Train a model synchronously. Blocks until training completes.

**Authentication:** Required (Researcher, Operator, or Admin role)  
**Rate limit:** 5 requests / minute  
**Content-Type:** `application/json`

**Request body:**

```json
{
  "model_name":       "mambavision",
  "epochs":           30,
  "batch_size":       32,
  "learning_rate":    0.0001,
  "dataset_dir":      null,
  "fine_tune":        true,
  "fine_tune_layers": 20,
  "fine_tune_epochs": 10
}
```

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model_name` | string | `"mambavision"` | `mambavision`, `cnn`, `vgg16`, `resnet50`, `efficientnet` | Architecture |
| `epochs` | int | `30` | 1–500 | Training epochs |
| `batch_size` | int | `32` | 1–256 | Images per batch |
| `learning_rate` | float | `0.0001` | (0, 1) | Initial learning rate |
| `dataset_dir` | string | `null` | — | Override dataset directory |
| `fine_tune` | bool | `true` | — | Enable Phase 2 backbone fine-tuning |
| `fine_tune_layers` | int | `20` | 1–200 | Backbone layers to unfreeze in Phase 2 |
| `fine_tune_epochs` | int | `10` | 1–200 | Additional epochs for Phase 2 |

**Response 200:**
```json
{
  "success": true,
  "message": "Training complete",
  "data": {
    "model_name":     "mambavision",
    "epochs_run":     30,
    "best_val_acc":   0.9847,
    "best_val_loss":  0.0512,
    "training_time_s": 842.3,
    "history": {
      "accuracy":     [0.72, 0.84, 0.91, "..."],
      "val_accuracy": [0.68, 0.81, 0.89, "..."],
      "loss":         [0.91, 0.52, 0.31, "..."],
      "val_loss":     [1.02, 0.61, 0.38, "..."]
    }
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mambavision", "epochs": 30, "fine_tune": true}'
```

---

### POST /train/start

Start an async training job. Returns immediately with a `job_id`.

**Authentication:** Required  
**Rate limit:** 5 requests / minute

**Request body:** Same as `POST /train`

**Response 202:**
```json
{
  "success":    true,
  "job_id":     "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message":    "Training job started",
  "status":     "pending",
  "started_at": "2026-07-14T12:00:00Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/train/start \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "resnet50", "epochs": 50}'
```

---

### GET /train/status/{job_id}

Poll an async training job.

**Authentication:** Required  
**Path parameter:** `job_id` — UUID returned by `/train/start`

**Response 200:**
```json
{
  "success": true,
  "data": {
    "job_id":       "a1b2c3d4-...",
    "status":       "running",
    "model_name":   "resnet50",
    "current_epoch": 12,
    "total_epochs":  50,
    "current_acc":   0.8741,
    "current_loss":  0.3821,
    "started_at":    "2026-07-14T12:00:00Z",
    "elapsed_s":     183.4,
    "error":         null
  }
}
```

**Status values:** `pending`, `running`, `completed`, `failed`

**Example:**
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/train/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

### GET /train/experiments

List all experiment runs.

**Authentication:** Required  
**Response 200:**
```json
{
  "success": true,
  "data": {
    "experiments": [
      {
        "experiment_id": "a1b2c3d4-...",
        "model_name":    "mambavision",
        "status":        "completed",
        "best_val_acc":  0.9847,
        "epochs_run":    30,
        "started_at":    "2026-07-14T10:00:00Z",
        "completed_at":  "2026-07-14T10:14:02Z"
      }
    ],
    "total": 1
  }
}
```

---

### GET /train/experiments/{experiment_id}

Get full details for one experiment including per-epoch history.

**Authentication:** Required

**Response 200:**
```json
{
  "success": true,
  "data": {
    "experiment_id": "a1b2c3d4-...",
    "model_name":    "mambavision",
    "config": {
      "epochs": 30, "batch_size": 32, "learning_rate": 0.0001
    },
    "status":       "completed",
    "best_val_acc": 0.9847,
    "history": {
      "accuracy":     [...],
      "val_accuracy": [...],
      "loss":         [...],
      "val_loss":     [...]
    },
    "started_at":   "2026-07-14T10:00:00Z",
    "completed_at": "2026-07-14T10:14:02Z"
  }
}
```

---

### POST /evaluate

Evaluate a trained model on the test dataset split.

**Authentication:** Required  
**Content-Type:** `application/json`

**Request body:**
```json
{
  "model_name":  "mambavision",
  "batch_size":  32,
  "dataset_dir": null
}
```

**Response 200:**
```json
{
  "success": true,
  "message": "Evaluation complete",
  "data": {
    "model_name":  "mambavision",
    "test_acc":    0.9847,
    "test_loss":   0.0512,
    "num_samples": 394,
    "confusion_matrix": [
      [98, 1, 0, 1],
      [0, 97, 2, 1],
      [0, 1, 99, 0],
      [1, 0, 0, 99]
    ],
    "class_names":      ["glioma", "meningioma", "notumor", "pituitary"],
    "per_class_acc":    [0.98, 0.97, 0.99, 0.99]
  }
}
```

---

## Dataset Management

### GET /dataset/info

Return saved dataset metadata.

**Authentication:** Optional  
**Response 200:**
```json
{
  "success": true,
  "data": {
    "total_images": 5712,
    "split": {
      "train": 3998,
      "val":   857,
      "test":  857
    },
    "class_counts": {
      "glioma":     1426,
      "meningioma": 1339,
      "notumor":    1595,
      "pituitary":  1352
    },
    "class_weights": {
      "glioma":     1.0,
      "meningioma": 1.07,
      "notumor":    0.89,
      "pituitary":  1.05
    },
    "prepared_at": "2026-07-14T09:00:00Z"
  }
}
```

---

### POST /dataset/validate

Validate the raw dataset directory structure.

**Authentication:** Optional

**Response 200:**
```json
{
  "success": true,
  "data": {
    "valid":    true,
    "warnings": [],
    "errors":   [],
    "summary": {
      "training_classes_found":   ["glioma", "meningioma", "notumor", "pituitary"],
      "testing_classes_found":    ["glioma", "meningioma", "notumor", "pituitary"],
      "missing_classes":          [],
      "total_training_images":    5712,
      "total_testing_images":     1311,
      "unsupported_files":        0
    }
  }
}
```

---

### POST /dataset/prepare

Split the raw dataset into train/val/test partitions.

**Authentication:** Required  
**Content-Type:** `application/json`

**Request body:**
```json
{
  "train_ratio":  0.70,
  "val_ratio":    0.15,
  "test_ratio":   0.15,
  "random_seed":  42,
  "stratify":     true
}
```

**Response 200:**
```json
{
  "success": true,
  "message": "Dataset prepared successfully",
  "data": {
    "train": 3998,
    "val":   857,
    "test":  857,
    "total": 5712,
    "output_dir": "/path/to/dataset/processed"
  }
}
```

---

## Preprocessing

### POST /preprocess/quality-check

Check an image for quality issues before inference.

**Authentication:** Optional  
**Content-Type:** `multipart/form-data`

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | Yes | Image to check |

**Response 200:**
```json
{
  "success": true,
  "data": {
    "passed":       true,
    "blur_score":   142.7,
    "blur_ok":      true,
    "intensity_min": 12,
    "intensity_max": 248,
    "intensity_ok": true,
    "variance":     1847.3,
    "variance_ok":  true,
    "warnings":     [],
    "file_size_kb": 87.4,
    "dimensions":   [512, 512]
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/preprocess/quality-check \
  -F "file=@scan.jpg"
```

---

### POST /preprocess/preview

Preview the result of the preprocessing pipeline.

**Authentication:** Optional  
**Content-Type:** `multipart/form-data`

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | Yes | Image to preprocess |

**Response 200:**
```json
{
  "success": true,
  "data": {
    "original_size":    [512, 512],
    "processed_size":   [224, 224],
    "preview_b64":      "data:image/png;base64,...",
    "pipeline_steps":   ["resize", "denoise", "clahe", "normalize"],
    "processing_ms":    8.3
  }
}
```

---

## Metrics & Dashboard

All dashboard endpoints require authentication.

### GET /dashboard/overview

Composite dashboard snapshot covering all subsystems.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "generated_at": "2026-07-14T12:00:00Z",
    "system": {
      "cpu_percent":  23.4,
      "ram_used_mb":  2341,
      "ram_total_mb": 16384,
      "disk_used_gb": 42.1,
      "disk_total_gb": 500
    },
    "inference": {
      "total_predictions": 1247,
      "success_rate":       0.998,
      "avg_latency_ms":     48.2,
      "p95_latency_ms":     73.1,
      "class_distribution": {
        "glioma": 312, "meningioma": 287, "notumor": 441, "pituitary": 207
      }
    },
    "training": {
      "jobs_run":      4,
      "best_val_acc":  0.9847,
      "last_run_at":   "2026-07-14T10:14:02Z"
    }
  }
}
```

---

### GET /dashboard/system

System resource metrics only.

### GET /dashboard/inference

Inference metrics only.

### GET /dashboard/training

Training metrics only.

### GET /dashboard/history

Rolling time-series history (last 100 data points per metric).

---

## Performance Monitoring

All performance endpoints require authentication. Write operations require Admin or Operator role.

### GET /performance/summary

Full JSON performance report across all subsystems.

**Response 200:** Composite report including system, inference, cache, memory, API stats, profiler, and concurrency data.

---

### GET /performance/report/html

Self-contained HTML performance report.

**Response 200:** `text/html` — open in browser or embed in an iframe.

---

### GET /performance/profiler

Function-level profiler summary.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top` | int | `20` | Number of functions to return (max 200) |

**Response 200:**
```json
{
  "success": true,
  "data": {
    "total_functions": 47,
    "functions": [
      {
        "label":   "preprocess_image",
        "calls":   1247,
        "avg_ms":  8.3,
        "max_ms":  42.1,
        "total_ms": 10350.1
      }
    ]
  }
}
```

---

### GET /performance/memory

Memory usage and leak-detection report.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "current_rss_mb":  1247.3,
    "warning_count":   0,
    "operations": [
      {
        "label":     "load_model",
        "delta_mb":  214.7,
        "timestamp": "2026-07-14T10:01:00Z"
      }
    ],
    "top_allocations": []
  }
}
```

---

### GET /performance/cache

Cache hit/miss/eviction statistics.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "model_cache": {
      "hits": 1241, "misses": 6, "evictions": 0, "hit_rate": 0.9952
    },
    "prediction_cache": {
      "hits": 87, "misses": 1160, "hit_rate": 0.0698
    },
    "recommendations": []
  }
}
```

---

### GET /performance/api-stats

Per-endpoint request latency and RPS statistics.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `slow_only` | bool | `false` | Return only endpoints where p95 > 500ms |

---

### GET /performance/concurrency

Last concurrency/stress-test report.

---

### POST /performance/benchmark/run

Run the full benchmark suite.

**Authentication:** Admin or Operator  
**Content-Type:** `application/json`

**Request body:**
```json
{
  "n_inference":  10,
  "n_preprocess": 20,
  "n_cache":      50,
  "batch_sizes":  [4, 8, 16],
  "background":   false
}
```

**Response 200:**
```json
{
  "success": true,
  "message": "Benchmark suite completed. 8/8 benchmarks passed.",
  "background": false,
  "data": {
    "generated_at": "2026-07-14T12:05:00Z",
    "total_ms":     4823.1,
    "benchmarks": [
      {
        "name":    "single_inference",
        "status":  "ok",
        "n":       10,
        "avg_ms":  48.2,
        "min_ms":  41.3,
        "max_ms":  73.1,
        "p95_ms":  67.4
      }
    ]
  }
}
```

---

### GET /performance/benchmark/result

Retrieve the last benchmark suite result.

---

### POST /performance/benchmark/single

Run a single named benchmark.

**Available benchmark names:** `preprocessing`, `image_quality_check`, `single_inference`, `cache_get_hit`, `cache_stats_call`, `dataset_metadata`, `system_metrics`, `inference_metrics`

**Request body:**
```json
{
  "name": "single_inference",
  "n":    20
}
```

---

### DELETE /performance/profiler/reset

Clear all profiler and API-optimizer data. Admin only.

---

## Auth Endpoints

### POST /auth/login

Authenticate and receive a token pair.

**Rate limit:** 5 requests / minute (per IP)  
**Content-Type:** `application/json`

**Request body:**
```json
{
  "username": "admin",
  "password": "secret"
}
```

**Response 200:**
```json
{
  "access_token":  "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type":    "bearer",
  "expires_in":    1800,
  "user": {
    "user_id":  "uuid",
    "username": "admin",
    "email":    "admin@example.com",
    "role":     "admin",
    "is_active": true
  }
}
```

**Error 401:** Invalid credentials  
**Error 403:** Account locked (too many failed attempts)

---

### POST /auth/logout

Revoke the current access token (and optionally the refresh token).

**Authentication:** Required

**Request body (optional):**
```json
{
  "refresh_token": "eyJhbGci..."
}
```

**Response 200:**
```json
{ "message": "Logged out successfully." }
```

---

### POST /auth/refresh

Exchange a refresh token for a new access token.

**Rate limit:** 30 requests / minute  
**Content-Type:** `application/json`

**Request body:**
```json
{ "refresh_token": "eyJhbGci..." }
```

**Response 200:**
```json
{
  "access_token": "eyJhbGci...",
  "token_type":   "bearer",
  "expires_in":   1800
}
```

---

### GET /auth/me

Return the authenticated user's profile.

**Authentication:** Required

**Response 200:**
```json
{
  "success": true,
  "data": {
    "user_id":    "uuid",
    "username":   "admin",
    "email":      "admin@example.com",
    "role":       "admin",
    "is_active":  true,
    "created_at": "2026-07-01T00:00:00Z"
  }
}
```

---

### POST /auth/change-password

Change the authenticated user's password.

**Authentication:** Required  
**Content-Type:** `application/json`

**Request body:**
```json
{
  "current_password": "old-password",
  "new_password":     "NewStr0ng!Pass"
}
```

**Response 200:**
```json
{ "message": "Password changed successfully." }
```

**Error 400:** Incorrect current password  
**Error 422:** New password too weak

---

### GET /auth/users

List all user accounts. Admin only.

**Response 200:**
```json
{
  "users": [ { "user_id": "...", "username": "admin", "role": "admin" } ],
  "total": 1
}
```

---

### POST /auth/users

Create a new user account. Admin only.

**Request body:**
```json
{
  "username": "researcher1",
  "email":    "researcher1@example.com",
  "password": "SecurePass123!",
  "role":     "researcher"
}
```

**Response 201:**
```json
{
  "success": true,
  "data": { "user_id": "uuid", "username": "researcher1", "role": "researcher" }
}
```

**Error 409:** Username already exists

---

### POST /auth/users/{user_id}/unlock

Unlock a locked user account. Admin only.

**Response 200:**
```json
{ "message": "Account 'researcher1' has been unlocked." }
```

---

## Error Responses

All error responses follow this format:

```json
{
  "success": false,
  "error": {
    "code":    422,
    "message": "Validation error",
    "detail":  "field 'epochs' must be >= 1"
  }
}
```

Or for FastAPI validation errors:
```json
{
  "detail": [
    {
      "loc":   ["body", "epochs"],
      "msg":   "Input should be greater than or equal to 1",
      "type":  "greater_than_equal"
    }
  ]
}
```

### HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 201 | Created |
| 400 | Bad request — invalid input |
| 401 | Unauthorised — missing or invalid token |
| 403 | Forbidden — insufficient role |
| 404 | Not found |
| 409 | Conflict — resource already exists |
| 422 | Unprocessable entity — schema validation failed |
| 429 | Too many requests — rate limit exceeded |
| 500 | Internal server error |

---

## Rate Limits

| Endpoint Group | Limit |
|---|---|
| `POST /auth/login` | 5 / minute per IP |
| `POST /auth/refresh` | 30 / minute per IP |
| `POST /predict` | 60 / minute per IP |
| `POST /predict/batch` | 10 / minute per IP |
| `POST /train`, `POST /train/start` | 5 / minute per IP |
| `GET /dashboard/*`, `GET /performance/*` | 120 / minute per IP |
| All other auth endpoints | 30 / minute per IP |

When a rate limit is exceeded, the response is:
```json
{
  "error": "Rate limit exceeded: 5 per 1 minute"
}
```

With header: `Retry-After: 60`
