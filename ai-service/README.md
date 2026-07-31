# Brain Tumour Detection — AI Service

Python / PyTorch service for deep-learning MRI brain tumour classification using the official
MambaVision implementation. Exposes a FastAPI REST API consumed by the Node.js backend.

---

## Table of Contents

1. [Quick start](#quick-start)
2. [Project structure](#project-structure)
3. [Dataset preparation](#dataset-preparation)
4. [Training workflow](#training-workflow)
   - [CLI training](#cli-training)
   - [Makefile targets](#makefile-targets)
   - [Python API](#python-api)
   - [REST API (async)](#rest-api-async)
5. [Hyperparameter reference](#hyperparameter-reference)
6. [Transfer learning & fine-tuning](#transfer-learning--fine-tuning)
7. [Experiment tracking](#experiment-tracking)
8. [Model artefact locations](#model-artefact-locations)
9. [Evaluation](#evaluation)
10. [Inference & prediction](#inference--prediction)
    - [Inference CLI](#inference-cli)
    - [Inference Makefile targets](#inference-makefile-targets)
    - [Python API (inference)](#python-api-inference)
    - [REST API (inference v2)](#rest-api-inference-v2)
    - [Model management](#model-management)
    - [Expected response schemas](#expected-response-schemas)
11. [Running the API server](#running-the-api-server)
12. [Running tests](#running-tests)
13. [Metrics & Monitoring Dashboard](#metrics--monitoring-dashboard)

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset (raw → train/val/test split)
python scripts/prepare_dataset.py

# 3. Train the default model (MambaVision-T, 30 epochs)
python -m training.trainer

# 4. Evaluate the trained model
make evaluate

# 5. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project structure

```
ai-service/
├── app/
│   ├── api/
│   │   └── routes.py          # All FastAPI routes (training v2 + inference v2 + auth)
│   ├── core/
│   │   ├── config.py          # Pydantic-Settings singleton (active_model="mambavision")
│   │   └── logging.py
│   ├── dataset/               # Dataset validation, splitting, metadata
│   ├── inference/             # Production inference package
│   │   ├── __init__.py        # Public package surface
│   │   ├── config.py          # InferenceConfig dataclass
│   │   ├── pipeline.py        # InferencePipeline + CLI entry point
│   │   ├── batch.py           # BatchInferenceRunner (parallel, CSV/JSON export)
│   │   ├── cache.py           # LRU ModelCache with hot-reload
│   │   └── results.py         # PredictionResult, BatchPredictionResult, etc.
│   ├── models/
│   │   ├── mambavision/       # Official MambaVision sub-package
│   │   │   ├── config.py      # MambaVisionHFConfig
│   │   │   ├── factory.py     # build_mambavision_model()
│   │   │   ├── predictor.py   # TorchImageClassifier wrapper
│   │   │   └── storage.py     # HF local-dir save / load helpers
│   │   ├── architectures.py   # build_model() factory for all 5 architectures
│   │   ├── evaluate.py        # evaluate_model()
│   │   ├── load_model.py      # load_model() with in-memory cache
│   │   ├── predict.py         # predict() single-image inference
│   │   ├── save_model.py      # save_model() to HF local-dir format
│   │   └── train.py           # train_model() two-phase PyTorch training loop
│   ├── preprocessing/         # Preprocess, augmentation, transforms, quality
│   ├── metrics/               # System, inference, training, dashboard metrics
│   ├── performance/           # Profiler, benchmark, cache, memory, reports
│   ├── security/              # JWT, auth, roles, rate limiting, audit log
│   ├── training/              # Training job store, experiment registry
│   └── utils/                 # Grad-CAM (PyTorch), GLCM features
│   └── main.py                # FastAPI application factory
├── training/                  # Core training package (standalone CLI + API)
│   ├── __init__.py
│   ├── config.py              # TrainingConfig dataclass
│   ├── callbacks.py           # build_callbacks() factory
│   ├── checkpoints.py         # Checkpoint save / load / list / delete
│   ├── experiment.py          # Experiment dataclass + ExperimentRegistry
│   └── trainer.py             # Trainer class + train() wrapper + CLI
├── dataset/
│   ├── raw/                   # Original images (one sub-folder per class)
│   └── processed/             # Split dataset (train/ val/ test/)
├── saved_models/              # PyTorch model weights (HF local-dir format)
│   └── mambavision/           # Default model directory
│       ├── config.json
│       ├── model.safetensors
│       ├── model_info.json
│       ├── checkpoints/
│       │   ├── best_phase1.pt
│       │   └── best_phase2.pt
│       └── gradcam/
├── gradcam_output/            # Grad-CAM overlay PNG files
├── logs/
│   ├── experiments/           # Experiment JSON records + registry
│   ├── training/              # Per-epoch CSV logs
│   ├── audit/                 # Security audit log
│   └── metrics/               # Rolling metric snapshots (*.jsonl)
├── tests/
├── Makefile
└── requirements.txt
```

---

## Dataset preparation

Place raw images under `dataset/raw/` with one sub-folder per class:

```
dataset/raw/
    glioma/       *.jpg / *.png
    meningioma/
    notumor/
    pituitary/
```

Then prepare the dataset (validate + stratified split):

```bash
python scripts/prepare_dataset.py
# or
make prepare-dataset
```

The split is written to `dataset/processed/train/`, `val/`, and `test/`.
Default ratios: 70% train / 15% val / 15% test.

---

## Training workflow

### CLI training

```bash
# Default: MambaVision-T, 30 epochs, batch 32, lr 1e-4, Phase-2 fine-tuning enabled
python -m training.trainer

# Override architecture and epochs
python -m training.trainer --architecture resnet50 --epochs 20

# Full flag reference
python -m training.trainer \
    --architecture    mambavision \   # mambavision | cnn | vgg16 | resnet50 | efficientnet
    --epochs          30 \
    --batch-size      32 \
    --learning-rate   1e-4 \
    --fine-tune-layers 20 \
    --fine-tune-epochs 10 \
    --seed            42 \
    --dataset-dir     dataset/processed   # optional override

# Disable Phase-2 fine-tuning
python -m training.trainer --architecture vgg16 --no-fine-tune
```

### Makefile targets

```bash
make train                                  # MambaVision defaults
make train ARCH=resnet50 EPOCHS=20          # ResNet-50
make train ARCH=cnn BS=16 EPOCHS=50         # Lightweight CNN, smaller batch
make train ARCH=efficientnet LR=5e-5        # EfficientNet-B3, lower LR

make evaluate                               # Evaluate active model (mambavision)
make evaluate ARCH=resnet50                 # Evaluate a specific architecture

make predict IMAGE=scan.jpg                 # Single-image inference
make predict-batch DIR=dataset/test/        # Batch inference from directory
make predict-zip ZIP=images.zip             # Batch inference from ZIP

make models                                 # List all models + cache status
make reload-model                           # Hot-reload active model

make test                                   # Run pytest suite
make lint                                   # ruff + black check
make format                                 # auto-format with ruff
make clean                                  # remove __pycache__, *.pyc
```

### Python API

```python
from training import Trainer, TrainingConfig

cfg = TrainingConfig(
    architecture="mambavision",
    epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    fine_tune=True,
    fine_tune_layers=20,
    fine_tune_epochs=10,
)

trainer = Trainer(cfg)
result  = trainer.run()

print(result["experiment_id"])       # "mambavision-20260101-143022-ab12cd34"
print(result["best_val_accuracy"])   # 0.982
print(result["training_duration_s"]) # 2400.5
```

Or use the convenience wrapper:

```python
from training import train

result = train(
    architecture="mambavision",
    epochs=30,
    class_weights={"glioma": 1.5, "notumor": 0.8},  # handle class imbalance
)
```

### REST API (async)

#### Start a training job

```http
POST /api/v1/train/start
Content-Type: application/json

{
  "architecture":     "mambavision",
  "epochs":           30,
  "batch_size":       32,
  "learning_rate":    0.0001,
  "fine_tune":        true,
  "fine_tune_layers": 20,
  "fine_tune_epochs": 10
}
```

Response (202 Accepted):

```json
{
  "success":       true,
  "message":       "Training job queued for 'mambavision'. Poll GET /api/v1/train/status/... for progress.",
  "job_id":        "a3f2b1c0d4e5f6a7b8c9d0e1f2a3b4c5",
  "experiment_id": "mambavision-20260101-143022-ab12cd34"
}
```

#### Poll job status

```http
GET /api/v1/train/status/{job_id}
```

`status` values: `queued` → `running` → `completed` / `failed`

#### List experiments

```http
GET /api/v1/train/experiments
GET /api/v1/train/experiments?architecture=mambavision
GET /api/v1/train/experiments?exp_status=completed&limit=10
```

---

## Hyperparameter reference

| Parameter | Default | Description |
|---|---|---|
| `architecture` | `mambavision` | `mambavision` \| `cnn` \| `vgg16` \| `resnet50` \| `efficientnet` |
| `epochs` | `30` | Maximum Phase-1 epochs |
| `batch_size` | `32` | Mini-batch size |
| `learning_rate` | `1e-4` | Phase-1 Adam learning rate |
| `dropout_rate` | `0.5` | Dropout in the classification head |
| `image_size` | `224` | Input resolution (H = W, pixels) |
| `seed` | `42` | Random seed |
| `class_weights` | `null` | `{"class_name": float}` for imbalanced data |
| `early_stopping_patience` | `10` | Epochs without val_loss improvement before stopping |
| `reduce_lr_patience` | `5` | Epochs before ReduceLROnPlateau fires |
| `reduce_lr_factor` | `0.5` | LR multiplier on plateau |

---

## Transfer learning & fine-tuning

All architectures except `cnn` use a two-phase transfer learning strategy:

**Phase 1 — head training**
The ImageNet backbone is frozen. Only the classification head is trained.

**Phase 2 — fine-tuning**
The top `fine_tune_layers` of the backbone are unfrozen and trained at
`learning_rate / 10`. For MambaVision, this adapts the Mamba blocks to the MRI domain.

| Parameter | Default | Description |
|---|---|---|
| `fine_tune` | `true` | Enable Phase-2 |
| `fine_tune_layers` | `20` | Backbone layers to unfreeze |
| `fine_tune_epochs` | `10` | Maximum Phase-2 epochs |

To skip fine-tuning:

```bash
python -m training.trainer --no-fine-tune
```

---

## Experiment tracking

Every training run creates an experiment record in `logs/experiments/`:

```
logs/experiments/
    experiment_registry.json                    ← lightweight index (newest first)
    mambavision-20260101-143022-ab12cd34/
        experiment.json                         ← full metadata
        training_config.json                    ← config snapshot
```

---

## Model artefact locations

```
saved_models/
    mambavision/                                ← default (active) model
        config.json                             ← HF AutoModel config
        model.safetensors                       ← serialised PyTorch weights
        model_info.json                         ← accuracy, params, training config
        checkpoints/
            best_phase1.pt                      ← best Phase-1 checkpoint
            best_phase2.pt                      ← best Phase-2 checkpoint
        gradcam/                                ← Grad-CAM heatmap outputs
    efficientnet/                               ← optional alternative model
        config.json
        model.safetensors
        model_info.json
        checkpoints/
```

All models are stored in the **Hugging Face local-directory format** (compatible with
`AutoModelForImageClassification.from_pretrained(path)`). PyTorch `.pt` checkpoints are
stored alongside in the `checkpoints/` sub-directory.

---

## Evaluation

Evaluation runs automatically at the end of training (against `dataset/processed/test/`).
It can also be triggered manually:

```bash
make evaluate

# Python
from app.models.evaluate import evaluate_model
metrics = evaluate_model("mambavision")
print(metrics["accuracy"], metrics["f1"], metrics["auc_roc"])

# REST
curl -X POST /api/v1/evaluate \
     -H "Content-Type: application/json" \
     -d '{"model_name": "mambavision", "batch_size": 32}'
```

Metrics returned: `accuracy`, `precision`, `recall`, `f1` (macro), `auc_roc` (macro OvR),
`confusion_matrix`, `per_class` breakdown.

---

## Inference & prediction

### Inference CLI

```bash
# Single image
python -m app.inference.pipeline scan.jpg

# With model override and Grad-CAM
python -m app.inference.pipeline scan.jpg --model mambavision --top-k 3 --gradcam

# Batch inference
python -m app.inference.pipeline --batch dataset/test/glioma/
python -m app.inference.pipeline --zip images.zip --output-dir output/
```

### Inference Makefile targets

```bash
make predict IMAGE=path/to/scan.jpg
make predict-batch DIR=dataset/test/
make predict-zip ZIP=images.zip
make models
make reload-model
```

### Python API (inference)

```python
from app.inference import predict, InferencePipeline, InferenceConfig

# Convenience function
result = predict("path/to/scan.jpg", model_name="mambavision", top_k=3)
print(result.predicted_class, result.confidence)

# Full pipeline
cfg = InferenceConfig(model_name="mambavision", top_k=3, generate_gradcam=True)
pipeline = InferencePipeline(cfg)
result = pipeline.predict(open("scan.jpg", "rb").read())

print(result.predicted_class)       # "glioma"
print(result.confidence)            # 0.9821
print(result.timing_ms)             # 38.4
```

### REST API (inference v2)

```bash
# Single-image inference
curl -X POST http://localhost:8000/api/v1/predict/image \
     -F "image=@scan.jpg" \
     -F "model_name=mambavision" \
     -F "top_k=3" \
     -F "generate_gradcam=true"

# Batch inference from multiple files
curl -X POST http://localhost:8000/api/v1/predict/batch \
     -F "images=@scan1.jpg" \
     -F "images=@scan2.png"

# Batch from ZIP archive
curl -X POST http://localhost:8000/api/v1/predict/zip \
     -F "archive=@images.zip"
```

### Model management

```bash
# List all available models with cache status
curl http://localhost:8000/api/v1/models

# Hot-reload after retraining (no server restart required)
curl -X POST http://localhost:8000/api/v1/models/reload \
     -H "Content-Type: application/json" \
     -d '{"model_name": "mambavision"}'

# Get active model info
curl http://localhost:8000/api/v1/models/active
```

### Expected response schemas

#### Single-image response (200 OK)

```json
{
  "success": true,
  "data": {
    "image_id":              "3f8a1c2d-4e5b-6789-abcd-ef0123456789",
    "predicted_class":       "glioma",
    "predicted_class_index": 0,
    "confidence":            0.9821,
    "is_high_confidence":    true,
    "probabilities": {
      "glioma":     0.9821,
      "meningioma": 0.0112,
      "notumor":    0.0047,
      "pituitary":  0.0020
    },
    "timing_ms": 38.4,
    "error": null,
    "metadata": {
      "model_name":    "mambavision",
      "model_version": "2026-07-01T10:00:00Z",
      "image_size":    224,
      "class_names":   ["glioma", "meningioma", "notumor", "pituitary"],
      "predicted_at":  "2026-07-31T15:00:01.123456Z",
      "gradcam_path":  "/abs/path/gradcam_output/3f8a1c2d.png"
    }
  }
}
```

---

## Running the API server

```bash
# Development (auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (2 workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Swagger UI: `http://localhost:8000/docs`

---

## Running tests

```bash
# Run the full suite
make test

# Run a specific test module
python -m pytest tests/test_training_config.py -v
python -m pytest tests/test_inference.py -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

| Test file | Coverage area |
|---|---|
| `test_imports.py` | All modules import cleanly (smoke test) |
| `test_health.py` | Health, predict, train, evaluate endpoints |
| `test_training_*.py` | Config, callbacks, checkpoints, experiment, trainer, API |
| `test_inference.py` | InferenceConfig, pipeline, batch runner, REST endpoints |
| `test_preprocessing.py` | Config, transforms, quality, augmentation |
| `test_dataset.py` | Validator, splitter, metadata, stats |
| `test_metrics.py` | System, inference, training, dashboard, storage |
| `test_security.py` | JWT, auth, roles, rate limiting, audit |
| `test_gradcam.py` | Grad-CAM heatmap generation |

---

## Metrics & Monitoring Dashboard

The monitoring dashboard provides live visibility into system health, inference
performance, and training activity.

### API endpoints

```
GET /api/v1/dashboard/overview     Composite snapshot (system + inference + training + alerts)
GET /api/v1/dashboard/system       CPU / RAM / disk / GPU metrics
GET /api/v1/dashboard/inference    Prediction counts, latency, class distribution
GET /api/v1/dashboard/training     Training job counts and experiment summaries
GET /api/v1/dashboard/history      Rolling time-series for a metric type
```

### Alert thresholds

| Metric | Warning | Critical |
|---|---|---|
| CPU usage | ≥ 80% | ≥ 95% |
| RAM usage | ≥ 85% | ≥ 95% |
| Disk usage | ≥ 85% | ≥ 95% |
| Inference success rate | < 80% (≥ 10 predictions) | — |
| Average inference latency | > 2 000 ms | — |
