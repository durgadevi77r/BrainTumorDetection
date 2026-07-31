# Brain Tumour Detection — AI Service

Python / TensorFlow service for deep-learning MRI brain tumour classification.
Exposes a FastAPI REST API consumed by the Node.js backend.

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
13. [Metrics & Monitoring Dashboard](#metrics--monitoring-dashboard-module-8)

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset (raw → train/val/test split)
python scripts/prepare_dataset.py

# 3. Train the default model (EfficientNetB3, 30 epochs)
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
│   │   └── routes.py          # All FastAPI routes (including training v2 + inference v2)
│   ├── core/
│   │   ├── config.py          # Pydantic-settings singleton
│   │   └── logging.py
│   ├── dataset/               # Dataset validation, splitting, metadata
│   ├── inference/             # Production inference package (Module 6)
│   │   ├── __init__.py        # Public package surface
│   │   ├── config.py          # InferenceConfig dataclass
│   │   ├── pipeline.py        # InferencePipeline + CLI entry point
│   │   ├── batch.py           # BatchInferenceRunner (parallel, CSV/JSON export)
│   │   ├── cache.py           # LRU ModelCache with hot-reload
│   │   └── results.py         # PredictionResult, BatchPredictionResult, etc.
│   ├── models/
│   │   ├── architectures.py   # build_model(), unfreeze_top_layers()
│   │   ├── evaluate.py        # evaluate_model()
│   │   ├── load_model.py      # load_model() with cache
│   │   ├── predict.py         # predict() legacy inference shim
│   │   ├── save_model.py      # save_model()
│   │   └── train.py           # Legacy synchronous train_model() shim
│   ├── preprocessing/         # Preprocess, augmentation, transforms
│   ├── metrics/               # Metrics & monitoring package (Module 8)
│   │   ├── __init__.py        # Public API surface
│   │   ├── system.py          # CPU / RAM / disk / GPU metrics via psutil
│   │   ├── inference.py       # In-process prediction accumulator
│   │   ├── training.py        # Training job aggregator
│   │   ├── dashboard.py       # Composite overview + alert engine
│   │   └── storage.py         # Rolling JSON-Lines persistence
│   └── training/
│       ├── __init__.py        # Re-exports from training/
│       └── job_store.py       # In-process async job registry
├── training/                  # Core training package
│   ├── __init__.py
│   ├── config.py              # TrainingConfig dataclass
│   ├── callbacks.py           # build_callbacks() factory
│   ├── checkpoints.py         # Checkpoint save / load / list / delete
│   ├── experiment.py          # Experiment dataclass + ExperimentRegistry
│   └── trainer.py             # Trainer class + train() wrapper + CLI
├── dataset/
│   ├── raw/                   # Original images (one sub-folder per class)
│   └── processed/             # Split dataset (train/ val/ test/)
├── saved_models/              # Saved model artefacts
├── gradcam_output/            # Grad-CAM overlay PNG files
├── logs/
│   ├── experiments/           # Experiment JSON records + registry
│   ├── tensorboard/           # TensorBoard event files
│   ├── training/              # Per-epoch CSV logs
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
Default ratios: 70 % train / 15 % val / 15 % test.

---

## Training workflow

### CLI training

```bash
# Default: EfficientNetB3, 30 epochs, batch 32, lr 1e-4, Phase-2 fine-tuning on
python -m training.trainer

# Override architecture and epochs
python -m training.trainer --architecture resnet50 --epochs 20

# Full flag reference
python -m training.trainer \
    --architecture    efficientnet \   # cnn | vgg16 | resnet50 | efficientnet
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
make train                                # EfficientNetB3 defaults
make train ARCH=resnet50 EPOCHS=20        # ResNet-50
make train ARCH=cnn BS=16 EPOCHS=50       # Lightweight CNN, smaller batch
make train ARCH=vgg16 LR=5e-5 FT=false   # VGG-16, lower LR, no fine-tuning
make train DATA_DIR=path/to/processed    # Custom dataset directory

make evaluate                             # Evaluate active model
make evaluate ARCH=resnet50              # Evaluate a specific architecture

make predict IMAGE=scan.jpg              # Single-image inference
make predict-batch DIR=dataset/test/     # Batch inference from directory
make predict-zip ZIP=images.zip          # Batch inference from ZIP

make models                              # List all models + cache status
make reload-model                        # Hot-reload active model
make reload-model ARCH=resnet50          # Hot-reload a specific model

make test                                 # Run pytest suite
make lint                                 # ruff + mypy
make format                              # auto-format with ruff
make clean                               # remove __pycache__, *.pyc
```

### Python API

```python
from training import Trainer, TrainingConfig

cfg = TrainingConfig(
    architecture="efficientnet",
    epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    fine_tune=True,
    fine_tune_layers=20,
    fine_tune_epochs=10,
)

trainer = Trainer(cfg)
result  = trainer.run()

print(result["experiment_id"])       # "efficientnet-20240715-143022-ab12cd34"
print(result["best_val_accuracy"])   # 0.973
print(result["training_duration_s"]) # 1909.4
```

Or use the convenience wrapper:

```python
from training import train

result = train(
    architecture="resnet50",
    epochs=20,
    class_weights={"glioma": 1.5, "notumor": 0.8},  # handle class imbalance
)
```

### REST API (async)

#### Start a training job

```http
POST /api/v1/train/start
Content-Type: application/json

{
  "architecture":    "efficientnet",
  "epochs":          30,
  "batch_size":      32,
  "learning_rate":   0.0001,
  "fine_tune":       true,
  "fine_tune_layers": 20,
  "fine_tune_epochs": 10,
  "class_weights":   {"glioma": 1.5}
}
```

Response (202 Accepted):

```json
{
  "success":       true,
  "message":       "Training job queued for 'efficientnet'. Poll GET /api/v1/train/status/... for progress.",
  "job_id":        "a3f2b1c0d4e5f6a7b8c9d0e1f2a3b4c5",
  "experiment_id": "efficientnet-20240715-143022-ab12cd34"
}
```

#### Poll job status

```http
GET /api/v1/train/status/{job_id}
```

Response:

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

`status` values: `queued` → `running` → `completed` / `failed`

#### List experiments

```http
GET /api/v1/train/experiments
GET /api/v1/train/experiments?architecture=efficientnet
GET /api/v1/train/experiments?exp_status=completed&limit=10
```

#### Get experiment details

```http
GET /api/v1/train/experiments/{experiment_id}
```

Returns the full record including per-epoch training history, evaluation metrics,
and saved model paths.

---

## Hyperparameter reference

| Parameter | Default | Description |
|---|---|---|
| `architecture` | `efficientnet` | `cnn` \| `vgg16` \| `resnet50` \| `efficientnet` |
| `epochs` | `30` | Maximum Phase-1 epochs (EarlyStopping may end earlier) |
| `batch_size` | `32` | Mini-batch size for train and validation generators |
| `learning_rate` | `1e-4` | Phase-1 Adam learning rate |
| `optimiser` | `adam` | `adam` \| `sgd` \| `rmsprop` \| `adamw` |
| `dropout_rate` | `0.5` | Dropout in the classification head |
| `l2_reg` | `1e-4` | L2 kernel regularisation coefficient |
| `image_size` | `224` | Input resolution (H = W, pixels) |
| `seed` | `42` | Random seed for all generators |
| `class_weights` | `null` | `{"class_name": float}` for imbalanced data |
| `early_stopping_patience` | `10` | Epochs without val_loss improvement before stopping |
| `reduce_lr_patience` | `5` | Epochs before ReduceLROnPlateau fires |
| `reduce_lr_factor` | `0.5` | LR multiplier on plateau |
| `csv_log` | `true` | Write per-epoch CSV alongside TensorBoard |

---

## Transfer learning & fine-tuning

All architectures except `cnn` use a two-phase transfer learning strategy:

**Phase 1 — head training**
The ImageNet backbone is frozen. Only the classification head (`GlobalAveragePooling2D` → `BatchNorm` → `Dense(256/512)` → `Dropout` → `Dense(num_classes, softmax)`) is trained. This converges quickly and avoids destroying pretrained features.

**Phase 2 — fine-tuning**
The top `fine_tune_layers` (default: 20) of the backbone are unfrozen and trained at a lower learning rate (`learning_rate / 10` by default). This adapts the high-level features to the MRI domain.

| Parameter | Default | Description |
|---|---|---|
| `fine_tune` | `true` | Enable Phase-2 |
| `fine_tune_layers` | `20` | Backbone layers to unfreeze |
| `fine_tune_epochs` | `10` | Maximum Phase-2 epochs |
| `fine_tune_lr` | `lr / 10` | Phase-2 learning rate |

To skip fine-tuning (Phase 1 only):

```bash
python -m training.trainer --no-fine-tune
make train FT=false
```

---

## Experiment tracking

Every training run creates an experiment record in `logs/experiments/`:

```
logs/experiments/
    experiment_registry.json          ← lightweight index (newest first)
    efficientnet-20240715-143022-ab12cd34/
        experiment.json               ← full metadata
        training_config.json          ← config snapshot
```

`experiment.json` includes:
- Unique ID, timestamps, duration
- Full `TrainingConfig` snapshot
- Dataset provenance (paths, split counts, class weights)
- Per-epoch metrics for Phase 1 and Phase 2
- Post-training evaluation metrics (accuracy, F1, AUC-ROC, confusion matrix)
- Paths to saved model artefacts

Access programmatically:

```python
from training.experiment import ExperimentRegistry

registry = ExperimentRegistry()

# List recent runs
runs = registry.list_experiments(architecture="efficientnet", status="completed")

# Load a specific run
data = registry.get("efficientnet-20240715-143022-ab12cd34")
print(data["eval_metrics"]["accuracy"])
```

---

## Model artefact locations

```
saved_models/
    efficientnet/
        saved_model.pb          ← TensorFlow SavedModel (primary)
        variables/
        efficientnet.h5         ← HDF5 copy (portability)
        model_info.json         ← metadata: accuracy, params, training config
        checkpoints/
            <experiment_id>/
                best_weights.h5     ← best epoch weights
                checkpoint_info.json
```

| Artefact | Purpose |
|---|---|
| `saved_model.pb` | Primary inference artefact, loaded by `load_model()` |
| `*.h5` | Portable weight file for sharing / deployment |
| `model_info.json` | Metadata surfaced by `/health` and `/evaluate` endpoints |
| `best_weights.h5` | Best-epoch weights (via `ModelCheckpoint` callback) |
| `checkpoint_info.json` | Metrics and config at the checkpoint epoch |

---

## Evaluation

Evaluation runs automatically at the end of training (against `dataset/processed/test/`).
It can also be triggered manually:

```bash
make evaluate ARCH=efficientnet

# Python
from app.models.evaluate import evaluate_model
metrics = evaluate_model("efficientnet")
print(metrics["accuracy"], metrics["f1"], metrics["auc_roc"])

# REST
curl -X POST /api/v1/evaluate -H "Content-Type: application/json" \
     -d '{"model_name": "efficientnet", "batch_size": 32}'
```

Metrics returned: `accuracy`, `precision`, `recall`, `f1` (macro), `auc_roc` (macro OvR),
`confusion_matrix`, `per_class` breakdown.

---

## Inference & prediction

The `app/inference/` package provides a production-grade inference pipeline that sits on top of the
existing preprocessing and model-loading modules. It is independent of FastAPI and can be used
directly from Python or the CLI.

### Architecture overview

```
InferencePipeline          ← single-image and batch-list prediction
BatchInferenceRunner       ← parallel batch with progress tracking + export
InferenceConfig            ← all inference settings in one dataclass
ModelCache (LRU)           ← capacity-bounded cache with hot-reload
PredictionResult           ← typed result with top-K, timing, Grad-CAM path
BatchPredictionResult      ← aggregate: counts, class distribution, export paths
```

### Inference CLI

Run inference from the command line (no server required):

```bash
# Single image
python -m app.inference.pipeline scan.jpg

# With model override and top-3 predictions
python -m app.inference.pipeline scan.jpg --model resnet50 --top-k 3

# With Grad-CAM heatmap
python -m app.inference.pipeline scan.jpg --gradcam

# Batch inference — all images in a directory
python -m app.inference.pipeline --batch dataset/test/glioma/

# Batch from a ZIP archive
python -m app.inference.pipeline --zip images.zip

# Save batch results to a directory
python -m app.inference.pipeline --batch dataset/test/ --output-dir output/
```

Sample output (single image):

```json
{
  "image_id":           "3f8a1c2d-...",
  "predicted_class":    "glioma",
  "confidence":         0.9732,
  "is_high_confidence": true,
  "top_k": [
    {"rank": 1, "class_name": "glioma",     "class_index": 0, "probability": 0.9732},
    {"rank": 2, "class_name": "meningioma", "class_index": 1, "probability": 0.0153}
  ],
  "timing_ms":  42.1,
  "model":      "efficientnet",
  "gradcam_path": null
}
```

### Inference Makefile targets

```bash
# Single-image prediction
make predict IMAGE=path/to/scan.jpg
make predict IMAGE=scan.jpg ARCH=resnet50 TOP_K=3
make predict IMAGE=scan.jpg GRADCAM=--gradcam

# Batch inference from a directory
make predict-batch DIR=dataset/test/
make predict-batch DIR=dataset/test/ OUT_DIR=output/ ARCH=vgg16

# Batch inference from a ZIP archive
make predict-zip ZIP=images.zip
make predict-zip ZIP=images.zip OUT_DIR=output/ GRADCAM=--gradcam

# List all models with availability + cache status
make models

# Hot-reload the active model (after retraining)
make reload-model
make reload-model ARCH=resnet50
```

### Python API (inference)

#### Single-image prediction

```python
from app.inference import predict

# Convenience function — one line
result = predict("path/to/scan.jpg", model_name="efficientnet", top_k=3)
print(result.predicted_class, result.confidence)

# Full pipeline control
from app.inference import InferencePipeline, InferenceConfig

cfg = InferenceConfig(
    model_name="efficientnet",
    top_k=3,
    generate_gradcam=True,
    confidence_threshold=0.7,
)
pipeline = InferencePipeline(cfg)

# From file path, bytes, or pathlib.Path
result = pipeline.predict(open("scan.jpg", "rb").read())

print(result.predicted_class)            # "glioma"
print(result.confidence)                 # 0.9732
print(result.is_high_confidence)         # True
print(result.probabilities)              # {"glioma": 0.9732, ...}
print(result.top_k[0].class_name)        # "glioma"
print(result.timing_ms)                  # 42.1
print(result.metadata.gradcam_path)      # "/abs/path/overlay.png"
print(result.metadata.model_version)     # "2024-07-15T14:30:22Z"
```

#### Batch prediction from a directory

```python
result = pipeline.predict_directory("dataset/test/glioma/")

print(result.total)              # 300
print(result.succeeded)          # 298
print(result.failed)             # 2
print(result.success_rate)       # 0.9933
print(result.class_distribution) # {"glioma": 250, "notumor": 48}
print(result.timing_ms)          # 3412.5
```

#### Batch prediction from a ZIP archive

```python
result = pipeline.predict_zip("images.zip")
print(result.to_json(indent=2))
```

#### BatchInferenceRunner with export

```python
from app.inference import BatchInferenceRunner, InferenceConfig

cfg    = InferenceConfig(model_name="efficientnet", max_workers=4)
runner = BatchInferenceRunner(cfg, progress_callback=lambda done, tot: print(f"{done}/{tot}"))

result = runner.run_directory("dataset/test/")

# Export to JSON and CSV
paths = runner.export(result, output_dir="output/", formats=("json", "csv"))
print(paths["json_path"])   # "output/batch_results.json"
print(paths["csv_path"])    # "output/batch_results.csv"
```

#### Model cache management

```python
from app.inference import get_model, reload_model, cache_stats, list_available_models

# Load (cached after first call)
model = get_model("efficientnet")

# Hot-reload after retraining
reload_model("efficientnet")

# Inspect cache
stats = cache_stats()
print(stats["size"])         # 1
print(stats["hit_rate"])     # 0.875
print(stats["cached_models"])# ["efficientnet"]

# List all architectures
for m in list_available_models():
    print(m["name"], m["available"], m["cached"])
```

### REST API (inference v2)

All endpoints are prefixed with `/api/v1`.

#### POST /predict/image — Single-image inference

```bash
curl -X POST http://localhost:8000/api/v1/predict/image \
     -F "image=@scan.jpg" \
     -F "model_name=efficientnet" \
     -F "top_k=3" \
     -F "generate_gradcam=true" \
     -F "confidence_threshold=0.7"
```

Form fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | JPEG or PNG MRI scan |
| `model_name` | string | active model | `cnn` \| `vgg16` \| `resnet50` \| `efficientnet` |
| `top_k` | int | `1` | Number of top predictions (1–4) |
| `generate_gradcam` | bool | `false` | Generate Grad-CAM heatmap overlay |
| `confidence_threshold` | float | `0.5` | Threshold for `is_high_confidence` flag |

#### POST /predict/batch — Multi-file batch

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
     -F "images=@scan1.jpg" \
     -F "images=@scan2.png" \
     -F "model_name=efficientnet" \
     -F "top_k=1"
```

#### POST /predict/zip — ZIP archive batch

```bash
curl -X POST http://localhost:8000/api/v1/predict/zip \
     -F "archive=@images.zip" \
     -F "model_name=efficientnet"
```

Returns **400** when the archive contains no valid images.
Returns **422** when the uploaded file is not a valid ZIP.

### Model management

#### GET /models — List all models

```bash
curl http://localhost:8000/api/v1/models
```

#### POST /models/reload — Hot-reload a model

Use this after a training run completes to make updated weights available
without restarting the server:

```bash
curl -X POST http://localhost:8000/api/v1/models/reload \
     -H "Content-Type: application/json" \
     -d '{"model_name": "efficientnet"}'
```

#### GET /models/active — Active model details

```bash
curl http://localhost:8000/api/v1/models/active
```

### Expected response schemas

#### Single-image response (200 OK)

```json
{
  "success": true,
  "data": {
    "image_id":             "3f8a1c2d-4e5b-6789-abcd-ef0123456789",
    "predicted_class":      "glioma",
    "predicted_class_index": 0,
    "confidence":           0.9732,
    "is_high_confidence":   true,
    "probabilities": {
      "glioma":      0.9732,
      "meningioma":  0.0153,
      "notumor":     0.0082,
      "pituitary":   0.0033
    },
    "top_k": [
      {"rank": 1, "class_name": "glioma",     "class_index": 0, "probability": 0.9732},
      {"rank": 2, "class_name": "meningioma", "class_index": 1, "probability": 0.0153},
      {"rank": 3, "class_name": "notumor",    "class_index": 2, "probability": 0.0082}
    ],
    "timing_ms": 42.1,
    "error": null,
    "metadata": {
      "model_name":    "efficientnet",
      "model_version": "2024-07-15T14:30:22Z",
      "image_size":    224,
      "class_names":   ["glioma", "meningioma", "notumor", "pituitary"],
      "predicted_at":  "2024-07-15T15:00:01.123456Z",
      "source_path":   "scan.jpg",
      "gradcam_path":  "/abs/path/gradcam_output/3f8a1c2d.png"
    }
  }
}
```

#### Batch response (200 OK)

```json
{
  "success": true,
  "data": {
    "total":              10,
    "succeeded":          9,
    "failed":             1,
    "success_rate":       0.9,
    "timing_ms":          384.2,
    "model_name":         "efficientnet",
    "source_type":        "zip",
    "class_distribution": {"glioma": 5, "meningioma": 2, "notumor": 2},
    "export_paths":       {},
    "results": [
      {
        "filename": "scan1.jpg",
        "success":  true,
        "result":   { "...": "full PredictionResult" },
        "error":    null
      },
      {
        "filename": "corrupt.jpg",
        "success":  false,
        "result":   null,
        "error":    "ValueError: could not decode image"
      }
    ]
  }
}
```

#### Model list response (200 OK)

```json
{
  "success": true,
  "cache_stats": {
    "capacity":      4,
    "size":          1,
    "cached_models": ["efficientnet"],
    "total_hits":    42,
    "total_misses":  3,
    "hit_rate":      0.9333
  },
  "data": [
    {
      "name":          "efficientnet",
      "available":     true,
      "cached":        true,
      "model_version": "2024-07-15T14:30:22Z",
      "total_params":  12341232,
      "model_dir":     "/abs/path/saved_models/efficientnet"
    },
    {"name": "resnet50",  "available": false, "cached": false, "model_version": null, "total_params": null, "model_dir": "..."},
    {"name": "vgg16",     "available": false, "cached": false, "model_version": null, "total_params": null, "model_dir": "..."},
    {"name": "cnn",       "available": false, "cached": false, "model_version": null, "total_params": null, "model_dir": "..."}
  ]
}
```

---

## Running the API server

```bash
# Development (auto-reload on file changes)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (2 workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Running tests

```bash
# Run the full suite
make test

# Run a specific module
python -m pytest tests/test_training_config.py -v
python -m pytest tests/test_training_trainer.py -v
python -m pytest tests/test_training_api.py -v

# Run with coverage
python -m pytest tests/ --cov=training --cov=app/training --cov-report=term-missing
```

Test modules:

| File | What it tests |
|---|---|
| `test_training_config.py` | `TrainingConfig` defaults, validation, serialisation |
| `test_training_callbacks.py` | `build_callbacks()` factory, callback types, phases |
| `test_training_checkpoints.py` | Save/load/list/delete checkpoint utilities |
| `test_training_experiment.py` | `Experiment` lifecycle, `ExperimentRegistry` CRUD |
| `test_training_trainer.py` | `Trainer` init, mocked run, CLI arg parser |
| `test_training_api.py` | REST endpoints: start, status, list, get experiment |
| `test_inference.py` | `InferenceConfig`, `PredictionResult`, `ModelCache`, `InferencePipeline`, `BatchInferenceRunner`, export, inference v2 REST endpoints |
| `test_preprocessing.py` | Config, transforms, quality checks, augmentation, pipeline |
| `test_dataset.py` | Dataset validator, splitter, metadata, stats |
| `test_health.py` | `/health`, `/predict`, `/train`, `/evaluate` endpoints |
| `test_imports.py` | All package symbols importable without side effects |
| `test_metrics.py` | `SystemMetrics`, `InferenceMetricsStore`, `TrainingMetrics`, `MetricsStorage`, `DashboardOverview`, dashboard REST endpoints |

---

## Metrics & Monitoring Dashboard (Module 8)

The monitoring dashboard provides live visibility into system health, inference performance, and training activity. It is built as a production-grade analytics layer on top of the existing inference and training subsystems.

### Architecture overview

```
app/metrics/
├── __init__.py        # Public API: get_system_metrics, get_inference_metrics, ...
├── system.py          # CPU / RAM / disk / GPU / process metrics via psutil
├── inference.py       # In-process prediction accumulator (InferenceMetricsStore)
├── training.py        # Training job aggregator (reads JobStore + experiment logs)
├── dashboard.py       # Composite overview + alert engine + history queries
└── storage.py         # Rolling JSON-Lines persistence (MetricsStorage)
```

**Data flow:**

```
Inference endpoints  ──► record_prediction()  ──► InferenceMetricsStore (in-memory)
Training jobs        ──► JobStore / experiments ──► get_training_metrics()
psutil               ──► get_system_metrics()
                                 │
                         get_dashboard_overview()
                                 │
                         MetricsStorage.save_snapshot()  ──► logs/metrics/*.jsonl
                                 │
                         GET /dashboard/history  ──► time-series charts
```

### Metric definitions

#### System metrics (`GET /api/v1/dashboard/system`)

| Field | Description |
|---|---|
| `cpu_percent` | Overall CPU utilisation (%) |
| `cpu_per_core` | Per-logical-core utilisation (%) |
| `ram_percent` | RAM usage (%) |
| `ram_used_mb` | RAM used (MB) |
| `disk_percent` | Disk usage for the project filesystem (%) |
| `gpu_available` | Whether TensorFlow can see a GPU |
| `gpus[].utilization_percent` | Per-GPU utilisation (requires `pynvml`) |
| `gpus[].memory_used_mb` | Per-GPU VRAM used (MB) |
| `uptime_seconds` | Seconds since the AI service process started |
| `process_ram_mb` | RSS of the current Python process (MB) |
| `process_threads` | Thread count for the current process |

#### Inference metrics (`GET /api/v1/dashboard/inference`)

| Field | Description |
|---|---|
| `total_predictions` | Total single-image predictions since service start |
| `succeeded` / `failed` | Counts of successful and failed predictions |
| `success_rate` | `succeeded / total_predictions` |
| `avg_latency_ms` | Mean inference time (ms) over last 1 000 samples |
| `p95_latency_ms` | 95th-percentile inference time (requires ≥ 20 samples) |
| `confidence_distribution` | Histogram (6 buckets: `<50%` → `95–100%`) |
| `class_distribution` | Prediction count per class |
| `top_classes` | Top-10 predicted classes by count |
| `per_model_counts` | Predictions per architecture |
| `batch_runs` | Total batch prediction runs |
| `batch_images_processed` | Total images across all batch runs |
| `recent_predictions` | Rolling window of the last 100 predictions |

#### Training metrics (`GET /api/v1/dashboard/training`)

| Field | Description |
|---|---|
| `total_jobs` | Total training jobs submitted via `/train/start` |
| `running_jobs` | Currently running jobs |
| `completed_jobs` / `failed_jobs` | Terminal state counts |
| `best_val_accuracy` | Best validation accuracy across all completed experiments |
| `avg_job_duration_s` | Mean duration of completed + failed jobs (seconds) |
| `architecture_counts` | Jobs submitted per architecture |
| `recent_jobs` | Last 10 jobs with status and duration |
| `recent_experiments` | Last 10 experiment records |

### Dashboard API endpoints

All endpoints are prefixed with `/api/v1`.

```
GET /dashboard/overview     Composite snapshot (system + inference + training + alerts)
GET /dashboard/system       System resource metrics
GET /dashboard/inference    Prediction and latency metrics
GET /dashboard/training     Training job and experiment metrics
GET /dashboard/history      Rolling time-series for a metric type
```

#### GET /dashboard/overview

Returns a single composite payload covering all metric domains plus threshold-based alerts.

```bash
curl http://localhost:8000/api/v1/dashboard/overview
```

```json
{
  "success": true,
  "data": {
    "timestamp":       "2024-07-14T12:00:00Z",
    "service_version": "1.0.0",
    "system":    { "cpu_percent": 34.1, "ram_percent": 61.2, "disk_percent": 42.7, "gpu_available": false, "uptime_seconds": 3612.4 },
    "inference": { "total_predictions": 142, "success_rate": 0.9718, "avg_latency_ms": 38.4, "batch_runs": 3 },
    "training":  { "total_jobs": 7, "running_jobs": 0, "completed_jobs": 6, "best_val_accuracy": 0.9732 },
    "models":    { "capacity": 4, "size": 1, "hit_rate": 0.83 },
    "alerts":    []
  }
}
```

#### GET /dashboard/history

```bash
# System metrics for the last 6 hours
curl "http://localhost:8000/api/v1/dashboard/history?metric_type=system&hours=6"

# Inference metrics for the last 24 hours (default)
curl "http://localhost:8000/api/v1/dashboard/history?metric_type=inference"
```

Query parameters:

| Parameter | Values | Default | Description |
|---|---|---|---|
| `metric_type` | `system` \| `inference` \| `training` \| `overview` | `system` | Metric domain |
| `hours` | 1–168 | `24` | Lookback window (max 7 days) |

### Alert thresholds

The overview endpoint computes threshold-based alerts automatically:

| Metric | Warning | Critical |
|---|---|---|
| CPU usage | ≥ 80% | ≥ 95% |
| RAM usage | ≥ 85% | ≥ 95% |
| Disk usage | ≥ 85% | ≥ 95% |
| Inference success rate | < 80% (when ≥ 10 predictions) | — |
| Average inference latency | > 2 000 ms | — |

Each alert has `level` (`"warning"` or `"critical"`), `domain` (e.g. `"system"`), and `message`.

### Metric storage

Snapshots are written to `logs/metrics/` as JSON-Lines files keyed by type and date:

```
logs/metrics/
    system_2024-07-14.jsonl
    inference_2024-07-14.jsonl
    training_2024-07-14.jsonl
    overview_2024-07-14.jsonl
```

Each line is one snapshot. Files older than 30 days are pruned by `MetricsStorage.purge_old_files()`.

```python
from app.metrics.storage import get_metrics_store

store = get_metrics_store()

# Read last 6 hours of system snapshots
history = store.load_history(metric_type="system", hours=6)

# Daily summary
summary = store.load_daily_summary("2024-07-14")

# Dates with stored data
dates = store.get_available_dates("inference")

# Purge old files
removed = store.purge_old_files(keep_days=30)
```

### Recording inference events

Hook the metrics accumulator into your prediction code:

```python
from app.metrics import record_prediction, record_batch_prediction

# After a single inference
result = pipeline.predict(image_bytes)
record_prediction(result)            # accepts PredictionResult dataclass or dict

# After a batch run
batch = runner.run(sources)
record_batch_prediction(batch)       # accepts BatchPredictionResult or dict
```

### Python API

```python
from app.metrics import (
    get_system_metrics,      # → dict with CPU/RAM/GPU/disk/process fields
    get_inference_metrics,   # → dict with predictions, latency, class dist
    get_training_metrics,    # → dict with job counts and best accuracy
    get_dashboard_overview,  # → composite overview dict
    metrics_store,           # → module-level MetricsStorage singleton
)

# Live snapshots
sys_snapshot = get_system_metrics()
print(sys_snapshot["cpu_percent"], sys_snapshot["ram_percent"])

inf_snapshot = get_inference_metrics()
print(inf_snapshot["total_predictions"], inf_snapshot["avg_latency_ms"])

trn_snapshot = get_training_metrics()
print(trn_snapshot["best_val_accuracy"])

overview = get_dashboard_overview()
for alert in overview["alerts"]:
    print(alert["level"], alert["message"])
```

### Frontend monitoring page

The React frontend exposes a `/monitoring` route with five tabs:

| Tab | Content |
|---|---|
| **Overview** | KPI cards for CPU/RAM/disk, inference counts, training jobs, active alerts |
| **System** | CPU/RAM/disk/GPU gauges, per-core bars, process info, uptime |
| **Inference** | Success rate, latency (avg + p95), confidence histogram, class pie chart, model breakdown, recent predictions table |
| **Training** | Job counts, architecture bar chart, best accuracy, recent jobs and experiments tables |
| **History** | Configurable time-series line chart (1h–7d lookback, selectable metric type) |

Live polling refreshes all metrics every 5 seconds. Manual refresh and automatic error recovery are also supported.

Export buttons in the page header:
- **JSON** — Downloads the full dashboard snapshot (`metrics_YYYY-MM-DD.json`)
- **CSV** — Downloads the recent predictions table (`predictions_YYYY-MM-DD.csv`)

Key frontend files:

```
frontend/src/
├── api/dashboard.ts                 # API functions: getDashboardOverview, getSystemMetrics, ...
├── hooks/useDashboard.ts            # Polling hook with per-domain control
├── components/
│   ├── MetricGauge.tsx              # SVG arc gauge (CPU / RAM / disk / GPU)
│   ├── SystemHealthPanel.tsx        # Gauges + per-core bars + info rows
│   ├── InferenceMetricsPanel.tsx    # Confidence histogram, class pie, recent table
│   ├── TrainingMetricsPanel.tsx     # Job counts, arch bar chart, experiment table
│   ├── MetricsHistoryChart.tsx      # Recharts line chart for rolling history
│   └── AlertsBanner.tsx             # Threshold alert display
├── pages/Monitoring.tsx             # 5-tab monitoring page
└── utils/metricsExport.ts           # JSON + CSV download helpers
```

### Monitoring workflow

1. Start the AI service: `uvicorn app.main:app --port 8000 --reload`
2. Open the frontend at `http://localhost:3000`
3. Click **Monitoring** in the navbar or from the Dashboard quick-actions
4. The Overview tab loads automatically. Metrics refresh every 5 seconds.
5. Switch tabs to drill into System, Inference, Training, or History views.
6. Use the **History** tab to inspect rolling time-series (select type + window, click Load).
7. Use **⬇ JSON** / **⬇ CSV** to export the current snapshot or recent predictions.

### Running tests

```bash
# Python metrics tests only
python -m pytest tests/test_metrics.py -v

# Full Python suite (669 tests)
make test

# Frontend metrics tests
cd frontend
npx vitest run src/api/dashboard.test.ts
npx vitest run src/hooks/useDashboard.test.ts
npx vitest run src/components/MetricGauge.test.tsx
npx vitest run src/components/AlertsBanner.test.tsx
npx vitest run src/components/SystemHealthPanel.test.tsx
npx vitest run src/components/InferenceMetricsPanel.test.tsx
npx vitest run src/components/TrainingMetricsPanel.test.tsx
npx vitest run src/components/MetricsHistoryChart.test.tsx
npx vitest run src/pages/Monitoring.test.tsx
npx vitest run src/utils/metricsExport.test.ts
```
