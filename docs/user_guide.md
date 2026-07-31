# User Guide

Brain Tumour Detection — complete workflow guide for end users.

---

## Table of Contents

1. [First-Time Setup](#first-time-setup)
2. [Navigating the Interface](#navigating-the-interface)
3. [Dataset Preparation](#dataset-preparation)
4. [Training a Model](#training-a-model)
5. [Running Inference](#running-inference)
6. [Batch Inference](#batch-inference)
7. [Understanding Grad-CAM](#understanding-grad-cam)
8. [Monitoring the Dashboard](#monitoring-the-dashboard)
9. [User Account Management](#user-account-management)
10. [Exporting Results](#exporting-results)
11. [Troubleshooting](#troubleshooting)
12. [Frequently Asked Questions](#frequently-asked-questions)

---

## First-Time Setup

After completing installation, follow these steps before using the application:

### 1. Start all services

**Docker (recommended):**
```bash
docker compose -f docker/docker-compose.yml up -d
```

**Local:**
```bash
# Three separate terminals:
# Terminal 1: cd ai-service && uvicorn app.main:app --port 8000 --reload
# Terminal 2: cd backend && npm run dev
# Terminal 3: cd frontend && npm run dev
```

### 2. Check service health

Open http://localhost:8000/api/v1/health in your browser or run:
```bash
curl http://localhost:8000/api/v1/health
```

A healthy response looks like:
```json
{
  "success": true,
  "status": "ok",
  "service": "Brain Tumour Detection AI Service",
  "version": "1.0.0",
  "environment": "development",
  "active_model": "mambavision",
  "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
  "models_available": {
    "mambavision": false,
    "cnn": false,
    "vgg16": false,
    "resnet50": false,
    "efficientnet": false
  }
}
```

All models show `false` until you train them — this is expected.

### 3. Open the application

Navigate to http://localhost:3000.

---

## Navigating the Interface

The React frontend has five main sections accessible from the navigation bar:

| Section | Path | Purpose |
|---|---|---|
| **Predict** | `/predict` | Upload and classify a single MRI scan |
| **Batch** | `/batch` | Upload and classify multiple scans at once |
| **Train** | `/train` | Configure and launch model training |
| **Dataset** | `/dataset` | Manage and validate the MRI dataset |
| **Dashboard** | `/dashboard` | Live system and inference metrics |

---

## Dataset Preparation

The AI model needs a properly organised dataset before training.

### Expected directory structure

```
ai-service/dataset/raw/
├── Training/
│   ├── glioma/        ← JPEG/PNG MRI images
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

A commonly used dataset is the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

### Step 1 — Validate the dataset

Using the UI:
1. Go to the **Dataset** page.
2. Click **Validate Dataset**.
3. Review the validation report — it lists missing classes, unsupported file formats, and empty directories.

Using the API:
```bash
curl -X POST http://localhost:8000/api/v1/dataset/validate
```

### Step 2 — Prepare (split) the dataset

Using the UI:
1. On the **Dataset** page, set split ratios (e.g., 70 / 15 / 15).
2. Click **Prepare Dataset**.

Using the API:
```bash
curl -X POST http://localhost:8000/api/v1/dataset/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,
    "random_seed": 42
  }'
```

This creates `dataset/processed/` with `train/`, `val/`, and `test/` subdirectories, each containing class subdirectories with symbolic links or copies of the source images.

### Step 3 — View dataset statistics

```bash
curl http://localhost:8000/api/v1/dataset/info
```

The response includes per-class counts, total images, class weights for imbalanced training, and the split ratios used.

---

## Training a Model

### Using the UI

1. Go to the **Train** page.
2. Select a model architecture from the dropdown:
   - **MambaVision-T** — state-of-the-art accuracy, recommended default (NVIDIA vision Mamba)
   - **EfficientNet-B3** — excellent accuracy/speed tradeoff
   - **ResNet50** — good generalisation, higher memory
   - **VGG16** — classic, slowest training
   - **Custom CNN** — fastest, good for experimentation
3. Set training parameters:
   - **Epochs** — number of full dataset passes (default 30)
   - **Batch Size** — images per gradient update (default 32)
   - **Learning Rate** — step size (default 0.0001)
   - **Fine-tune** — enable two-phase backbone fine-tuning (recommended)
4. Click **Start Training**.
5. Training progress updates in real time.

### Using the API — synchronous (legacy)

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name":      "mambavision",
    "epochs":          30,
    "batch_size":      32,
    "learning_rate":   0.0001,
    "fine_tune":       true,
    "fine_tune_layers": 20,
    "fine_tune_epochs": 10
  }'
```

### Using the API — asynchronous (recommended for long runs)

```bash
# 1. Start the job
curl -X POST http://localhost:8000/api/v1/train/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mambavision", "epochs": 50}'

# Returns: {"job_id": "abc123", ...}

# 2. Poll status
curl http://localhost:8000/api/v1/train/status/abc123

# 3. List all experiment runs
curl http://localhost:8000/api/v1/train/experiments

# 4. Get full experiment details
curl http://localhost:8000/api/v1/train/experiments/abc123
```

### Training output

After training completes:
- Model weights are saved to `ai-service/saved_models/{model_name}/`
- Training history (loss, accuracy per epoch) is stored in the experiment registry
- The model is automatically cached for inference

### Evaluating a trained model

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mambavision", "batch_size": 32}'
```

Returns accuracy, loss, and a per-class confusion matrix on the test split.

---

## Running Inference

### Using the UI

1. Go to the **Predict** page.
2. Drag and drop an MRI scan image (JPEG, PNG, BMP, TIFF supported).
3. Click **Analyse**.
4. Results appear showing:
   - Predicted class with confidence score
   - Probability bar chart for all four classes
   - Grad-CAM heatmap overlay (if enabled)

### Using the API

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@mri_scan.jpg"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction":   "glioma",
    "confidence":   0.9423,
    "probabilities": {
      "glioma":      0.9423,
      "meningioma":  0.0312,
      "notumor":     0.0189,
      "pituitary":   0.0076
    },
    "model_used":   "mambavision",
    "inference_ms": 47.3,
    "gradcam_b64":  "data:image/png;base64,..."
  }
}
```

### Preprocessing preview

Before submitting a scan for inference, preview what the preprocessing pipeline produces:

```bash
curl -X POST http://localhost:8000/api/v1/preprocess/preview \
  -F "file=@mri_scan.jpg"
```

Returns a base64-encoded preview image of the preprocessed scan.

### Image quality check

```bash
curl -X POST http://localhost:8000/api/v1/preprocess/quality-check \
  -F "file=@mri_scan.jpg"
```

Returns quality flags: blur level, intensity range, variance — useful to catch corrupted images before inference.

---

## Batch Inference

### Using the UI

1. Go to the **Batch** page.
2. Drag and drop multiple image files, or upload a ZIP archive.
3. Click **Run Batch**.
4. Download the results CSV when processing completes.

### Using the API

```bash
# Submit multiple files
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -F "files=@scan1.jpg" \
  -F "files=@scan2.jpg" \
  -F "files=@scan3.jpg"

# Submit a ZIP archive
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -F "zip_file=@scans.zip"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total":      3,
    "completed":  3,
    "failed":     0,
    "results": [
      {"filename": "scan1.jpg", "prediction": "glioma",     "confidence": 0.94},
      {"filename": "scan2.jpg", "prediction": "notumor",    "confidence": 0.98},
      {"filename": "scan3.jpg", "prediction": "pituitary",  "confidence": 0.87}
    ]
  }
}
```

---

## Understanding Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image regions the model focused on when making its prediction.

### What the heatmap shows

- **Red / warm areas** — regions strongly associated with the predicted class
- **Blue / cool areas** — regions the model found less relevant
- Overlaid on the original MRI as a semi-transparent mask

### When to trust the heatmap

The heatmap should highlight the tumour region. If it highlights background or skull regions, consider:
- The image may need better cropping
- The model may benefit from more training data for that class
- Image quality may be poor (run the quality check first)

### Generating Grad-CAM via the API

The `/predict` endpoint always returns a `gradcam_b64` field. To get only the Grad-CAM image:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@mri_scan.jpg" | python3 -c "
import sys, json, base64
resp = json.load(sys.stdin)
img = base64.b64decode(resp['data']['gradcam_b64'].split(',')[1])
open('gradcam.png', 'wb').write(img)
print('Saved gradcam.png')
"
```

---

## Monitoring the Dashboard

The Dashboard page (`/dashboard`) shows live system metrics refreshed every 30 seconds.

### System metrics

| Metric | Description |
|---|---|
| CPU Usage | Current CPU utilisation % |
| RAM Usage | Used / total RAM in MB |
| Disk Usage | Used / total disk in GB |
| GPU (if available) | GPU utilisation and VRAM |
| Process RSS | AI service memory footprint |

### Inference metrics

| Metric | Description |
|---|---|
| Total Predictions | Cumulative prediction count |
| Success Rate | % of predictions that completed without error |
| Avg Latency | Mean inference time in ms |
| P95 Latency | 95th percentile inference time |
| Class Distribution | Pie chart of predicted classes |

### Training metrics

| Metric | Description |
|---|---|
| Jobs Run | Total training jobs executed |
| Best Accuracy | Highest validation accuracy achieved |
| Experiment History | Timeline of training runs |

### Using the dashboard API directly

```bash
# Full composite snapshot
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/dashboard/overview

# System metrics only
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/dashboard/system

# Inference metrics only
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/dashboard/inference
```

---

## User Account Management

### Logging in

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
```

Returns `access_token` (30 min lifetime) and `refresh_token` (7 day lifetime).

### Using authenticated endpoints

```bash
curl -H "Authorization: Bearer <access_token>" \
  http://localhost:8000/api/v1/auth/me
```

### Refreshing a token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
```

### Changing your password

```bash
curl -X POST http://localhost:8000/api/v1/auth/change-password \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"current_password": "old", "new_password": "NewStr0ng!"}'
```

### User roles

| Role | Permissions |
|---|---|
| **Admin** | Full access: user management, benchmarks, all operations |
| **Researcher** | Training, inference, dataset management, metrics |
| **Operator** | Inference, batch processing, benchmarks |
| **Viewer** | Read-only: health, metrics, experiment history |

### Account lockout

After 5 consecutive failed login attempts the account is locked for 15 minutes. An Admin can unlock an account immediately:

```bash
curl -X POST http://localhost:8000/api/v1/auth/users/{user_id}/unlock \
  -H "Authorization: Bearer <admin_token>"
```

---

## Exporting Results

### Download batch results as CSV

The batch inference response includes a CSV-compatible results array. To export:

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -F "zip_file=@scans.zip" | python3 -c "
import sys, json, csv
resp = json.load(sys.stdin)
rows = resp['data']['results']
w = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
w.writeheader()
w.writerows(rows)
" > results.csv
```

### Download performance report as HTML

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/performance/report/html \
  -o performance_report.html
```

---

## Troubleshooting

See the full [Troubleshooting Guide](troubleshooting.md) for detailed solutions.

**Quick fixes:**

| Problem | Solution |
|---|---|
| "No model available" on predict | Train a model first via `/train` |
| Frontend shows blank page | Check `VITE_API_BASE_URL` in `frontend/.env.local` |
| AI service won't start | Check `JWT_SECRET_KEY` is set in `ai-service/.env` |
| Training is very slow | Ensure GPU is detected — run `nvidia-smi` |
| 401 Unauthorized | Token expired — refresh or log in again |

---

## Frequently Asked Questions

See the full [FAQ](faq.md).

**Q: What image formats are supported?**
JPEG, PNG, BMP, TIFF. Images are automatically resized to 224×224 pixels.

**Q: How accurate is the model?**
MambaVision-T trained on the Kaggle Brain Tumor MRI Dataset typically achieves 97–99% validation accuracy. Accuracy on your own data will vary.

**Q: Can I add my own model architecture?**
Yes — see the [Developer Guide](developer_guide.md#adding-new-models).

**Q: How do I update the active model?**
Set `ACTIVE_MODEL` in `ai-service/.env` to `mambavision`, `cnn`, `vgg16`, `resnet50`, or `efficientnet`, then restart the service.
