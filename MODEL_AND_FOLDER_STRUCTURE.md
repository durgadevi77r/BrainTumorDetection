# Models Used

## AI Service (Python / PyTorch)

### Default model
- **Model key/name**: `mambavision`
- **Architecture**: MambaVision-T (official NVIDIA implementation)
- **Pretrained weights**: ImageNet-1K (loaded via Hugging Face Hub on first run)
- **Saved directory**: `ai-service/saved_models/mambavision/`
  - `config.json` — Hugging Face model configuration
  - `model.safetensors` — serialised PyTorch weights (safetensors format)
  - `model_info.json` — training metadata (accuracy, epochs, timestamp, etc.)
  - `gradcam/` — Grad-CAM visualisation outputs
- **Architecture factory**: [app/models/architectures.py](ai-service/app/models/architectures.py)
- **Inference loader**: [app/models/load_model.py](ai-service/app/models/load_model.py)
- **Hugging Face integration**: [app/models/mambavision/](ai-service/app/models/mambavision/)

### Supported architectures
All architectures are implemented in PyTorch via `app/models/architectures.py`.

| Key | Backbone | Weights source | Notes |
|---|---|---|---|
| `mambavision` | MambaVision-T (NVIDIA) | Hugging Face Hub (ImageNet) | **Default** — state-of-the-art vision Mamba |
| `efficientnet` | EfficientNet-B3 | torchvision (ImageNet) | Excellent accuracy/speed tradeoff |
| `resnet50` | ResNet-50 | torchvision (ImageNet) | Solid baseline |
| `vgg16` | VGG-16 | torchvision (ImageNet) | Classic reference |
| `cnn` | Custom 4-block CNN | random init | Lightweight, CPU-friendly |

### Checkpoint format
All saved models are stored in the **Hugging Face local directory format**:
```
saved_models/<model_name>/
    config.json            ← AutoModelForImageClassification config
    model.safetensors      ← weights (safetensors binary)
    model_info.json        ← metadata written by save_model()
```

PyTorch `.pt` / `.pth` checkpoints written during training are stored alongside:
```
saved_models/<model_name>/
    checkpoints/
        best_phase1.pt     ← best checkpoint from Phase 1 (head training)
        best_phase2.pt     ← best checkpoint from Phase 2 (fine-tuning)
```

## Backend (Node.js)
- The Node.js backend proxies AI requests to the FastAPI service.
- Classifier stubs exist for the legacy EDN-SVM pipeline (not production model):
  - `backend/pipeline/classifier/ednSvm.js`
  - `backend/pipeline/classifier/modelSerializer.js`

---

# Folder Structure (Top Level)

```text
BrainTumor/
  .github/                          GitHub Actions workflows
    workflows/
      ci.yml                        Continuous Integration (lint + test + docker build)
      cd.yml                        Continuous Deployment (staging + production)
      release.yml                   Release automation (versioned Docker images)

  ai-service/                       Python / FastAPI / PyTorch AI service
    app/
      api/                          REST route handlers (routes, auth, performance)
      core/                         Config (Pydantic-Settings), logging (Loguru)
      dataset/                      Dataset validation, splitting, stats
      inference/                    Inference pipeline, batch, cache, results
      metrics/                      System, inference, training, dashboard metrics
      models/
        mambavision/                Official MambaVision sub-package
          config.py                 MambaVisionHFConfig (HF local-dir config)
          factory.py                build_mambavision_model()
          predictor.py              TorchImageClassifier wrapper
          storage.py                save / load helpers
        architectures.py            build_model() factory for all 5 architectures
        evaluate.py                 evaluate_model() on test set
        load_model.py               load_model() with in-memory cache
        predict.py                  predict() single-image inference
        save_model.py               save_model() to HF local-dir format
        train.py                    train_model() two-phase training loop
      performance/                  Profiler, benchmark, cache, memory, reports
      preprocessing/                Image pipeline, augmentation, transforms, quality
      security/                     JWT, auth, roles, rate limiting, audit log
      training/                     Training job store, experiment registry
      utils/                        Grad-CAM (PyTorch), GLCM features
      main.py                       FastAPI application factory
    dataset/
      raw/                          Original MRI images (Training/ + Testing/)
      processed/                    Split dataset (train/ + val/ + test/)
    saved_models/                   PyTorch model weights (HF local-dir format)
    tests/                          pytest suite (1,100+ tests)
    Dockerfile                      Multi-stage PyTorch container (CPU + GPU)
    requirements.txt                PyTorch + MambaVision + FastAPI deps
    pyproject.toml                  Project metadata, pytest, ruff, black, isort

  backend/                          Node.js / Express / SQLite
    api/                            Route handlers (9 modules)
    database/                       Schema SQL, migrations, db.js
    middleware/                     Upload, error handling, validation
    pipeline/                       Preprocessing, segmentation, classifier
    server.js
    package.json

  frontend/                         React 18 / Vite / TypeScript
    src/
      components/                   Reusable UI components
      pages/                        Route-level page components
      hooks/                        Custom React hooks
      api/                          Axios API clients
      context/                      React context providers
      types/                        TypeScript type definitions
    package.json
    vite.config.ts

  docker/                           Docker Compose configurations
    docker-compose.yml              Base configuration
    docker-compose.dev.yml          Development overrides
    docker-compose.prod.yml         Production hardening overlay
    Dockerfile.backend              Node.js backend image
    Dockerfile.frontend             React + nginx image
    scripts/
      ai-entrypoint.sh              AI service container startup script

  docs/                             Project documentation
  scripts/                          Deployment and maintenance scripts
  .github/workflows/                CI/CD pipeline definitions
  Makefile                          Developer task automation
  CHANGELOG.md
  CONTRIBUTING.md
  LICENSE
  SECURITY.md
  VERSION
  README.md
```
