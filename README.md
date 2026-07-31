# Brain Tumour Detection

> Deep-learning MRI classification with explainable AI, production inference pipeline, and full-stack web interface.

[![CI](https://github.com/your-org/brain-tumor-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/brain-tumor-detection/actions/workflows/ci.yml)
[![CD](https://github.com/your-org/brain-tumor-detection/actions/workflows/cd.yml/badge.svg)](https://github.com/your-org/brain-tumor-detection/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20_LTS-green)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1492%2B_passing-brightgreen)](#testing)

---

## Overview

Brain Tumour Detection is a production-ready, full-stack application that classifies MRI brain scans into four categories using deep learning:

| Class | Description |
|---|---|
| **Glioma** | Tumour originating from glial cells |
| **Meningioma** | Tumour arising from the meninges |
| **Pituitary** | Tumour of the pituitary gland |
| **No Tumour** | Healthy scan — no detectable mass |

The system supports five model architectures (including MambaVision, the official state-of-the-art vision Mamba model), provides Grad-CAM visual explanations, offers async training with experiment tracking, and ships with a React dashboard for real-time monitoring.

---

## Architecture

```
Browser (React 18 + Vite + Tailwind CSS)
         │  port 3000
         ▼
  Node.js / Express API          ← port 5000
  ├── /api/upload                 Multer file upload + SQLite storage
  ├── /api/preprocess             OpenCV preprocessing pipeline
  ├── /api/segment                Image segmentation
  ├── /api/features               Feature extraction
  ├── /api/classify               EDN-SVM classifier
  ├── /api/batch                  Batch processing
  ├── /api/results                Result retrieval
  ├── /api/metrics                Aggregated metrics
  └── /api/compare                Model comparison
         │  proxies AI requests
         ▼
  Python / FastAPI / PyTorch     ← port 8000
  ├── /api/v1/health              Liveness probe
  ├── /api/v1/predict             Single-image inference + Grad-CAM
  ├── /api/v1/train               Model training (sync + async)
  ├── /api/v1/evaluate            Model evaluation on test set
  ├── /api/v1/dataset/*           Dataset management
  ├── /api/v1/preprocess/*        Image preprocessing & quality check
  ├── /api/v1/auth/*              JWT authentication + RBAC
  ├── /api/v1/performance/*       Performance monitoring + benchmarking
  └── /api/v1/dashboard/*         Metrics & monitoring dashboard
```

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| **Frontend** | React + Vite + TypeScript | 18 / 5 / 5.3 |
| **Frontend UI** | Tailwind CSS + Recharts | 3.4 / 2.12 |
| **Backend** | Node.js + Express | 20 LTS / 4.21 |
| **Database** | SQLite via better-sqlite3 | — |
| **AI Service** | Python + FastAPI | 3.12 / 0.115.5 |
| **Deep Learning** | PyTorch + torchvision | 2.4.1 / 0.19.1 |
| **MambaVision** | Official NVIDIA implementation | 1.2.0 |
| **Transformers** | Hugging Face (model loader) | 4.43.3 |
| **Computer Vision** | OpenCV + Pillow | 4.10 / 11.0 |
| **Explainability** | Grad-CAM (PyTorch) | custom |
| **Auth** | JWT (python-jose) + bcrypt | HS256 / 4.0 |
| **Rate Limiting** | SlowAPI | 0.1.9 |
| **Container** | Docker + Docker Compose | 24+ / v2 |
| **CI/CD** | GitHub Actions | — |
| **Testing (Python)** | pytest + pytest-asyncio | 8.3 / 0.24 |
| **Testing (Frontend)** | Vitest + Testing Library | 1.3 / 14 |
| **Testing (Backend)** | Jest + Supertest | 30 / 7.1 |

---

## Prerequisites

| Tool | Minimum | Install |
|---|---|---|
| Python | 3.12 | [python.org](https://www.python.org/downloads/) |
| Node.js | 20 LTS | [nodejs.org](https://nodejs.org/) |
| npm | 10 | bundled with Node |
| Docker | 24 | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | v2.20 | bundled with Docker Desktop |
| GNU Make | any | Windows: [gnuwin32](https://gnuwin32.sourceforge.net/packages/make.htm) or WSL |

---

## Quick Start — Docker (Recommended)

The fastest way to run everything:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/brain-tumor-detection.git
cd brain-tumor-detection

# 2. Copy environment templates
cp ai-service/.env.example  ai-service/.env
cp backend/.env.example     backend/.env
cp frontend/.env.example    frontend/.env.local

# 3. Set a strong JWT secret
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> ai-service/.env

# 4. Build and start all services
docker compose -f docker/docker-compose.yml up --build -d

# 5. Follow startup logs
docker compose -f docker/docker-compose.yml logs -f
```

Once healthy:
- **Frontend** → http://localhost:3000
- **Backend API** → http://localhost:5000
- **AI Service + Swagger** → http://localhost:8000/docs

---

## Quick Start — Local Development

### 1. Bootstrap all environments

```bash
make setup
```

On Windows without Make:

```powershell
# AI service
cd ai-service
.\setup_env.ps1

# Backend
cd ..\backend
npm ci
copy .env.example .env

# Frontend
cd ..\frontend
npm ci
copy .env.example .env.local
```

### 2. Configure environment files

```
ai-service/.env    ← AI service config
backend/.env       ← Backend config
frontend/.env.local ← Frontend config
```

See the [Environment Variables Reference](#environment-variables) below for all options.

### 3. Run database migrations

```bash
make migrate
# or
cd backend && node database/migrate.js
```

### 4. Start all three services

```bash
# Terminal 1 — AI Service (port 8000)
cd ai-service
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Backend (port 5000)
cd backend
npm run dev

# Terminal 3 — Frontend (port 3000)
cd frontend
npm run dev
```

---

## Usage

### Train a model

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mambavision",
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "fine_tune": true
  }'
```

### Run inference

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@mri_scan.jpg"
```

### Async training (experiment-tracked)

```bash
# Start async job
curl -X POST http://localhost:8000/api/v1/train/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "resnet50", "epochs": 50}'

# Poll status
curl http://localhost:8000/api/v1/train/status/{job_id}

# List all experiments
curl http://localhost:8000/api/v1/train/experiments
```

---

## Model Architectures

| Architecture | Params (approx.) | Notes |
|---|---|---|
| **MambaVision-T** (default) | 31.8M | State-of-the-art vision Mamba, official NVIDIA implementation |
| **EfficientNet-B3** | 12M | Excellent accuracy/speed tradeoff (torchvision) |
| **ResNet50** | 25.6M | Solid baseline, good generalisation |
| **VGG16** | 138M | Classic, high memory usage |
| **Custom CNN** | ~500K | Lightweight, fast training |

All transfer-learning architectures support two-phase fine-tuning: head training first, then selective backbone unfreezing.

---

## Dataset

The model expects MRI images organised in class subdirectories:

```
dataset/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Prepare the dataset via the API:

```bash
# Validate structure
curl -X POST http://localhost:8000/api/v1/dataset/validate

# Split into train/val/test
curl -X POST http://localhost:8000/api/v1/dataset/prepare \
  -H "Content-Type: application/json" \
  -d '{"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}'
```

---

## Testing

```bash
# Run all test suites
make test

# Python tests only (with coverage)
make test-ai-coverage

# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && npm test
```

| Suite | Framework | Tests |
|---|---|---|
| AI Service | pytest | 1,100+ |
| Frontend | Vitest | 280+ |
| Backend | Jest | 112+ |
| **Total** | | **1,492+** |

---

## Environment Variables

### AI Service (`ai-service/.env`)

| Variable | Default | Description |
|---|---|---|
| `AI_SERVICE_HOST` | `0.0.0.0` | Bind address |
| `AI_SERVICE_PORT` | `8000` | Listen port |
| `AI_SERVICE_ENV` | `development` | `development` \| `production` |
| `AI_SERVICE_DEBUG` | `true` | Enable debug mode |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | Comma-separated CORS origins |
| `ACTIVE_MODEL` | `mambavision` | `mambavision` \| `cnn` \| `vgg16` \| `resnet50` \| `efficientnet` |
| `IMAGE_SIZE` | `224` | Input image dimension (pixels) |
| `CLASS_NAMES` | `glioma,meningioma,notumor,pituitary` | Comma-separated class labels |
| `SAVED_MODELS_DIR` | `./saved_models` | Trained PyTorch model directory |
| `DATASET_RAW_DIR` | `./dataset/raw` | Raw MRI image dataset |
| `DATASET_PROCESSED_DIR` | `./dataset/processed` | Preprocessed images |
| `GRADCAM_OUTPUT_DIR` | `./gradcam_output` | Grad-CAM PNG output |
| `JWT_SECRET_KEY` | *(change me!)* | HS256 signing secret |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token lifetime |
| `BCRYPT_ROUNDS` | `12` | bcrypt cost factor |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

### Backend (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Express listen port |
| `NODE_ENV` | `development` | `development` \| `production` \| `test` |
| `FRONTEND_URL` | `http://localhost:3000` | CORS allowed origin |
| `UPLOAD_DIR` | `../uploads` | Multer upload directory |
| `DB_PATH` | `./database/brain_tumor.db` | SQLite database file |
| `AI_SERVICE_URL` | `http://localhost:8000` | AI service base URL |

### Frontend (`frontend/.env.local`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:5000` | Backend API base URL |
| `VITE_AI_SERVICE_URL` | `http://localhost:8000` | AI service base URL |
| `VITE_APP_NAME` | `Brain Tumour Detection` | App display name |
| `VITE_ENABLE_GRADCAM` | `true` | Show Grad-CAM overlay |

---

## Makefile Targets

```
make setup               Bootstrap all three environments
make dev                 Start all services (tmux session)
make test                Run all test suites
make build               Build all production artefacts
make docker-up           Build and start all containers (detached)
make docker-down         Stop and remove containers
make docker-logs         Tail logs from all containers
make migrate             Run SQLite database migrations
make clean               Remove build artefacts and logs
```

Run `make help` for the full list.

---

## Documentation

| Document | Description |
|---|---|
| [Installation Guide](docs/installation.md) | Step-by-step setup for all platforms |
| [User Guide](docs/user_guide.md) | Dataset prep, training, inference, dashboard |
| [Developer Guide](docs/developer_guide.md) | Project structure, standards, extending |
| [API Reference](docs/api_reference.md) | All REST endpoints with examples |
| [Architecture](docs/project_architecture.md) | System design and data flows |
| [Deployment Guide](docs/deployment.md) | Docker, CI/CD, production |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [FAQ](docs/faq.md) | Frequently asked questions |
| [Release Notes](docs/release_notes.md) | Version history |
| [Final Report](docs/final_report.md) | Project statistics and summary |
| [Performance Guide](docs/performance.md) | Profiling, benchmarking, optimisation |
| [Security Architecture](docs/authentication_architecture.md) | Auth, JWT, RBAC details |
| [CI/CD Guide](docs/cicd-guide.md) | GitHub Actions pipeline reference |
| [Docker Guide](docs/docker-guide.md) | Container configuration reference |
| [Production Checklist](docs/production-checklist.md) | Pre-launch checklist |
| [Changelog](CHANGELOG.md) | Module-by-module feature changelog |

---

## Project Structure

```
brain-tumor-detection/
├── ai-service/                  Python / FastAPI / PyTorch
│   ├── app/
│   │   ├── api/                 REST route handlers
│   │   ├── core/                Config, logging
│   │   ├── dataset/             Validation, splitting, stats
│   │   ├── inference/           Pipeline, batch, cache, results
│   │   ├── metrics/             System, inference, training, dashboard
│   │   ├── models/              Architectures, train, predict, evaluate
│   │   ├── performance/         Profiler, benchmark, cache, memory
│   │   ├── preprocessing/       Image pipeline, augmentation, quality
│   │   ├── security/            JWT, auth, roles, rate limiting, audit
│   │   ├── training/            Job store, experiment registry
│   │   └── utils/               Grad-CAM
│   ├── dataset/                 raw/ and processed/ MRI images
│   ├── saved_models/            Trained PyTorch checkpoints
│   ├── tests/                   pytest suite (1,100+ tests)
│   ├── Dockerfile
│   └── requirements.txt
│
├── backend/                     Node.js / Express / SQLite
│   ├── api/                     Route handlers (9 modules)
│   ├── database/                Schema SQL, migrations, db.js
│   ├── middleware/              Upload, error handling, validation
│   ├── pipeline/                Preprocessing, segmentation, classifier
│   ├── server.js
│   └── package.json
│
├── frontend/                    React 18 / Vite / TypeScript
│   ├── src/
│   │   ├── components/          Reusable UI components
│   │   ├── pages/               Route-level page components
│   │   ├── hooks/               Custom React hooks
│   │   ├── api/                 Axios API clients
│   │   ├── context/             React context providers
│   │   └── types/               TypeScript type definitions
│   ├── package.json
│   └── vite.config.ts
│
├── docker/                      Docker Compose configurations
│   ├── docker-compose.yml       Base configuration
│   ├── docker-compose.dev.yml   Development overrides
│   ├── docker-compose.prod.yml  Production overrides
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── docs/                        Project documentation
├── scripts/                     Deployment and maintenance scripts
├── .github/workflows/           CI/CD pipeline definitions
├── Makefile                     Developer task automation
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues, submitting pull requests, and the development workflow.

---

## Security

To report a security vulnerability, please follow the process described in [SECURITY.md](SECURITY.md). Do not open a public GitHub issue for security concerns.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle) for training data
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- [NVIDIA](https://github.com/NVlabs/MambaVision) for the official MambaVision implementation
- [Hugging Face](https://huggingface.co/) for the Transformers library and model hub
- [FastAPI](https://fastapi.tiangolo.com/) for the elegant Python web framework
