# Project Architecture

Brain Tumour Detection — system design, data flows, and folder structure.

---

## Table of Contents

1. [Overall System Architecture](#overall-system-architecture)
2. [Three-Tier Application Stack](#three-tier-application-stack)
3. [AI Pipeline](#ai-pipeline)
4. [Dataset Flow](#dataset-flow)
5. [Training Workflow](#training-workflow)
6. [Inference Workflow](#inference-workflow)
7. [Frontend Architecture](#frontend-architecture)
8. [Backend Architecture](#backend-architecture)
9. [Security Architecture](#security-architecture)
10. [Deployment Architecture](#deployment-architecture)
11. [Folder Structure Reference](#folder-structure-reference)
12. [Technology Decision Log](#technology-decision-log)

---

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser / Client                            │
│                    React 18 + Vite + TypeScript                     │
│                        http://localhost:3000                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTP (JSON / multipart)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Node.js / Express Backend                       │
│                          port 5000                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│   │  upload  │  │preprocess│  │ classify │  │ metrics/results  │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
│              SQLite (better-sqlite3)   Multer uploads               │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTP (JSON / multipart)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Python / FastAPI / PyTorch / MambaVision               │
│                     AI Service — port 8000                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  API Layer  (routes.py / auth_routes.py / perf_routes.py)   │   │
│   └────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│   ┌──────────┐  ┌─────────────────┐  ┌──────────┐  ┌───────────┐   │
│   │ Dataset  │  │  Preprocessing  │  │  Models  │  │ Inference │   │
│   │ Manager  │  │    Pipeline     │  │(PyTorch) │  │ Pipeline  │   │
│   └──────────┘  └─────────────────┘  └──────────┘  └───────────┘   │
│   ┌──────────┐  ┌─────────────────┐  ┌──────────┐  ┌───────────┐   │
│   │ Security │  │    Training     │  │ Metrics  │  │ Perf Mon  │   │
│   │  & Auth  │  │ (Async + Exp.)  │  │Dashboard │  │Benchmarks │   │
│   └──────────┘  └─────────────────┘  └──────────┘  └───────────┘   │
│                                                                     │
│   PyTorch weights   Dataset (raw/processed)   Logs / Audit         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Three-Tier Application Stack

### Tier 1 — Frontend (Presentation)

- **Technology:** React 18, Vite 5, TypeScript 5, Tailwind CSS 3
- **Responsibilities:** User interface, file upload, result display, Grad-CAM overlay, metrics charts
- **Communicates with:** Node.js Backend via `VITE_API_BASE_URL`; AI Service directly via `VITE_AI_SERVICE_URL` for some calls
- **Build output:** Static files served by nginx in production

### Tier 2 — Backend (Application Logic)

- **Technology:** Node.js 20, Express 4, SQLite (better-sqlite3)
- **Responsibilities:** File storage, SQLite persistence, request orchestration, traditional ML pipeline (EDN-SVM classifier), proxying to AI service
- **Communicates with:** AI Service via `AI_SERVICE_URL`; SQLite for persistence
- **State:** SQLite database, uploads directory

### Tier 3 — AI Service (Deep Learning)

- **Technology:** Python 3.12, FastAPI 0.115, PyTorch 2.x, MambaVision 1.2 (HuggingFace Transformers)
- **Responsibilities:** Model training, inference, Grad-CAM, dataset management, preprocessing pipeline, authentication, monitoring
- **State:** PyTorch weights (saved_models/), dataset (dataset/), logs, Grad-CAM output

---

## AI Pipeline

```
                 ┌─────────────┐
  MRI Image ───▶│ Quality     │
  (JPEG/PNG)    │ Check       │
                └──────┬──────┘
                       │ pass
                       ▼
                ┌─────────────┐
                │Preprocessing│ resize → denoise → CLAHE → normalise
                │ Pipeline    │ (OpenCV + Pillow)
                └──────┬──────┘
                       │ 224×224×3 float32 tensor
                       ▼
                ┌─────────────┐
                │  PyTorch    │ MambaVision (default) / CNN / ResNet50
                │   Model     │ softmax output (4 classes)
                └──────┬──────┘
                       │ class probabilities
                       ▼
                ┌─────────────┐
                │  Grad-CAM   │ PyTorch hooks: guided backpropagation
                │  Heatmap    │ → PNG overlay base64
                └──────┬──────┘
                       │
                       ▼
               Prediction Response
               { prediction, confidence, probabilities, gradcam_b64 }
```

---

## Dataset Flow

```
Raw Dataset (dataset/raw/)
├── Training/
│   ├── glioma/     [images]
│   ├── meningioma/ [images]
│   ├── notumor/    [images]
│   └── pituitary/  [images]
└── Testing/
    └── ... (same structure)

         │
         ▼  POST /dataset/validate
         │
         ▼  POST /dataset/prepare
         │  Stratified split
         │  (70% train / 15% val / 15% test by default)
         │  Computes class weights for imbalanced training
         │
Processed Dataset (dataset/processed/)
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── val/
│   └── ... (same)
└── test/
    └── ... (same)

         │
         ▼  GET /dataset/info → dataset_info.json
         │  { total_images, split counts, class_weights, prepared_at }
```

---

## Training Workflow

### Synchronous (POST /train)

```
Client request
     │
     ▼
Validate request parameters (Pydantic)
     │
     ▼
Build model (architectures.py MODEL_REGISTRY)
     │
     ▼
Phase 1 — Head training
   Backbone frozen
   Train classification head
   Early stopping on val_loss
   save_best_checkpoint() → saved_models/{name}/weights.pt
     │
     ▼
Phase 2 — Fine-tuning (if fine_tune=true)
   Unfreeze last N backbone layers
   Lower learning rate (÷10)
   Additional epochs
   save_best_checkpoint() updates if val_acc improves
     │
     ▼
Save final weights → saved_models/{name}/
Save history → ExperimentRegistry
     │
     ▼
Return training history + metrics
```

### Asynchronous (POST /train/start)

```
Client POST /train/start
     │
     ▼
Create job entry in JobStore (status=pending)
Return job_id immediately (202)
     │
     ▼ (background thread via FastAPI BackgroundTasks)
Same training flow as above
JobStore updates status: pending → running → completed/failed
Per-epoch progress recorded in JobStore
     │
Client polls GET /train/status/{job_id}
     │
Client reads GET /train/experiments/{id} for full history
```

---

## Inference Workflow

```
Single image:
POST /predict (multipart file)
         │
         ▼
InferencePipeline.predict()
         │
         ├─▶ LRU Model Cache
         │   ├── MISS → load_model() → cache → use
         │   └── HIT  → use cached model
         │
         ├─▶ Preprocessing Pipeline
         │   resize → denoise → CLAHE → normalise → tensor
         │
         ├─▶ model.predict(tensor) → probabilities[4]
         │
         ├─▶ Grad-CAM generation
         │   PyTorch hooks GradCAM → PNG → base64
         │
         ├─▶ Record to InferenceMetrics
         │   (latency, class distribution, success/failure)
         │
         └─▶ Return PredictionResponse

Batch (POST /predict/batch):
         │
         ▼
BatchInferenceRunner (ThreadPoolExecutor)
For each image → InferencePipeline.predict()
Aggregates results → returns list
```

---

## Frontend Architecture

```
src/
├── main.tsx              App bootstrap, React Router setup
├── App.tsx               Root component, layout
│
├── api/                  Axios API clients
│   ├── aiService.ts      Calls to FastAPI (predict, train, metrics)
│   └── backendApi.ts     Calls to Express (upload, results)
│
├── context/              React Context providers
│   ├── AuthContext.tsx   JWT token state, login/logout
│   └── AppContext.tsx    Global app state
│
├── hooks/                Custom React hooks
│   ├── usePredict.ts     Prediction request + state management
│   ├── useTraining.ts    Training job polling
│   └── useMetrics.ts     Dashboard data polling (30s interval)
│
├── pages/                Route-level components
│   ├── PredictPage.tsx   Single-image inference UI
│   ├── BatchPage.tsx     Batch inference UI
│   ├── TrainPage.tsx     Training configuration + progress
│   ├── DatasetPage.tsx   Dataset management UI
│   └── DashboardPage.tsx Metrics charts + system stats
│
├── components/           Reusable UI components
│   ├── ImageUpload.tsx   Dropzone with preview
│   ├── GradCAMOverlay.tsx Heatmap overlay canvas
│   ├── ProbabilityChart.tsx Recharts bar chart
│   ├── MetricsCard.tsx   Dashboard stat card
│   ├── TrainingProgress.tsx Live training chart
│   └── NavBar.tsx        Top navigation
│
└── types/                TypeScript type definitions
    ├── prediction.ts
    ├── training.ts
    └── metrics.ts
```

**State management:** React Context API + custom hooks. No external state library (Zustand/Redux) needed at current scale.

**Data fetching:** Axios with interceptors that attach `Authorization: Bearer <token>` automatically.

**Routing:** React Router v6 with lazy-loaded routes.

---

## Backend Architecture

```
server.js                  Express app factory
├── Middleware
│   ├── helmet             Security headers
│   ├── cors               CORS for frontend origin
│   ├── morgan             HTTP request logging
│   └── multer             Multipart file upload handler
│
├── Routes (api/)
│   ├── /api/upload        File upload → saves to uploads/, records in SQLite
│   ├── /api/preprocess    OpenCV preprocessing via jimp
│   ├── /api/segment       Image segmentation
│   ├── /api/features      Feature vector extraction
│   ├── /api/classify      EDN-SVM classifier (models/edn_svm.json)
│   ├── /api/batch         Batch processing coordinator
│   ├── /api/results       Result retrieval from SQLite
│   ├── /api/metrics       Aggregated metrics queries
│   └── /api/compare       Model comparison endpoint
│
├── Pipeline (pipeline/)
│   ├── preprocessing.js   Image normalisation, resizing
│   ├── segmentation.js    Region-of-interest detection
│   ├── features.js        HOG / LBP feature extraction
│   └── dl_bridge.js       HTTP client → AI service
│
└── Database (database/)
    ├── schema.sql         SQLite table definitions
    ├── migrate.js         Idempotent migration runner
    └── db.js              better-sqlite3 connection singleton
```

---

## Security Architecture

See [Authentication Architecture](authentication_architecture.md) for full detail.

```
Request
   │
   ▼ SlowAPI RateLimiter (per-endpoint, per-IP)
   │
   ▼ JWT Bearer Token validation (HTTPBearer dependency)
   │   ├── Decode HS256 signed token (python-jose)
   │   ├── Check token revocation set (in-process dict)
   │   └── Resolve user from UserStore
   │
   ▼ Role-Based Access Control
   │   ├── Viewer    — read-only endpoints
   │   ├── Operator  — inference + benchmarks
   │   ├── Researcher — training + dataset + inference
   │   └── Admin     — full access + user management
   │
   ▼ Endpoint handler
   │
   ▼ Audit Log (JSONL → logs/audit/)
       AuditEvent: LOGIN, LOGOUT, PREDICTION, TRAINING, USER_CREATED…
```

**Password security:** bcrypt with 12 rounds. Passwords never stored in plaintext.  
**Account lockout:** 5 consecutive failures → 15-minute lockout (cleared by Admin).  
**Token revocation:** In-process revocation set (extend with Redis for multi-node deployments).

---

## Deployment Architecture

```
Production:

Internet
   │
   ▼
nginx (port 80/443)
   ├── / ──────────────▶ frontend static files
   ├── /api/* ─────────▶ backend:5000  (Node.js)
   └── /ai/* ──────────▶ ai-service:8000 (FastAPI)

Docker Compose services:
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  frontend   │  │  backend    │  │ ai-service  │  │    nginx    │
   │  (nginx)    │  │ (Node 20)   │  │  (Python)   │  │  (proxy)    │
   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
         │                │                 │
         └────────────────┴─────────────────┘
                          │
                   Docker networks: app-network
                   Docker volumes:
                      models_volume      ← PyTorch weights
                      dataset_volume     ← MRI images
                      uploads_volume     ← User uploads
                      db_volume          ← SQLite DB
                      gradcam_volume     ← Heatmaps
                      logs_volume        ← Application logs

CI/CD (GitHub Actions):
   Push to main  ──▶ ci.yml (lint → test → build → security scan)
                 ──▶ cd.yml (auto-deploy to staging)
   Release tag   ──▶ release.yml (versioned Docker push → GitHub Release)
                 ──▶ cd.yml (deploy to production)
```

---

## Folder Structure Reference

```
brain-tumor-detection/                Root
├── ai-service/                       Python AI service
│   ├── app/                          Application source
│   │   ├── api/                      REST route handlers (3 files)
│   │   ├── core/                     Config singleton, logging setup
│   │   ├── dataset/                  Dataset manager (4 modules)
│   │   ├── inference/                Inference pipeline (5 modules)
│   │   ├── metrics/                  Monitoring dashboard (5 modules)
│   │   ├── models/                   DL architectures + train/predict (7 modules)
│   │   ├── performance/              Performance monitoring (7 modules)
│   │   ├── preprocessing/            Image preprocessing (5 modules)
│   │   ├── security/                 Auth, JWT, RBAC (8 modules)
│   │   ├── training/                 Async job management (1 module)
│   │   ├── utils/                    Grad-CAM utility (1 module)
│   │   └── main.py                   Application factory
│   ├── dataset/                      raw/ and processed/ MRI data
│   ├── saved_models/                 Trained PyTorch weights (.pt / HF format)
│   ├── gradcam_output/               Generated heatmap PNGs
│   ├── logs/                         Application + audit logs
│   ├── tests/                        pytest test suite (15+ files)
│   ├── Dockerfile                    Multi-stage Python 3.12-slim image
│   ├── requirements.txt              Production Python dependencies
│   ├── requirements-dev.txt          Development tools (ruff, black…)
│   └── pyproject.toml               Tool configuration
│
├── backend/                          Node.js Express service
│   ├── api/                          Route modules (9 files)
│   ├── baselines/                    Baseline model JSON files
│   ├── database/                     SQLite schema, migrations, connection
│   ├── middleware/                   Upload, error, validation middleware
│   ├── models/                       EDN-SVM model JSON
│   ├── pipeline/                     Preprocessing, segmentation, classifier
│   ├── training/                     Backend training scripts
│   ├── utils/                        Shared utilities
│   ├── uploads/                      User-uploaded files
│   ├── server.js                     Express app entry point
│   └── package.json
│
├── frontend/                         React + Vite + TS
│   ├── src/
│   │   ├── api/                      Axios API clients
│   │   ├── components/               Reusable UI components
│   │   ├── context/                  React context providers
│   │   ├── hooks/                    Custom React hooks
│   │   ├── pages/                    Route-level page components
│   │   └── types/                    TypeScript type definitions
│   ├── index.html                    Vite entry HTML
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── package.json
│
├── docker/                           Container configuration
│   ├── docker-compose.yml            Base compose file
│   ├── docker-compose.dev.yml        Development overrides
│   ├── docker-compose.prod.yml       Production overrides
│   ├── Dockerfile.backend            Multi-stage Node.js image
│   └── Dockerfile.frontend           Multi-stage Vite → nginx image
│
├── docs/                             Documentation
│   ├── installation.md
│   ├── user_guide.md
│   ├── developer_guide.md
│   ├── api_reference.md
│   ├── project_architecture.md       (this file)
│   ├── deployment.md
│   ├── troubleshooting.md
│   ├── faq.md
│   ├── release_notes.md
│   ├── final_report.md
│   ├── performance.md
│   ├── authentication_architecture.md
│   ├── cicd-guide.md
│   ├── docker-guide.md
│   └── production-checklist.md
│
├── scripts/                          Operational scripts
│   ├── deploy.sh / deploy.ps1        Deployment with rollback
│   ├── backup.sh                     Docker volume backup
│   ├── restore.sh                    Volume restore
│   ├── validate-env.sh / .ps1        Environment file validation
│   └── bump-version.sh               Semantic version bump
│
├── .github/workflows/
│   ├── ci.yml                        CI pipeline
│   ├── cd.yml                        CD pipeline
│   └── release.yml                   Release automation
│
├── notebooks/                        Jupyter research notebooks
├── tests/                            Cross-service integration tests
├── uploads/                          Shared upload directory
├── Makefile                          Developer task automation
├── VERSION                           Current semantic version
├── CHANGELOG.md                      Version history
├── CONTRIBUTING.md                   Contribution guidelines
├── CODE_OF_CONDUCT.md                Community standards
├── SECURITY.md                       Vulnerability reporting
├── LICENSE                           MIT license
└── README.md                         Project overview
```

---

## Technology Decision Log

| Decision | Choice | Alternative Considered | Reason |
|---|---|---|---|
| AI framework | PyTorch 2.x + HuggingFace Transformers | TensorFlow 2 / Keras | State-of-the-art vision transformer support; native Grad-CAM hooks; MambaVision integration; active research ecosystem |
| Model architecture | MambaVision (default) | EfficientNetB0, ResNet50, CNN | Best accuracy on brain tumour benchmarks; SSM-based hybrid transformer; HuggingFace pretrained weights |
| Grad-CAM | PyTorch native hooks | tf-explain | Framework consistency; no additional dependency; full control over layer selection |
| Python web framework | FastAPI | Flask, Django | Async support; automatic OpenAPI docs; Pydantic validation |
| Node.js DB | SQLite (better-sqlite3) | PostgreSQL | Zero-ops for development; scales to production with swap |
| Frontend build | Vite 5 | Create React App, Next.js | Fastest HMR; lightweight; SSR not needed |
| CSS framework | Tailwind CSS 3 | styled-components, MUI | Utility-first; no CSS-in-JS overhead; good with TypeScript |
| Auth approach | JWT HS256 | Session cookies, OAuth | Stateless; works across microservices; simple for API clients |
| Rate limiting | SlowAPI | express-rate-limit | Native FastAPI/Starlette integration |
| Containerisation | Docker + Compose | Kubernetes | Sufficient for current scale; easy local dev |
| CI | GitHub Actions | Jenkins, CircleCI | Free for public repos; native GitHub integration |

> **Migration note:** The project originally used TensorFlow 2.20 / Keras 3 with EfficientNetB0 as the default model.
> The framework was migrated to PyTorch + MambaVision (Module 5) to gain access to state-of-the-art
> vision transformer architectures and simplify the Grad-CAM implementation.
> Legacy TF scripts (`download_and_train.py`, `post_train_validate.py`, `restore_checkpoint.py`) are retained
> in the repository root marked as DEPRECATED and will be removed in a future release.
