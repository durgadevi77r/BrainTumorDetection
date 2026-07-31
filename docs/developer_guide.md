# Developer Guide

Brain Tumour Detection — development setup, coding standards, project conventions, and extension points.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Environment Setup](#development-environment-setup)
3. [Coding Standards](#coding-standards)
4. [Running Tests](#running-tests)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Docker Usage](#docker-usage)
7. [Adding New Model Architectures](#adding-new-model-architectures)
8. [Adding New API Endpoints](#adding-new-api-endpoints)
9. [Adding Frontend Pages](#adding-frontend-pages)
10. [Database Migrations](#database-migrations)
11. [Environment Variables](#environment-variables)
12. [Logging](#logging)
13. [Security Considerations](#security-considerations)
14. [Pre-commit Hooks](#pre-commit-hooks)
15. [Release Process](#release-process)

---

## Project Structure

```
brain-tumor-detection/
├── ai-service/                       Python / FastAPI / PyTorch / MambaVision
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py             Core endpoints (health, predict, train…)
│   │   │   ├── auth_routes.py        Authentication endpoints
│   │   │   ├── performance_routes.py Performance monitoring endpoints
│   │   │   └── __init__.py
│   │   ├── core/
│   │   │   ├── config.py             Pydantic-Settings configuration singleton
│   │   │   └── logging.py            Loguru logger setup
│   │   ├── dataset/
│   │   │   ├── validator.py          Raw dataset structure validation
│   │   │   ├── splitter.py           Stratified train/val/test split
│   │   │   ├── stats.py              Per-class statistics
│   │   │   └── metadata.py           dataset_info.json read/write
│   │   ├── inference/
│   │   │   ├── pipeline.py           InferencePipeline with LRU model cache
│   │   │   ├── batch.py              BatchInferenceRunner (thread pool + ZIP)
│   │   │   ├── cache.py              Prediction result cache
│   │   │   ├── config.py             InferenceConfig validation
│   │   │   └── results.py            Result storage
│   │   ├── metrics/
│   │   │   ├── system.py             CPU, RAM, disk, GPU metrics
│   │   │   ├── inference.py          Latency percentiles, class distribution
│   │   │   ├── training.py           Job counts, best accuracy
│   │   │   ├── storage.py            JSONL time-series storage
│   │   │   └── dashboard.py          Composite dashboard endpoint
│   │   ├── models/
│   │   │   ├── architectures.py      MambaVision-T, EfficientNet-B3, ResNet50, VGG16, CNN builders
│   │   │   ├── train.py              Training loop, callbacks, checkpoints
│   │   │   ├── predict.py            Single-image prediction
│   │   │   ├── evaluate.py           Test-set evaluation
│   │   │   ├── load_model.py         PyTorch model loading + cache (HF + PT formats)
│   │   │   └── save_model.py         PyTorch model persistence (HF + state_dict)
│   │   ├── performance/
│   │   │   ├── profiler.py           cProfile-based function timer
│   │   │   ├── benchmark.py          BenchmarkSuite across all modules
│   │   │   ├── optimizer.py          APIOptimizer per-endpoint stats
│   │   │   ├── cache.py              Cache analytics and recommendations
│   │   │   ├── memory.py             RSS + tracemalloc leak detection
│   │   │   ├── concurrency.py        ThreadPool stress testing
│   │   │   └── reports.py            JSON + HTML report generation
│   │   ├── preprocessing/
│   │   │   ├── preprocess.py         Denoise + CLAHE spatial pipeline
│   │   │   ├── augmentation.py       Training augmentation stack
│   │   │   ├── transforms.py         Individual image transforms
│   │   │   ├── quality.py            Blur, intensity, variance checks
│   │   │   └── config.py             Preprocessing configuration
│   │   ├── security/
│   │   │   ├── auth.py               User store, authenticate_user
│   │   │   ├── jwt.py                Token creation, decoding, revocation
│   │   │   ├── password.py           bcrypt hashing, strength validation
│   │   │   ├── roles.py              Role enum (Admin, Researcher, Operator, Viewer)
│   │   │   ├── dependencies.py       FastAPI dependency injection
│   │   │   ├── rate_limit.py         SlowAPI limiter + limit constants
│   │   │   └── audit.py              JSONL audit event logging
│   │   ├── training/
│   │   │   └── job_store.py          Async job state + experiment registry
│   │   ├── utils/
│   │   │   └── gradcam.py            Grad-CAM heatmap generation
│   │   └── main.py                   FastAPI app factory + lifespan
│   ├── tests/                        pytest suite
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── pyproject.toml               pytest / ruff / black / isort config
│   └── setup_env.sh / .ps1
│
├── backend/                          Node.js / Express
│   ├── api/                          Route modules (9 files)
│   ├── database/
│   │   ├── schema.sql
│   │   ├── migrate.js
│   │   └── db.js
│   ├── middleware/
│   ├── pipeline/                     Preprocessing, segmentation, classifier
│   ├── tests/
│   ├── server.js
│   └── package.json
│
├── frontend/                         React 18 / Vite / TypeScript
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── api/
│   │   ├── context/
│   │   └── types/
│   ├── package.json
│   └── vite.config.ts
│
├── docker/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── docs/                             All documentation
├── scripts/                          Operational scripts
│   ├── deploy.sh / deploy.ps1
│   ├── backup.sh / restore.sh
│   ├── validate-env.sh / .ps1
│   └── bump-version.sh
│
├── .github/workflows/
│   ├── ci.yml                        Lint → Test → Docker build → Security scan
│   ├── cd.yml                        Staging auto-deploy + production release
│   └── release.yml                   Versioned Docker tags + GitHub Release
│
├── Makefile
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── VERSION
└── README.md
```

---

## Development Environment Setup

### Full bootstrap

```bash
# Linux / macOS
make setup

# Windows — run each manually
cd ai-service && .\setup_env.ps1
cd backend    && npm ci && copy .env.example .env
cd frontend   && npm ci && copy .env.example .env.local
```

### Python virtual environment

```bash
cd ai-service
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### VS Code recommended extensions

- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Ruff** (charliermarsh.ruff)
- **ESLint** (dbaeumer.vscode-eslint)
- **Prettier** (esbenp.prettier-vscode)
- **Tailwind CSS IntelliSense** (bradlc.vscode-tailwindcss)
- **Docker** (ms-azuretools.vscode-docker)

---

## Coding Standards

### Python (ai-service)

| Tool | Config | Purpose |
|---|---|---|
| **ruff** | `pyproject.toml [tool.ruff]` | Linting (replaces flake8 + pylint) |
| **black** | `pyproject.toml [tool.black]` | Code formatting (line length 100) |
| **isort** | `pyproject.toml [tool.isort]` | Import sorting (black-compatible) |
| **mypy** | — | Type checking (optional but encouraged) |

Run all checks:
```bash
cd ai-service
ruff check app/ tests/
black --check app/ tests/
isort --check-only app/ tests/
```

Auto-fix:
```bash
ruff check --fix app/ tests/
black app/ tests/
isort app/ tests/
```

Key conventions:
- Use `from __future__ import annotations` at the top of all modules
- All public functions and classes must have docstrings
- Use Pydantic models for all request/response schemas
- Prefer `Path` objects over string paths
- Import the `settings` singleton rather than reading env vars directly
- Use `logger` from `app.core.logging` — never `print()`

### TypeScript / React (frontend)

| Tool | Config | Purpose |
|---|---|---|
| **ESLint** | `.eslintrc.cjs` | Linting with jsx-a11y and react-hooks rules |
| **Prettier** | `.prettierrc` | Formatting (2-space indent, single quotes) |
| **TypeScript** | `tsconfig.json` | Strict mode enabled |

Run checks:
```bash
cd frontend
npm run lint
npm run format:check
npm run type-check
```

Auto-fix:
```bash
npm run lint:fix
npm run format
```

Key conventions:
- No `any` types — use proper TypeScript types
- Use React functional components with hooks
- All API calls go through the `src/api/` layer
- Error boundaries must wrap all route-level components
- Use `aria-*` attributes on interactive elements for accessibility

### Node.js (backend)

| Tool | Config | Purpose |
|---|---|---|
| **ESLint** | `.eslintrc.js` | Linting |
| **Prettier** | `.prettierrc` | Formatting |

---

## Running Tests

### AI Service (pytest)

```bash
cd ai-service
source .venv/bin/activate

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html

# Single test file
pytest tests/test_security.py -v

# Single test
pytest tests/test_health.py::test_health_endpoint -v

# Skip slow tests (e.g. model loading)
pytest tests/ -v -m "not slow"
```

### Frontend (Vitest)

```bash
cd frontend

# Single run
npm test

# Watch mode
npm run test:watch

# With coverage
npm run test:coverage
```

### Backend (Jest)

```bash
cd backend

# All tests (runs sequentially due to SQLite)
npm test

# With verbose output
npx jest --verbose
```

### All suites at once

```bash
# From project root
make test
```

---

## CI/CD Pipeline

Three GitHub Actions workflows live in `.github/workflows/`:

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | Push / PR to `main`, `develop` | lint-ai, lint-frontend, lint-backend, test-ai, test-frontend, test-backend, docker-build, security-scan |
| `cd.yml` | Push to `main` → staging; release tag → production | deploy-staging, deploy-production |
| `release.yml` | Git tag `v*.*.*` | Build + push versioned Docker images, create GitHub Release |

See [CI/CD Guide](cicd-guide.md) for full pipeline documentation.

---

## Docker Usage

### Build individual images

```bash
# AI service
docker build -f ai-service/Dockerfile -t brain-tumor-ai:latest ai-service/

# Backend
docker build -f docker/Dockerfile.backend -t brain-tumor-backend:latest backend/

# Frontend
docker build -f docker/Dockerfile.frontend -t brain-tumor-frontend:latest frontend/
```

### Development stack (with hot reload)

```bash
docker compose -f docker/docker-compose.yml \
               -f docker/docker-compose.dev.yml \
               up --build
```

### Production stack

```bash
docker compose -f docker/docker-compose.yml \
               -f docker/docker-compose.prod.yml \
               up --build -d
```

### Useful commands

```bash
# Shell into running container
docker compose exec ai-service bash

# View AI service logs
docker compose logs -f ai-service

# Rebuild only one service
docker compose up --build --no-deps ai-service

# Check health status
docker compose ps
```

---

## Adding New Model Architectures

### 1. Define the architecture

Add a builder function to `ai-service/app/models/architectures.py`:

```python
import torch.nn as nn

def build_my_model(
    num_classes: int = 4,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    MyModel — brief description.

    Args:
        num_classes: Number of output classes.
        dropout_rate: Dropout regularisation rate.

    Returns:
        Initialised nn.Module (weights are random — load or train before use).
    """
    return nn.Sequential(
        # ... add backbone layers ...
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes),
    )
```

### 2. Register the architecture

In `architectures.py`, add your builder to the `_BUILDERS` dict:

```python
_BUILDERS: Dict[str, Callable] = {
    "mambavision": build_mambavision,
    "cnn":          build_cnn,
    "vgg16":        build_vgg16,
    "resnet50":     build_resnet50,
    "efficientnet": build_efficientnet,
    "my_model":     build_my_model,   # ← add here
}
```

### 3. Update validation

In `ai-service/app/core/config.py`, add the new name to the allowed set:

```python
@field_validator("active_model")
@classmethod
def validate_model_name(cls, v: str) -> str:
    allowed = {"mambavision", "cnn", "vgg16", "resnet50", "efficientnet", "my_model"}
    ...
```

Also update the `TrainRequest.model_name` field description in `routes.py`.

### 4. Update the health endpoint

The `health_check` endpoint in `routes.py` builds `models_available` from a hardcoded list — add the new name:

```python
supported = ["mambavision", "cnn", "vgg16", "resnet50", "efficientnet", "my_model"]
```

### 5. Write tests

Add a test in `ai-service/tests/` following the pattern in `test_training_trainer.py`.

---

## Adding New API Endpoints

### AI Service (FastAPI)

#### 1. Choose the right router file

| File | Use for |
|---|---|
| `app/api/routes.py` | Core AI operations |
| `app/api/auth_routes.py` | Authentication |
| `app/api/performance_routes.py` | Performance monitoring |

Or create a new router file for a new feature group.

#### 2. Define request/response models

```python
class MyRequest(BaseModel):
    param_a: str = Field(..., description="Description of param_a")
    param_b: int = Field(default=10, ge=1, le=100)

class MyResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
```

#### 3. Implement the endpoint

```python
@router.post(
    "/my-endpoint",
    response_model=MyResponse,
    summary="Short summary for Swagger",
    tags=["MyGroup"],
)
@limiter.limit(limits.DASHBOARD)          # choose appropriate rate limit
async def my_endpoint(
    request: Request,
    body: MyRequest,
    current_user: UserInDB = Depends(get_current_active_user),  # require auth
) -> MyResponse:
    """
    Full docstring appears in Swagger UI.
    Describe what the endpoint does, parameters, and return values.
    """
    # implementation
    return MyResponse(success=True, data={...})
```

#### 4. Register a new router

If you created a new router file, register it in `app/main.py`:

```python
from app.api.my_routes import my_router
app.include_router(my_router, prefix="/api/v1")
```

### Backend (Express)

#### 1. Create a route file

```javascript
// backend/api/myFeature.js
const express = require('express');
const router = express.Router();

router.post('/my-endpoint', async (req, res) => {
    try {
        // implementation
        res.json({ success: true, data: {} });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

module.exports = router;
```

#### 2. Register in server.js

```javascript
const myFeatureRouter = require('./api/myFeature');
app.use('/api/my-feature', myFeatureRouter);
```

---

## Adding Frontend Pages

### 1. Create the page component

```tsx
// frontend/src/pages/MyPage.tsx
import React from 'react';

const MyPage: React.FC = () => {
    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-2xl font-bold mb-6">My Page</h1>
            {/* content */}
        </div>
    );
};

export default MyPage;
```

### 2. Add an API client (if needed)

```typescript
// frontend/src/api/myApi.ts
import axios from 'axios';

const API_BASE = import.meta.env.VITE_AI_SERVICE_URL;

export const fetchMyData = async (param: string) => {
    const response = await axios.get(`${API_BASE}/api/v1/my-endpoint`, {
        params: { param },
    });
    return response.data;
};
```

### 3. Register the route

In `frontend/src/main.tsx` (or your router config):

```tsx
import MyPage from './pages/MyPage';

// Inside your <Routes>:
<Route path="/my-page" element={<MyPage />} />
```

### 4. Add to navigation

Add a `<NavLink>` in the navigation bar component.

### 5. Write tests

```tsx
// frontend/src/pages/__tests__/MyPage.test.tsx
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import MyPage from '../MyPage';

describe('MyPage', () => {
    it('renders the heading', () => {
        render(<MemoryRouter><MyPage /></MemoryRouter>);
        expect(screen.getByRole('heading', { name: /my page/i })).toBeInTheDocument();
    });
});
```

---

## Database Migrations

The backend uses SQLite with a custom migration system.

### Create a migration

Add a new SQL file or extend `backend/database/schema.sql`.

### Run migrations

```bash
cd backend
node database/migrate.js
```

Migrations are idempotent — safe to run multiple times.

---

## Environment Variables

Never hard-code secrets. Always:
1. Add new variables to `.env.example` (with a safe default or placeholder)
2. Document them in `README.md` and `docs/installation.md`
3. Validate them in `app/core/config.py` (AI service) using Pydantic validators

For the AI service, all settings are accessed via:
```python
from app.core.config import settings
print(settings.jwt_secret_key)
```

---

## Logging

### AI Service (Loguru)

```python
from app.core.logging import logger

logger.info("Something happened: {}", value)
logger.warning("Watch out: {}", warning)
logger.error("Failed: {}", error)
logger.exception("Unhandled error")    # includes stack trace
```

Log files rotate daily at `ai-service/logs/`. Format: `YYYY-MM-DD.log`.

### Backend (Winston)

```javascript
const logger = require('./utils/logger');
logger.info('Something happened', { context: value });
logger.error('Failed', { error: err.message });
```

---

## Security Considerations

When adding new endpoints, follow these rules:

1. **Always validate input** — use Pydantic models; never trust raw request data
2. **Apply rate limiting** — add `@limiter.limit(limits.APPROPRIATE_LIMIT)`
3. **Require authentication** — add `Depends(get_current_active_user)` unless the endpoint is intentionally public
4. **Apply role-based access** — use `Depends(require_roles(Role.ADMIN))` for admin-only endpoints
5. **Log audit events** — call `log_audit(AuditEvent.X, ...)` for all sensitive operations
6. **Never log secrets** — use `logger.info("key={}", key_name)` not the value
7. **Parameterise all queries** — never build SQL strings with user input

See [Security Architecture](authentication_architecture.md) for full details.

---

## Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Install hooks (one-time)
pip install pre-commit
pre-commit install

# Run manually against all files
pre-commit run --all-files
```

Configured checks (`.pre-commit-config.yaml`):
- **ruff** — Python linting
- **black** — Python formatting
- **isort** — Python import sorting
- **prettier** — TypeScript/JSON/YAML formatting
- **shellcheck** — Shell script linting
- **hadolint** — Dockerfile linting

---

## Release Process

1. Update `VERSION` file with the new semantic version
2. Update `CHANGELOG.md` — move `[Unreleased]` items under the new version heading
3. Run `scripts/bump-version.sh <major|minor|patch>`
4. Commit: `git commit -m "chore: release v1.2.0"`
5. Tag: `git tag -a v1.2.0 -m "Release v1.2.0"`
6. Push: `git push origin main --tags`
7. The `release.yml` GitHub Actions workflow publishes Docker images and creates a GitHub Release automatically

See [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md) for the full pre-release checklist.
