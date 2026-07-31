# Changelog

All notable changes to Brain Tumour Detection are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Module 10: Deployment & CI/CD infrastructure
  - Optimized multi-stage Dockerfiles with tini, non-root users, OCI labels
  - Docker Compose: base / dev / prod / override configuration
  - Nginx reverse proxy with gzip, security headers, rate limiting, HTTPS-ready
  - GitHub Actions CI pipeline (lint, test, Docker build, security scan)
  - GitHub Actions CD pipeline (staging auto-deploy, production on release)
  - GitHub Actions release workflow (semantic versioning, changelog, GHCR push)
  - deploy.sh / deploy.ps1 — cross-platform deployment scripts with rollback
  - backup.sh / restore.sh — Docker volume backup and restore
  - validate-env.sh / validate-env.ps1 — environment validation scripts
  - bump-version.sh — semantic version bumping across all manifests
  - Pre-commit hooks (ruff, black, isort, prettier, shellcheck, hadolint)
  - Code quality configuration: ruff, black, isort (Python); ESLint, Prettier (TS)
  - Production environment templates (.env.production)
  - Docker secrets integration for JWT and DB keys
  - Deployment validation test suite (pytest)
  - Documentation: Deployment Guide, CI/CD Guide, Docker Guide, Production Checklist

---

## [1.0.0] - 2026-07-14

### Added
- Module 9: Security & Authentication
  - JWT authentication with access + refresh token pair
  - Role-based access control (Admin, Researcher, Operator, Viewer)
  - bcrypt password hashing (12 rounds in production)
  - SlowAPI rate limiting per endpoint
  - Audit logging to JSONL files
  - Account lockout after 5 failed logins

- Module 8: Metrics & Monitoring Dashboard
  - System metrics (CPU, RAM, disk, GPU)
  - Inference metrics (latency percentiles, class distribution)
  - Training metrics (job counts, best accuracy, experiment history)
  - Rolling time-series history with JSONL storage
  - Dashboard overview composite endpoint

- Module 7: Frontend Integration
  - React 18 + Vite 5 + TypeScript frontend
  - Prediction upload interface with Grad-CAM overlay
  - Training dashboard with live progress polling
  - Batch prediction (multi-file + ZIP)
  - Dataset manager UI
  - Metrics dashboard with recharts

- Module 6: Production Inference Pipeline
  - InferencePipeline with LRU model cache
  - BatchInferenceRunner (thread-pool, ZIP support)
  - InferenceConfig validation
  - Hot-reload endpoint for model updates

- Module 5: Deep Learning Training Framework
  - MambaVision-T (NVIDIA, ImageNet pretrained) as primary architecture with custom classification head
  - EfficientNet-B3, ResNet50, VGG16, custom CNN as additional architectures (all PyTorch/torchvision)
  - Two-phase fine-tuning (head → backbone)
  - ExperimentRegistry with per-epoch history
  - Async training via FastAPI BackgroundTasks
  - Checkpointing with best-model save (PyTorch state_dict + HuggingFace formats)

- Module 4: Advanced Image Preprocessing
  - Denoise + CLAHE spatial pipeline
  - Training augmentation stack
  - Quality validation checks (blur, intensity, variance)
  - Base64 preview endpoint

- Module 3: Dataset Manager
  - Raw dataset validation
  - Stratified train/val/test split
  - Class weight computation
  - dataset_info.json metadata

- Module 2: Core FastAPI Service
  - FastAPI application factory with lifespan
  - CORS middleware, rate limiting, request logging
  - Pydantic settings with env-file support
  - Loguru rotating log files

- Module 1: Project Scaffold
  - Three-tier architecture (React / Node.js / FastAPI)
  - Docker multi-stage builds
  - pytest suite with 1,128+ passing tests

---

[Unreleased]: https://github.com/your-org/brain-tumor-detection/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/brain-tumor-detection/releases/tag/v1.0.0
