# ─── ai-service/Dockerfile ────────────────────────────────────────────────────
#
# Multi-stage build for the Python / PyTorch AI service.
#
# Stages
# ──────
#   deps     — resolves and pins transitive dependencies
#   builder  — compiles native extensions into a clean /install prefix
#   runtime  — minimal python:3.12-slim with only runtime artifacts
#
# CPU-only build (PyTorch CPU wheels).
# For GPU: swap base images to nvidia/cuda:12.x-cudnn9-runtime-ubuntu22.04 and
# adjust torch installation accordingly.
#
# Build:
#   docker build -t brain-tumor-ai-service:latest ./ai-service
#   docker build --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
#                --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
#                -t brain-tumor-ai-service:latest ./ai-service
#
# Run (standalone):
#   docker run --rm -p 8000:8000 --env-file ai-service/.env brain-tumor-ai-service:latest

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: compile native extensions
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Build-time dependencies for native extensions (numpy, scipy, opencv, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Upgrade pip and install wheel tools once — cached unless base image changes
RUN pip install --upgrade --no-cache-dir pip setuptools wheel

# Copy requirements first to exploit layer caching:
# Only re-runs pip install when requirements.txt changes.
COPY requirements.txt .

# Install into a clean prefix — no --user, no system pollution
RUN pip install \
        --prefix=/install \
        --no-cache-dir \
        --compile \
        -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime: minimal production image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# ── Build-time metadata arguments ────────────────────────────────────────────
ARG BUILD_DATE="unknown"
ARG GIT_COMMIT="unknown"
ARG VERSION="1.0.0"

# OCI image labels (visible in `docker inspect` and container registries)
LABEL org.opencontainers.image.title="Brain Tumour Detection — AI Service" \
      org.opencontainers.image.description="FastAPI / PyTorch MRI classification service" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.source="https://github.com/your-org/brain-tumor-detection" \
      org.opencontainers.image.licenses="MIT"

# Runtime OS libraries required by OpenCV-headless and PyTorch (CPU)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
        tini \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Copy installed Python packages from builder (wheels only, no build tools)
COPY --from=builder /install /usr/local

# ── Non-root user (security best practice) ───────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup \
             --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy application source (excludes .venv, dataset, saved_models, logs via .dockerignore)
COPY --chown=appuser:appgroup app/          ./app/
COPY --chown=appuser:appgroup pyproject.toml .

# Copy startup script
COPY --chown=appuser:appgroup docker/scripts/ai-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create runtime directories — these are bind-mounted in docker-compose
# but must exist in the image for standalone `docker run` usage.
RUN mkdir -p \
        saved_models \
        dataset/raw \
        dataset/processed \
        gradcam_output \
        logs/audit \
        logs/metrics \
        logs/experiments \
 && chown -R appuser:appgroup /app

USER appuser

# ── Environment defaults (all overridden by docker-compose / .env) ────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    AI_SERVICE_HOST=0.0.0.0 \
    AI_SERVICE_PORT=8000 \
    AI_SERVICE_ENV=production \
    AI_SERVICE_DEBUG=false \
    LOG_LEVEL=INFO \
    ACTIVE_MODEL=efficientnet \
    IMAGE_SIZE=224 \
    IMAGE_CHANNELS=3 \
    CLASS_NAMES=glioma,meningioma,notumor,pituitary \
    UVICORN_WORKERS=2

EXPOSE 8000

# Use tini as PID 1 — properly forwards signals and reaps zombies
ENTRYPOINT ["tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-access-log", \
     "--timeout-keep-alive", "75"]

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
