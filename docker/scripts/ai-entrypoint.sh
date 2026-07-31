#!/bin/sh
# ─── docker/scripts/ai-entrypoint.sh ──────────────────────────────────────────
#
# Entrypoint for the AI service container.
# Runs pre-flight checks then exec's the CMD passed from the Dockerfile.
#
# Responsibilities:
#   1. Validate required environment variables
#   2. Ensure runtime directories exist
#   3. Wait for any dependent services (if configured)
#   4. Exec the application process (becomes PID 1 under tini)
#
set -e

# ── Colour output helpers ──────────────────────────────────────────────────────
log_info()  { echo "[AI-SERVICE] INFO:  $1"; }
log_warn()  { echo "[AI-SERVICE] WARN:  $1" >&2; }
log_error() { echo "[AI-SERVICE] ERROR: $1" >&2; }

log_info "=== Brain Tumour AI Service — startup ==="
log_info "Environment : ${AI_SERVICE_ENV:-production}"
log_info "Port        : ${AI_SERVICE_PORT:-8000}"
log_info "Active model: ${ACTIVE_MODEL:-mambavision}"

# ── Required directory checks ──────────────────────────────────────────────────
for dir in \
    "${SAVED_MODELS_DIR:-/app/saved_models}" \
    "${LOG_DIR:-/app/logs}" \
    "${GRADCAM_OUTPUT_DIR:-/app/gradcam_output}"; do
    if [ ! -d "$dir" ]; then
        log_info "Creating directory: $dir"
        mkdir -p "$dir" 2>/dev/null || log_warn "Could not create $dir (may be read-only volume)"
    fi
done

# ── JWT secret strength check in production ───────────────────────────────────
if [ "${AI_SERVICE_ENV:-production}" = "production" ]; then
    if [ "${JWT_SECRET_KEY:-change-me-in-production-use-a-long-random-secret}" \
         = "change-me-in-production-use-a-long-random-secret" ]; then
        log_error "JWT_SECRET_KEY is still the default placeholder!"
        log_error "Set a strong random secret before running in production."
        exit 1
    fi
    JWT_LEN=$(echo -n "${JWT_SECRET_KEY}" | wc -c)
    if [ "$JWT_LEN" -lt 32 ]; then
        log_error "JWT_SECRET_KEY is too short (${JWT_LEN} chars). Minimum 32 required."
        exit 1
    fi
fi

# ── Python import sanity check ─────────────────────────────────────────────────
log_info "Running import sanity check..."
python -c "import fastapi, torch, uvicorn; print('Core imports OK')" \
    || { log_error "Core Python imports failed — check requirements installation"; exit 1; }

log_info "Pre-flight checks passed — starting application..."
log_info "CMD: $*"

# exec replaces this shell so tini (PID 1) manages the app process
exec "$@"
