"""
main.py — FastAPI application entry point.

Creates and configures the FastAPI app instance:
  - CORS middleware (origins read from settings)
  - Request / response logging middleware
  - API router mounted at /api/v1
  - Global exception handlers
  - Startup / shutdown lifecycle events

Run in development:
    cd ai-service
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Run in production:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import router
from app.api.auth_routes import auth_router
from app.api.performance_routes import performance_router
from app.core.config import settings
from app.core.logging import logger
from app.security.rate_limit import limiter


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup (before first request) and on shutdown.
    Used to warm up heavy resources (model weights, GPU context).
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Brain Tumour Detection — AI Service")
    logger.info(f"  Environment : {settings.ai_service_env}")
    logger.info(f"  Host        : {settings.ai_service_host}:{settings.ai_service_port}")
    logger.info(f"  Active model: {settings.active_model}")
    logger.info(f"  Classes     : {settings.classes}")
    logger.info(f"  Image size  : {settings.image_size}×{settings.image_size}")
    logger.info("=" * 60)

    # Pre-load the active model into cache if weights are available.
    # A missing model is non-fatal at startup — the /train endpoint creates one.
    from app.models.load_model import is_model_available, load_model
    if is_model_available(settings.active_model):
        try:
            load_model(settings.active_model)
            logger.info(f"Active model '{settings.active_model}' pre-loaded into cache.")
        except Exception as exc:
            logger.warning(f"Could not pre-load model '{settings.active_model}': {exc}")
    else:
        logger.info(
            f"No saved weights found for '{settings.active_model}'. "
            "Use POST /api/v1/train to train the model first."
        )

    logger.info("AI Service startup complete — ready to accept requests.")

    yield  # ← server is running

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("AI Service shutting down.")


# ─── Application factory ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""

    app = FastAPI(
        title="Brain Tumour Detection — AI Service",
        description=(
            "Python / PyTorch backend for deep-learning-based MRI brain "
            "tumour classification. Exposes REST endpoints consumed by the "
            "Node.js Express backend."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Rate limiter ──────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        # Feed the API optimizer so /performance/api-stats has live data.
        try:
            from app.performance.optimizer import record_request
            record_request(
                path=request.url.path,
                method=request.method,
                elapsed_ms=elapsed_ms,
                status_code=response.status_code,
            )
        except Exception:
            pass  # never let metrics collection crash a real request
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": "An unexpected internal error occurred.",
                    "detail": str(exc) if settings.ai_service_debug else None,
                },
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    # All AI endpoints are prefixed with /api/v1
    app.include_router(router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(performance_router, prefix="/api/v1")

    return app


# ─── Singleton app instance (imported by uvicorn) ────────────────────────────
app = create_app()
