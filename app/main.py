"""MedDiag - AI-Based Medical Diagnosis Assistant (ECG / EEG / Chest X-ray).

FastAPI application entry point.  Run with::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import settings
from app.core.audit import init_db
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.ml.registry import registry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare storage, the audit database and the models."""
    setup_logging()
    settings.ensure_directories()
    init_db()
    registry.load_all()
    logger.info("%s %s ready on %s:%s", settings.app_name, settings.version, settings.host, settings.port)
    yield
    logger.info("%s shutting down", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_title,
        version=settings.version,
        description=(
            "Clinical decision-support backend for cardiovascular (ECG), neurological (EEG) "
            "and pulmonary (chest X-ray) analysis with explainable AI and automated reporting."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router)

    static_dir = settings.static_dir
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/dashboard", include_in_schema=False)
        async def dashboard() -> FileResponse:
            """Serve the unified healthcare dashboard."""
            index = static_dir / "index.html"
            if not index.exists():
                return JSONResponse(status_code=404, content={"error": "Dashboard assets are missing."})
            return FileResponse(str(index))

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Any:
            icon = static_dir / "favicon.svg"
            if icon.exists():
                return FileResponse(str(icon), media_type="image/svg+xml")
            return JSONResponse(status_code=404, content={"error": "not found"})

    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, Any]:
        return {
            "app": settings.app_name,
            "title": settings.app_title,
            "version": settings.version,
            "docs": "/docs",
            "dashboard": "/dashboard",
            "api_prefix": "/api/v1",
        }

    return app


app = create_app()
