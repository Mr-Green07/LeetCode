"""Liveness endpoints (kept outside the API-key protected router)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["System"])


@router.get("/health", summary="Liveness probe")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "app": settings.app_name,
        "title": settings.app_title,
        "version": settings.version,
        "docs": "/docs",
        "dashboard": "/dashboard",
    }
