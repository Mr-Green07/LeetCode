"""Model registry and system health endpoints."""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key
from app.config import settings
from app.core.audit import count_analyses
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.ml.registry import registry
from app.schemas import HealthResponse, ModelInfo

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["System"])

_STARTED_AT = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service health and model status")
async def health() -> HealthResponse:
    model_info = registry.info()
    uptime = round(time.time() - _STARTED_AT, 2)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        uptime_seconds=uptime,
        device=registry.get("ecg").device,
        models={key: ModelInfo(**value) for key, value in model_info.items() if "error" not in value},
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/models", summary="List every registered model")
async def list_models() -> Dict[str, Any]:
    return {
        "count": len(registry.available_modalities()),
        "models": registry.info(),
    }


@router.get("/models/{modality}", response_model=ModelInfo, summary="Describe one model")
async def describe_model(modality: str) -> ModelInfo:
    if modality not in registry.available_modalities():
        from app.core.errors import NotFoundError

        raise NotFoundError(f"Unknown modality '{modality}'")
    return ModelInfo(**registry.get(modality).info())


@router.get("/stats", summary="Aggregate statistics for the dashboard")
async def stats() -> Dict[str, Any]:
    snapshot = metrics.snapshot()
    per_modality = {modality: count_analyses(modality=modality) for modality in registry.available_modalities()}
    return {
        "analyses_total": sum(per_modality.values()),
        "analyses_by_modality": per_modality,
        "metrics": snapshot,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": registry.get("ecg").device,
        },
        "models": registry.info(),
    }
