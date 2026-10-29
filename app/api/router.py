"""Router assembly."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import (
    routes_demo,
    routes_ecg,
    routes_eeg,
    routes_health,
    routes_history,
    routes_models,
    routes_reports,
    routes_xray,
)

api_router = APIRouter()
api_router.include_router(routes_health.router)
api_router.include_router(routes_models.router)
api_router.include_router(routes_ecg.router)
api_router.include_router(routes_eeg.router)
api_router.include_router(routes_xray.router)
api_router.include_router(routes_reports.router)
api_router.include_router(routes_history.router)
api_router.include_router(routes_demo.router)

__all__ = ["api_router"]
