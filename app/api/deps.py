"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Enforce the optional shared-secret API key.

    When ``MEDDIAG_API_KEY`` is unset the service is open (useful for local
    development and demos).  When it is set, every ``/api`` route demands a
    matching ``X-API-Key`` header.
    """
    expected = settings.api_key
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
