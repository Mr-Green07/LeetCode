"""Domain exceptions and the FastAPI exception handlers that render them."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class MedDiagError(Exception):
    """Base class for every error raised by this service."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "meddiag_error"

    def __init__(self, message: str, *, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationError(MedDiagError):
    """Input payload failed validation."""

    status_code = 422
    code = "validation_error"


class UnsupportedFormatError(MedDiagError):
    """The uploaded file / signal container is not supported."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_format"


class ModelNotReadyError(MedDiagError):
    """A required model artifact is missing or unreadable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"


class NotFoundError(MedDiagError):
    """A referenced resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach uniform JSON error rendering to the FastAPI app."""

    @app.exception_handler(MedDiagError)
    async def _handle_meddiag_error(_: Request, exc: MedDiagError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(ValueError)
    async def _handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        logger.warning("ValueError: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "value_error", "message": str(exc), "details": {}}},
        )
