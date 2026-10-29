"""Chest X-ray analysis endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import require_api_key
from app.config import settings
from app.core.audit import log_event
from app.core.errors import UnsupportedFormatError, ValidationError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.storage import save_upload
from app.ml.registry import registry
from app.ml.xray.model import CLASS_DESCRIPTIONS, XRAY_CLASSES
from app.schemas import DiagnosisResponse, PatientInfo
from app.services.diagnosis import diagnosis_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/xray", tags=["X-ray"], dependencies=[Depends(require_api_key)])


@router.post(
    "/analyze",
    response_model=DiagnosisResponse,
    summary="Analyse a chest radiograph",
    description="Accepts a PNG, JPEG or DICOM chest X-ray and returns a diagnostic impression with Grad-CAM localisation.",
)
async def analyze_xray(
    file: UploadFile = File(..., description="Chest radiograph (PNG / JPEG / DICOM)"),
    patient_id: str = Form(..., description="Local patient reference"),
    age: int | None = Form(None),
    sex: str = Form("unknown"),
    view: str = Form("UNKNOWN"),
    referring_clinician: str | None = Form(None),
    clinical_notes: str | None = Form(None),
) -> DiagnosisResponse:
    started = time.perf_counter()
    metrics.increment("xray.analysis.requests")

    content_type = (file.content_type or "").lower()
    if content_type and not any(content_type.startswith(t.split("/")[0]) for t in settings.allowed_image_types):
        raise UnsupportedFormatError(f"Unsupported content type: {content_type}")

    content = await file.read()
    if not content:
        raise ValidationError("The uploaded image is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationError(f"The image exceeds the {settings.max_upload_mb} MB upload limit.")

    patient = PatientInfo(
        patient_id=patient_id,
        age=age,
        sex=sex.lower() if sex else "unknown",
        referring_clinician=referring_clinician,
        clinical_notes=clinical_notes,
        metadata={"view": view.upper()} if view else {},
    )

    analysis_id = f"xray-upload-{len(content)}"
    try:
        save_upload(content, analysis_id=analysis_id, filename=file.filename or "upload.png")
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Could not store the uploaded radiograph: %s", exc)

    try:
        response = diagnosis_service.analyze_xray(content, patient)
    except ValueError as exc:
        metrics.record_error("xray.analysis")
        raise ValidationError(str(exc)) from exc

    log_event("xray_analysed", analysis_id=response.analysis_id, detail=str(file.filename))
    metrics.observe_latency("xray.analysis", time.perf_counter() - started)
    return response


@router.get("/classes", summary="List the chest X-ray diagnostic classes")
async def xray_classes() -> Dict[str, Any]:
    model = registry.get("xray")
    return {
        "modality": "xray",
        "classes": [{"label": label, "description": CLASS_DESCRIPTIONS.get(label, "")} for label in XRAY_CLASSES],
        "model": model.info(),
    }
