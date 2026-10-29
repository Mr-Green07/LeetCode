"""EEG analysis endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import require_api_key
from app.config import settings
from app.core.audit import log_event
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.parsers import parse_signal_upload
from app.core.storage import save_upload
from app.ml.eeg.model import CLASS_DESCRIPTIONS, EEG_CLASSES
from app.ml.registry import registry
from app.schemas import DiagnosisResponse, PatientInfo, SignalAnalysisRequest
from app.services.diagnosis import diagnosis_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/eeg", tags=["EEG"], dependencies=[Depends(require_api_key)])


@router.post(
    "/analyze",
    response_model=DiagnosisResponse,
    summary="Analyse a raw EEG signal",
    description="Accepts time-ordered EEG samples and returns a neurological screening result with spectral findings and explainable-AI output.",
)
async def analyze_eeg(payload: SignalAnalysisRequest) -> DiagnosisResponse:
    started = time.perf_counter()
    metrics.increment("eeg.analysis.requests")
    try:
        response = diagnosis_service.analyze_eeg(
            payload.samples,
            payload.sampling_rate,
            payload.patient,
            channel_names=payload.lead_names,
        )
    except ValueError as exc:
        metrics.record_error("eeg.analysis")
        raise ValidationError(str(exc)) from exc
    metrics.observe_latency("eeg.analysis", time.perf_counter() - started)
    return response


@router.post(
    "/analyze/upload",
    response_model=DiagnosisResponse,
    summary="Analyse an uploaded EEG file",
    description="Accepts a CSV, JSON, TXT or MAT file containing EEG samples.",
)
async def analyze_eeg_upload(
    file: UploadFile = File(..., description="EEG file (CSV/JSON/TXT/MAT)"),
    patient_id: str = Form(..., description="Local patient reference"),
    sampling_rate: float = Form(settings.eeg_default_sampling_rate, description="Sampling rate in Hz when the file does not declare one"),
    age: int | None = Form(None),
    sex: str = Form("unknown"),
    referring_clinician: str | None = Form(None),
    clinical_notes: str | None = Form(None),
) -> DiagnosisResponse:
    started = time.perf_counter()
    metrics.increment("eeg.upload.requests")
    content = await file.read()
    if not content:
        raise ValidationError("The uploaded file is empty.")

    parsed = parse_signal_upload(file.filename or "upload.csv", content, settings.eeg_default_sampling_rate)
    patient = PatientInfo(
        patient_id=patient_id,
        age=age,
        sex=sex.lower() if sex else "unknown",
        referring_clinician=referring_clinician,
        clinical_notes=clinical_notes,
    )

    analysis_id = f"eeg-upload-{len(content)}"
    try:
        save_upload(content, analysis_id=analysis_id, filename=file.filename or "upload.csv")
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Could not store the uploaded EEG: %s", exc)

    try:
        response = diagnosis_service.analyze_eeg(
            parsed.samples,
            parsed.sampling_rate or sampling_rate,
            patient,
            channel_names=parsed.channel_names,
        )
    except ValueError as exc:
        metrics.record_error("eeg.upload")
        raise ValidationError(str(exc)) from exc

    log_event("eeg_upload_analysed", analysis_id=response.analysis_id, detail=str(parsed.source_format))
    metrics.observe_latency("eeg.upload", time.perf_counter() - started)
    return response


@router.get("/classes", summary="List the EEG diagnostic classes")
async def eeg_classes() -> Dict[str, Any]:
    model = registry.get("eeg")
    return {
        "modality": "eeg",
        "classes": [{"label": label, "description": CLASS_DESCRIPTIONS.get(label, "")} for label in EEG_CLASSES],
        "model": model.info(),
    }
