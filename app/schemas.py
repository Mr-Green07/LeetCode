"""Pydantic request / response contracts for the public API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import Modality

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Severity(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class FindingStatus(str, Enum):
    NORMAL = "normal"
    BORDERLINE = "borderline"
    ABNORMAL = "abnormal"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Patient + input payloads
# --------------------------------------------------------------------------- #


class PatientInfo(BaseModel):
    """De-identified patient context sent with every analysis."""

    patient_id: str = Field(..., min_length=1, max_length=64, description="Local patient reference (never a real national ID)")
    age: Optional[int] = Field(None, ge=0, le=130)
    sex: Sex = Sex.UNKNOWN
    referring_clinician: Optional[str] = Field(None, max_length=120)
    clinical_notes: Optional[str] = Field(None, max_length=4000)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("clinical_notes")
    @classmethod
    def _strip_notes(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value else value


class SignalAnalysisRequest(BaseModel):
    """Raw biosignal (ECG or EEG) submitted as plain samples."""

    patient: PatientInfo
    samples: List[float] = Field(..., description="Flat, time-ordered samples of the primary signal")
    sampling_rate: float = Field(..., gt=0, description="Samples per second in Hz")
    lead_names: Optional[List[str]] = Field(None, description="Optional channel names for multichannel input")
    channel: Optional[str] = Field(None, description="Name of the primary channel to analyse")
    window_seconds: Optional[float] = Field(None, gt=0, description="Length of the analysis window")
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("samples")
    @classmethod
    def _require_samples(cls, value: List[float]) -> List[float]:
        if len(value) < 32:
            raise ValueError("at least 32 samples are required")
        return value


class ImageAnalysisRequest(BaseModel):
    """Image analysis options (the image itself is uploaded separately)."""

    patient: PatientInfo
    view: Literal["PA", "AP", "LATERAL", "UNKNOWN"] = "UNKNOWN"
    context: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Analysis outputs
# --------------------------------------------------------------------------- #


class ClassScore(BaseModel):
    """Probability attached to one diagnostic class."""

    label: str
    probability: float = Field(..., ge=0.0, le=1.0)
    severity: Severity = Severity.NORMAL


class Prediction(BaseModel):
    """Primary model output."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: Severity = Severity.NORMAL
    description: str = ""
    class_probabilities: Dict[str, float] = Field(default_factory=dict)


class ClinicalFinding(BaseModel):
    """One interpretable measurement taken from the signal or image."""

    name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: FindingStatus = FindingStatus.NORMAL
    explanation: str = ""


class FeatureContribution(BaseModel):
    """How much a single feature moved the prediction."""

    name: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]
    detail: str = ""


class Explainability(BaseModel):
    """Explainable-AI payload attached to a diagnosis."""

    methods: List[str] = Field(default_factory=list)
    summary: str = ""
    top_features: List[FeatureContribution] = Field(default_factory=list)
    class_relevance: Dict[str, float] = Field(default_factory=dict)
    saliency_png_base64: Optional[str] = None
    overlay_png_base64: Optional[str] = None
    waveform_png_base64: Optional[str] = None
    attribution_method: str = ""


class QualityMetrics(BaseModel):
    """Input quality assessment (garbage in -> unreliable out)."""

    signal_quality_index: float = Field(..., ge=0.0, le=1.0)
    usable: bool = True
    issues: List[str] = Field(default_factory=list)
    snr_db: Optional[float] = None
    clipped_samples: int = 0


class DiagnosisResponse(BaseModel):
    """Full diagnosis payload returned by every analysis endpoint."""

    analysis_id: str
    modality: Modality
    patient: PatientInfo
    prediction: Prediction
    differential: List[ClassScore] = Field(default_factory=list)
    findings: List[ClinicalFinding] = Field(default_factory=list)
    explainability: Explainability = Field(default_factory=Explainability)
    recommendations: List[str] = Field(default_factory=list)
    quality: QualityMetrics
    model_version: str = ""
    model_source: Literal["trained", "heuristic_baseline"] = "trained"
    report_url: Optional[str] = None
    created_at: datetime
    disclaimer: str = (
        "Decision-support output for qualified healthcare professionals. "
        "Not a substitute for clinical judgement or a formal diagnosis."
    )


class ReportResponse(BaseModel):
    """Report metadata returned after a report is rendered."""

    report_id: str
    analysis_id: str
    format: Literal["pdf", "html", "json"]
    download_url: str
    created_at: datetime
    size_bytes: int


class ModelInfo(BaseModel):
    """Metadata describing one loaded model."""

    modality: Modality
    architecture: str
    version: str
    source: Literal["trained", "heuristic_baseline"]
    classes: List[str] = Field(default_factory=list)
    input_shape: List[int] = Field(default_factory=list)
    parameters: int = 0
    checkpoint_path: Optional[str] = None
    trained: bool = False
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Service health payload."""

    status: str
    app: str
    version: str
    uptime_seconds: float
    device: str
    models: Dict[str, ModelInfo]
    timestamp: datetime


class AnalysisSummary(BaseModel):
    """One row in the dashboard history table."""

    analysis_id: str
    modality: Modality
    patient_id: str
    predicted_label: str
    confidence: float
    severity: Severity
    created_at: datetime


class HistoryResponse(BaseModel):
    """Paginated analysis history."""

    total: int
    items: List[AnalysisSummary]
    limit: int
    offset: int
