"""Analysis history endpoints used by the dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import require_api_key
from app.core.audit import count_analyses, list_analyses
from app.schemas import AnalysisSummary, HistoryResponse, Severity

router = APIRouter(prefix="/api/v1/history", tags=["History"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=HistoryResponse, summary="Recent analyses, newest first")
async def history(
    modality: Optional[str] = Query(None, description="Filter by ecg, eeg or xray"),
    patient_ref: Optional[str] = Query(None, description="Filter by patient reference"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> HistoryResponse:
    rows = list_analyses(modality=modality, patient_ref=patient_ref, limit=limit, offset=offset)
    items = [
        AnalysisSummary(
            analysis_id=row["analysis_id"],
            modality=row["modality"],
            patient_id=row["patient_ref"],
            predicted_label=row["predicted"],
            confidence=float(row["confidence"]),
            severity=Severity(row["severity"]) if row["severity"] in {s.value for s in Severity} else Severity.NORMAL,
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return HistoryResponse(
        total=count_analyses(modality=modality),
        items=items,
        limit=limit,
        offset=offset,
    )


@router.get("/summary", summary="Label distribution for the dashboard charts")
async def summary() -> Dict[str, Any]:
    rows = list_analyses(limit=1000)
    distribution: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    modality_counts: Dict[str, int] = {}
    for row in rows:
        distribution[row["predicted"]] = distribution.get(row["predicted"], 0) + 1
        severity_counts[row["severity"]] = severity_counts.get(row["severity"], 0) + 1
        modality_counts[row["modality"]] = modality_counts.get(row["modality"], 0) + 1
    return {
        "total": len(rows),
        "label_distribution": distribution,
        "severity_distribution": severity_counts,
        "modality_distribution": modality_counts,
    }
