"""Diagnostic report endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse

from app.api.deps import require_api_key
from app.core.audit import list_reports
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.storage import new_id
from app.schemas import ReportResponse 
from app.services.report import generate_report, reports_for_analysis

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"], dependencies=[Depends(require_api_key)])


@router.post(
    "/{analysis_id}",
    response_model=ReportResponse,
    summary="Generate a diagnostic report",
    description="Renders the stored analysis into a PDF, HTML or JSON report and returns its download URL.",
)
async def create_report(
    analysis_id: str,
    format: str = Query("pdf", pattern="^(pdf|html|json)$", description="Report format"),
) -> ReportResponse:
    result = generate_report(analysis_id, format)
    return ReportResponse(
        report_id=result["report_id"],
        analysis_id=result["analysis_id"],
        format=result["format"],
        download_url=f"/api/v1/reports/{result['report_id']}/download",
        created_at=result["created_at"],
        size_bytes=result["size_bytes"],
    )


@router.get(
    "/{report_id}/download",
    summary="Download a generated report",
    response_class=FileResponse,
)
async def download_report(report_id: str) -> FileResponse:
    rows = list_reports()
    match = next((row for row in rows if row["report_id"] == report_id), None)
    if match is None:
        raise NotFoundError(f"No report found with id {report_id}")
    path = match["path"]
    media_types = {"pdf": "application/pdf", "html": "text/html", "json": "application/json"}
    return FileResponse(
        path,
        media_type=media_types.get(match["format"], "application/octet-stream"),
        filename=f"{report_id}.{match['format']}",
    )


@router.get(
    "/analysis/{analysis_id}",
    summary="List reports generated for an analysis",
)
async def reports_for(analysis_id: str) -> Dict[str, Any]:
    rows = reports_for_analysis(analysis_id)
    return {
        "analysis_id": analysis_id,
        "count": len(rows),
        "reports": [
            {
                "report_id": row["report_id"],
                "format": row["format"],
                "size_bytes": row["size_bytes"],
                "created_at": row["created_at"],
                "download_url": f"/api/v1/reports/{row['report_id']}/download",
            }
            for row in rows
        ],
    }


@router.get("/list", summary="List all generated reports")
async def list_all_reports(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    rows = list_reports()[:limit]
    return {
        "count": len(rows),
        "reports": [
            {
                "report_id": row["report_id"],
                "analysis_id": row["analysis_id"],
                "format": row["format"],
                "size_bytes": row["size_bytes"],
                "created_at": row["created_at"],
                "download_url": f"/api/v1/reports/{row['report_id']}/download",
            }
            for row in rows
        ],
    }
