"""Filesystem helpers for uploaded inputs and stored analysis artefacts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings
from app.core.errors import ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def analysis_dir(analysis_id: str) -> Path:
    """Directory that holds every artefact of one analysis."""
    return _ensure(Path(settings.storage_dir) / "analyses" / analysis_id)


def save_upload(data: bytes, *, analysis_id: str, filename: str) -> Path:
    """Persist an uploaded payload and return its path."""
    safe_name = Path(filename).name or "upload.bin"
    target = _ensure(analysis_dir(analysis_id) / "input" / safe_name)
    target.write_bytes(data)
    return target


def save_analysis_payload(analysis_id: str, payload: Dict[str, Any]) -> Path:
    """Store the full analysis result as JSON so it can be re-rendered later."""
    target = _ensure(analysis_dir(analysis_id) / "analysis.json")
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def load_analysis_payload(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Load a previously stored analysis payload."""
    target = analysis_dir(analysis_id) / "analysis.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("Unable to read analysis payload %s: %s", analysis_id, exc)
        return None


def new_id(prefix: str = "") -> str:
    """Generate a short, sortable, unique identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    token = uuid.uuid4().hex[:10]
    return f"{prefix}{stamp}-{token}" if prefix else f"{stamp}-{token}"


def safe_join(base: Path, *parts: str) -> Path:
    """Join paths while refusing to escape the base directory."""
    target = base.joinpath(*parts).resolve()
    base_resolved = base.resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise ValidationError("Illegal path traversal attempt")
    return target


def read_upload_limit() -> int:
    """Maximum accepted upload size in bytes."""
    return int(settings.max_upload_mb) * 1024 * 1024
