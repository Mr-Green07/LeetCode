"""SQLite-backed audit trail.

Every analysis and every report render is recorded so a clinical deployment can
answer "who looked at what, when, and what did the model say".
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()
_SCHEMA_READY = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id   TEXT PRIMARY KEY,
    modality      TEXT NOT NULL,
    patient_ref   TEXT NOT NULL,
    predicted     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    severity      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    payload       TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    report_id    TEXT PRIMARY KEY,
    analysis_id  TEXT NOT NULL,
    format       TEXT NOT NULL,
    path         TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id  TEXT PRIMARY KEY,
    analysis_id TEXT,
    action    TEXT NOT NULL,
    actor     TEXT NOT NULL DEFAULT 'system',
    detail    TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_analysis ON audit_events(analysis_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    settings.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.audit_db_path), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the audit schema (idempotent)."""
    global _SCHEMA_READY
    with _LOCK, _connect() as conn:
        conn.executescript(_SCHEMA)
    _SCHEMA_READY = True
    logger.info("Audit database ready at %s", settings.audit_db_path)


def _ensure_schema() -> None:
    """Create the schema on first use so scripts work without the app lifespan."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    try:
        init_db()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Could not initialise the audit schema: %s", exc)


def record_analysis(
    *,
    analysis_id: str,
    modality: str,
    patient_ref: str,
    predicted: str,
    confidence: float,
    severity: str,
    model_version: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist one finished analysis."""
    _ensure_schema()
    try:
        with _LOCK, _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analyses "
                "(analysis_id, modality, patient_ref, predicted, confidence, severity, "
                " model_version, created_at, payload) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    modality,
                    patient_ref,
                    predicted,
                    float(confidence),
                    severity,
                    model_version,
                    _utcnow(),
                    json.dumps(payload, default=str) if payload else None,
                ),
            )
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        logger.error("Failed to record analysis: %s", exc)


def record_report(*, report_id: str, analysis_id: str, fmt: str, path: str, size_bytes: int) -> None:
    """Persist one rendered report."""
    _ensure_schema()
    try:
        with _LOCK, _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reports "
                "(report_id, analysis_id, format, path, size_bytes, created_at) VALUES (?,?,?,?,?,?)",
                (report_id, analysis_id, fmt, path, int(size_bytes), _utcnow()),
            )
    except sqlite3.Error as exc:  # pragma: no cover
        logger.error("Failed to record report: %s", exc)


def log_event(action: str, *, analysis_id: Optional[str] = None, actor: str = "system", detail: str = "") -> str:
    """Append a generic audit event and return its id."""
    event_id = uuid.uuid4().hex
    _ensure_schema()
    try:
        with _LOCK, _connect() as conn:
            conn.execute(
                "INSERT INTO audit_events (event_id, analysis_id, action, actor, detail, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (event_id, analysis_id, action, actor, detail, _utcnow()),
            )
    except sqlite3.Error as exc:  # pragma: no cover
        logger.error("Failed to log audit event: %s", exc)
    return event_id


def list_analyses(
    *, modality: Optional[str] = None, patient_ref: Optional[str] = None, limit: int = 50, offset: int = 0
) -> List[Dict[str, Any]]:
    """Return recent analyses, newest first."""
    clauses, params = [], []
    if modality:
        clauses.append("modality = ?")
        params.append(modality)
    if patient_ref:
        clauses.append("patient_ref = ?")
        params.append(patient_ref)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([int(limit), int(offset)])
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            f"SELECT analysis_id, modality, patient_ref, predicted, confidence, severity, "
            f"model_version, created_at FROM analyses {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def count_analyses(*, modality: Optional[str] = None) -> int:
    """Total number of stored analyses, optionally filtered by modality."""
    sql = "SELECT COUNT(*) AS n FROM analyses"
    params: List[Any] = []
    if modality:
        sql += " WHERE modality = ?"
        params.append(modality)
    with _LOCK, _connect() as conn:
        return int(conn.execute(sql, params).fetchone()["n"])


def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one stored analysis (including its JSON payload)."""
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    if data.get("payload"):
        try:
            data["payload"] = json.loads(data["payload"])
        except (TypeError, ValueError):
            data["payload"] = None
    return data


def list_reports(analysis_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return report metadata rows."""
    sql = "SELECT * FROM reports"
    params: List[Any] = []
    if analysis_id:
        sql += " WHERE analysis_id = ?"
        params.append(analysis_id)
    sql += " ORDER BY created_at DESC"
    with _LOCK, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one report row by id."""
    with _LOCK, _connect() as conn:
        row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
    return dict(row) if row else None
