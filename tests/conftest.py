"""Shared pytest fixtures."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.audit import init_db  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_runtime():
    """Create the runtime directories and the audit schema once per session."""
    from app.config import settings

    setup_logging()
    settings.ensure_directories()
    init_db()


@pytest.fixture(scope="session")
def client():
    """A FastAPI test client bound to the application."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def ecg_signal():
    """A synthetic 10-second NORM ECG recording."""
    from app.data.synthetic import synth_ecg

    signal, fs = synth_ecg("NORM", duration_s=10, seed=101)
    return signal, fs


@pytest.fixture(scope="session")
def eeg_signal():
    """A synthetic 10-second NORMAL EEG recording."""
    from app.data.synthetic import synth_eeg

    signal, fs = synth_eeg("NORMAL", duration_s=10, seed=202)
    return signal, fs


@pytest.fixture(scope="session")
def xray_bytes():
    """A synthetic normal chest radiograph encoded as PNG."""
    from PIL import Image

    from app.data.synthetic import synth_xray

    buffer = io.BytesIO()
    Image.fromarray(synth_xray("NORMAL", seed=303), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
