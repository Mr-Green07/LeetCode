"""Synthetic demo-sample endpoints.

They let a reviewer (or the dashboard's "load demo sample" button) exercise the
full pipeline instantly, with no dataset download.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from app.api.deps import require_api_key
from app.data.synthetic import synth_ecg, synth_eeg, synth_xray
from app.ml.ecg.model import ECG_CLASSES
from app.ml.eeg.model import EEG_CLASSES
from app.ml.xray.model import XRAY_CLASSES

router = APIRouter(prefix="/api/v1/demo", tags=["Demo"], dependencies=[Depends(require_api_key)])


@router.get("/ecg/{label}", summary="Fetch a synthetic ECG sample")
async def demo_ecg(label: str, seconds: float = Query(10.0, ge=2.0, le=60.0), seed: int = 7) -> Dict[str, Any]:
    label = label.upper()
    if label not in ECG_CLASSES:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"Unknown ECG class '{label}'")
    signal, rate = synth_ecg(label, duration_s=seconds, seed=seed)
    return {
        "modality": "ecg",
        "label": label,
        "sampling_rate": rate,
        "samples": [round(float(value), 6) for value in signal],
    }


@router.get("/eeg/{label}", summary="Fetch a synthetic EEG sample")
async def demo_eeg(label: str, seconds: float = Query(10.0, ge=2.0, le=60.0), seed: int = 11) -> Dict[str, Any]:
    label = label.upper()
    if label not in EEG_CLASSES:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"Unknown EEG class '{label}'")
    signal, rate = synth_eeg(label, duration_s=seconds, seed=seed)
    return {
        "modality": "eeg",
        "label": label,
        "sampling_rate": rate,
        "samples": [round(float(value), 6) for value in signal],
    }


@router.get("/xray/{label}", summary="Fetch a synthetic chest radiograph")
async def demo_xray(label: str, seed: int = 13) -> Dict[str, Any]:
    label = label.upper()
    if label not in XRAY_CLASSES:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"Unknown X-ray class '{label}'")

    image = synth_xray(label, seed=seed)
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image.astype("uint8"), mode="L").save(buffer, format="PNG")
    return {
        "modality": "xray",
        "label": label,
        "content_type": "image/png",
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


@router.get("/classes", summary="List every synthetic demo sample")
async def demo_classes() -> Dict[str, Any]:
    return {
        "ecg": list(ECG_CLASSES),
        "eeg": list(EEG_CLASSES),
        "xray": list(XRAY_CLASSES),
    }
