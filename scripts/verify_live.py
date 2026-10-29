"""End-to-end verification against a running MedDiag server.

Usage::

    python scripts/verify_live.py --base-url http://127.0.0.1:8000

Exercises the dashboard, the static assets, all three analysis endpoints (JSON and
upload), report generation in all three formats, the history endpoints, the model
registry and the statistics endpoint.  Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    status = "OK  " if condition else "FAIL"
    print(f"  {status} {name}" + (f"  -- {detail}" if detail and not condition else ""))


def _has_content(fmt: str, body: bytes) -> bool:
    """True when the downloaded bytes look like the requested format."""
    head = body.lstrip()[:256]
    if fmt == "pdf":
        return head.startswith(b"%PDF")
    if fmt == "html":
        return b"<html" in head.lower()
    return head.startswith(b"{")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a running MedDiag server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=args.timeout)

    from app.data.synthetic import synth_ecg, synth_eeg, synth_xray

    # ---------------------------------------------------------------- dashboard
    print("== dashboard & static ==")
    response = client.get("/dashboard")
    check("GET /dashboard returns 200", response.status_code == 200, str(response.status_code))
    check("dashboard is the SPA shell", "<html" in response.text.lower())
    for asset in ("/static/js/app.js", "/static/css/app.css"):
        asset_response = client.get(asset)
        check(f"GET {asset} returns 200", asset_response.status_code == 200)
    docs = client.get("/docs")
    check("GET /docs returns 200", docs.status_code == 200)

    # ------------------------------------------------------------------- health
    print("== health & models ==")
    health = client.get("/api/v1/health")
    check("GET /api/v1/health returns 200", health.status_code == 200)
    health_body = health.json()
    check("health reports every modality", set(health_body.get("models", {})) >= {"ecg", "eeg", "xray"})
    for modality, info in health_body.get("models", {}).items():
        source = info.get("source") if isinstance(info, dict) else None
        check(f"  {modality} model source is reported", bool(source), str(source))
        print(f"       {modality}: source={source} accuracy={info.get('accuracy')} auc={info.get('auc')}")

    models = client.get("/api/v1/models")
    check("GET /api/v1/models returns 200", models.status_code == 200)
    check("models payload lists three models", len(models.json().get("models", {})) == 3)

    # ---------------------------------------------------------------------- ECG
    print("== ECG ==")
    signal, fs = synth_ecg("AFIB", duration_s=10, seed=99)
    payload = {
        "patient": {"patient_id": "PT-LIVE-ECG", "age": 61, "sex": "male"},
        "samples": [float(v) for v in signal],
        "sampling_rate": float(fs),
    }
    response = client.post("/api/v1/ecg/analyze", json=payload)
    check("POST /api/v1/ecg/analyze returns 200", response.status_code == 200, response.text[:300])
    ecg = response.json()
    check("ECG label is AFIB", ecg["prediction"]["label"] == "AFIB", str(ecg["prediction"]["label"]))
    check("ECG model_source is trained", ecg.get("model_source") == "trained", str(ecg.get("model_source")))
    check("ECG returns findings", len(ecg.get("findings", [])) > 0)
    check("ECG returns recommendations", len(ecg.get("recommendations", [])) > 0)
    check("ECG returns quality metrics", "signal_quality_index" in ecg.get("quality", {}))
    explanation = ecg.get("explainability", {})
    check("ECG saliency image is present", bool(explanation.get("saliency_png_base64")))
    check("ECG waveform image is present", bool(explanation.get("waveform_png_base64")))
    check("ECG top_features are present", len(explanation.get("top_features", [])) > 0)
    check("Grad-CAM is among the methods", any("Grad-CAM" in m for m in explanation.get("methods", [])))
    ecg_id = ecg["analysis_id"]

    # ECG upload
    csv_body = "sampling_rate\n" + "\n".join(f"{v:.6f}" for v in signal)
    response = client.post(
        "/api/v1/ecg/analyze/upload",
        files={"file": ("ecg.csv", csv_body.encode(), "text/csv")},
        data={"patient_id": "PT-LIVE-UPLOAD", "sampling_rate": str(fs), "age": "61", "sex": "male"},
    )
    check("POST /api/v1/ecg/analyze/upload returns 200", response.status_code == 200, response.text[:300])
    if response.status_code == 200:
        check("ECG upload label is AFIB", response.json()["prediction"]["label"] == "AFIB")

    # ---------------------------------------------------------------------- EEG
    print("== EEG ==")
    signal, fs = synth_eeg("EPILEPSY", duration_s=10, seed=77)
    response = client.post(
        "/api/v1/eeg/analyze",
        json={
            "patient": {"patient_id": "PT-LIVE-EEG", "age": 28, "sex": "female"},
            "samples": [float(v) for v in signal],
            "sampling_rate": float(fs),
        },
    )
    check("POST /api/v1/eeg/analyze returns 200", response.status_code == 200, response.text[:300])
    eeg = response.json()
    check("EEG label is EPILEPSY", eeg["prediction"]["label"] == "EPILEPSY", str(eeg["prediction"]["label"]))
    check("EEG model_source is trained", eeg.get("model_source") == "trained", str(eeg.get("model_source")))
    check("EEG saliency image is present", bool(eeg["explainability"].get("saliency_png_base64")))
    check("EEG top_features are present", len(eeg["explainability"].get("top_features", [])) > 0)
    eeg_id = eeg["analysis_id"]

    # --------------------------------------------------------------------- X-ray
    print("== X-ray ==")
    from PIL import Image

    image = synth_xray("PNEUMONIA", seed=55)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    response = client.post(
        "/api/v1/xray/analyze",
        files={"file": ("chest.png", buffer.getvalue(), "image/png")},
        data={"patient_id": "PT-LIVE-XRAY", "age": "72", "sex": "male", "view": "PA"},
    )
    check("POST /api/v1/xray/analyze returns 200", response.status_code == 200, response.text[:300])
    xray = response.json()
    check("X-ray label is PNEUMONIA", xray["prediction"]["label"] == "PNEUMONIA", str(xray["prediction"]["label"]))
    check("X-ray model_source is trained", xray.get("model_source") == "trained", str(xray.get("model_source")))
    check("X-ray Grad-CAM overlay is present", bool(xray["explainability"].get("overlay_png_base64")))
    xray_id = xray["analysis_id"]

    # ------------------------------------------------------------------- reports
    print("== reports ==")
    for fmt in ("pdf", "html", "json"):
        response = client.post(f"/api/v1/reports/{ecg_id}?format={fmt}")
        check(f"POST /api/v1/reports ({fmt}) returns 200", response.status_code == 200, response.text[:300])
        if response.status_code != 200:
            continue
        report = response.json()
        check(f"  {fmt} report has a download_url", bool(report.get("download_url")))
        download = client.get(report["download_url"])
        check(f"  {fmt} report downloads", download.status_code == 200)
        check(
            f"  {fmt} report has the expected content",
            _has_content(fmt, download.content),
            repr(download.content[:32]),
        )

    listing = client.get("/api/v1/reports/list")
    check("GET /api/v1/reports/list returns 200", listing.status_code == 200)
    check("report list is not empty", len(listing.json().get("reports", [])) > 0)

    # ------------------------------------------------------------ history & stats
    print("== history & stats ==")
    history = client.get("/api/v1/history?limit=50")
    check("GET /api/v1/history returns 200", history.status_code == 200)
    items = history.json().get("items", [])
    check("history contains the analyses", len(items) >= 3, str(len(items)))
    summary = client.get("/api/v1/history/summary")
    check("GET /api/v1/history/summary returns 200", summary.status_code == 200)
    stats = client.get("/api/v1/stats")
    check("GET /api/v1/stats returns 200", stats.status_code == 200)

    # --------------------------------------------------------------------- demo
    print("== demo samples ==")
    demo = client.get("/api/v1/demo/classes")
    check("GET /api/v1/demo/classes returns 200", demo.status_code == 200)
    for modality, label in (("ecg", "AFIB"), ("eeg", "EPILEPSY"), ("xray", "PNEUMONIA")):
        response = client.get(f"/api/v1/demo/{modality}/{label}")
        check(f"GET /api/v1/demo/{modality}/{label} returns 200", response.status_code == 200)

    # ------------------------------------------------------------------ summary
    failures = [name for name, ok, _ in CHECKS if not ok]
    print("\n" + "=" * 62)
    print(f"checks: {len(CHECKS)}   passed: {len(CHECKS) - len(failures)}   failed: {len(failures)}")
    if failures:
        for name in failures:
            print(f"  FAILED: {name}")
    print("=" * 62)
    client.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
