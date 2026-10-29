"""End-to-end API tests."""

from __future__ import annotations

import io

import pytest


# --------------------------------------------------------------------------- #
# Service health
# --------------------------------------------------------------------------- #
def test_liveness_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_every_modality(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    for modality in ("ecg", "eeg", "xray"):
        assert modality in payload["models"]
        assert payload["models"][modality]["classes"]


def test_model_registry(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert set(models) == {"ecg", "eeg", "xray"}
    assert models["ecg"]["classes"] == ["NORM", "AFIB", "MI", "STDE", "PVC"]
    assert models["eeg"]["classes"] == ["NORMAL", "EPILEPSY", "PARKINSON", "DEPRESSION"]
    assert len(models["xray"]["classes"]) == 6


# --------------------------------------------------------------------------- #
# Signal analysis
# --------------------------------------------------------------------------- #
def test_ecg_analysis(client, ecg_signal):
    signal, fs = ecg_signal
    response = client.post(
        "/api/v1/ecg/analyze",
        json={
            "patient": {"patient_id": "PT-TEST-ECG", "age": 58, "sex": "male"},
            "samples": [float(v) for v in signal],
            "sampling_rate": fs,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["modality"] == "ecg"
    assert payload["prediction"]["label"] in ["NORM", "AFIB", "MI", "STDE", "PVC"]
    assert 0.0 <= payload["prediction"]["confidence"] <= 1.0
    assert len(payload["findings"]) > 10
    assert payload["explainability"]["saliency_png_base64"]
    assert payload["recommendations"]
    assert payload["analysis_id"]


def test_eeg_analysis(client, eeg_signal):
    signal, fs = eeg_signal
    response = client.post(
        "/api/v1/eeg/analyze",
        json={
            "patient": {"patient_id": "PT-TEST-EEG"},
            "samples": [float(v) for v in signal],
            "sampling_rate": fs,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["modality"] == "eeg"
    assert payload["prediction"]["label"] in ["NORMAL", "EPILEPSY", "PARKINSON", "DEPRESSION"]
    assert payload["findings"]
    assert payload["explainability"]["saliency_png_base64"]


def test_ecg_upload(client, ecg_signal):
    signal, fs = ecg_signal
    body = "\n".join([f"fs={fs:.0f}", "samples", *[f"{v:.6f}" for v in signal]])
    response = client.post(
        "/api/v1/ecg/analyze/upload",
        files={"file": ("ecg.csv", body.encode("utf-8"), "text/csv")},
        data={"patient_id": "PT-UPLOAD", "sampling_rate": str(fs)},
    )
    assert response.status_code == 200
    assert response.json()["patient"]["patient_id"] == "PT-UPLOAD"


def test_xray_analysis(client, xray_bytes):
    response = client.post(
        "/api/v1/xray/analyze",
        files={"file": ("cxr.png", xray_bytes, "image/png")},
        data={"patient_id": "PT-TEST-XRAY", "age": 63, "sex": "female", "view": "PA"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["modality"] == "xray"
    assert payload["prediction"]["label"]
    assert payload["findings"]
    assert payload["explainability"]["saliency_png_base64"]


def test_xray_rejects_non_image(client):
    response = client.post(
        "/api/v1/xray/analyze",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        data={"patient_id": "PT-BAD"},
    )
    assert response.status_code in (415, 422)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_rejects_too_few_samples(client):
    response = client.post(
        "/api/v1/ecg/analyze",
        json={
            "patient": {"patient_id": "PT-SHORT"},
            "samples": [0.1, 0.2, 0.3],
            "sampling_rate": 500,
        },
    )
    assert response.status_code == 422


def test_rejects_missing_patient(client, ecg_signal):
    signal, fs = ecg_signal
    response = client.post(
        "/api/v1/ecg/analyze",
        json={"samples": [float(v) for v in signal], "sampling_rate": fs},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def test_report_generation_and_download(client, ecg_signal):
    signal, fs = ecg_signal
    analysis = client.post(
        "/api/v1/ecg/analyze",
        json={
            "patient": {"patient_id": "PT-REPORT"},
            "samples": [float(v) for v in signal],
            "sampling_rate": fs,
        },
    ).json()
    analysis_id = analysis["analysis_id"]

    response = client.post(f"/api/v1/reports/{analysis_id}?format=pdf")
    assert response.status_code == 200
    report = response.json()
    assert report["format"] == "pdf"
    assert report["size_bytes"] > 1000

    download = client.get(report["download_url"])
    assert download.status_code == 200
    assert download.content[:4] == b"%PDF"


def test_html_report(client, ecg_signal):
    signal, fs = ecg_signal
    analysis = client.post(
        "/api/v1/eeg/analyze",
        json={
            "patient": {"patient_id": "PT-REPORT-HTML"},
            "samples": [float(v) for v in signal],
            "sampling_rate": fs,
        },
    ).json()
    response = client.post(f"/api/v1/reports/{analysis['analysis_id']}?format=html")
    assert response.status_code == 200
    download = client.get(response.json()["download_url"])
    assert download.status_code == 200
    assert "<html" in download.text.lower()


def test_report_for_unknown_analysis(client):
    response = client.post("/api/v1/reports/does-not-exist?format=pdf")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# History and dashboard support
# --------------------------------------------------------------------------- #
def test_history_contains_the_analyses(client):
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 1
    assert payload["items"]


def test_history_summary(client):
    response = client.get("/api/v1/history/summary")
    assert response.status_code == 200
    assert "label_distribution" in response.json()


def test_stats_endpoint(client):
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    assert "models" in response.json()


# --------------------------------------------------------------------------- #
# Demo samples and dashboard assets
# --------------------------------------------------------------------------- #
def test_demo_samples(client):
    response = client.get("/api/v1/demo/classes")
    assert response.status_code == 200
    classes = response.json()
    assert "NORM" in classes["ecg"]
    assert "NORMAL" in classes["eeg"]
    assert "COVID19" in classes["xray"]


def test_dashboard_is_served(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "MedDiag" in response.text


def test_static_assets_are_served(client):
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/static/css/app.css").status_code == 200


# --------------------------------------------------------------------------- #
# Report download links
# --------------------------------------------------------------------------- #
def test_download_url_is_a_complete_path(client):
    """`download_url` already contains the API prefix.

    The dashboard uses the value as-is. If a future change makes the API
    return a *relative* path, the dashboard would have to prefix it again -
    and the two must never both happen, or the URL becomes
    "/api/v1/api/v1/..." and every download 404s.
    """
    analysis = client.post(
        "/api/v1/ecg/analyze",
        json={
            "samples": [0.0, 0.1, -0.1, 0.2, -0.2] * 120,
            "sampling_rate": 500,
            "patient": {"patient_id": "TEST-1", "age": 50, "sex": "male"},
        },
    )
    assert analysis.status_code == 200
    analysis_id = analysis.json()["analysis_id"]

    created = client.post(f"/api/v1/reports/{analysis_id}?format=pdf")
    assert created.status_code == 200
    download_url = created.json()["download_url"]
    assert download_url.startswith("/api/v1/reports/")

    # The URL the dashboard builds must resolve.
    assert client.get(download_url).status_code == 200


def test_dashboard_does_not_double_prefix_the_download_url(client):
    """Regression: `API + report.download_url` produced /api/v1/api/v1/..."""
    script = client.get("/static/js/app.js").text
    assert "API + report.download_url" not in script
    assert "API + item.download_url" not in script

