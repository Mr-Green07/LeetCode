# MedDiag — AI-Based Medical Diagnosis Assistant (ECG / EEG / Chest X-ray)

A Python **backend system** that turns raw cardiac signals, brain signals and chest
radiographs into structured clinical decision support: a diagnosis, the findings that
support it, an explainable-AI visual, and an automatically generated diagnostic report —
all behind one REST API and one dashboard for healthcare professionals.

> **Clinical safety notice.** MedDiag is a **research and educational prototype**.
> It is **not** a medical device, its output is **not** a diagnosis, and it must not be
> used to make clinical decisions about real patients. The models shipped with this
> repository are trained on **synthetic data** — see [Model provenance](#model-provenance).

---

## Table of contents

1. [Objectives → how they are met](#objectives--how-they-are-met)
2. [Feature overview](#feature-overview)
3. [Architecture](#architecture)
4. [Project layout](#project-layout)
5. [Quick start](#quick-start)
6. [Running the server](#running-the-server)
7. [Using the API](#using-the-api)
8. [Using the dashboard](#using-the-dashboard)
9. [Training the models](#training-the-models)
10. [Dataset adapters](#dataset-adapters)
11. [Model provenance](#model-provenance)
12. [Configuration](#configuration)
13. [Testing](#testing)
14. [Benchmarking](#benchmarking-for-research--papers)
15. [Troubleshooting](#troubleshooting)
16. [Extending the project](#extending-the-project)
17. [Limitations and future work](#limitations-and-future-work)

> Full testing parameters, evaluation protocol and measured results are in
> **[TESTING.md](TESTING.md)**; the complete function inventory is in
> **[functions.md](functions.md)**.

---

## Objectives → how they are met

| # | Stated objective | Implementation | Where to look |
|---|---|---|---|
| 1 | Detect cardiovascular diseases from **ECG** signals | `ResNet1D` + clinical-feature fusion, 5 classes (`NORM`, `AFIB`, `MI`, `STDE`, `PVC`) | `app/ml/ecg/` |
| 2 | Detect neurological disorders from **EEG** signals | `EEGNet` + clinical-feature fusion, 4 classes (`NORMAL`, `EPILEPSY`, `PARKINSON`, `DEPRESSION`) | `app/ml/eeg/` |
| 3 | Detect lung diseases from **chest X-rays** | 4-stage 2-D CNN + clinical-feature fusion, 6 classes (`NORMAL`, `PNEUMONIA`, `CONSOLIDATION`, `EFFUSION`, `TUBERCULOSIS`, `COVID19`) | `app/ml/xray/` |
| 4 | Generate **automated diagnostic reports** | PDF (ReportLab), HTML and JSON renderers, stored on disk and downloadable | `app/services/report.py` |
| 5 | Provide **explainable-AI visualizations** | Grad-CAM (1-D and 2-D), integrated gradients, occlusion sensitivity, per-feature network attribution, all rendered to base64 PNG | `app/ml/xai/` |
| 6 | Provide a **unified dashboard** for healthcare professionals | Dependency-free SPA served by FastAPI — tabs for all three modalities, history, reports, model info | `app/web/static/` |
| 7 | Deliver a **Python backend** | FastAPI + Uvicorn, Pydantic v2 contracts, SQLite audit trail, optional API-key auth | `app/` |

---

## Feature overview

### Clinical analysis
- **Signal conditioning** — band-pass filtering, notch filtering, AHA-standard 0.05 Hz
  baseline-wander removal, resampling, z-score normalisation, unit auto-detection
  (mV / µV / raw counts).
- **Rhythm and morphology measurement** — Pan-Tompkins R-peak detection with adaptive
  thresholds, median-beat template analysis, TP-referenced ST measurement, QRS duration,
  QT/PR intervals, HRV (RMSSD, pNN50, LF/HF), PVC detection.
- **EEG spectral analysis** — band powers (delta/theta/alpha/beta/gamma), spectral
  entropy, Hjorth parameters, spike-wave index, alpha asymmetry, artefact detection.
- **Radiograph analysis** — CLAHE enhancement, lung-field-relative zone statistics
  (apical / perihilar / peripheral / costophrenic), image-quality scoring and a
  "is this a chest X-ray?" plausibility check.
- **Signal-quality scoring** — every response carries quality metrics; low-quality
  inputs reduce the reported confidence rather than producing a confident wrong answer.

### Explainable AI
| Technique | Modality | Output |
|---|---|---|
| Grad-CAM / Grad-CAM1D | ECG, EEG, X-ray | Heat map over the waveform or the radiograph |
| Integrated Gradients | ECG | Per-sample input attribution map |
| Occlusion sensitivity | any signal | Ranked list of the most influential time segments |
| Feature attribution | all | Per-clinical-feature contribution to the predicted class score |
| Feature deviation analysis | all | Which measured features fall outside their reference band |

All visualisations are returned inline as base64-encoded PNGs, so a client never has to
fetch or render them separately.

### Reporting and audit
- Every analysis is persisted as JSON, recorded in a SQLite audit trail (WAL mode) and
  can be exported as a **PDF**, **HTML** or **JSON** report.
- Reports embed the probability breakdown, the clinical findings table, the
  recommendations list and the explainability images.

### Operational
- Optional shared-secret authentication (`X-API-Key`), **open by default** for local use.
- CORS enabled, upload size limits, structured error handling, request metrics.
- Runs on CPU with no external services and no CDN dependencies — fully offline.

---

## Architecture

```
                    ┌──────────────────────────┐
   Browser  ──────► │  FastAPI  /dashboard      │
                    │  (static SPA, no CDN)     │
                    └────────────┬─────────────┘
                                 │  REST /api/v1
                    ┌────────────▼─────────────┐
                    │  API routes               │
                    │  ecg · eeg · xray ·       │
                    │  reports · history · demo │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  DiagnosisService         │  orchestration + XAI wiring
                    └──────┬──────────────┬──────┘
                           │              │
        ┌──────────────────▼───┐   ┌──────▼───────────────────┐
        │  ModalityModel (ML)   │   │  ReportService           │
        │  ├ ECGModel           │   │  PDF · HTML · JSON        │
        │  ├ EEGModel           │   └──────┬───────────────────┘
        │  └ XrayModel          │          │
        │  trained ckpt OR      │   ┌──────▼───────────────────┐
        │  heuristic baseline   │   │  Storage + SQLite audit  │
        └──────────────────┬───┘   └──────────────────────────┘
                           │
   ┌───────────────────────┼───────────────────────────────┐
   ▼                       ▼                               ▼
Preprocess              Features                    XAI (Grad-CAM, IG,
(filter, resample,       (18 ECG / 16 EEG /           occlusion, feature
R-peak, spectral,        11 X-ray numeric             attribution)
zones)                   features)
```

**Dual inference path.** Every modality first tries to load a trained checkpoint
(`models/<modality>_model.pt`). If no checkpoint exists — or PyTorch is unavailable —
it falls back to a deterministic **clinical-rule heuristic**. The active path is always
reported in the response as `model_source` (`"trained"` or `"heuristic_baseline"`), so a
baseline can never be mistaken for a validated model.

---

## Project layout

```
meddiag/
├── app/
│   ├── main.py                 FastAPI app factory, lifespan, CORS, static mount
│   ├── config.py               Settings (env prefix MEDDIAG_), directory bootstrap
│   ├── schemas.py              All Pydantic request/response contracts
│   ├── api/
│   │   ├── router.py            Router assembly
│   │   ├── deps.py              API-key dependency
│   │   ├── routes_ecg.py        ECG endpoints
│   │   ├── routes_eeg.py        EEG endpoints
│   │   ├── routes_xray.py       X-ray endpoints
│   │   ├── routes_reports.py    Report generation & download
│   │   ├── routes_history.py    Analysis history & summary
│   │   ├── routes_models.py     Health, model registry, stats
│   │   ├── routes_health.py     Root liveness probes
│   │   └── routes_demo.py       Synthetic demo samples
│   ├── core/
│   │   ├── logging.py           Structured logging setup
│   │   ├── errors.py            Error hierarchy + FastAPI exception handlers
│   │   ├── audit.py             SQLite audit trail (WAL)
│   │   ├── storage.py           File storage, safe path joins, ID generation
│   │   ├── metrics.py           Request counters / latency registry
│   │   └── parsers.py           CSV / JSON / TXT / MAT signal parsers
│   ├── ml/
│   │   ├── base.py              ModalityModel ABC, checkpoint loading, softmax
│   │   ├── registry.py          Lazy, thread-safe model registry
│   │   ├── training.py          Shared training loop, metrics, checkpoint writer
│   │   ├── ecg/                 preprocess · features · model · inference
│   │   ├── eeg/                 preprocess · features · model · inference
│   │   ├── xray/                preprocess · features · model · inference
│   │   └── xai/                 gradcam.py · plots.py
│   ├── data/
│   │   ├── synthetic.py         Physiological signal + radiograph generators
│   │   ├── splitting.py         Leakage-safe (grouped) train/validation splits
│   │   ├── ecg_datasets.py      PTB-XL / MIT-BIH / folder adapters
│   │   ├── eeg_datasets.py      Bonn / MNE-sample / folder adapters
│   │   └── xray_datasets.py     NIH ChestX-ray14 / folder adapters
│   ├── services/
│   │   ├── diagnosis.py         Orchestration, findings, recommendations, XAI
│   │   └── report.py            PDF / HTML / JSON report rendering
│   └── web/static/              Dependency-free dashboard (HTML, CSS, JS, SVG)
├── scripts/
│   ├── run_server.py            Start Uvicorn with the project settings
│   ├── train_ecg.py             ECG training entry point
│   ├── train_eeg.py             EEG training entry point
│   ├── train_xray.py            X-ray training entry point
│   ├── make_demo_data.py        Write synthetic demo files to disk
│   ├── download_datasets.py     Fetch public benchmarks
│   ├── evaluate_models.py       Score the checkpoints on freshly seeded data
│   ├── run_benchmark.py         Full benchmark suite -> results/ (CSV + Markdown)
│   ├── fill_testing_section.py  Refresh the results tables inside TESTING.md
│   ├── verify_live.py           End-to-end check against a running server
│   └── smoke_check.py           End-to-end heuristic sanity check (15 cases)
├── tests/                       pytest suite (API, features, XAI, splitting)
├── models/                      Trained checkpoints (created by the training scripts)
├── reports/                     Generated PDF / HTML reports
├── results/                    Benchmark artefacts (CSV + Markdown tables)
├── storage/                     Audit database + analysis payloads
├── requirements.txt
├── .env.example
└── README.md · functions.md · TESTING.md
```

---

## Quick start

### Prerequisites
- Python **3.10+** (developed and verified on 3.12)
- ~2 GB of free disk for the virtual environment (PyTorch is the bulk of it)

### 1. Create a virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

> On Windows, if `python -m venv` fails, use the `py` launcher as shown above — a
> mismatched interpreter on `PATH` is the usual cause.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

CPU-only machines (no CUDA) should install PyTorch from the CPU wheel index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Configure (optional)

```bash
cp .env.example .env        # Linux / macOS
copy .env.example .env      # Windows
```

Every variable in `.env` is optional — the defaults work out of the box.

### 4. Run the server

```bash
python scripts/run_server.py
```

Then open:

| URL | What it is |
|---|---|
| <http://localhost:8000/dashboard> | The unified clinical dashboard |
| <http://localhost:8000/docs> | Interactive OpenAPI / Swagger documentation |
| <http://localhost:8000/redoc> | ReDoc reference |
| <http://localhost:8000/api/v1/health> | Service + model health check |

---

## Running the server

`scripts/run_server.py` reads the same settings as the app, so `MEDDIAG_HOST`,
`MEDDIAG_PORT` and `MEDDIAG_DEBUG` all apply:

```bash
python scripts/run_server.py                 # defaults from .env
python scripts/run_server.py --port 9000     # override the port
python scripts/run_server.py --reload        # auto-reload for development
```

You can also start Uvicorn directly:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On Windows, run this after activating the project environment (`.venv\Scripts\Activate.ps1`
in PowerShell or `source .venv/Scripts/activate` in Git Bash). Using `python -m uvicorn`
ensures Uvicorn runs with the same interpreter where the project dependencies were installed.

---

## Using the API

All endpoints live under `/api/v1`. Interactive documentation with working "Try it out"
buttons is at `/docs`.

### ECG analysis — JSON body

```bash
curl -X POST http://localhost:8000/api/v1/ecg/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "patient": {"patient_id": "PT-001", "age": 58, "sex": "male"},
        "samples": [0.0, 0.12, 0.05, ...],
        "sampling_rate": 500.0
      }'
```

### ECG analysis — file upload

```bash
curl -X POST http://localhost:8000/api/v1/ecg/analyze/upload \
  -F "file=@recording.csv" \
  -F "patient_id=PT-001" -F "age=58" -F "sex=male" -F "sampling_rate=500"
```

Accepted signal formats: `.csv`, `.json`, `.txt` (whitespace separated), `.mat`.

### EEG analysis — same shape, `sampling_rate` typically `256.0`

```bash
curl -X POST http://localhost:8000/api/v1/eeg/analyze \
  -H "Content-Type: application/json" \
  -d '{"patient": {"patient_id": "PT-002", "age": 34, "sex": "female"},
       "samples": [0.01, -0.02, ...], "sampling_rate": 256.0}'
```

### Chest X-ray analysis — multipart upload

```bash
curl -X POST http://localhost:8000/api/v1/xray/analyze \
  -F "file=@chest.png" \
  -F "patient_id=PT-003" -F "age=67" -F "sex=" -F "view=PA"
```

Accepted image formats: PNG, JPEG, DICOM.

### Response shape (abridged)

```jsonc
{
  "analysis_id": "ana_9f2c1b7e4a",
  "modality": "ecg",
  "patient": { "patient_id": "PT-001", "age": 58, "sex": "male" },
  "prediction": {
    "label": "AFIB",
    "confidence": 0.887,
    "severity": "moderate",
    "description": "Atrial fibrillation - irregularly irregular rhythm with absent P waves.",
    "class_probabilities": { "NORM": 0.063, "AFIB": 0.887, "MI": 0.018, "STDE": 0.016, "PVC": 0.016 }
  },
  "differential": [
    { "label": "AFIB", "probability": 0.887, "severity": "moderate" }
  ],
  "findings": [
    { "name": "Rr Irregularity", "value": "0.31", "unit": null,
      "reference_range": "0.0 – 0.12", "status": "abnormal",
      "explanation": "..." }
  ],
  "quality": { "signal_quality_index": 0.91, "usable": true, "issues": [],
                "snr_db": 21.4, "clipped_samples": 0 },
  "explainability": {
    "methods": ["Grad-CAM", "Feature attribution (trained network)"],
    "summary": "The prediction 'AFIB' was driven mainly by Rr Irregularity, ...",
    "top_features": [ { "name": "Rr Irregularity", "contribution": 0.83,
                        "direction": "increases_risk", "detail": "..." } ],
    "class_relevance": { "NORM": 0.06, "AFIB": 0.89, "MI": 0.02 },
    "saliency_png_base64": "iVBORw0KGgo...",
    "overlay_png_base64": null,
    "waveform_png_base64": "iVBORw0KGgo...",
    "attribution_method": "Grad-CAM over the final convolutional block"
  },
  "recommendations": [ "..." ],
  "model_version": "1.0.0",
  "model_source": "trained",
  "report_url": "/api/v1/reports/ana_9f2c1b7e4a",
  "created_at": "2026-01-01T12:00:00Z",
  "disclaimer": "Decision-support output for qualified healthcare professionals. ..."
}
```

Key fields:

| Field | Meaning |
|---|---|
| `prediction.label` | The predicted diagnostic class |
| `prediction.confidence` | Probability of the predicted class (0–1) |
| `prediction.class_probabilities` | Full probability map over every class |
| `differential` | Ranked alternatives with their probability and severity |
| `findings[]` | Individual clinical measurements with reference ranges and status |
| `quality` | Input quality (`signal_quality_index`, `usable`, `issues`, `snr_db`) — read this before trusting the label |
| `explainability` | Methods, summary, ranked feature contributions and the visualisation images |
| `model_source` | `"trained"` or `"heuristic_baseline"` |
| `disclaimer` | Clinical-safety notice included in every response |
```

### Diagnostic report

```bash
# Create (returns a report id + download url)
curl -X POST "http://localhost:8000/api/v1/reports/ana_9f2c1b7e4a?format=pdf"

# Download the bytes
curl -OJ "http://localhost:8000/api/v1/reports/<report_id>/download"

# List every report, or the ones for one analysis
curl "http://localhost:8000/api/v1/reports/list"
curl "http://localhost:8000/api/v1/reports/ana_9f2c1b7e4a"
```

`format` accepts `pdf`, `html` and `json`.

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Service status and per-model readiness |
| `GET` | `/api/v1/models` | All registered models with architecture, metrics, source |
| `GET` | `/api/v1/models/{modality}` | One model in detail |
| `GET` | `/api/v1/stats` | Aggregate counters for the dashboard |
| `GET` | `/api/v1/history` | Recent analyses, newest first (filterable) |
| `GET` | `/api/v1/history/summary` | Label distribution for the charts |
| `GET` | `/api/v1/ecg/classes` | ECG class list with descriptions |
| `GET` | `/api/v1/eeg/classes` | EEG class list with descriptions |
| `GET` | `/api/v1/xray/classes` | X-ray class list with descriptions |
| `GET` | `/api/v1/demo/ecg/{label}` | Synthetic ECG sample (for demos) |
| `GET` | `/api/v1/demo/eeg/{label}` | Synthetic EEG sample |
| `GET` | `/api/v1/demo/xray/{label}` | Synthetic chest radiograph (PNG) |
| `GET` | `/api/v1/demo/classes` | Every demo sample that can be fetched |

### Authentication (optional)

The service is **open by default** for local use. To require a key, set it in `.env`:

```bash
MEDDIAG_API_KEY=change-me
```

Every `/api/v1` request must then send the header:

```bash
curl -H "X-API-Key: change-me" http://localhost:8000/api/v1/ecg/classes
```

A missing or wrong key returns `401`.

---

## Using the dashboard

Open <http://localhost:8000/dashboard>. The dashboard is a single page with five tabs:

- **ECG / EEG / X-ray** — pick a modality, load a built-in demo sample or upload your
  own file, then read the prediction, the probability breakdown, the clinical findings
  table, the explainability images and the recommendations.
- **Case history** — every analysis performed in this session, filterable by modality,
  with one-click re-opening of a past result.
- **Reports** — generate and download PDF/HTML/JSON reports for any analysis.
- **About** — model architecture, class lists and the active `model_source`.

The dashboard uses **no CDN**: charts are hand-rolled SVG and CSS, so it works on an
air-gapped machine or behind a proxy that blocks external hosts.

---

## Training the models

The service works out of the box without training (it uses the heuristic baseline).
Training produces real CNN checkpoints, which switches `model_source` to `"trained"`
and **activates the Grad-CAM visualisations**.

### Synthetic corpus (no download required)

```bash
python scripts/train_ecg.py  --dataset synthetic --per-class 140 --epochs 25 --batch-size 32
python scripts/train_eeg.py  --dataset synthetic --per-class 140 --epochs 25 --batch-size 32
python scripts/train_xray.py --dataset synthetic --per-class 120 --epochs 20 --batch-size 32
```

Checkpoints are written to `models/ecg_model.pt`, `models/eeg_model.pt` and
`models/xray_model.pt`, each with a `*.metrics.json` sidecar holding the validation
metrics. Restart the server afterwards so the new checkpoints are loaded.

### Real public benchmarks

```bash
python scripts/train_ecg.py  --dataset ptbxl  --data-root data/datasets/ptb-xl
python scripts/train_ecg.py  --dataset mitbih --data-root data/datasets/mitdb \
    --validation-ratio 0.15 --test-ratio 0.15 --output models/ecg_model_mitbih.pt
python scripts/train_eeg.py  --dataset bonn   --data-root data/datasets/bonn \
    --validation-ratio 0.15 --test-ratio 0.15 --output models/eeg_model_bonn.pt
python scripts/train_xray.py --dataset nih    --data-root data/datasets/chestxray14
```

`scripts/download_datasets.py` fetches the public benchmarks into `data/datasets/`.

For the grouped datasets (`mitbih`, `ptbxl`, `bonn`) the split is made on the
**group** — record id or subject id — so no recording or subject can appear on
both sides. With `--test-ratio > 0` the split becomes three-way: the test
partition is carved out first and is never looked at during training, early
stopping or model selection, and the checkpoint stores the metrics it measured
there (`test_accuracy`, `test_f1_macro`, `test_auc`).

### Useful training flags

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `synthetic` | `synthetic` \| `folder` \| `ptbxl` \| `mitbih` \| `bonn` \| `nih` |
| `--data-root` | – | Directory holding the dataset |
| `--per-class` | 140–160 | Synthetic samples per class |
| `--epochs` | 30 | Maximum epochs (early stopping applies) |
| `--batch-size` | 32 | Mini-batch size |
| `--lr` | `1e-3` | AdamW learning rate |
| `--patience` | 6 | Early-stopping patience |
| `--seed` | 42 | Reproducibility seed (applied **before** the network is built) |
| `--validation-ratio` | 0.2 | Share of the data reserved for early stopping |
| `--test-ratio` | 0.0 | Share of the data reserved as an untouched test partition; the checkpoint then records its own `test_*` metrics |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--limit` | – | Cap the number of records for real datasets |
| `--output` | `models/<modality>_model.pt` | Checkpoint path |

### Sanity check without training

```bash
python scripts/smoke_check.py
```

Runs all 15 synthetic classes through the loaded model — the trained checkpoint when one
is present in `models/`, otherwise the heuristic baseline — and prints a pass/fail line
per class. Expected output: `failures: 0`.

### Score the shipped checkpoints on unseen data

```bash
python scripts/evaluate_models.py
```

Generates fresh synthetic samples from a **different seed range** than the training
corpus and reports the trained models' accuracy on them:

```
ECG    source=trained              60/60 = 100.0%
EEG    source=trained              48/48 = 100.0%
X-ray  source=trained              60/60 = 100.0%

OVERALL 168/168 = 100.0%
```

> This measures how well the networks generalise **within the synthetic domain**. It is
> not a measure of clinical accuracy — see [Model provenance](#model-provenance).

---

## Dataset adapters

| Adapter | Dataset | Labels used |
|---|---|---|
| `synthetic` | Built-in generator (seeded, deterministic) | All 15 classes |
| `ptbxl` | PTB-XL (`wfdb`) | `NORM`, `AFIB`, `MI`, `STDE`, `PVC` |
| `mitbih` | MIT-BIH Arrhythmia Database (`wfdb`) | `NORM`, `AFIB` (record-level), `PVC` (beat-level) |
| `bonn` | Bonn EEG Database | `NORMAL`, `EPILEPSY` |
| `nih` | NIH ChestX-ray14 | `NORMAL`, `PNEUMONIA`, `CONSOLIDATION`, `EFFUSION` |
| `folder` | `<root>/<LABEL>/<file>` for any modality | Folder names mapped to the class list |

Two label-mapping decisions are worth stating explicitly, because both are
clinically load-bearing:

- **MIT-BIH atrial fibrillation is a record-level label.** The beat symbol `A`
  means *atrial premature beat*, which is not atrial fibrillation, so it is not
  mapped to `AFIB`. Instead every window taken from one of the 20 known AFIB
  records inherits the record-level `AFIB` label.
- **Bonn "interictal" carries epileptiform activity.** Sets `C`/`D`/`E`/`N`/`F`
  show epileptiform spikes between seizures, so they map to `EPILEPSY`; only the
  healthy sets (`A`/`B`/`H`/`K`/`O`/`Z`) map to `NORMAL`.

The MIT-BIH loader also spreads its per-class selection evenly across each
record instead of taking the first beats, so the rare ectopic classes are
represented rather than drowned out by normal beats.

Synthetic generators model real physiology rather than random noise: ECG beats are
built from P-QRS-T morphology with per-condition PR/QT/ST changes, AFIB uses irregular
RR intervals with absent P waves, PVCs are premature and wide with atypical R amplitude;
EEG combines band-limited noise with the spectral signature of each disorder; chest
radiographs are rendered with lung-geometry-relative opacity patterns.

---

## Model provenance

This matters, so it is stated plainly:

- The default checkpoints shipped in `models/` are trained on the **built-in
  synthetic corpus**. Their validation metrics therefore measure **synthetic-data**
  separation, not clinical accuracy against real patients. They demonstrate the
  full pipeline end to end.
- **Real-data checkpoints are also trained and shipped** for the two benchmarks
  that fit on a normal machine: `models/ecg_model_mitbih.pt` (MIT-BIH) and
  `models/eeg_model_bonn.pt` (Bonn). These are the ones to quote for real-data
  claims, and their measured scores are in [TESTING.md](TESTING.md). They are
  trained on a **70 / 15 / 15 grouped split**: the validation part drives early
  stopping, the test part is never touched during training, and the checkpoint
  records its own test-partition metrics (`test_accuracy`, `test_f1_macro`,
  `test_auc`) so a reported number can be traced back to the file it came from.
- Training is **bit-reproducible**: the training scripts seed NumPy and
  PyTorch *before* the network is constructed, so re-running the same command
  with the same seed produces the same weights and the same metrics.
- The clinical numbers behind each finding (reference ranges, thresholds, feature
  definitions) follow published conventions, but they have **not** been validated
  against a clinical cohort.
- **Before any real-world use, retrain on the public benchmarks** listed above
  (`--dataset ptbxl`, `mitbih`, `bonn`, `nih`) and re-run a proper external validation.
  The adapters are already implemented for this. NIH ChestX-ray14 (~45 GB) was
  not downloaded here, so the X-ray results are synthetic-domain only.
- The API always exposes `model_source` so a deployment can distinguish a trained model
  from the heuristic fallback.

**Diagnostic classes**

| Modality | Classes |
|---|---|
| ECG | `NORM`, `AFIB`, `MI`, `STDE`, `PVC` |
| EEG | `NORMAL`, `EPILEPSY`, `PARKINSON`, `DEPRESSION` |
| X-ray | `NORMAL`, `PNEUMONIA`, `CONSOLIDATION`, `EFFUSION`, `TUBERCULOSIS`, `COVID19` |

**Architectures**

| Modality | Architecture | Network input | Clinical features |
|---|---|---|---|
| ECG | ResNet1D (3 residual stages, avg+max pool) + fusion head | `1 × 2500` (5 s @ 500 Hz) | 18 |
| EEG | EEGNet (depthwise-separable, 3 blocks) + fusion head | `1 × 1 × 2560` (10 s @ 256 Hz) | 16 |
| X-ray | 4-stage CNN + global pooling + fusion head | `1 × 224 × 224` | 11 |

Both streams are fused, so the CNN reads raw morphology while the head reasons over the
same numbers a cardiologist uses.

---

## Configuration

All settings are environment variables with the `MEDDIAG_` prefix, or keys in `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `MEDDIAG_HOST` | `0.0.0.0` | Bind address |
| `MEDDIAG_PORT` | `8000` | Bind port |
| `MEDDIAG_DEBUG` | `false` | Debug mode |
| `MEDDIAG_API_KEY` | *(unset)* | When set, `/api/v1` requires `X-API-Key` |
| `MEDDIAG_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `MEDDIAG_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `MEDDIAG_XAI_ENABLED` | `true` | Toggle explainability computation |
| `MEDDIAG_XAI_MAX_FEATURES` | `8` | How many feature contributions to report |
| `MEDDIAG_ECG_DEFAULT_SAMPLING_RATE` | `500` | Default ECG sampling rate |
| `MEDDIAG_EEG_DEFAULT_SAMPLING_RATE` | `256` | Default EEG sampling rate |
| `MEDDIAG_RETAIN_ANALYSES_DAYS` | `30` | Retention window for stored analyses |

Paths (`data/`, `models/`, `reports/`, `storage/`) are also configurable and are created
automatically on startup.

---

## Testing

```bash
pytest -q
```

The suite covers the API contract (health, ECG/EEG/X-ray analysis, reports, history),
the clinical feature extractors, the XAI layer and the evaluation protocol itself
(grouped splitting, metric conventions, reproducible initialisation).
`scripts/smoke_check.py` adds an end-to-end pass over all 15 synthetic classes.

### Benchmarking (for research / papers)

`TESTING.md` documents every test parameter, the evaluation protocol and the
measured results. The harness produces paper-ready Markdown tables and CSV:

```bash
# fetch the public benchmarks (~130 MB)
python scripts/download_datasets.py --dataset mitbih
python scripts/download_datasets.py --dataset bonn

# train the real-data checkpoints (70/15/15 grouped split; the test part is untouched)
python scripts/train_ecg.py --dataset mitbih --data-root data/datasets/mitdb \
    --epochs 40 --patience 8 --validation-ratio 0.15 --test-ratio 0.15 \
    --output models/ecg_model_mitbih.pt
python scripts/train_eeg.py --dataset bonn --data-root data/datasets/bonn \
    --epochs 40 --patience 8 --validation-ratio 0.15 --test-ratio 0.15 \
    --output models/eeg_model_bonn.pt

# run the full suite -> results/
python scripts/run_benchmark.py --suite full --per-class 40 \
    --validation-ratio 0.15 --test-ratio 0.15 --out results
```

Evaluation sets are **split by group** — record id for MIT-BIH, subject id for
Bonn — so no recording or subject appears on both sides of the split. The real
datasets use a **three-way** grouped split (train / validation / test): the
validation side drives early stopping and the test side is never touched
during training, so the reported score is a genuine held-out number rather
than a model-selection score. Each result row records the active
`model_source`, so a heuristic baseline can never be quoted as a learned-model
result.

### Verifying a running server

With the server up, an end-to-end script exercises the dashboard, the static assets,
all three analysis endpoints (JSON **and** upload), report generation in all three
formats, the history and statistics endpoints, and the demo routes — asserting that
the **trained** model path is active and that the Grad-CAM images are produced:

```bash
python scripts/verify_live.py --base-url http://127.0.0.1:8000
```

It prints a line per check and exits non-zero if any check fails.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python -m venv` fails or creates a broken environment | The interpreter on `PATH` is the wrong version. Use `py -3.12 -m venv .venv` on Windows. |
| `RuntimeError: ... expected input[1, 1, 2500] to have 1 channels` | An old checkpoint trained with the previous kernel layout. Delete `models/*.pt` and retrain. |
| `No module named 'app'` when running a script | Run from the project root, or use the provided `scripts/*.py` wrappers (they add the root to `sys.path`). |
| `No module named 'fastapi'` when starting Uvicorn | Activate `.venv` first and run `python -m uvicorn ...`; a bare `uvicorn` command may use a different Python installation. |
| `train_ecg.py --synthetic` → "unrecognized arguments" | The flag does not exist. Use `--dataset synthetic`. |
| Response shows `model_source: "heuristic_baseline"` | No checkpoint in `models/`. Train one and restart the server. |
| Grad-CAM images are missing | Grad-CAM requires a **trained** checkpoint; the heuristic path has no gradients. |
| Upload rejected with 413 | Raise `MEDDIAG_MAX_UPLOAD_MB`. |
| `401` from every `/api/v1` call | `MEDDIAG_API_KEY` is set — send the `X-API-Key` header. |
| Torch install is huge / fails | Install the CPU wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. |

---

## Extending the project

- **A new class** — add it to the modality's class list, give it a description and a
  severity, add a synthetic generator branch, then retrain.
- **A new modality** — subclass `ModalityModel` in `app/ml/<modality>/`, implement
  `predict_proba` and `analyze`, and register it in `app/ml/registry.py`.
- **A new dataset** — implement a `load_*` function in `app/data/` returning the
  dataset dataclass, then wire it into the matching `scripts/train_*.py`.
- **A new report format** — add a renderer next to `render_pdf` / `render_html` in
  `app/services/report.py` and register the format in `generate_report`.

---

## Limitations and future work

**Current limitations**
- Synthetic-data checkpoints — see [Model provenance](#model-provenance).
- Single-lead ECG analysis: the service selects the most diagnostic lead of a
  multi-lead recording but does not yet reason across leads jointly.
- No DICOM viewer integration beyond basic decoding.
- The heuristic fallback is a coarse rule engine, not a validated classifier.
- No federated-learning, no DICOM/HL7/FHIR integration, no multi-user authentication.

**Natural next steps**
- Retrain on PTB-XL / MIT-BIH / Bonn / NIH ChestX-ray14 and report external validation.
- Multi-lead ECG reasoning with lead-to-lead concordance features.
- FHIR `DiagnosticReport` export for EHR integration.
- Per-request rate limiting and role-based access control.
- Model cards and bias analysis across patient subgroups.

---

## License

Released for research and educational use. See the repository for details.
**Not a medical device — do not use for clinical decision-making.**
