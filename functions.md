# MedDiag — Function Reference

Complete inventory of every function, method and class in the **AI-Based Medical
Diagnosis Assistant (ECG / EEG / Chest X-ray)** backend.

**Conventions used below**

| Notation | Meaning |
|---|---|
| `def name(...) -> T` | Module-level function |
| `Class.method(...)` | Method on a class |
| `*` | Keyword-only argument |
| `_name` | Private helper (internal, not part of the public contract) |

**Project root:** `D:\minimaxc\meddiag` · **Python:** 3.10+ (verified on 3.12)

---

## Table of contents

1. [Entrypoints and scripts](#1-entrypoints-and-scripts)
2. [Application layer](#2-application-layer)
3. [API layer](#3-api-layer)
4. [Core layer](#4-core-layer)
5. [ML — shared plumbing](#5-ml--shared-plumbing)
6. [ML — ECG](#6-ml--ecg)
7. [ML — EEG](#7-ml--eeg)
8. [ML — X-ray](#8-ml--xray)
9. [ML — explainable AI](#9-ml--explainable-ai)
10. [Data layer](#10-data-layer)
11. [Services layer](#11-services-layer)
12. [Schemas](#12-schemas)
13. [Dashboard](#13-dashboard)
14. [Tests](#14-tests)

---

## 1. Entrypoints and scripts

### `app/main.py`

| Function | Purpose |
|---|---|
| `lifespan(app)` | Async context manager run at startup/shutdown: creates runtime directories, initialises the audit database, pre-loads the model registry. |
| `create_app() -> FastAPI` | Builds and configures the application: metadata, CORS middleware, exception handlers, routers, static mount, `/dashboard`, `/favicon.ico` and `/` routes. |
| `app` | Module-level application instance (`uvicorn app.main:app`). |
| `root() -> Dict[str, Any]` | Returns service name, version, docs URL, dashboard URL and API prefix. |

### `scripts/train_ecg.py` · `scripts/train_eeg.py` · `scripts/train_xray.py`

| Function | Purpose |
|---|---|
| `build_synthetic(per_class, duration_s) -> (signals, features, labels)` | Generates the synthetic corpus and preprocesses it into model-ready tensors. |
| `build_from_folder(root) -> (signals, features, labels)` | Loads a `<root>/<LABEL>/<file>` corpus and preprocesses it. |
| `build_from_dataset(name, root, limit) -> (signals, features, labels)` | Loads a public benchmark (PTB-XL, MIT-BIH, Bonn, NIH) and preprocesses it. |
| `main() -> int` | CLI entry point: parses arguments, builds the dataset, trains, saves the checkpoint and metrics sidecar. |

### `scripts/run_server.py`

| Function | Purpose |
|---|---|
| `main() -> int` | Starts Uvicorn with the host, port and reload settings taken from `MEDDIAG_*` / CLI flags. |

### `scripts/make_demo_data.py`

| Function | Purpose |
|---|---|
| `main() -> int` | Writes synthetic ECG/EEG/X-ray demo files (CSV + PNG) into `data/demo/` for offline testing. |

### `scripts/download_datasets.py`

| Function | Purpose |
|---|---|
| `_fetch(url, destination, *, retries=3) -> bool` | Streams one file to disk with retries; skips files that already exist. |
| `download_archive(url, destination) -> bool` | Downloads and extracts a single zip archive (PTB-XL, NIH ChestX-ray14). |
| `download_mitbih(target, limit=None) -> bool` | Fetches the 40 MIT-BIH records file by file from PhysioNet (`.dat`, `.hea`, `.atr`). |
| `download_bonn(target, limit=None) -> bool` | Fetches the 500 Bonn EEG text recordings set by set. |
| `main() -> int` | Downloads the public benchmarks (MIT-BIH, Bonn, PTB-XL, NIH ChestX-ray14) into `data/datasets/`. |

### `scripts/smoke_check.py`

| Function | Purpose |
|---|---|
| `main() -> int` | Runs all 15 synthetic classes through the loaded model (trained checkpoint if present, otherwise the heuristic baseline) and prints a pass/fail line per class; exits non-zero on any failure. |

### `scripts/verify_live.py`

| Function | Purpose |
|---|---|
| `check(name, condition, detail="")` | Records and prints one verification check. |
| `_has_content(fmt, body)` | True when the downloaded bytes look like the requested report format. |
| `main() -> int` | Drives a running server through the dashboard, static assets, all three analysis endpoints (JSON and upload), all three report formats, the history/statistics endpoints and the demo routes; exits non-zero on any failure. |

### `scripts/evaluate_models.py`

| Function | Purpose |
|---|---|
| `evaluate(name, corpus, modality) -> (correct, total, errors)` | Scores one modality's checkpoint against a freshly seeded corpus and lists the misclassifications. |
| `main() -> int` | Evaluates all three trained models on synthetic data generated outside the training seed range. |

### `scripts/run_benchmark.py`

| Function / Class | Purpose |
|---|---|
| `class EvalSet` | One evaluation set: raw inputs, labels, class names, grouping keys, sampling rate and description. |
| `class ModelVariant` | One model configuration to evaluate: key, label, modality, checkpoint path and description. |
| `_split_mitbih(root, validation_ratio, test_ratio, seed) -> EvalSet` | Builds the held-out MIT-BIH evaluation set (grouped by record). With `test_ratio > 0` it returns the untouched **test** partition of the three-way split; otherwise the validation partition. |
| `_split_bonn(root, validation_ratio, test_ratio, seed) -> EvalSet` | Builds the held-out Bonn evaluation set (grouped by subject), using the same test/validation partition rule as MIT-BIH. |
| `_synthetic(modality, per_class, seed) -> EvalSet` | Builds a synthetic evaluation set from a seed outside the training range. |
| `_load_model(variant)` | Instantiates a modality model with the requested checkpoint, or disables the checkpoint to force the heuristic baseline. |
| `_confusion(y_true, y_pred, n_classes) -> np.ndarray` | Builds the confusion matrix. |
| `_prf(matrix, class_index) -> Dict[str, float]` | Precision / recall / F1 / support for one class. |
| `_macro_auc(y_true, probabilities, n_classes) -> float` | Macro one-vs-rest ROC AUC computed from the rank statistic. |
| `_evaluate(model, dataset) -> Dict[str, Any]` | Runs a model over an evaluation set and collects every metric plus the confusion matrix and the latencies. |
| `_write_csv(path, header, rows)` | Writes one CSV artefact. |
| `_markdown_tables(results, datasets) -> str` | Renders the benchmark results as Markdown tables. |
| `main() -> int` | Runs the suite and writes `summary.csv`, the per-class CSVs, the confusion matrices, `results.md` and `results.json`. |

---

## 2. Application layer

### `app/config.py`

| Function / Class | Purpose |
|---|---|
| `class Settings(BaseSettings)` | Environment-driven configuration (`MEDDIAG_` prefix): server, paths, auth, limits, device, XAI switches, retention. |
| `Settings._split_types(cls, value)` | Field validator that turns a comma-separated `allowed_image_types` string into a list. |
| `Settings.ensure_directories()` | Creates `data/`, `models/`, `reports/`, `storage/` and the audit-DB parent directory. |
| `get_settings() -> Settings` | Returns the cached settings singleton (`lru_cache`). |
| `settings` | Module-level settings instance used across the app. |

---

## 3. API layer

### `app/api/router.py`

| Function / Class | Purpose |
|---|---|
| `api_router` | `APIRouter` that mounts every route module. |

### `app/api/deps.py`

| Function | Purpose |
|---|---|
| `require_api_key(x_api_key: Optional[str] = Header(alias="X-API-Key"))` | FastAPI dependency: enforces the shared secret when `MEDDIAG_API_KEY` is set; raises `401` otherwise. |

### `app/api/routes_ecg.py` — prefix `/api/v1/ecg`

| Function | Endpoint | Purpose |
|---|---|---|
| `analyze_ecg(payload: SignalAnalysisRequest) -> DiagnosisResponse` | `POST /analyze` | Analyses a raw ECG array supplied as JSON. |
| `analyze_ecg_upload(file, patient_id, sampling_rate, age, sex, referring_clinician, clinical_notes) -> DiagnosisResponse` | `POST /analyze/upload` | Parses an uploaded CSV/JSON/TXT/MAT ECG file and analyses it. |
| `ecg_classes() -> Dict[str, Any]` | `GET /classes` | Lists the ECG diagnostic classes with descriptions and severities. |

### `app/api/routes_eeg.py` — prefix `/api/v1/eeg`

| Function | Endpoint | Purpose |
|---|---|---|
| `analyze_eeg(payload: SignalAnalysisRequest) -> DiagnosisResponse` | `POST /analyze` | Analyses a raw EEG array supplied as JSON. |
| `analyze_eeg_upload(file, patient_id, sampling_rate, age, sex, referring_clinician, clinical_notes) -> DiagnosisResponse` | `POST /analyze/upload` | Parses an uploaded EEG file and analyses it. |
| `eeg_classes() -> Dict[str, Any]` | `GET /classes` | Lists the EEG diagnostic classes. |

### `app/api/routes_xray.py` — prefix `/api/v1/xray`

| Function | Endpoint | Purpose |
|---|---|---|
| `analyze_xray(file, patient_id, age, sex, view, referring_clinician, clinical_notes) -> DiagnosisResponse` | `POST /analyze` | Analyses an uploaded chest radiograph (PNG/JPEG/DICOM). |
| `xray_classes() -> Dict[str, Any]` | `GET /classes` | Lists the chest X-ray diagnostic classes. |

### `app/api/routes_reports.py` — prefix `/api/v1/reports`

| Function | Endpoint | Purpose |
|---|---|---|
| `create_report(analysis_id, format) -> ReportResponse` | `POST /{analysis_id}` | Renders and persists a PDF/HTML/JSON report for an analysis. |
| `download_report(report_id) -> FileResponse` | `GET /{report_id}/download` | Streams the stored report file back to the client. |
| `reports_for(analysis_id) -> Dict[str, Any]` | `GET /{analysis_id}` | Lists the reports generated for one analysis. |
| `list_all_reports(limit) -> Dict[str, Any]` | `GET /list` | Lists the most recent reports across all analyses. |

### `app/api/routes_history.py` — prefix `/api/v1/history`

| Function | Endpoint | Purpose |
|---|---|---|
| `history(modality, limit, offset) -> HistoryResponse` | `GET ""` | Returns recent analyses, newest first, with optional modality filter. |
| `summary() -> Dict[str, Any]` | `GET /summary` | Returns the label distribution used by the dashboard charts. |

### `app/api/routes_models.py` — prefix `/api/v1`

| Function | Endpoint | Purpose |
|---|---|---|
| `health() -> HealthResponse` | `GET /health` | Service status plus per-model readiness, source and metrics. |
| `list_models() -> Dict[str, Any]` | `GET /models` | Metadata for every registered model. |
| `describe_model(modality) -> ModelInfo` | `GET /models/{modality}` | Metadata for one model. |
| `stats() -> Dict[str, Any]` | `GET /stats` | Aggregate counters (analyses, reports, label counts, metrics). |

### `app/api/routes_health.py`

| Function | Endpoint | Purpose |
|---|---|---|
| `health() -> Dict[str, Any]` | `GET /health` | Lightweight liveness probe (no model loading). |
| `root() -> Dict[str, Any]` | `GET /` | Service banner with version and navigation links. |

### `app/api/routes_demo.py` — prefix `/api/v1/demo`

| Function | Endpoint | Purpose |
|---|---|---|
| `demo_ecg(label, seconds, seed) -> Dict[str, Any]` | `GET /ecg/{label}` | Returns a synthetic ECG sample for the requested class. |
| `demo_eeg(label, seconds, seed) -> Dict[str, Any]` | `GET /eeg/{label}` | Returns a synthetic EEG sample. |
| `demo_xray(label, seed) -> Dict[str, Any]` | `GET /xray/{label}` | Returns a synthetic chest radiograph (PNG bytes). |
| `demo_classes() -> Dict[str, Any]` | `GET /classes` | Lists every demo sample that can be fetched. |

---

## 4. Core layer

### `app/core/logging.py`

| Function | Purpose |
|---|---|
| `setup_logging(level: int | None = None)` | Configures root logging with a timestamped console format and a file handler. |
| `get_logger(name: str) -> logging.Logger` | Returns a namespaced logger. |

### `app/core/errors.py`

| Function / Class | Purpose |
|---|---|
| `class MedDiagError(Exception)` | Base error carrying a message and a details dict. |
| `MedDiagError.__init__(message, *, details=None)` | Stores the message and details. |
| `MedDiagError.to_payload() -> Dict[str, Any]` | Serialises the error into the API error envelope. |
| `class ValidationError(MedDiagError)` | Input validation failure → HTTP 422. |
| `class UnsupportedFormatError(MedDiagError)` | Unsupported file/format → HTTP 415. |
| `class ModelNotReadyError(MedDiagError)` | Model unavailable → HTTP 503. |
| `class NotFoundError(MedDiagError)` | Unknown resource → HTTP 404. |
| `register_exception_handlers(app: FastAPI)` | Installs handlers that map every error type onto its HTTP status and the standard JSON envelope. |

### `app/core/audit.py`

| Function | Purpose |
|---|---|
| `_utcnow() -> str` | Current UTC timestamp in ISO-8601 form. |
| `_connect() -> Iterator[sqlite3.Connection]` | Context-managed SQLite connection with WAL enabled and row access by name. |
| `init_db()` | Creates the audit database and its schema if they do not exist. |
| `_ensure_schema()` | Lazily creates the `analyses` / `reports` / `events` tables on first use. |
| `record_analysis(...) -> str` | Persists an analysis (id, modality, label, confidence, patient, timestamps) and returns its id. |
| `record_report(*, report_id, analysis_id, fmt, path, size_bytes)` | Persists a generated report record. |
| `log_event(action, *, analysis_id=None, actor="system", detail="") -> str` | Appends an audit event and returns its id. |
| `list_analyses(...) -> List[Dict[str, Any]]` | Returns analyses newest-first with optional modality/label filters and pagination. |
| `count_analyses(*, modality=None) -> int` | Counts stored analyses, optionally per modality. |
| `get_analysis(analysis_id) -> Optional[Dict[str, Any]]` | Fetches one analysis record. |
| `list_reports(analysis_id=None) -> List[Dict[str, Any]]` | Lists reports, optionally for one analysis. |
| `get_report(report_id) -> Optional[Dict[str, Any]]` | Fetches one report record. |

### `app/core/storage.py`

| Function | Purpose |
|---|---|
| `_ensure(path: Path) -> Path` | Creates the parent directory and returns the path. |
| `analysis_dir(analysis_id: str) -> Path` | Returns (and creates) the per-analysis storage directory. |
| `save_upload(data: bytes, *, analysis_id: str, filename: str) -> Path` | Stores an uploaded file inside the analysis directory. |
| `save_analysis_payload(analysis_id: str, payload: Dict[str, Any]) -> Path` | Writes the analysis payload as JSON. |
| `load_analysis_payload(analysis_id: str) -> Optional[Dict[str, Any]]` | Reads back a stored analysis payload. |
| `new_id(prefix: str = "") -> str` | Generates a short, collision-resistant identifier with an optional prefix. |
| `safe_join(base: Path, *parts: str) -> Path` | Joins path fragments while refusing to escape `base` (path-traversal guard). |
| `read_upload_limit() -> int` | Returns the maximum upload size in bytes from the settings. |

### `app/core/metrics.py`

| Function / Class | Purpose |
|---|---|
| `class MetricsRegistry` | In-process counters, latency histograms and error counters. |
| `MetricsRegistry.__init__()` | Initialises the counter store. |
| `MetricsRegistry.increment(name, value=1)` | Increments a named counter. |
| `MetricsRegistry.observe_latency(name, seconds)` | Records a timing sample and updates the rolling average. |
| `MetricsRegistry.record_error(name)` | Increments the error counter for an operation. |
| `MetricsRegistry.snapshot() -> Dict[str, object]` | Returns a JSON-serialisable snapshot of all metrics. |
| `MetricsRegistry.reset()` | Clears every counter. |
| `metrics` | Module-level registry instance. |

### `app/core/parsers.py`

| Function / Class | Purpose |
|---|---|
| `class ParsedSignal` | Result of parsing a signal file: `samples`, `sampling_rate`, `channel_names`, `source_format`. |
| `ParsedSignal.channels` | Number of channels present in the parsed signal. |
| `_to_array(values) -> np.ndarray` | Converts a nested list into a 2-D float array. |
| `_guess_sampling_rate(header, default) -> float` | Extracts a declared sampling rate from a file header, falling back to the default. |
| `parse_csv(data: bytes, default_rate: float) -> ParsedSignal` | Parses a CSV of samples, with optional header/metadata rows. |
| `parse_json(data: bytes, default_rate: float) -> ParsedSignal` | Parses a JSON document of samples or `{samples, sampling_rate}` objects. |
| `parse_txt(data: bytes, default_rate: float) -> ParsedSignal` | Parses whitespace-separated numeric text. |
| `parse_mat(data: bytes, default_rate: float) -> ParsedSignal` | Parses a MATLAB `.mat` file via SciPy. |
| `parse_signal_upload(filename: str, data: bytes, default_rate: float) -> ParsedSignal` | Dispatches to the right parser by file extension. |
| `_is_number(value: str) -> bool` | True when a token parses as a finite float. |

---

## 5. ML — shared plumbing

### `app/ml/base.py`

| Function / Class | Purpose |
|---|---|
| `resolve_device() -> str` | Picks the torch device from `MEDDIAG_DEVICE` (`auto` → CUDA when available). |
| `softmax(x, axis=-1) -> np.ndarray` | Numerically stable softmax. |
| `class ModalityModel(ABC)` | Base class shared by the three modality models. |
| `ModalityModel.__init__(checkpoint_name: str)` | Resolves the device, sets the checkpoint path and loads the checkpoint if present. |
| `ModalityModel.is_trained -> bool` | True when a real neural checkpoint backs the model. |
| `ModalityModel.source -> str` | `"trained"` or `"heuristic_baseline"`. |
| `ModalityModel._load_checkpoint()` | Loads a checkpoint from disk; falls back to the heuristic path on any failure. |
| `ModalityModel._build_and_load(state)` | Builds the concrete network and loads its weights (implemented per modality). |
| `ModalityModel.save_checkpoint(path=None, extra=None) -> Path` | Persists weights plus metadata for later inference. |
| `ModalityModel.predict_proba(samples, context=None) -> np.ndarray` | **Abstract.** Returns a probability vector over the class list. |
| `ModalityModel.predict(samples, context=None) -> Dict[str, float]` | Wraps `predict_proba` into a normalised label → probability mapping. |
| `ModalityModel.parameter_count -> int` | Number of trainable parameters (0 on the heuristic path). |
| `ModalityModel.info() -> Dict[str, Any]` | Metadata payload served by `/api/v1/models`. |
| `ModalityModel.write_metadata(path, metadata)` | Writes a JSON metadata sidecar next to a checkpoint. |

### `app/ml/registry.py`

| Function / Class | Purpose |
|---|---|
| `class ModelRegistry` | Lazy, thread-safe holder for the three modality models. |
| `ModelRegistry.__init__()` | Initialises the internal cache and lock. |
| `ModelRegistry.get(modality: Modality) -> ModalityModel` | Returns the model for a modality, constructing it on first use. |
| `ModelRegistry.load_all()` | Eagerly loads all three models (used at startup). |
| `ModelRegistry.info() -> Dict[str, dict]` | Metadata for every loaded model. |
| `ModelRegistry.available_modalities() -> List[str]` | Lists the modalities that are currently loaded. |
| `registry` | Module-level registry instance. |

### `app/ml/training.py`

| Function / Class | Purpose |
|---|---|
| `class TrainConfig` | Hyper-parameters: epochs, batch size, learning rate, weight decay, patience, validation ratio, **test ratio**, seed, device, class weighting, label smoothing. |
| `TrainConfig.resolve_device() -> str` | Resolves the training device, falling back to CPU with a warning. |
| `set_seed(seed: int)` | Seeds NumPy and PyTorch for reproducibility. Must be called **before** the network is constructed — the weight initialisation draws from the global RNG, so seeding afterwards would leave every run starting from different weights. |
| `stratified_split(labels, validation_ratio, seed) -> (train_idx, val_idx)` | Splits indices into train/validation while preserving the class balance. |
| `compute_metrics(y_true, y_pred, probabilities, n_classes) -> Dict[str, float]` | Accuracy, macro F1, per-class F1 and macro one-vs-rest ROC AUC. The macro F1 averages only over the classes that are **present** in the evaluation set, matching the benchmark convention. |
| `train_network(network, signals, features, labels, class_names, config, groups=None) -> (metrics, history)` | Shared training loop: AdamW, label smoothing, gradient clipping, `ReduceLROnPlateau`, early stopping, best-checkpoint tracking. When `groups` is given the split is grouped; when `config.test_ratio > 0` it is a three-way split and the untouched test partition is scored at the end, with the results returned under the `test_*` metric keys. |
| `evaluate(network, loader, device, n_classes) -> Dict[str, float]` | Evaluates a network on a data loader and returns the metric dict. |
| `save_checkpoint(network, path, class_names, architecture, metrics, metadata=None) -> Path` | Writes the `.pt` checkpoint plus a `*.metrics.json` sidecar. |

---

## 6. ML — ECG

### `app/ml/ecg/preprocess.py`

| Function / Class | Purpose |
|---|---|
| `class PreprocessedECG` | Dataclass carrying the conditioned signal, per-lead windows, R-peaks, quality metrics and detected unit. |
| `PreprocessedECG.primary() -> np.ndarray` | The conditioned single-channel signal. |
| `_as_float_array(samples) -> np.ndarray` | Coerces input into a float array. |
| `bandpass(samples, fs, low, high, order=3) -> np.ndarray` | Butterworth band-pass filter (0.5–40 Hz by default at call sites). |
| `remove_baseline_wander(samples, fs, cutoff=0.05) -> np.ndarray` | High-pass baseline-wander removal at the AHA-standard 0.05 Hz. |
| `notch_filter(samples, fs, freq=50.0, q=30.0) -> np.ndarray` | Removes mains interference. |
| `resample_signal(samples, orig_fs, target_fs) -> np.ndarray` | Polyphase resampling to the target rate. |
| `zscore(samples) -> np.ndarray` | Zero-mean, unit-variance normalisation. |
| `to_millivolts(samples, unit="auto") -> np.ndarray` | Converts raw/µV counts to millivolts, auto-detecting the unit. |
| `detect_r_peaks(signal, fs, min_distance_s=0.25) -> np.ndarray` | Pan-Tompkins detection with adaptive SPKI/NPKI thresholds and R-peak refinement. |
| `segment_beats(signal, r_peaks, fs, before=0.25, after=0.45) -> List[np.ndarray]` | Extracts individual beat windows around each R-peak. |
| `_quality_of(clean, raw) -> Dict[str, float]` | Signal-quality index from crest factor, SNR and the quietest-decile noise floor. |
| `preprocess_ecg(samples, sampling_rate, *, lead_names=None, unit="auto") -> PreprocessedECG` | **Main ECG conditioning pipeline**: unit conversion → filtering → resampling → R-peak detection → per-lead model windows → quality metrics. |
| `_select_window(signal, fs, length) -> int` | Picks the highest-energy window (used to avoid artefact-dominated segments). |
| `build_model_input(window) -> np.ndarray` | Normalises one window into the `(1, 2500)` network layout. |
| `summarize_signal(pre, max_points=1200) -> List[Tuple[float, float]]` | Down-samples the waveform into a JSON-friendly point list for the dashboard. |

### `app/ml/ecg/features.py`

| Function / Class | Purpose |
|---|---|
| `class FeatureValue` | One measured clinical quantity with its unit, reference band, status and explanation. |
| `FeatureValue.to_dict() -> Dict[str, object]` | Serialises the feature for the API. |
| `_status(value, low, high, tolerance=0.15) -> str` | Classifies a value as `normal` / `borderline` / `abnormal` against its band. |
| `_hrv_from_rr(rr_intervals, fs) -> Dict[str, float]` | Computes RMSSD, pNN50 and LF/HF from the RR series. |
| `_median_beat_template(signal, r_peaks, fs) -> np.ndarray` | Builds the median beat used for interval and ST measurement. |
| `_envelope(signal, fs, window_s=0.040) -> np.ndarray` | Smoothed amplitude envelope of the signal. |
| `_envelope_width_ms(onset, offset, fs) -> float` | Width in milliseconds between two envelope crossings. |
| `_qrs_bounds(signal, r_peak, fs) -> (onset, offset)` | Locates the QRS onset and offset around an R-peak. |
| `_st_index_placeholder(size, j_point, fs) -> int` | Maps a J-point to the ST measurement index. |
| `_j_point(template, r_index, fs, threshold_ratio=0.15) -> int` | Locates the J point on the median beat template. |
| `_beat_windows(signal, r_peaks, fs) -> Dict[str, float]` | Measures QRS duration, QT, PR, ST deviation, T and R amplitudes, and the P wave. |
| `_qrs_widths(signal, r_peaks, fs) -> List[float]` | Per-beat QRS width estimates. |
| `_qrs_widths_robust(signal, r_peaks, fs) -> List[float]` | Biased-compensated QRS widths from the local-max envelope. |
| `_ectopic_beats(signal, r_peaks, fs) -> float` | Fraction of premature, wide, morphologically atypical (PVC) beats. |
| `extract_features(pre: PreprocessedECG) -> Dict[str, FeatureValue]` | Computes all 18 ECG features with reference bands and explanations. |
| `feature_vector(features) -> np.ndarray` | Packs the feature dict into the fixed, scale-normalised numeric vector for the model. |
| `feature_importance_hint(features) -> List[Tuple[str, float, str]]` | Ranks features by deviation from their reference band (heuristic explainability prior). |

### `app/ml/ecg/model.py`

| Function / Class | Purpose |
|---|---|
| `ECG_CLASSES` | `["NORM", "AFIB", "MI", "STDE", "PVC"]`. |
| `CLASS_DESCRIPTIONS` | Clinical description per ECG class. |
| `CLASS_SEVERITY` | Severity label per ECG class (`normal` / `mild` / `moderate` / `severe` / `critical`). |
| `class ResNet1DBlock(nn.Module)` | Pre-activation residual block (odd kernel, length-exact identity path). |
| `ResNet1DBlock.forward(x)` | Conv-BN-ReLU → Conv-BN, plus the identity branch, then ReLU. |
| `class ECGNet(nn.Module)` | ResNet1D backbone with avg+max pooling and a clinical-feature fusion head. |
| `ECGNet.__init__(num_classes=5, n_features=18, dropout=0.3)` | Builds the stem, three residual stages and the fusion head. |
| `ECGNet.forward(signal, features)` | Fuses the pooled CNN activations with the projected clinical features and returns class logits. |

### `app/ml/ecg/inference.py`

| Function / Class | Purpose |
|---|---|
| `class ECGAnalysis` | Result of an ECG analysis: label, probabilities, features, class relevance, quality, warnings and recommendations. |
| `class ECGModel(ModalityModel)` | ECG model implementation. |
| `ECGModel.__init__()` | Registers the ECG class list and the `ecg_model.pt` checkpoint name. |
| `ECGModel._build_and_load(state)` | Builds `ECGNet`, loads the weights and switches the network to eval mode. |
| `ECGModel.predict_proba(samples, context=None) -> np.ndarray` | Preprocesses, extracts features and returns the class-probability vector. |
| `ECGModel._network_proba(pre, features) -> np.ndarray` | Runs the trained network. |
| `ECGModel._pick_analysis_lead(pre) -> int` | Chooses the most diagnostic lead of a multi-lead recording. |
| `ECGModel._heuristic_proba(features, sqi) -> np.ndarray` | Deterministic clinical-rule scorer used when no checkpoint exists. |
| `ECGModel.analyze(samples, sampling_rate, lead_names=None, unit="auto") -> ECGAnalysis` | **Main entry point**: full analysis returning the label, findings and recommendations. |
| `ECGModel._class_relevance(features) -> Dict[str, float]` | Per-class relevance scores used for ranking and explanation. |
| `ECGModel._recommendations(label, features, relevance) -> List[str]` | Generates the clinical recommendation list for the predicted label. |

---

## 7. ML — EEG

### `app/ml/eeg/preprocess.py`

| Function / Class | Purpose |
|---|---|
| `class PreprocessedEEG` | Dataclass: conditioned signal, model window, band powers, artefact ratio, quality metrics. |
| `PreprocessedEEG.primary() -> np.ndarray` | The conditioned single-channel signal. |
| `_as_float_array(samples) -> np.ndarray` | Coerces input into a float array. |
| `bandpass(samples, fs, low, high, order=4) -> np.ndarray` | Butterworth band-pass filter. |
| `notch_filter(samples, fs, freq=50.0, q=30.0) -> np.ndarray` | Removes mains interference. |
| `resample_signal(samples, orig_fs, target_fs) -> np.ndarray` | Resamples to the target rate. |
| `average_reference(signal) -> np.ndarray` | Average reference across channels. |
| `detect_artifacts(signal, fs) -> float` | Moving-RMS outlier comparison returning the artefact ratio. |
| `signal_quality(signal, fs, artifact_ratio) -> Dict[str, float]` | Signal-quality index from crest factor, SNR and artefact ratio. |
| `preprocess_eeg(samples, sampling_rate, *, channel_names=None, unit="auto") -> PreprocessedEEG` | **Main EEG conditioning pipeline**: filtering → resampling → artefact detection → window selection → quality metrics. |
| `_select_window(signal, fs, length) -> int` | Picks the cleanest (lowest-outlier) window. |
| `build_model_input(window) -> np.ndarray` | Normalises one window into the `(1, 1, 2560)` network layout. |

### `app/ml/eeg/features.py`

| Function / Class | Purpose |
|---|---|
| `class FeatureValue` | One measured EEG quantity with unit, reference band, status and explanation. |
| `FeatureValue.to_dict() -> Dict[str, object]` | Serialises the feature for the API. |
| `_status(value, low, high, tolerance=0.15) -> str` | Classifies a value against its reference band. |
| `band_powers(signal, fs) -> Dict[str, float]` | Relative powers in the delta/theta/alpha/beta/gamma bands. |
| `spectral_entropy(powers) -> float` | Normalised spectral entropy of the band-power distribution. |
| `hjorth_parameters(signal) -> (activity, mobility, complexity)` | Hjorth activity/mobility/complexity descriptors. |
| `spike_wave_index(signal, fs) -> float` | Index of spike-and-wave discharges (epileptiform activity). |
| `alpha_asymmetry(signal, fs) -> float` | Asymmetry of left/right alpha power. |
| `extract_features(pre: PreprocessedEEG) -> Dict[str, FeatureValue]` | Computes all 16 EEG features with reference bands and explanations. |
| `feature_vector(features) -> np.ndarray` | Packs the feature dict into the fixed, scale-normalised numeric vector. |
| `feature_importance_hint(features) -> List[Tuple[str, float, str]]` | Ranks features by deviation from their reference band. |

### `app/ml/eeg/model.py`

| Function / Class | Purpose |
|---|---|
| `EEG_CLASSES` | `["NORMAL", "EPILEPSY", "PARKINSON", "DEPRESSION"]`. |
| `CLASS_DESCRIPTIONS` | Clinical description per EEG class. |
| `CLASS_SEVERITY` | Severity label per EEG class. |
| `class EEGNet(nn.Module)` | Depthwise-separable EEGNet core with global temporal pooling and a fusion head. |
| `EEGNet.__init__(num_classes=4, n_features=16, dropout=0.25)` | Builds the three convolutional blocks, the pooling layer and the fusion head. |
| `EEGNet.forward(signal, features)` | Fuses the pooled convolutions with the projected clinical features and returns class logits. |

### `app/ml/eeg/inference.py`

| Function / Class | Purpose |
|---|---|
| `class EEGAnalysis` | Result of an EEG analysis: label, probabilities, features, class relevance, quality, warnings and recommendations. |
| `class EEGModel(ModalityModel)` | EEG model implementation. |
| `EEGModel.__init__()` | Registers the EEG class list and the `eeg_model.pt` checkpoint name. |
| `EEGModel._build_and_load(state)` | Builds `EEGNet`, loads the weights and switches to eval mode. |
| `EEGModel.predict_proba(samples, context=None) -> np.ndarray` | Preprocesses, extracts features and returns the probability vector. |
| `EEGModel._network_proba(pre, features) -> np.ndarray` | Runs the trained network. |
| `EEGModel._heuristic_proba(features, sqi) -> np.ndarray` | Deterministic clinical-rule scorer used when no checkpoint exists. |
| `EEGModel.analyze(samples, sampling_rate, channel_names=None, unit="auto") -> EEGAnalysis` | **Main entry point**: full analysis with findings and recommendations. |
| `EEGModel._class_relevance(features) -> Dict[str, float]` | Per-class relevance scores. |
| `EEGModel._recommendations(label, features, relevance) -> List[str]` | Generates the clinical recommendation list. |

---

## 8. ML — X-ray

### `app/ml/xray/preprocess.py`

| Function / Class | Purpose |
|---|---|
| `class PreprocessedXray` | Dataclass: enhanced greyscale array, original size, quality metrics, chest-X-ray plausibility flag, source format. |
| `PreprocessedXray.height -> int` | Height of the original image. |
| `PreprocessedXray.width -> int` | Width of the original image. |
| `decode_image(data: bytes) -> PIL.Image.Image` | Decodes PNG/JPEG/DICOM bytes into a Pillow image. |
| `clahe(image, clip_limit=2.0, tile_grid=(8, 8)) -> np.ndarray` | Contrast-limited adaptive histogram equalisation. |
| `to_grayscale(image) -> np.ndarray` | Converts any image mode to 8-bit greyscale. |
| `resize_image(gray, size=224) -> np.ndarray` | Resizes to the model input size. |
| `normalize(gray) -> np.ndarray` | Scales to `[0, 1]`. |
| `looks_like_chest_xray(gray) -> bool` | Heuristic plausibility check (lung-field geometry). |
| `image_quality(gray) -> Dict[str, float]` | Sharpness, contrast and brightness quality metrics. |
| `preprocess_xray(data: bytes, *, size=224) -> PreprocessedXray` | **Main X-ray pipeline** for uploaded bytes: decode → CLAHE → resize → quality metrics. |
| `preprocess_xray_array(gray, *, size=224) -> PreprocessedXray` | Same pipeline for an in-memory array (training / tests). |
| `build_model_input(gray) -> np.ndarray` | Normalises a greyscale view into the `(1, 224, 224)` network layout. |

### `app/ml/xray/features.py`

| Function / Class | Purpose |
|---|---|
| `class FeatureValue` | One measured radiograph quantity with unit, reference band, status and explanation. |
| `FeatureValue.to_dict() -> Dict[str, object]` | Serialises the feature for the API. |
| `_status(value, low, high, tolerance=0.15) -> str` | Classifies a value against its reference band. |
| `_normalised_grid(height, width) -> (nx, ny)` | Normalised coordinate grids in `[-1, 1]`. |
| `_lung_mask(nx, ny) -> np.ndarray` | Boolean mask of the two lung fields. |
| `_region_mean(gray, mask) -> float` | Mean intensity inside a mask. |
| `extract_features(pre: PreprocessedXray) -> Dict[str, FeatureValue]` | Computes all 11 X-ray features from lung-relative zone statistics (apical / perihilar / peripheral / costophrenic). |
| `feature_vector(features) -> np.ndarray` | Packs the feature dict into the fixed, scale-normalised numeric vector. |
| `feature_importance_hint(features) -> List[Tuple[str, float, str]]` | Ranks features by deviation from their reference band. |

### `app/ml/xray/model.py`

| Function / Class | Purpose |
|---|---|
| `XRAY_CLASSES` | `["NORMAL", "PNEUMONIA", "CONSOLIDATION", "EFFUSION", "TUBERCULOSIS", "COVID19"]`. |
| `CLASS_DESCRIPTIONS` | Clinical description per X-ray class. |
| `CLASS_SEVERITY` | Severity label per X-ray class. |
| `class ConvBlock(nn.Module)` | Conv → BatchNorm → ReLU → MaxPool block. |
| `ConvBlock.forward(x)` | Applies the convolution block. |
| `class XrayCNN(nn.Module)` | Four-stage 2-D CNN with global average pooling and a fusion head. |
| `XrayCNN.__init__(num_classes=6, n_features=11, dropout=0.3)` | Builds the four stages, the pooling layer and the fusion head. |
| `XrayCNN.forward(image, features)` | Fuses the pooled convolutions with the projected clinical features and returns class logits. |

### `app/ml/xray/inference.py`

| Function / Class | Purpose |
|---|---|
| `class XrayAnalysis` | Result of an X-ray analysis: label, probabilities, features, class relevance, quality, warnings and recommendations. |
| `class XrayModel(ModalityModel)` | X-ray model implementation. |
| `XrayModel.__init__()` | Registers the X-ray class list and the `xray_model.pt` checkpoint name. |
| `XrayModel._build_and_load(state)` | Builds `XrayCNN`, loads the weights and switches to eval mode. |
| `XrayModel.predict_proba(samples, context=None) -> np.ndarray` | Preprocesses, extracts features and returns the probability vector. |
| `XrayModel._network_proba(pre, features) -> np.ndarray` | Runs the trained network. |
| `XrayModel._heuristic_proba(features, sqi) -> np.ndarray` | Deterministic zone-statistics scorer used when no checkpoint exists. |
| `XrayModel.analyze(data: bytes) -> XrayAnalysis` | **Main entry point**: full analysis of an uploaded radiograph. |
| `XrayModel._class_relevance(features) -> Dict[str, float]` | Per-class relevance scores. |
| `XrayModel._warnings(pre, features) -> List[str]` | Quality and plausibility warnings (e.g. low quality, not a chest X-ray). |
| `XrayModel._recommendations(label, features, relevance) -> List[str]` | Generates the clinical recommendation list. |

---

## 9. ML — explainable AI

### `app/ml/xai/gradcam.py`

| Function | Purpose |
|---|---|
| `_spatial_dims(activation) -> Tuple[int, ...]` | Indices of the spatial axes of an activation tensor. |
| `grad_cam(network, input_tensor, feature_tensor, target_class, target_layer) -> np.ndarray` | **Core Grad-CAM**: captures the target layer's activation and gradient, weights the channels, upsamples to the input size and normalises to `[0, 1]`. |
| `grad_cam_for_ecg(network, window, features, target_class, device) -> np.ndarray` | Grad-CAM1D over the ECG ResNet's final residual block. |
| `grad_cam_for_eeg(network, window, features, target_class, device) -> np.ndarray` | Grad-CAM over the EEGNet's last convolutional block. |
| `grad_cam_for_xray(network, image, features, target_class, device) -> np.ndarray` | Grad-CAM over the X-ray CNN's final block. |
| `integrated_gradients(network, input_tensor, feature_tensor, target_class, baseline=None, steps=32) -> np.ndarray` | Integrated-gradients attribution map over the model input. |
| `occlusion_sensitivity(score_fn, signal, window_size, stride) -> (sensitivity, segments)` | Occlusion-based segment importance with the ranked segments returned. |
| `feature_attributions(network, input_tensor, feature_tensor, target_class, baseline_features=None) -> np.ndarray` | Per-clinical-feature attribution: how much the target-class score drops when each feature is reset to its baseline. |

### `app/ml/xai/plots.py`

| Function | Purpose |
|---|---|
| `_to_base64(buffer: io.BytesIO) -> str` | Encodes a PNG buffer as a base64 string. |
| `plot_signal(signal, sampling_rate, r_peaks=None, title="", color="#4f7cff") -> str` | Renders the conditioned waveform with optional R-peak markers; returns base64 PNG. |
| `plot_saliency(signal, saliency, sampling_rate, title="") -> str` | Renders the waveform with the Grad-CAM heat map underneath. |
| `plot_overlay(grayscale, cam, title="Grad-CAM localisation") -> str` | Renders the radiograph with the Grad-CAM heat map overlaid. |
| `plot_image(grayscale, title="Radiograph") -> str` | Renders the plain radiograph. |
| `plot_class_probabilities(probabilities, title="Class probabilities") -> str` | Renders the class-probability bar chart. |
| `plot_feature_contributions(contributions) -> str` | Renders the feature-contribution bar chart. |
| `plot_band_spectrum(frequencies, powers, title="Power spectral density") -> str` | Renders the EEG band-power spectrum. |

---

## 10. Data layer

### `app/data/synthetic.py`

| Function | Purpose |
|---|---|
| `_ecg_beat(label, fs, rng, ...)` | Builds a single P-QRS-T beat with per-condition PR/QT/ST/P-wave morphology. |
| `_ecg_intervals(label, rng, count) -> np.ndarray` | Generates the RR-interval series for a class (irregular for AFIB, etc.). |
| `synth_ecg(label, duration_s=10.0, seed=0, noise_scale=1.0, amplitude_scale=1.0) -> (signal, fs)` | Generates a physiologically plausible ECG recording for one class. |
| `_pink_noise(n, rng) -> np.ndarray` | 1/f noise generator used for realistic backgrounds. |
| `synth_eeg(label, duration_s=10.0, seed=0, noise_scale=1.0) -> (signal, fs)` | Generates an EEG recording with the spectral signature of a class. |
| `synth_xray(label, size=224, seed=0, noise_scale=1.0) -> np.ndarray` | Renders a synthetic chest radiograph with class-specific opacity patterns. |
| `build_ecg_corpus(per_class=160, *, duration_s=10.0, seed=2024) -> List[(signal, label, name)]` | Builds the full ECG training corpus with per-sample augmentation. |
| `build_eeg_corpus(per_class=160, *, duration_s=10.0, seed=2025) -> List[(signal, label, name)]` | Builds the full EEG training corpus. |
| `build_xray_corpus(per_class=140, *, size=224, seed=2026) -> List[(image, label, name)]` | Builds the full X-ray training corpus with translation jitter. |
| `write_demo_dataset(output_dir=None, *, per_class=3, ...) -> Dict[str, Any]` | Writes demo CSV/PNG files to disk for offline testing. |
| `_write_csv(path, signal, rate)` | Writes a signal as CSV with a metadata header. |
| `_write_png(path, image)` | Writes a greyscale image as PNG. |

### `app/data/splitting.py`

| Function | Purpose |
|---|---|
| `grouped_split_indices(labels, groups, validation_ratio, seed) -> (train_idx, val_idx)` | Leakage-safe split: groups are assigned to the validation side greedily so the class balance stays close to the target and no group is ever split across the boundary. |
| `grouped_split_indices_3way(labels, groups, validation_ratio, test_ratio, seed) -> (train_idx, val_idx, test_idx)` | Three-way grouped split. Carves out a **test** partition that influences neither the weights nor the early-stopping decision, so the reported number is a genuine held-out result. Falls back to the two-way split when `test_ratio <= 0`. |
| `_normalise_groups(groups) -> np.ndarray` | Returns the group ids as a string array so integer and string ids compare safely. |
| `_group_profile(labels, groups) -> (unique_groups, group_class_counts, group_sizes)` | Per-group class histogram used by the greedy split assignment. |
| `class_distribution(labels, class_names) -> Dict[str, int]` | Counts the samples per class name. |

### `app/data/ecg_datasets.py`

| Function / Class | Purpose |
|---|---|
| `class ECGDataset` | Dataset container: `signals`, `labels`, `label_names`, `records`, `metadata`. |
| `_resize_window(signal, n_timesteps) -> np.ndarray` | Truncates/pads a signal to the fixed model length. |
| `load_ptbxl(root, *, limit=None) -> ECGDataset` | Loads PTB-XL via `wfdb` and maps its diagnostic statements onto the ECG classes. |
| `load_mitbih(root, *, records=MITBIH_DEFAULT_RECORDS, n_timesteps=2500, beats_per_class=20) -> ECGDataset` | Loads the MIT-BIH Arrhythmia Database via `wfdb`; builds 5-second beat-centred windows, applies record-level AFIB labelling and spreads the per-class selection evenly across each record. |
| `load_folder_dataset(root) -> ECGDataset` | Loads a `<root>/<LABEL>/<file>` corpus. |
| `write_dataset_manifest(dataset, path) -> None` | Writes a JSON manifest describing the loaded dataset. |

### `app/data/eeg_datasets.py`

| Function / Class | Purpose |
|---|---|
| `class EEGDataset` | Dataset container: `signals`, `labels`, `label_names`, `records`, `metadata`. |
| `_resize(signal, n_timesteps) -> np.ndarray` | Truncates/pads a signal to the fixed model length. |
| `_read_signal(path) -> Optional[np.ndarray]` | Reads a `.edf`/`.fif`/`.txt`/`.csv` EEG file. |
| `_label_for(path) -> Optional[str]` | Derives the class label from the file or folder name. |
| `load_bonn(root, *, n_timesteps=2560, sampling_rate=128.0) -> EEGDataset` | Loads the Bonn EEG Database and maps its sets onto `NORMAL` / `EPILEPSY`; records the subject of every segment as the grouping key. |
| `load_mne_sample(*, n_timesteps=2560, n_windows=40) -> EEGDataset` | Loads the MNE sample dataset and slices it into labelled windows. |
| `load_folder_dataset(root, *, n_timesteps=2560, suffix=".csv") -> EEGDataset` | Loads a `<root>/<LABEL>/<file>` corpus. |

### `app/data/xray_datasets.py`

| Function / Class | Purpose |
|---|---|
| `class XrayDataset` | Dataset container: `images`, `labels`, `label_names`, `records`, `metadata`. |
| `_normalise_label(folder_name) -> Optional[str]` | Maps a folder name onto the X-ray class list. |
| `_load_image(path, size=224) -> Optional[np.ndarray]` | Loads and resizes an image file to a greyscale array. |
| `load_nih_chestxray14(root, *, limit=None) -> XrayDataset` | Loads NIH ChestX-ray14 and maps its finding labels onto the X-ray classes. |
| `load_folder_dataset(root) -> XrayDataset` | Loads a `<root>/<LABEL>/<file>` corpus. |

---

## 11. Services layer

### `app/services/diagnosis.py`

| Function / Class | Purpose |
|---|---|
| `class DiagnosisService` | Orchestrates preprocessing → inference → findings → explainability → persistence. |
| `DiagnosisService.analyze_ecg(samples, sampling_rate, patient, lead_names=None, unit="auto") -> DiagnosisResponse` | **ECG workflow.** |
| `DiagnosisService.analyze_eeg(samples, sampling_rate, patient, channel_names=None, unit="auto") -> DiagnosisResponse` | **EEG workflow.** |
| `DiagnosisService.analyze_xray(data: bytes, patient: PatientInfo) -> DiagnosisResponse` | **X-ray workflow.** |
| `DiagnosisService._finalize(...) -> DiagnosisResponse` | Builds the `DiagnosisResponse`, persists the payload, writes the audit record and returns the result. |
| `DiagnosisService._signal_explainability(...) -> Explainability` | ECG/EEG attribution: Grad-CAM saliency, network feature attribution, waveform plot and summary. |
| `DiagnosisService._xray_explainability(model, analysis) -> Explainability` | X-ray attribution: Grad-CAM overlay, network feature attribution and probability chart. |
| `DiagnosisService.list_analyses(**kwargs) -> List[Dict[str, Any]]` | Lists stored analyses from the audit trail. |
| `_rank_features(features) -> List[Tuple[str, float]]` | Ranks features by how far they deviate from their reference band. |
| `_feature_order_for(modality) -> List[str]` | Returns the ordered clinical feature names for a modality. |
| `_features_from_attributions(attributions, feature_names) -> List[FeatureContribution]` | Converts raw network attributions into ranked, human-readable explanations. |
| `_explain_summary(modality, analysis, top_features) -> str` | Builds the plain-language explanation sentence. |
| `_ecg_feature_vector(features) -> np.ndarray` | Packs ECG features into the model vector. |
| `_eeg_feature_vector(features) -> np.ndarray` | Packs EEG features into the model vector. |
| `_severity_for(label, modality) -> str` | Maps a predicted label onto its severity level. |
| `diagnosis_service` | Module-level service instance. |

### `app/services/report.py`

| Function | Purpose |
|---|---|
| `_decode_png(data: Optional[str]) -> Optional[bytes]` | Decodes a base64 PNG payload back into bytes. |
| `_format_probability(value: float) -> str` | Formats a probability as a percentage string. |
| `_load_payload(analysis_id) -> Dict[str, Any]` | Loads the stored analysis payload for report rendering. |
| `_report_header_rows(payload) -> List[Tuple[str, str]]` | Builds the patient/metadata key-value rows used in every report. |
| `render_pdf(payload) -> bytes` | Renders the analysis as a formatted PDF (ReportLab) with tables, images and the disclaimer. |
| `render_html(payload) -> str` | Renders the analysis as a standalone HTML document. |
| `generate_report(analysis_id, fmt="pdf") -> Dict[str, Any]` | **Service entry point**: renders, persists and records the report; returns its metadata and download URL. |
| `reports_for_analysis(analysis_id) -> List[Dict[str, Any]]` | Lists the reports previously generated for an analysis. |

---

## 12. Schemas

### `app/schemas.py`

| Class / Enum | Purpose |
|---|---|
| `Severity` | Enum: `normal`, `mild`, `moderate`, `severe`, `critical`. |
| `FindingStatus` | Enum: `normal`, `borderline`, `abnormal`. |
| `Sex` | Enum: `male`, `female`, `other`, `unknown`. |
| `class PatientInfo` | Patient demographics and referring-clinician metadata. |
| `class SignalAnalysisRequest` | Request body for ECG/EEG analysis: samples, sampling rate, patient, lead/channel names. |
| `class ImageAnalysisRequest` | Request body for X-ray analysis metadata (file supplied separately). |
| `class ClassScore` | One class with its probability and description. |
| `class Prediction` | The predicted label, confidence, description and per-class scores. |
| `class ClinicalFinding` | One clinical finding with value, unit, reference range, status and explanation. |
| `class FeatureContribution` | How much one feature moved the prediction (`increases_risk` / `decreases_risk` / `neutral`). |
| `class Explainability` | Methods, summary, top features, class relevance and the base64 visualisations. |
| `class QualityMetrics` | Signal/image quality metrics returned with every analysis. |
| `class DiagnosisResponse` | The full analysis response: prediction, findings, quality, explainability, recommendations. |
| `class ReportResponse` | Report metadata plus the download URL. |
| `class ModelInfo` | Model metadata: architecture, classes, source, metrics, parameter count. |
| `class HealthResponse` | Service and model health status. |
| `class AnalysisSummary` | One row of the history list. |
| `class HistoryResponse` | Paginated analysis history. |

---

## 13. Dashboard

### `app/web/static/`

| Asset | Purpose |
|---|---|
| `index.html` | The single-page dashboard shell with the five tabs (ECG / EEG / X-ray / History / Reports / About). |
| `css/app.css` | Full stylesheet — layout, cards, probability bars, findings tables, responsive breakpoints. |
| `js/app.js` | Vanilla-JS application: API calls, tab switching, file upload, result rendering, history and report management. |
| `favicon.svg` | The application icon. |

**No external dependencies** — charts are rendered as hand-rolled SVG, so the dashboard
works fully offline.

---

## 14. Tests

### `tests/conftest.py`

| Fixture | Purpose |
|---|---|
| `_prepare_runtime` (session, autouse) | Creates the runtime directories and the audit schema once per session. |
| `client` (session) | A FastAPI `TestClient` bound to the application. |
| `ecg_signal` (session) | A synthetic 10-second NORM ECG recording. |
| `eeg_signal` (session) | A synthetic 10-second NORMAL EEG recording. |
| `xray_bytes` (session) | A synthetic normal chest radiograph encoded as PNG. |

### `tests/test_api.py`

| Test | Purpose |
|---|---|
| `test_liveness_endpoint(client)` | The root liveness probe responds. |
| `test_health_reports_every_modality(client)` | `/api/v1/health` reports all three modalities. |
| `test_model_registry(client)` | The model registry lists architecture, classes and source. |
| `test_ecg_analysis(client, ecg_signal)` | An ECG request returns a complete `DiagnosisResponse`. |
| `test_eeg_analysis(client, eeg_signal)` | An EEG request returns a complete `DiagnosisResponse`. |
| `test_ecg_upload(client, ecg_signal)` | An uploaded ECG file is parsed and analysed. |
| `test_xray_analysis(client, xray_bytes)` | An uploaded radiograph returns a complete `DiagnosisResponse`. |
| `test_xray_rejects_non_image(client)` | A non-image upload is rejected with a clear error. |
| `test_rejects_too_few_samples(client)` | A too-short signal is rejected with HTTP 422. |
| `test_rejects_missing_patient(client, ecg_signal)` | A request without patient data is rejected. |
| `test_report_generation_and_download(client, ecg_signal)` | A PDF report is created and downloads as PDF bytes. |
| `test_html_report(client, ecg_signal)` | An HTML report renders as a standalone document. |
| `test_report_for_unknown_analysis(client)` | An unknown analysis id returns HTTP 404. |
| `test_history_contains_the_analyses(client)` | The history route returns the recorded analyses. |
| `test_history_summary(client)` | The summary route returns the label distribution. |
| `test_stats_endpoint(client)` | The stats route returns the aggregate counters. |
| `test_demo_samples(client)` | The demo routes serve synthetic samples for every class. |
| `test_dashboard_is_served(client)` | `/dashboard` serves the SPA shell. |
| `test_static_assets_are_served(client)` | The CSS and JS assets are served from `/static`. |

### `tests/test_features.py`

| Test | Purpose |
|---|---|
| `test_bandpass_removes_out_of_band_energy` | The band-pass filter attenuates out-of-band content. |
| `test_notch_filter_leaves_signal_intact` | The notch filter removes mains interference without destroying the signal. |
| `test_resample_changes_the_length` | Resampling produces the expected sample count. |
| `test_zscore_is_standardised` | z-score normalisation yields zero mean and unit variance. |
| `test_r_peak_detection_finds_every_beat` | R-peak detection finds every synthetic beat. |
| `test_ecg_preprocessing_shapes_and_quality` | The ECG pipeline returns the documented shapes and quality metrics. |
| `test_ecg_features_separate_the_classes` | ECG features separate the synthetic classes. |
| `test_ecg_features_have_a_complete_reference_table` | Every ECG feature carries a reference band and explanation. |
| `test_eeg_preprocessing_shapes` | The EEG pipeline returns the documented shapes. |
| `test_eeg_band_powers_separate_the_classes` | EEG band powers separate the synthetic classes. |
| `test_eeg_artifact_ratio_is_low_for_clean_data` | Clean EEG is not flagged as artefact. |
| `test_clahe_preserves_dimensions` | CLAHE preserves the image dimensions. |
| `test_xray_preprocessing_and_features` | The X-ray pipeline returns the documented shapes and features. |
| `test_xray_regional_features_detect_the_lesion` | The regional features detect the synthetic lesion. |

### `tests/test_xai.py`

| Test | Purpose |
|---|---|
| `test_plot_signal_returns_base64_png` | The plot helpers return valid base64 PNGs. |
| `test_plot_saliency_handles_mismatched_lengths` | Saliency rendering tolerates length mismatches. |
| `test_plot_overlay_and_probabilities` | Overlay and probability plots render. |
| `test_plot_feature_contributions` | The feature-contribution plot renders. |
| `test_grad_cam_runs_on_the_ecg_network` | Grad-CAM1D returns a normalised `(2500,)` map for the ECG network. |
| `test_grad_cam_runs_on_the_xray_network` | Grad-CAM returns a normalised `(224, 224)` map for the X-ray network. |
| `test_integrated_gradients_returns_an_attribution_map` | Integrated gradients returns an attribution map matching the input shape. |
| `test_feature_attributions_rank_the_inputs` | Feature attributions return one value per clinical feature. |
| `test_occlusion_sensitivity_returns_segments` | Occlusion sensitivity returns a per-sample curve and ranked segments. |
| `test_download_url_is_a_complete_path` | The report `download_url` already contains the API prefix and resolves as-is. |
| `test_dashboard_does_not_double_prefix_the_download_url` | The dashboard never prefixes `download_url` with the API base again (which would 404). |

### `tests/test_splitting.py`

| Test | Purpose |
|---|---|
| `test_two_way_split_is_disjoint_and_covers_everything` | The two-way grouped split produces disjoint index sets that cover every sample. |
| `test_two_way_split_never_splits_a_group` | No group appears on both sides of the two-way split. |
| `test_three_way_split_partitions_are_disjoint_and_complete` | The train/validation/test partitions are pairwise disjoint and cover every sample. |
| `test_three_way_split_never_splits_a_group` | No group appears in more than one of the three partitions. |
| `test_three_way_split_is_deterministic_for_a_fixed_seed` | The same seed reproduces the same split. |
| `test_three_way_split_respects_the_sample_budgets` | The validation and test partitions stay within their sample budgets. |
| `test_three_way_split_accepts_integer_group_ids` | Integer group ids produce the same partition as their string equivalents. |
| `test_three_way_split_falls_back_to_two_way_when_test_ratio_is_zero` | A zero test ratio reproduces the two-way split exactly. |
| `test_three_way_split_rejects_ratios_that_do_not_fit` | `validation_ratio + test_ratio >= 1` raises `ValueError`. |
| `test_class_distribution_counts_every_class` | The per-class histogram counts every class, including absent ones. |
| `test_macro_f1_only_averages_over_present_classes` | The macro F1 ignores classes that are absent from the evaluation set. |
| `test_network_initialisation_is_reproducible_after_set_seed` | Two networks built after `set_seed(42)` have identical weights. |

---

## Quick function index

Counts are generated from the source with an AST pass (public names only — no
leading underscore), covering module-level functions and public class methods.

| Layer | Module(s) | Public functions | Public classes |
|---|---|---|---|
| Config | `app/config.py` | 2 | 1 |
| Schemas | `app/schemas.py` | 0 | 18 |
| API | `app/api/*.py` | 25 | 0 |
| Core | `app/core/*.py` | 31 | 7 |
| ML shared | `app/ml/base.py`, `registry.py`, `training.py` | 21 | 3 |
| ECG | `app/ml/ecg/*.py` | 18 | 4 |
| EEG | `app/ml/eeg/*.py` | 20 | 4 |
| X-ray | `app/ml/xray/*.py` | 18 | 4 |
| XAI | `app/ml/xai/*.py` | 14 | 0 |
| Data | `app/data/*.py` | 22 | 3 |
| Services | `app/services/*.py` | 8 | 1 |
| **Total** | | **179** | **45** |

Plus 10 CLI entry points in `scripts/` (`train_ecg`, `train_eeg`, `train_xray`,
`run_server`, `make_demo_data`, `download_datasets`, `smoke_check`,
`evaluate_models`, `run_benchmark`, `verify_live`).

---

*Generated for the MedDiag project — AI-Based Medical Diagnosis Assistant (ECG / EEG / Chest X-ray).*
