# MedDiag — Testing Protocol, Parameters and Results

Reference document for evaluating the **AI-Based Medical Diagnosis Assistant
(ECG / EEG / Chest X-ray)** and for producing the numbers reported in a research
paper.

Everything below is reproducible from this repository with three commands; no
manual step is required.

---

## Table of contents

1. [Scope](#1-scope)
2. [Quick reproduction](#2-quick-reproduction)
3. [Test parameters](#3-test-parameters)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [Results](#5-results)
6. [Ablations and domain transfer](#6-ablations-and-domain-transfer)
7. [Automated test suite](#7-automated-test-suite)
8. [API-level test parameters](#8-api-level-test-parameters)
9. [Service configuration parameters](#9-service-configuration-parameters)
10. [Reporting guidance and limitations](#10-reporting-guidance-and-limitations)
11. [Reproducibility checklist](#11-reproducibility-checklist)

---

## 1. Scope

MedDiag predicts one of the following diagnostic classes and returns the
measurements, confidence and explainability that support the prediction.

| Modality | Classes | Network | Clinical features fused |
|---|---|---|---|
| ECG | `NORM`, `AFIB`, `MI`, `STDE`, `PVC` | ResNet1D | 18 |
| EEG | `NORMAL`, `EPILEPSY`, `PARKINSON`, `DEPRESSION` | EEGNet | 16 |
| Chest X-ray | `NORMAL`, `PNEUMONIA`, `CONSOLIDATION`, `EFFUSION`, `TUBERCULOSIS`, `COVID19` | 4-stage CNN | 11 |

**What this document does and does not claim.** The numbers in
[§5](#5-results) measure how well the models separate classes *on the stated
evaluation sets*. For the synthetic sets that is a synthetic-domain result; for
MIT-BIH and Bonn they are real-data results on public benchmarks. None of them
is a clinical validation, and none of them supports a diagnostic claim.

---

## 2. Quick reproduction

```powershell
# 1. environment (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. fetch the public benchmarks (~130 MB total)
.venv\Scripts\python.exe scripts\download_datasets.py --dataset mitbih
.venv\Scripts\python.exe scripts\download_datasets.py --dataset bonn

# 3. train the real-data checkpoints (optional - the benchmark runs either way)
#    --validation-ratio / --test-ratio must match the benchmark flags below
.venv\Scripts\python.exe scripts\train_ecg.py --dataset mitbih --data-root data\datasets\mitdb --epochs 40 --patience 8 --validation-ratio 0.15 --test-ratio 0.15 --output models\ecg_model_mitbih.pt
.venv\Scripts\python.exe scripts\train_eeg.py --dataset bonn   --data-root data\datasets\bonn   --epochs 40 --patience 8 --validation-ratio 0.15 --test-ratio 0.15 --output models\eeg_model_bonn.pt

# 4. run the benchmark
.venv\Scripts\python.exe scripts\run_benchmark.py --suite full --per-class 40 --validation-ratio 0.15 --test-ratio 0.15 --out results
```

Artefacts written to `results/`:

| File | Contents |
|---|---|
| `summary.csv` | One row per (evaluation set × model) with every headline metric |
| `per_class_<set>__<model>.csv` | Precision / recall / F1 / support per class |
| `confusion_<set>__<model>.csv` | Confusion matrix |
| `results.md` | The same tables rendered as Markdown, ready to paste into a paper |
| `results.json` | Everything above plus the raw confusion matrices |

Two auxiliary checks complement the benchmark:

```powershell
.venv\Scripts\python.exe scripts\evaluate_models.py   # shipped checkpoints on unseen synthetic seeds
.venv\Scripts\python.exe scripts\smoke_check.py        # all 15 synthetic classes, pass/fail per class
.venv\Scripts\python.exe -m pytest -q                  # 56 unit / integration tests
```

After a new benchmark run, refresh the tables inside this document with:

```powershell
.venv\Scripts\python.exe scripts\fill_testing_section.py
```

That command rewrites the block between the `RESULTS-SUMMARY` markers from
`results/summary.csv` and `results/results.md`, so §5 can never drift away
from the numbers the harness actually produced.

---

## 3. Test parameters

### 3.1 Environment

| Parameter | Value |
|---|---|
| Python | 3.12.10 (64-bit) |
| PyTorch | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.0 |
| Pillow | 13.0.0 |
| Matplotlib | 3.11.2 |
| ReportLab | 5.1.0 |
| wfdb | 4.3.1 |
| mne | 1.13.2 |
| Compute device | CPU (`MEDDIAG_DEVICE=auto` resolves to `cpu`, no CUDA) |
| Global seed | 42 (training and split), 90210 (synthetic evaluation) |

### 3.2 Evaluation datasets

| Set | Source | Samples | Classes present | Grouping key | Split |
|---|---|---|---|---|---|
| `synthetic-ecg` | Built-in generator, seed 90210 | 40 per class × 5 = 200 | all 5 | per sample | whole set is held out (training used seeds 2024/2025/2026) |
| `synthetic-eeg` | Built-in generator, seed 90210 | 40 per class × 4 = 160 | all 4 | per sample | whole set is held out |
| `synthetic-xray` | Built-in generator, seed 90210 | 40 per class × 6 = 240 | all 6 | per sample | whole set is held out |
| `mit-bih` | MIT-BIH Arrhythmia Database, PhysioNet | 883 beat windows from 38 records → **145 in the test partition** | `NORM`, `AFIB`, `PVC` | record id | 70/15/15 grouped, seed 42 |
| `bonn` | Bonn University EEG Database | 500 recordings from 25 subjects → **80 in the test partition** | `NORMAL`, `EPILEPSY` | subject id | 70/15/15 grouped, seed 42 |

**Provenance and licence**

| Dataset | Origin | Licence |
|---|---|---|
| MIT-BIH Arrhythmia Database | PhysioNet, MIT Laboratory | Open Data Commons Attribution License v1.0 |
| Bonn University EEG Database | Andrzejak et al. (2001); redistributed via the `NeuralPsy/Bonn-EEG-Dataset` mirror | Free for academic and research use; cite the original publication |

**MIT-BIH label derivation.** Each annotated beat becomes a 5-second window
centred on the beat. The label comes from the beat symbol (`N` → `NORM`;
`V`/`e`/`E`/`j`/`r`/`F` → `PVC`). Atrial fibrillation is a *rhythm-level*
property, so it is applied at record level: every window taken from one of the
20 known AFIB records (`201`, `203`, `205`, `207`, `208`, `209`, `210`, `212`,
`213`, `214`, `215`, `217`, `219`, `220`, `223`, `230`, `231`, `232`, `233`,
`234`) inherits the `AFIB` label. The MIT-BIH symbol `A` (atrial premature
beat) is deliberately **not** mapped to `AFIB` — a single atrial ectopic beat
is not atrial fibrillation.

**Bonn label derivation.** The corpus uses two naming schemes; both are
supported. `Z`/`O` (or `A`/`B`/`H`/`K`) are healthy volunteers → `NORMAL`;
`N`/`F`/`S` (or `C`/`D`/`E`/`F`/`G`/`I`/`J`) carry epileptiform activity →
`EPILEPSY`.

### 3.3 Preprocessing parameters

#### ECG

| Parameter | Value |
|---|---|
| Input sampling rate | dataset native (360 Hz for MIT-BIH, 500 Hz default for the API) |
| Resampling | polyphase `resample_poly` to **500 Hz** |
| Unit handling | auto-detect (`mV` / `µV` / raw counts) → millivolts |
| Baseline wander removal | zero-phase high-pass, **0.05 Hz** (AHA standard) |
| Band-pass | zero-phase Butterworth, **0.05 – 40 Hz**, order 2 |
| Notch filter | **50 Hz**, Q = 30, applied when `fs ≥ 100 Hz` |
| Model window | **5 s = 2 500 samples** at 500 Hz |
| Window selection | highest-energy 5-second segment |
| Window normalisation | z-score (zero mean, unit variance) |
| R-peak detection | Pan-Tompkins: 5–15 Hz band → derivative → squaring → 150 ms moving-window integration → adaptive SPKI/NPKI threshold → refractory 250 ms → peak refinement |
| Heart rate | median RR over valid intervals (0.3 – 2.5 s) |
| HRV | RMSSD, pNN50, LF/HF from the RR series |
| Beat morphology | median-beat template, TP-segment-referenced ST measurement at J+30…J+80 ms, QRS onset/offset from the smoothed amplitude envelope |
| PVC detection | premature **and** wide **and** morphologically atypical |
| Signal quality index | crest factor × SNR × clipping penalty, noise floor from the quietest decile |

#### EEG

| Parameter | Value |
|---|---|
| Input sampling rate | dataset native (128 Hz for Bonn, 256 Hz default for the API) |
| Resampling | polyphase `resample_poly` to **256 Hz** |
| Unit handling | auto-detect (`mV` / `µV`) → microvolts |
| Band-pass | zero-phase Butterworth, **0.5 – 45 Hz**, order 4 |
| Notch filter | **50 Hz**, Q = 30, applied when `fs ≥ 100 Hz` |
| Model window | **10 s = 2 560 samples** at 256 Hz |
| Window selection | lowest-artefact 10-second segment (moving-RMS outlier score) |
| Window normalisation | z-score |
| Band powers | delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz |
| Spectral features | spectral entropy, Hjorth activity/mobility/complexity |
| Epileptiform features | spike-wave index, alpha asymmetry |
| Artefact detection | moving-RMS outlier comparison against the median (8× threshold) |
| Signal quality index | crest factor × SNR × (1 − artefact ratio) |

#### Chest X-ray

| Parameter | Value |
|---|---|
| Accepted formats | PNG, JPEG, DICOM |
| Decoding | Pillow (DICOM via the Pillow DICOM handler) |
| Enhancement | CLAHE, clip limit **2.0**, **8 × 8** tile grid |
| Model input | **224 × 224** greyscale, normalised to `[0, 1]` |
| Zone features | apical, perihilar, peripheral and costophrenic zones measured **relative to the lung fields** (not absolute pixel percentiles) |
| Plausibility check | `looks_like_chest_xray` — lung-field geometry heuristic |
| Image quality | sharpness, contrast and brightness scores |

### 3.4 Model and training parameters

| Parameter | ECG | EEG | X-ray |
|---|---|---|---|
| Architecture | ResNet1D, 3 residual stages, avg + max pool | EEGNet, 3 depthwise-separable blocks, global temporal pool | 4-stage CNN, global average pool |
| Trainable parameters | 6 686 341 | 30 884 | 505 830 |
| Fusion head | clinical features → BatchNorm1d(64) → ReLU → concat with pooled CNN features → MLP → logits | same | same |
| Loss | cross-entropy with **label smoothing 0.05** | same | same |
| Optimiser | AdamW, lr **1e-3** | same | same |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| LR schedule | `ReduceLROnPlateau` (factor 0.5, patience 2) | same | same |
| Gradient clipping | global norm 5.0 | same | same |
| Batch size | 32 | 32 | 32 |
| Epochs (max) | 40 (real) / 30 (synthetic) | 40 (real) / 30 (synthetic) | 20 (synthetic) |
| Early stopping | patience 8 (real) / 6 (synthetic) | same | same |
| Validation ratio | 0.2 | 0.2 | 0.2 |
| Split | **grouped** (record id / subject id) for real data | **grouped** (subject id) | stratified (samples are independent) |
| Model selection | best `(validation accuracy, −validation loss)` | same | same |
| Seed | 42 | 42 | 42 |

> Model selection uses accuracy **and** validation loss. Accuracy alone is
> insufficient: once it saturates at 1.000 it can never improve, so the saved
> checkpoint would stay on the first epoch that saturated it — often the least
> confident one. Adding the loss tie-break changed the shipped EEG checkpoint
> from `best_epoch 1, val_loss 1.19` to `best_epoch 6, val_loss 0.024`.

### 3.5 Inference parameters

| Parameter | Value |
|---|---|
| Model mode | `eval()` (BatchNorm uses running statistics) |
| Output | softmax over the full class list |
| Decision rule | `argmax` over all classes of the modality |
| Classes absent from a benchmark | predictions into an absent class count as **errors**, not as abstentions |
| Explainability | Grad-CAM (1-D for ECG/EEG, 2-D for X-ray) + per-feature network attribution |
| Grad-CAM upsampling | `linear` for 1-D activations, `bilinear` for 2-D |
| Image embedding | matplotlib Agg backend → base64 PNG |

### 3.6 Runtime parameters of the benchmark harness

| Parameter | Flag | Default |
|---|---|---|
| Evaluation suite | `--suite` | `full` (`synthetic` \| `real` \| `full`) |
| Output directory | `--out` | `results/` |
| Synthetic samples per class | `--per-class` | 20 |
| Synthetic evaluation seed | `--seed` | 90210 |
| Validation ratio for the real sets | `--validation-ratio` | 0.15 |
| Test ratio for the real sets | `--test-ratio` | 0.15 |
| MIT-BIH root | `--mitbih-root` | `data/datasets/mitdb` |
| Bonn root | `--bonn-root` | `data/datasets/bonn` |

---

## 4. Evaluation protocol

### 4.1 Grouped, leakage-safe, three-way splits

Windows cut from the same recording are near-duplicates of each other. A
window-level random split therefore places near-identical samples on both
sides and overstates the score. MedDiag splits on a **group identifier**:

| Dataset | Group identifier | Effect |
|---|---|---|
| MIT-BIH | record id (`100`, `201`, …) | all windows of one record stay together |
| Bonn | subject id (set + subject, 20 segments per subject) | all segments of one subject stay together |

A single train/validation split is not enough for a paper. The validation side
is the set the early-stopping rule and the best-epoch choice looked at, so a
score measured on it is a **model-selection** score and is optimistic by
construction. MedDiag therefore carves the data into **three** grouped
partitions:

| Partition | Share (MIT-BIH / Bonn) | Used for |
|---|---|---|
| train | 0.70 / 0.68 | weight updates |
| validation | 0.15 | early stopping, best-epoch selection, learning-rate schedule |
| **test** | 0.15 | **the reported number — nothing else touches it** |

The split is class-balanced by a greedy assignment: groups are added to the
validation and test sides in the order that best reduces the deviation between
their class counts and the targets, and the assignment stops once each sample
budget is reached. `app/data/splitting.py` contains the implementation;
`grouped_split_indices()` (two-way) and `grouped_split_indices_3way()`
(three-way) are the entry points.

Two invariants are asserted by the test suite and by the training logs:

1. the three partitions are **disjoint** and together cover every sample;
2. **no group appears in two partitions** — zero record/subject overlap.

The benchmark reports the **test** partition only, so the reported score never
contains a sample the model was trained on *or* a sample that influenced which
epoch was saved.

> **Reproducibility requirement.** The `--validation-ratio` and `--test-ratio`
> values must be identical in the training command and in the benchmark
> command, because the split is a deterministic function of
> (ratios, seed). Changing them re-partitions the data and the benchmark would
> score a different set of recordings than the checkpoint was selected on.

### 4.2 Metric definitions

| Metric | Definition |
|---|---|
| Accuracy | `correct / total` over all evaluated samples |
| Precision (per class) | `TP / (TP + FP)` |
| Recall (per class) | `TP / (TP + FN)` |
| F1 (per class) | `2 · P · R / (P + R)` |
| Macro-P / Macro-R / Macro-F1 | unweighted mean over the classes **present** in the evaluation set |
| Weighted-F1 | support-weighted mean F1 over all samples |
| Macro-AUC | mean of the per-class one-vs-rest ROC AUC, computed from the rank statistic (Mann–Whitney U equivalent) |
| Confusion matrix | rows = true class, columns = predicted class |
| Mean / P95 latency | wall-clock milliseconds per analysis, measured on the evaluation machine, single process, no batching |

### 4.3 Reproducibility

- Training, splitting and the synthetic generators are all seeded
  (`--seed`, default 42).
- The synthetic evaluation sets use seed **90210**, deliberately outside the
  training seed range (2024 / 2025 / 2026), so they are genuinely unseen.
- Model selection, split assignment and metric computation are deterministic
  for a fixed seed.
- The three-way split is a pure function of `(validation_ratio, test_ratio,
  seed)`. The same ratios and seed in the training and benchmark commands
  reproduce the exact same partitions, so the benchmark scores precisely the
  recordings the checkpoint was selected on — and nothing else.
- The trained checkpoints record their own split configuration
  (`metadata.validation_ratio`, `metadata.test_ratio`) and the metrics they
  measured on the untouched test partition (`test_accuracy`, `test_f1_macro`,
  `test_auc`), so a reported number can be traced back to the checkpoint file
  it came from.

---

## 5. Results

Every number below was produced by `scripts/run_benchmark.py` and is archived in
`results/` (`summary.csv`, `results.md`, `results.json`, plus one CSV per class
breakdown and per confusion matrix). The command that produced this run was:

```powershell
.venv\Scripts\python.exe scripts\run_benchmark.py --suite full --per-class 40 --validation-ratio 0.15 --test-ratio 0.15 --out results
```

<!-- RESULTS-SUMMARY -->
### 5.1 Headline results

| Evaluation set | Modality | Model | Source | Partition | n | Groups | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Macro-AUC | Latency ms |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic-ecg | ecg | trained (synthetic corpus) | trained | synthetic-holdout | 200 | 200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 30.0 |
| synthetic-ecg | ecg | trained (real benchmark) | trained | synthetic-holdout | 200 | 200 | 0.2100 | 0.2455 | 0.2100 | 0.0931 | 0.0931 | 0.7314 | 36.2 |
| synthetic-ecg | ecg | heuristic baseline | heuristic_baseline | synthetic-holdout | 200 | 200 | 0.9800 | 0.9818 | 0.9800 | 0.9799 | 0.9799 | 1.0000 | 5.8 |
| synthetic-eeg | eeg | trained (synthetic corpus) | trained | synthetic-holdout | 160 | 160 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.2 |
| synthetic-eeg | eeg | trained (real benchmark) | trained | synthetic-holdout | 160 | 160 | 0.5000 | 0.3333 | 0.5000 | 0.3750 | 0.3750 | 0.8176 | 5.3 |
| synthetic-eeg | eeg | heuristic baseline | heuristic_baseline | synthetic-holdout | 160 | 160 | 0.8938 | 0.9254 | 0.8938 | 0.8965 | 0.8965 | 1.0000 | 3.9 |
| synthetic-xray | xray | trained (synthetic corpus) | trained | synthetic-holdout | 240 | 240 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28.0 |
| synthetic-xray | xray | heuristic baseline | heuristic_baseline | synthetic-holdout | 240 | 240 | 0.6625 | 0.7655 | 0.6625 | 0.6348 | 0.6348 | 0.9198 | 10.0 |
| mit-bih | ecg | trained (synthetic corpus) | trained | test | 145 | 6 | 0.3310 | 0.5805 | 0.2978 | 0.3486 | 0.3832 | 0.6079 | 38.1 |
| mit-bih | ecg | trained (real benchmark) | trained | test | 145 | 6 | 0.5931 | 0.6118 | 0.6022 | 0.5770 | 0.6118 | 0.7289 | 40.8 |
| mit-bih | ecg | heuristic baseline | heuristic_baseline | test | 145 | 6 | 0.0828 | 0.4048 | 0.0667 | 0.1128 | 0.1400 | 0.5020 | 5.1 |
| bonn | eeg | trained (synthetic corpus) | trained | test | 80 | 4 | 0.4375 | 0.7708 | 0.4375 | 0.4791 | 0.4791 | 0.8250 | 7.6 |
| bonn | eeg | trained (real benchmark) | trained | test | 80 | 4 | 0.8875 | 0.9082 | 0.8875 | 0.8861 | 0.8861 | 0.9559 | 8.4 |
| bonn | eeg | heuristic baseline | heuristic_baseline | test | 80 | 4 | 0.0250 | 0.5000 | 0.0250 | 0.0476 | 0.0476 | 0.5000 | 5.3 |


### 5.2 Per-class results and confusion matrices

**synthetic-ecg - trained (synthetic corpus)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 1.0000 | 1.0000 | 1.0000 | 40 |
| AFIB | 1.0000 | 1.0000 | 1.0000 | 40 |
| MI | 1.0000 | 1.0000 | 1.0000 | 40 |
| STDE | 1.0000 | 1.0000 | 1.0000 | 40 |
| PVC | 1.0000 | 1.0000 | 1.0000 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORM | AFIB | MI | STDE | PVC |
|---|---|---|---|---|---|
| **NORM** | 40 | 0 | 0 | 0 | 0 |
| **AFIB** | 0 | 40 | 0 | 0 | 0 |
| **MI** | 0 | 0 | 40 | 0 | 0 |
| **STDE** | 0 | 0 | 0 | 40 | 0 |
| **PVC** | 0 | 0 | 0 | 0 | 40 |


**synthetic-ecg - trained (real benchmark)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 0.0000 | 0.0000 | 0.0000 | 40 |
| AFIB | 0.2273 | 1.0000 | 0.3704 | 40 |
| MI | 0.0000 | 0.0000 | 0.0000 | 40 |
| STDE | 0.0000 | 0.0000 | 0.0000 | 40 |
| PVC | 1.0000 | 0.0500 | 0.0952 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORM | AFIB | MI | STDE | PVC |
|---|---|---|---|---|---|
| **NORM** | 0 | 40 | 0 | 0 | 0 |
| **AFIB** | 0 | 40 | 0 | 0 | 0 |
| **MI** | 3 | 37 | 0 | 0 | 0 |
| **STDE** | 19 | 21 | 0 | 0 | 0 |
| **PVC** | 0 | 38 | 0 | 0 | 2 |


**synthetic-ecg - heuristic baseline**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 0.9091 | 1.0000 | 0.9524 | 40 |
| AFIB | 1.0000 | 1.0000 | 1.0000 | 40 |
| MI | 1.0000 | 0.9000 | 0.9474 | 40 |
| STDE | 1.0000 | 1.0000 | 1.0000 | 40 |
| PVC | 1.0000 | 1.0000 | 1.0000 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORM | AFIB | MI | STDE | PVC |
|---|---|---|---|---|---|
| **NORM** | 40 | 0 | 0 | 0 | 0 |
| **AFIB** | 0 | 40 | 0 | 0 | 0 |
| **MI** | 4 | 0 | 36 | 0 | 0 |
| **STDE** | 0 | 0 | 0 | 40 | 0 |
| **PVC** | 0 | 0 | 0 | 0 | 40 |


**synthetic-eeg - trained (synthetic corpus)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 1.0000 | 1.0000 | 1.0000 | 40 |
| EPILEPSY | 1.0000 | 1.0000 | 1.0000 | 40 |
| PARKINSON | 1.0000 | 1.0000 | 1.0000 | 40 |
| DEPRESSION | 1.0000 | 1.0000 | 1.0000 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | EPILEPSY | PARKINSON | DEPRESSION |
|---|---|---|---|---|
| **NORMAL** | 40 | 0 | 0 | 0 |
| **EPILEPSY** | 0 | 40 | 0 | 0 |
| **PARKINSON** | 0 | 0 | 40 | 0 |
| **DEPRESSION** | 0 | 0 | 0 | 40 |


**synthetic-eeg - trained (real benchmark)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.3333 | 1.0000 | 0.5000 | 40 |
| EPILEPSY | 1.0000 | 1.0000 | 1.0000 | 40 |
| PARKINSON | 0.0000 | 0.0000 | 0.0000 | 40 |
| DEPRESSION | 0.0000 | 0.0000 | 0.0000 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | EPILEPSY | PARKINSON | DEPRESSION |
|---|---|---|---|---|
| **NORMAL** | 40 | 0 | 0 | 0 |
| **EPILEPSY** | 0 | 40 | 0 | 0 |
| **PARKINSON** | 40 | 0 | 0 | 0 |
| **DEPRESSION** | 40 | 0 | 0 | 0 |


**synthetic-eeg - heuristic baseline**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.7018 | 1.0000 | 0.8247 | 40 |
| EPILEPSY | 1.0000 | 0.7250 | 0.8406 | 40 |
| PARKINSON | 1.0000 | 0.9750 | 0.9873 | 40 |
| DEPRESSION | 1.0000 | 0.8750 | 0.9333 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | EPILEPSY | PARKINSON | DEPRESSION |
|---|---|---|---|---|
| **NORMAL** | 40 | 0 | 0 | 0 |
| **EPILEPSY** | 11 | 29 | 0 | 0 |
| **PARKINSON** | 1 | 0 | 39 | 0 |
| **DEPRESSION** | 5 | 0 | 0 | 35 |


**synthetic-xray - trained (synthetic corpus)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 1.0000 | 1.0000 | 1.0000 | 40 |
| PNEUMONIA | 1.0000 | 1.0000 | 1.0000 | 40 |
| CONSOLIDATION | 1.0000 | 1.0000 | 1.0000 | 40 |
| EFFUSION | 1.0000 | 1.0000 | 1.0000 | 40 |
| TUBERCULOSIS | 1.0000 | 1.0000 | 1.0000 | 40 |
| COVID19 | 1.0000 | 1.0000 | 1.0000 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | PNEUMONIA | CONSOLIDATION | EFFUSION | TUBERCULOSIS | COVID19 |
|---|---|---|---|---|---|---|
| **NORMAL** | 40 | 0 | 0 | 0 | 0 | 0 |
| **PNEUMONIA** | 0 | 40 | 0 | 0 | 0 | 0 |
| **CONSOLIDATION** | 0 | 0 | 40 | 0 | 0 | 0 |
| **EFFUSION** | 0 | 0 | 0 | 40 | 0 | 0 |
| **TUBERCULOSIS** | 0 | 0 | 0 | 0 | 40 | 0 |
| **COVID19** | 0 | 0 | 0 | 0 | 0 | 40 |


**synthetic-xray - heuristic baseline**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.5195 | 1.0000 | 0.6838 | 40 |
| PNEUMONIA | 0.3043 | 0.3500 | 0.3256 | 40 |
| CONSOLIDATION | 1.0000 | 0.2000 | 0.3333 | 40 |
| EFFUSION | 1.0000 | 0.4250 | 0.5965 | 40 |
| TUBERCULOSIS | 1.0000 | 1.0000 | 1.0000 | 40 |
| COVID19 | 0.7692 | 1.0000 | 0.8696 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | PNEUMONIA | CONSOLIDATION | EFFUSION | TUBERCULOSIS | COVID19 |
|---|---|---|---|---|---|---|
| **NORMAL** | 40 | 0 | 0 | 0 | 0 | 0 |
| **PNEUMONIA** | 26 | 14 | 0 | 0 | 0 | 0 |
| **CONSOLIDATION** | 0 | 32 | 8 | 0 | 0 | 0 |
| **EFFUSION** | 11 | 0 | 0 | 17 | 0 | 12 |
| **TUBERCULOSIS** | 0 | 0 | 0 | 0 | 40 | 0 |
| **COVID19** | 0 | 0 | 0 | 0 | 0 | 40 |


**mit-bih - trained (synthetic corpus)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 1.0000 | 0.2167 | 0.3562 | 60 |
| AFIB | 0.4559 | 0.5167 | 0.4844 | 60 |
| PVC | 0.2857 | 0.1600 | 0.2051 | 25 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORM | AFIB | MI | STDE | PVC |
|---|---|---|---|---|---|
| **NORM** | 13 | 20 | 27 | 0 | 0 |
| **AFIB** | 0 | 31 | 19 | 0 | 10 |
| **PVC** | 0 | 17 | 4 | 0 | 4 |


**mit-bih - trained (real benchmark)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 0.8649 | 0.5333 | 0.6598 | 60 |
| AFIB | 0.6441 | 0.6333 | 0.6387 | 60 |
| PVC | 0.3265 | 0.6400 | 0.4324 | 25 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORM | AFIB | MI | STDE | PVC |
|---|---|---|---|---|---|
| **NORM** | 32 | 16 | 0 | 0 | 12 |
| **AFIB** | 1 | 38 | 0 | 0 | 21 |
| **PVC** | 4 | 5 | 0 | 0 | 16 |


**mit-bih - heuristic baseline**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 0.5000 | 0.1167 | 0.1892 | 60 |
| AFIB | 0.7143 | 0.0833 | 0.1493 | 60 |
| PVC | 0.0000 | 0.0000 | 0.0000 | 25 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORM | AFIB | MI | STDE | PVC |
|---|---|---|---|---|---|
| **NORM** | 7 | 0 | 53 | 0 | 0 |
| **AFIB** | 6 | 5 | 41 | 6 | 2 |
| **PVC** | 1 | 2 | 22 | 0 | 0 |


**bonn - trained (synthetic corpus)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 1.0000 | 0.2250 | 0.3673 | 40 |
| EPILEPSY | 0.5417 | 0.6500 | 0.5909 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | EPILEPSY | PARKINSON | DEPRESSION |
|---|---|---|---|---|
| **NORMAL** | 9 | 22 | 6 | 3 |
| **EPILEPSY** | 0 | 26 | 14 | 0 |


**bonn - trained (real benchmark)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 1.0000 | 0.7750 | 0.8732 | 40 |
| EPILEPSY | 0.8163 | 1.0000 | 0.8989 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | EPILEPSY | PARKINSON | DEPRESSION |
|---|---|---|---|---|
| **NORMAL** | 31 | 9 | 0 | 0 |
| **EPILEPSY** | 0 | 40 | 0 | 0 |


**bonn - heuristic baseline**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 1.0000 | 0.0500 | 0.0952 | 40 |
| EPILEPSY | 0.0000 | 0.0000 | 0.0000 | 40 |

Confusion matrix (rows = true, columns = predicted):

| true \ predicted | NORMAL | EPILEPSY | PARKINSON | DEPRESSION |
|---|---|---|---|---|
| **NORMAL** | 2 | 0 | 10 | 28 |
| **EPILEPSY** | 0 | 0 | 11 | 29 |

<!-- /RESULTS-SUMMARY -->

### 5.3 Headline real-data results

| Evaluation set | Model | n | Accuracy | Macro-F1 | Macro-AUC | Beats the heuristic baseline? |
|---|---|---:|---:|---:|---:|---|
| `mit-bih` (ECG) | `trained (real benchmark)` | 145 | **0.5931** | **0.5770** | **0.7289** | yes — 0.0828 / 0.1128 / 0.5020 for the baseline |
| `bonn` (EEG) | `trained (real benchmark)` | 80 | **0.8875** | **0.8861** | **0.9559** | yes — 0.0250 / 0.0476 / 0.5000 for the baseline |

Both real-data models beat the transparent rule-based baseline by a wide
margin on the untouched test partition, and both beat the *synthetic-trained*
model evaluated on the same recordings (`mit-bih`: 0.3310 accuracy;
`bonn`: 0.4375 accuracy) — which is the domain-transfer effect, measured.

**The numbers are self-consistent.** The checkpoint files record the metrics
they measured on their own untouched test partition, and the benchmark
reproduces them exactly:

| Checkpoint | Recorded in `models/*.metrics.json` | Reproduced by the benchmark |
|---|---|---|
| `ecg_model_mitbih.pt` | `test_accuracy` 0.5931 · `test_f1_macro` 0.5770 · `test_auc` 0.7289 | 0.5931 · 0.5770 · 0.7289 |
| `eeg_model_bonn.pt` | `test_accuracy` 0.8875 · `test_f1_macro` 0.8861 · `test_auc` 0.9559 | 0.8875 · 0.8861 · 0.9559 |

That equality is the practical proof that the benchmark is scoring the
partition the training run carved out and left untouched.

**Where the ECG model is still weak, honestly.** MIT-BIH accuracy of 0.5931 is
modest, and the confusion matrix shows where the errors go (rows = true,
columns = predicted; support: NORM 60, AFIB 60, PVC 25):

- `NORM` beats (60): 32 correct, but **16 called `AFIB`** and 12 called `PVC`.
- `AFIB` beats (60): 38 correct, but **21 leak into `PVC`** and 1 into `NORM`.
- `PVC` beats (25): only **16 correct**, with 5 called `AFIB` and 4 `NORM`.

Three structural reasons:

1. **AFIB is a rhythm-level property.** The evaluation window is 5 seconds
   centred on one beat, so it holds only about six heart beats — very little
   evidence of RR irregularity, which is what actually defines atrial
   fibrillation. The label is correct at record level but the window is too
   short to show it, and a beat that looks slightly premature is read as
   ectopic.
2. **PVC and AFIB are confused with each other in both directions.** Both
   conditions change beat morphology and timing, and a single 5-second window
   does not contain enough context to separate "wide, premature beat" from
   "beat in an irregular rhythm".
3. **Small test partition.** 145 windows from 6 records. One window is worth
   0.69 % accuracy, so the confidence around these figures is wide.

The Bonn EEG result is strong and stable (accuracy 0.8875, macro-F1 0.8861,
AUC 0.9559; zero false negatives on `EPILEPSY`), and it is the one to quote
as the headline real-data result.

### 5.4 Latency

Latency is measured on the evaluation machine (CPU only, no batching, one
process). Per-dataset values are in `results/summary.csv`
(`mean_latency_ms`, `p95_latency_ms`).

### 5.5 How to read these numbers

- **A perfect synthetic score is not a clinical score.** The `synthetic-*`
  rows are 1.0000 because the synthetic generator encodes each condition with
  a distinct, deliberately separable signature and the evaluation set uses
  seeds the training never saw. That demonstrates the pipeline works end to
  end; it says nothing about accuracy on real patients. Quote the `mit-bih`
  and `bonn` rows for anything clinical, and label the synthetic rows as
  synthetic-domain results.
- **The real-data rows are the ones to quote.** `mit-bih` and `bonn` are
  scored on the untouched test partition of a grouped three-way split; the
  synthetic rows measure separation inside the synthetic domain only.
- **Always read the `model_source` column.** A row produced by the
  `heuristic baseline` is a rule-engine result and must not be presented as a
  learned-model result.
- **Macro metrics average over the classes that are present.** MIT-BIH has no
  MI or STDE recordings and Bonn has no Parkinson or depression recordings, so
  those classes are absent from the evaluation set and excluded from the macro
  averages rather than counted as zero.
- **A single seed.** The figures come from seed 42. They are bit-reproducible
  — the training scripts seed the RNG before the network is constructed — but
  they are one draw; see §10.2.

---

## 6. Ablations and domain transfer

The benchmark evaluates three model variants per modality:

| Variant | Checkpoint | What it isolates |
|---|---|---|
| `trained (synthetic corpus)` | `models/{ecg,eeg,xray}_model.pt` | the shipped model |
| `trained (real benchmark)` | `models/ecg_model_mitbih.pt`, `models/eeg_model_bonn.pt` | the model retrained on real public data |
| `heuristic baseline` | none | the deterministic clinical-rule scorer that runs when no checkpoint exists |

Comparing them yields two useful results for a paper:

1. **Domain transfer.** The synthetic-trained ECG model evaluated on MIT-BIH
   (and the synthetic-trained EEG model on Bonn) measures the gap between
   synthetic training and real recordings. It is expected to be large; quoting
   it is more honest than quoting only the synthetic number.
2. **Heuristic baseline.** The rule-based scorer is a strong, completely
   transparent reference point. A neural model that does not beat it on a
   given dataset is not adding value on that dataset.

A third, complementary comparison is the **cross-domain collapse**: the real
checkpoints evaluated on the synthetic corpus. They were trained on two or
three classes, so the classes that do not exist in their training set can
never be predicted — the confusion matrix shows exactly that, and it is a
useful illustration of why a headline accuracy alone is not enough to
describe a model.

### 6.1 What the protocol isolates

| Comparison | What it holds constant | What it isolates |
|---|---|---|
| `trained (real benchmark)` vs `heuristic baseline` on the same set | the evaluation set, the split, the metric code | the value the neural network adds over transparent clinical rules |
| `trained (real benchmark)` vs `trained (synthetic corpus)` on the same set | the evaluation set, the split | the effect of training on real public recordings |
| `trained (real benchmark)` on `mit-bih` vs on `synthetic-ecg` | the checkpoint | the size of the domain gap |

All three rows read from the same `results/summary.csv`; nothing is computed
outside the harness.

---

## 7. Automated test suite

```powershell
.venv\Scripts\python.exe -m pytest -q
```

56 tests covering:

| File | Coverage |
|---|---|
| `tests/test_api.py` | health, model registry, ECG/EEG analysis (JSON **and** upload), X-ray analysis, input validation and rejection paths, PDF/HTML/JSON report generation and download, history, statistics, demo routes, dashboard and static asset serving |
| `tests/test_features.py` | filter behaviour, resampling, z-scoring, R-peak detection, preprocessing shapes and quality metrics, per-class feature separation, completeness of the reference tables |
| `tests/test_xai.py` | every plotting helper, Grad-CAM on the ECG and X-ray networks, integrated gradients, per-feature attribution, occlusion sensitivity |
| `tests/test_splitting.py` | the three-way grouped split (disjointness, completeness, zero group overlap, determinism, ratio budgets, integer group ids), the macro-F1 "present classes only" convention, and reproducible network initialisation |

Run only the protocol tests while iterating on the split:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_splitting.py -q
```

---

## 8. API-level test parameters

### 8.1 Request bodies

**ECG / EEG** — `POST /api/v1/ecg/analyze`, `POST /api/v1/eeg/analyze`

| Field | Type | Required | Notes |
|---|---|---|---|
| `samples` | `list[float]` | yes | raw samples; ≥ 8 samples enforced |
| `sampling_rate` | `float` | yes | Hz; ECG default 500, EEG default 256 |
| `patient.patient_id` | `string` | yes | free-form reference |
| `patient.age` | `int` | yes | 0 – 120 |
| `patient.sex` | `male` \| `female` \| `other` \| `unknown` | yes | |
| `patient.referring_clinician` | `string` | no | |
| `patient.clinical_notes` | `string` | no | |
| `lead_names` / `channel_names` | `list[string]` | no | one label per channel |

**Chest X-ray** — `POST /api/v1/xray/analyze` (multipart/form-data)

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | PNG / JPEG / DICOM, ≤ 25 MB |
| `patient_id` | `string` | yes | |
| `age` | `int` | yes | |
| `sex` | `string` | no | |
| `view` | `PA` \| `AP` \| `LAT` | no | acquisition view |
| `referring_clinician`, `clinical_notes` | `string` | no | |

**Upload variant** — `POST /api/v1/{ecg,eeg}/analyze/upload`

| Field | Type | Notes |
|---|---|---|
| `file` | file | `.csv`, `.json`, `.txt` (whitespace separated) or `.mat` |
| `patient_id`, `age`, `sex`, `sampling_rate`, `referring_clinician`, `clinical_notes` | form fields | mirror the JSON body |

### 8.2 Response fields used for reporting

| Field | Type | Use in a paper |
|---|---|---|
| `prediction.label` | `string` | predicted class |
| `prediction.confidence` | `float` | probability of the predicted class |
| `prediction.class_probabilities` | `dict[str, float]` | full probability vector (for ROC / calibration) |
| `differential[]` | `list[{label, probability, severity}]` | ranked alternatives |
| `findings[]` | `list[{name, value, unit, reference_range, status, explanation}]` | the clinical measurements |
| `quality.signal_quality_index` | `float` | input quality, 0 – 1 |
| `quality.usable` | `bool` | whether the input was good enough to trust |
| `quality.snr_db` | `float` | signal-to-noise ratio |
| `quality.issues` | `list[string]` | e.g. "clipped samples", "not a chest radiograph" |
| `explainability.methods` | `list[string]` | attribution methods actually computed |
| `explainability.top_features[]` | `list[{name, contribution, direction, detail}]` | per-feature contributions |
| `explainability.class_relevance` | `dict[str, float]` | per-class relevance scores |
| `model_source` | `trained` \| `heuristic_baseline` | **always report this** |
| `disclaimer` | `string` | included in every response |

### 8.3 Report generation

```bash
POST /api/v1/reports/{analysis_id}?format=pdf|html|json
GET  /api/v1/reports/{report_id}/download
GET  /api/v1/reports/{analysis_id}
GET  /api/v1/reports/list?limit=50
```

PDF is rendered with ReportLab, HTML as a standalone document, JSON as the
raw analysis payload. All three are persisted under `reports/` and recorded in
the audit trail.

---

## 9. Service configuration parameters

| Variable | Default | Effect on testing |
|---|---|---|
| `MEDDIAG_HOST` | `0.0.0.0` | bind address |
| `MEDDIAG_PORT` | `8000` | bind port |
| `MEDDIAG_DEBUG` | `false` | debug mode |
| `MEDDIAG_API_KEY` | *(unset)* | when set, every `/api/v1` call needs the `X-API-Key` header |
| `MEDDIAG_MAX_UPLOAD_MB` | `25` | upload size limit |
| `MEDDIAG_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` — `auto` picks CUDA when available, otherwise CPU |
| `MEDDIAG_XAI_ENABLED` | `true` | set to `false` to time the inference path without explainability |
| `MEDDIAG_XAI_MAX_FEATURES` | `8` | number of feature contributions returned |
| `MEDDIAG_ECG_DEFAULT_SAMPLING_RATE` | `500` | used when the request omits the rate |
| `MEDDIAG_EEG_DEFAULT_SAMPLING_RATE` | `256` | used when the request omits the rate |
| `MEDDIAG_RETAIN_ANALYSES_DAYS` | `30` | retention window for stored analyses |

---

## 10. Reporting guidance and limitations

### 10.1 What to quote

Always report `model_source` alongside any accuracy. A number produced by the
heuristic baseline is a rule-engine result and must not be presented as a
learned-model result. The API makes the distinction explicit, and the
benchmark writes it into `results/summary.csv` as the `model_source` column.

### 10.2 Limitations

- **The shipped checkpoints are trained on synthetic data.** Their scores on
  `synthetic-*` measure separation inside the synthetic domain only. The
  real-data checkpoints (`*_model_mitbih.pt`, `*_model_bonn.pt`) are trained
  on the public benchmarks and are the ones to quote for real-data claims.
- **No X-ray real-data checkpoint.** NIH ChestX-ray14 is ~45 GB and was not
  downloaded in this environment, so the X-ray results are synthetic-domain
  only. The adapter (`--dataset nih`) is implemented and ready.
- **Partial class coverage.** MIT-BIH contains no myocardial infarction or
  non-specific ST-depression recordings, and Bonn contains no Parkinson or
  depression recordings. Those classes are absent from the corresponding
  evaluation sets and are reported as "not evaluated" rather than as zeros.
- **Single-lead ECG.** The service selects the most diagnostic lead of a
  multi-lead recording but does not reason across leads jointly.
- **Grouped splits limit the achievable balance.** With 25 Bonn subjects (and
  38 MIT-BIH records), a leakage-free partition can only be as balanced as the
  group counts allow; the realised split is reported in
  `results/summary.csv` (`n_samples`, `n_groups`).
- **Small test partitions.** A 15 % grouped test partition is 80 Bonn
  recordings from 4 subjects and 145 MIT-BIH windows from 6 records. A single
  sample therefore moves the accuracy by 1.25 % (Bonn) or 0.69 % (MIT-BIH),
  so differences below a few percentage points are not meaningful. Quote the
  confusion matrix and the per-class breakdown, not only the accuracy.
- **Single seed.** The reported figures come from one fixed seed (42). They are
  bit-reproducible — the training scripts seed the RNG before the network is
  constructed — but they are still one draw from the seed distribution. Use
  the split as the protocol and repeat with other seeds if a confidence
  interval is required.
- **Not a medical device.** No output of this system is a diagnosis, and
  nothing here supports a clinical decision.

### 10.3 Suggested paper wording

> *We evaluate MedDiag on the MIT-BIH Arrhythmia Database (PhysioNet) and the
> Bonn University EEG Database using subject-disjoint three-way splits
> (70 % train / 15 % validation / 15 % test), so that no recording or subject
> appears in more than one partition and the test partition influences neither
> the model weights nor the early-stopping decision. All preprocessing,
> feature extraction and model parameters are listed in Table 3; the evaluation
> harness and every number reported here are produced by
> `scripts/run_benchmark.py` and archived in `results/`.*

---

## 11. Reproducibility checklist

| Check | Command | Expected |
|---|---|---|
| The split is leakage-free | `pytest tests/test_splitting.py -q` | all pass; zero group overlap asserted |
| Training is reproducible | run `scripts/train_eeg.py --dataset bonn --seed 42` twice | identical epoch metrics |
| The reported test number is untouched | compare `test_*` keys in `models/*.metrics.json` with the matching row of `results/summary.csv` | same accuracy / F1 / AUC |
| The shipped model is the trained one | `GET /api/v1/models` | `model_source: "trained"`, not `heuristic_baseline` |
| The unit suite passes | `pytest -q` | all pass |
| The synthetic corpus is fully classified | `scripts/smoke_check.py` | 15/15 classes, 0 failures |

---

*MedDiag — AI-Based Medical Diagnosis Assistant (ECG / EEG / Chest X-ray).*
