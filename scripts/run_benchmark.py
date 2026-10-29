"""Benchmark harness for MedDiag.

Evaluates the trained checkpoints (and the heuristic baseline) on

* the **synthetic** corpus, using seed ranges that the training run never saw;
* the **MIT-BIH Arrhythmia Database** (real ECG, PhysioNet);
* the **Bonn University EEG Database** (real EEG).

Every evaluation set is split by *group* (record id for MIT-BIH, subject id
for Bonn) so no recording or subject appears on both sides of the split.  When
``--test-ratio`` is greater than zero the real-data sets use a three-way split
(train / validation / test) and the **test** partition is the one that gets
scored: that partition influenced neither the weights nor the epoch that was
saved, so the number is a genuine held-out result.

Outputs
-------
``results/summary.csv``            one row per (dataset, model)
``results/per_class_*.csv``        precision / recall / F1 / support per class
``results/confusion_*.csv``        confusion matrices
``results/latency_*.csv``          per-sample inference latency
``results/results.md``             the tables above rendered as Markdown

Usage
-----
    python scripts/run_benchmark.py --suite full --out results
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.ml.ecg.inference import ECGModel  # noqa: E402
from app.ml.eeg.inference import EEGModel  # noqa: E402
from app.ml.xray.inference import XrayModel  # noqa: E402

logger = get_logger("benchmark")


# --------------------------------------------------------------------------- data
@dataclass
class EvalSet:
    """One evaluation set: raw inputs, labels and grouping information."""

    name: str
    modality: str
    samples: List[Any]
    labels: np.ndarray
    class_names: List[str]
    groups: Optional[np.ndarray] = None
    sampling_rate: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)


def _split_mitbih(
    root: Path, validation_ratio: float, test_ratio: float, seed: int
) -> EvalSet:
    """MIT-BIH held-out evaluation set (grouped by record)."""
    from app.data.ecg_datasets import load_mitbih
    from app.data.splitting import grouped_split_indices, grouped_split_indices_3way

    dataset = load_mitbih(root)
    if test_ratio > 0:
        train_idx, val_idx, eval_idx = grouped_split_indices_3way(
            dataset.labels, dataset.groups, validation_ratio, test_ratio, seed
        )
        partition = "test"
    else:
        train_idx, val_idx = grouped_split_indices(
            dataset.labels, dataset.groups, validation_ratio, seed
        )
        eval_idx = val_idx
        partition = "validation"

    classes = [dataset.class_names[i] for i in range(len(dataset.class_names))]
    return EvalSet(
        name="mit-bih",
        modality="ecg",
        samples=[dataset.signals[i][0] for i in eval_idx],
        labels=dataset.labels[eval_idx],
        class_names=classes,
        groups=dataset.groups[eval_idx],
        sampling_rate=float(dataset.metadata.get("sampling_rate", 360.0)),
        description=(
            f"MIT-BIH Arrhythmia Database (PhysioNet), {len(eval_idx)} beat windows "
            f"from {len(set(dataset.groups[eval_idx]))} held-out records ({partition} partition)"
        ),
        metadata={"train_indices": train_idx.tolist(), "dataset": "mit-bih", "partition": partition},
    )


def _split_bonn(root: Path, validation_ratio: float, test_ratio: float, seed: int) -> EvalSet:
    """Bonn held-out evaluation set (grouped by subject)."""
    from app.data.eeg_datasets import load_bonn
    from app.data.splitting import grouped_split_indices, grouped_split_indices_3way

    dataset = load_bonn(root)
    if test_ratio > 0:
        _, _, eval_idx = grouped_split_indices_3way(
            dataset.labels, dataset.groups, validation_ratio, test_ratio, seed
        )
        partition = "test"
    else:
        _, val_idx = grouped_split_indices(dataset.labels, dataset.groups, validation_ratio, seed)
        eval_idx = val_idx
        partition = "validation"

    classes = [dataset.class_names[i] for i in range(len(dataset.class_names))]
    return EvalSet(
        name="bonn",
        modality="eeg",
        samples=[dataset.signals[i][0] for i in eval_idx],
        labels=dataset.labels[eval_idx],
        class_names=classes,
        groups=dataset.groups[eval_idx],
        sampling_rate=float(dataset.metadata.get("sampling_rate", 128.0)),
        description=(
            f"Bonn University EEG Database, {len(eval_idx)} recordings from "
            f"{len(set(dataset.groups[eval_idx]))} held-out subjects ({partition} partition)"
        ),
        metadata={"dataset": "bonn", "partition": partition},
    )


def _split_xray(root: Path, validation_ratio: float, test_ratio: float, seed: int) -> EvalSet:
    """Real chest X-ray held-out evaluation set (grouped by patient)."""
    from app.data.splitting import grouped_split_indices, grouped_split_indices_3way
    from app.data.xray_datasets import load_xray_benchmark

    dataset = load_xray_benchmark(root)
    if test_ratio > 0:
        _, _, eval_idx = grouped_split_indices_3way(
            dataset.labels, np.asarray(dataset.groups), validation_ratio, test_ratio, seed
        )
        partition = "test"
    else:
        _, val_idx = grouped_split_indices(
            dataset.labels, np.asarray(dataset.groups), validation_ratio, seed
        )
        eval_idx = val_idx
        partition = "validation"

    classes = list(dataset.class_names)
    samples = []
    for index in eval_idx:
        image = dataset.images[index][0]
        samples.append((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))

    return EvalSet(
        name="xray-real",
        modality="xray",
        samples=samples,
        labels=dataset.labels[eval_idx],
        class_names=classes,
        groups=np.asarray([dataset.groups[i] for i in eval_idx]),
        sampling_rate=0.0,
        description=(
            f"Real chest radiographs (NIH ChestX-ray14 + tuberculosis + COVID-19), "
            f"{len(eval_idx)} images from {len(set(dataset.groups[eval_idx]))} "
            f"held-out patients ({partition} partition)"
        ),
        metadata={"dataset": "xray-real", "partition": partition},
    )


def _synthetic(modality: str, per_class: int, seed: int) -> EvalSet:
    """Synthetic corpus generated with a seed range outside the training one."""
    from app.data.synthetic import build_ecg_corpus, build_eeg_corpus, build_xray_corpus

    if modality == "ecg":
        corpus = build_ecg_corpus(per_class=per_class, duration_s=10.0, seed=seed)
        sampling_rate = 500.0
    elif modality == "eeg":
        corpus = build_eeg_corpus(per_class=per_class, duration_s=10.0, seed=seed)
        sampling_rate = 256.0
    else:
        corpus = build_xray_corpus(per_class=per_class, seed=seed)
        sampling_rate = 0.0

    class_names = {
        "ecg": ["NORM", "AFIB", "MI", "STDE", "PVC"],
        "eeg": ["NORMAL", "EPILEPSY", "PARKINSON", "DEPRESSION"],
        "xray": ["NORMAL", "PNEUMONIA", "CONSOLIDATION", "EFFUSION", "TUBERCULOSIS", "COVID19"],
    }[modality]
    return EvalSet(
        name=f"synthetic-{modality}",
        modality=modality,
        samples=[item[0] for item in corpus],
        labels=np.asarray([item[1] for item in corpus], dtype=np.int64),
        class_names=class_names,
        groups=np.asarray([f"{modality}-{index}" for index in range(len(corpus))]),
        sampling_rate=sampling_rate,
        description=(
            f"Synthetic {modality.upper()} corpus, {per_class} samples per class, "
            f"seed {seed} (outside the training seed range)"
        ),
        metadata={"seed": seed, "dataset": "synthetic", "partition": "synthetic-holdout"},
    )


# ------------------------------------------------------------------------- models
@dataclass
class ModelVariant:
    """One model configuration to evaluate."""

    key: str
    label: str
    modality: str
    checkpoint: Optional[str]
    description: str


def _load_model(variant: ModelVariant):
    """Instantiate a modality model with the requested checkpoint."""
    factory = {"ecg": ECGModel, "eeg": EEGModel, "xray": XrayModel}[variant.modality]
    model = factory()
    if variant.checkpoint:
        path = Path(variant.checkpoint)
        if not path.exists():
            logger.warning("Checkpoint %s is missing - using the heuristic baseline", path)
            model.checkpoint_path = path
            model.network = None
        else:
            model.checkpoint_path = path
            model._load_checkpoint()
    else:
        # Drop the checkpoint the constructor may have picked up so the run
        # really exercises the deterministic clinical-rule baseline.
        model.checkpoint_path = Path("__no_checkpoint__")
        model.network = None
        logger.info("Using the heuristic baseline for %s", variant.modality)
    return model


# ----------------------------------------------------------------------- metrics
def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[int(true_index), int(pred_index)] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(matrix: np.ndarray, class_index: int) -> Dict[str, float]:
    """Precision / recall / F1 for one class from a confusion matrix."""
    tp = float(matrix[class_index, class_index])
    predicted = float(matrix[:, class_index].sum())
    actual = float(matrix[class_index, :].sum())
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, actual)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(actual),
    }


def _macro_auc(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest ROC AUC computed from the rank statistic."""
    aucs: List[float] = []
    for class_index in range(n_classes):
        positives = y_true == class_index
        n_pos = int(positives.sum())
        n_neg = int((~positives).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = np.argsort(np.argsort(probabilities[:, class_index])) + 1
        auc = (ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else 0.0


def _evaluate(model, dataset: EvalSet, zero_features: bool = False) -> Dict[str, Any]:
    """Run the model over a dataset and collect the metrics."""
    n_classes = len(dataset.class_names)
    probabilities = np.zeros((len(dataset), n_classes), dtype=np.float64)
    latencies: List[float] = []

    for index, sample in enumerate(dataset.samples):
        started = time.perf_counter()
        if dataset.modality == "xray":
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(sample, mode="L").save(buffer, format="PNG")
            vector = model.predict_proba(sample, {"bytes": buffer.getvalue()})
        elif dataset.modality == "ecg":
            vector = model.predict_proba(sample, {"sampling_rate": dataset.sampling_rate})
        else:
            vector = model.predict_proba(sample, {"sampling_rate": dataset.sampling_rate})
        latencies.append((time.perf_counter() - started) * 1000.0)
        probabilities[index] = np.asarray(vector, dtype=np.float64)[:n_classes]

    predictions = probabilities.argmax(axis=1)
    matrix = _confusion(dataset.labels, predictions, n_classes)

    per_class = {dataset.class_names[i]: _prf(matrix, i) for i in range(n_classes)}
    present = [i for i in range(n_classes) if matrix[i, :].sum() > 0]

    macro_precision = float(np.mean([per_class[dataset.class_names[i]]["precision"] for i in present]))
    macro_recall = float(np.mean([per_class[dataset.class_names[i]]["recall"] for i in present]))
    macro_f1 = float(np.mean([per_class[dataset.class_names[i]]["f1"] for i in present]))

    total = matrix.sum()
    weighted_f1 = float(
        sum(per_class[dataset.class_names[i]]["f1"] * matrix[i, :].sum() for i in present) / total
    ) if total else 0.0

    latencies_array = np.asarray(latencies)
    return {
        "dataset": dataset.name,
        "modality": dataset.modality,
        "n_samples": int(total),
        "n_groups": int(len(set(dataset.groups.tolist()))) if dataset.groups is not None else 0,
        "accuracy": float((predictions == dataset.labels).mean()) if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": _macro_auc(dataset.labels, probabilities, n_classes),
        "mean_latency_ms": float(latencies_array.mean()) if latencies_array.size else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_array, 95)) if latencies_array.size else 0.0,
        "per_class": per_class,
        "confusion": matrix,
        "probabilities": probabilities,
        "predictions": predictions,
    }


# ----------------------------------------------------------------------- output
def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _markdown_tables(results: List[Dict[str, Any]], datasets: Dict[str, EvalSet]) -> str:
    """Render the benchmark results as Markdown tables."""
    lines: List[str] = ["# MedDiag benchmark results", ""]
    lines.append("Generated by `scripts/run_benchmark.py`. All numbers are produced by the")
    lines.append("project's own evaluation code on the stated evaluation sets.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Dataset | Modality | Model | n | Groups | Partition | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Macro-AUC | Mean latency (ms) |"
    )
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        lines.append(
            "| {dataset} | {modality} | {model} | {n} | {groups} | {partition} | {acc:.4f} | {mp:.4f} | {mr:.4f} | "
            "{mf:.4f} | {wf:.4f} | {auc:.4f} | {lat:.1f} |".format(
                dataset=result["dataset"],
                modality=result["modality"],
                model=result["model_label"],
                n=result["n_samples"],
                groups=result["n_groups"],
                partition=result.get("partition", "n/a"),
                acc=result["accuracy"],
                mp=result["macro_precision"],
                mr=result["macro_recall"],
                mf=result["macro_f1"],
                wf=result["weighted_f1"],
                auc=result["macro_auc"],
                lat=result["mean_latency_ms"],
            )
        )
    lines.append("")

    for result in results:
        lines.append(f"### {result['dataset']} - {result['model_label']}")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, values in result["per_class"].items():
            if values["support"] == 0:
                continue
            lines.append(
                f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |"
            )
        lines.append("")
        matrix = result["confusion"]
        names = [name for name in datasets[result["dataset"]].class_names]
        lines.append("Confusion matrix (rows = true, columns = predicted):")
        lines.append("")
        lines.append("| true \\ predicted | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for index, name in enumerate(names):
            if matrix[index, :].sum() == 0:
                continue
            lines.append(
                f"| **{name}** | " + " | ".join(str(int(v)) for v in matrix[index]) + " |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MedDiag benchmark suite.")
    parser.add_argument(
        "--suite",
        choices=["synthetic", "real", "full"],
        default="full",
        help="Which evaluation sets to run",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--per-class", type=int, default=20, help="Synthetic samples per class")
    parser.add_argument("--seed", type=int, default=90210, help="Synthetic evaluation seed")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help=(
            "Fraction of grouped samples in the untouched test partition. "
            "Must match the --test-ratio used when the checkpoint was trained."
        ),
    )
    parser.add_argument("--mitbih-root", type=Path, default=ROOT / "data/datasets/mitdb")
    parser.add_argument("--bonn-root", type=Path, default=ROOT / "data/datasets/bonn")
    parser.add_argument("--xray-root", type=Path, default=ROOT / "data/datasets")
    args = parser.parse_args()

    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, EvalSet] = {}
    if args.suite in {"synthetic", "full"}:
        for modality in ("ecg", "eeg", "xray"):
            dataset = _synthetic(modality, args.per_class, args.seed)
            datasets[dataset.name] = dataset
            logger.info("Evaluation set %s: %d samples", dataset.name, len(dataset))

    if args.suite in {"real", "full"}:
        if args.mitbih_root.exists():
            dataset = _split_mitbih(args.mitbih_root, args.validation_ratio, args.test_ratio, 42)
            datasets[dataset.name] = dataset
            logger.info("Evaluation set %s: %d samples", dataset.name, len(dataset))
        else:
            logger.warning("MIT-BIH not found at %s - skipping", args.mitbih_root)
        if args.bonn_root.exists():
            dataset = _split_bonn(args.bonn_root, args.validation_ratio, args.test_ratio, 42)
            datasets[dataset.name] = dataset
            logger.info("Evaluation set %s: %d samples", dataset.name, len(dataset))
        else:
            logger.warning("Bonn not found at %s - skipping", args.bonn_root)
        if (args.xray_root / "nih_cxr").exists():
            dataset = _split_xray(args.xray_root, args.validation_ratio, args.test_ratio, 42)
            datasets[dataset.name] = dataset
            logger.info("Evaluation set %s: %d samples", dataset.name, len(dataset))
        else:
            logger.warning("Real chest X-ray data not found under %s - skipping", args.xray_root)

    variants = [
        ModelVariant("synthetic", "trained (synthetic corpus)", "ecg", str(ROOT / "models/ecg_model.pt"), "shipped synthetic-data checkpoint"),
        ModelVariant("synthetic", "trained (synthetic corpus)", "eeg", str(ROOT / "models/eeg_model.pt"), "shipped synthetic-data checkpoint"),
        ModelVariant("synthetic", "trained (synthetic corpus)", "xray", str(ROOT / "models/xray_model.pt"), "shipped synthetic-data checkpoint"),
        ModelVariant("real", "trained (real benchmark)", "ecg", str(ROOT / "models/ecg_model_mitbih.pt"), "checkpoint trained on MIT-BIH"),
        ModelVariant("real", "trained (real benchmark)", "eeg", str(ROOT / "models/eeg_model_bonn.pt"), "checkpoint trained on Bonn"),
        ModelVariant("real", "trained (real benchmark)", "xray", str(ROOT / "models/xray_model_nih.pt"), "checkpoint trained on real radiographs (NIH + TB + COVID-19)"),
        ModelVariant("heuristic", "heuristic baseline", "ecg", None, "deterministic clinical-rule scorer"),
        ModelVariant("heuristic", "heuristic baseline", "eeg", None, "deterministic clinical-rule scorer"),
        ModelVariant("heuristic", "heuristic baseline", "xray", None, "deterministic clinical-rule scorer"),
    ]

    results: List[Dict[str, Any]] = []
    for dataset in datasets.values():
        applicable = [v for v in variants if v.modality == dataset.modality]
        for variant in applicable:
            model = _load_model(variant)
            logger.info(
                "Evaluating %s on %s (%s)", variant.label, dataset.name, model.source
            )
            result = _evaluate(model, dataset)
            result["model"] = variant.key
            result["model_label"] = variant.label
            result["model_description"] = variant.description
            result["model_source"] = model.source
            result["partition"] = dataset.metadata.get("partition", "synthetic-holdout")
            results.append(result)

    # ------------------------------------------------------------------ write out
    summary_rows = []
    for result in results:
        summary_rows.append(
            [
                result["dataset"],
                result["modality"],
                result["model_label"],
                result["model_source"],
                result.get("partition", "n/a"),
                result["n_samples"],
                result["n_groups"],
                f"{result['accuracy']:.6f}",
                f"{result['macro_precision']:.6f}",
                f"{result['macro_recall']:.6f}",
                f"{result['macro_f1']:.6f}",
                f"{result['weighted_f1']:.6f}",
                f"{result['macro_auc']:.6f}",
                f"{result['mean_latency_ms']:.3f}",
                f"{result['p95_latency_ms']:.3f}",
            ]
        )
    _write_csv(
        args.out / "summary.csv",
        [
            "dataset",
            "modality",
            "model",
            "model_source",
            "partition",
            "n_samples",
            "n_groups",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "macro_auc",
            "mean_latency_ms",
            "p95_latency_ms",
        ],
        summary_rows,
    )

    for result in results:
        stem = f"{result['dataset']}__{result['model']}".replace(" ", "_")
        _write_csv(
            args.out / f"per_class_{stem}.csv",
            ["class", "precision", "recall", "f1", "support"],
            [
                [
                    name,
                    f"{values['precision']:.6f}",
                    f"{values['recall']:.6f}",
                    f"{values['f1']:.6f}",
                    values["support"],
                ]
                for name, values in result["per_class"].items()
            ],
        )
        names = datasets[result["dataset"]].class_names
        _write_csv(
            args.out / f"confusion_{stem}.csv",
            ["true\\predicted", *names],
            [
                [names[index], *[int(v) for v in result["confusion"][index]]]
                for index in range(len(names))
            ],
        )

    markdown = _markdown_tables(results, datasets)
    (args.out / "results.md").write_text(markdown, encoding="utf-8")
    (args.out / "results.json").write_text(
        json.dumps(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"probabilities", "predictions"}
                }
                | {"confusion": result["confusion"].tolist()}
                for result in results
            ],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n" + markdown)
    print(f"\nWrote benchmark artefacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
