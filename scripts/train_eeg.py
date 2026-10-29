"""Train the EEG diagnostic model.

Examples
--------
    python scripts/train_eeg.py --dataset synthetic --per-class 140 --epochs 25
    python scripts/train_eeg.py --dataset bonn   --data-root D:/datasets/bonn
    python scripts/train_eeg.py --dataset folder --data-root D:/datasets/my_eeg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.ml.eeg.features import extract_features, feature_vector  # noqa: E402
from app.ml.eeg.model import EEG_CLASSES, EEGNet  # noqa: E402
from app.ml.eeg.preprocess import build_model_input, preprocess_eeg  # noqa: E402
from app.ml.training import TrainConfig, save_checkpoint, set_seed, train_network  # noqa: E402

logger = get_logger("train_eeg")


def build_synthetic(per_class: int, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the synthetic corpus into (signals, features, labels)."""
    from app.data.synthetic import build_eeg_corpus

    corpus = build_eeg_corpus(per_class=per_class, duration_s=duration_s)
    signals: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for index, (signal, label, name) in enumerate(corpus):
        pre = preprocess_eeg(signal, 256.0)
        features = extract_features(pre)
        signals.append(build_model_input(pre.window[0]))
        feature_rows.append(feature_vector(features))
        labels.append(label)
        if (index + 1) % 50 == 0:
            logger.info("prepared %d/%d samples", index + 1, len(corpus))

    return np.stack(signals), np.stack(feature_rows), np.asarray(labels, dtype=np.int64)


def build_from_folder(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a ``<root>/<LABEL>/<file>.csv`` corpus."""
    from app.data.eeg_datasets import load_folder_dataset

    dataset = load_folder_dataset(root)
    signals: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for index in range(len(dataset)):
        pre = preprocess_eeg(dataset.signals[index][0], 256.0)
        features = extract_features(pre)
        signals.append(build_model_input(pre.window[0]))
        feature_rows.append(feature_vector(features))
        labels.append(int(dataset.labels[index]))

    return np.stack(signals), np.stack(feature_rows), np.asarray(labels, dtype=np.int64)


def build_from_dataset(name: str, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Preprocess a public benchmark.

    Returns ``(signals, features, labels, groups)``.  ``groups`` identifies the
    source subject so the train/validation split cannot leak a subject.
    """
    from app.data import eeg_datasets

    if name == "bonn":
        dataset = eeg_datasets.load_bonn(root)
    elif name == "mne":
        dataset = eeg_datasets.load_mne_sample()
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    signals: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    # Use the dataset's native sampling rate so the band powers and spectral
    # features are computed against the correct time base.
    sampling_rate = float(dataset.metadata.get("sampling_rate", 256.0))

    for index in range(len(dataset)):
        pre = preprocess_eeg(dataset.signals[index][0], sampling_rate)
        features = extract_features(pre)
        signals.append(build_model_input(pre.window[0]))
        feature_rows.append(feature_vector(features))
        labels.append(int(dataset.labels[index]))

    logger.info("Preprocessed %d %s recordings at %.0f Hz", len(dataset), name, sampling_rate)
    return (
        np.stack(signals),
        np.stack(feature_rows),
        np.asarray(labels, dtype=np.int64),
        dataset.groups,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the MedDiag EEG model.")
    parser.add_argument("--dataset", choices=["synthetic", "folder", "bonn", "mne"], default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=160)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Fraction of grouped samples held out as an untouched test partition",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path(settings.model_dir) / "eeg_model.pt")
    args = parser.parse_args()

    setup_logging()
    logger.info("Preparing the EEG training set (%s)...", args.dataset)

    # Seed *before* the network is constructed: the weight initialisation is
    # drawn from the global RNG, so seeding inside the training loop would be
    # too late and every run would start from different weights.
    set_seed(args.seed)

    groups: np.ndarray | None = None
    if args.dataset == "synthetic":
        signals, features, labels = build_synthetic(args.per_class, args.duration)
    elif args.dataset == "folder":
        if not args.data_root:
            parser.error("--data-root is required for the folder dataset")
        signals, features, labels = build_from_folder(args.data_root)
    else:
        if not args.data_root and args.dataset != "mne":
            parser.error("--data-root is required for this dataset")
        signals, features, labels, groups = build_from_dataset(args.dataset, args.data_root)

    logger.info("Training set: %d samples, signal %s, features %s", len(signals), signals.shape[1:], features.shape[1])
    if groups is not None:
        logger.info("Grouped split over %d source subjects", len(set(groups.tolist())))
    elif args.test_ratio > 0:
        logger.warning(
            "--test-ratio needs a grouped dataset (Bonn); it is ignored for %s",
            args.dataset,
        )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        device=args.device,
        checkpoint_name=args.output.name,
    )
    network = EEGNet(num_classes=len(EEG_CLASSES))
    metrics, history = train_network(network, signals, features, labels, EEG_CLASSES, config, groups=groups)

    logger.info("Best validation metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    save_checkpoint(
        network,
        args.output,
        class_names=EEG_CLASSES,
        architecture="EEGNet + clinical-feature fusion head",
        metrics=metrics,
        metadata={
            "dataset": args.dataset,
            "input_shape": [1, 1, int(signals.shape[3])],
            "version": "1.0.0",
            "grouped_split": groups is not None,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "history": history["history"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
