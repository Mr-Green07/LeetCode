"""Train the chest X-ray diagnostic model.

Examples
--------
    python scripts/train_xray.py --dataset synthetic --per-class 120 --epochs 20
    python scripts/train_xray.py --dataset nih   --data-root D:/datasets/chestxray14
    python scripts/train_xray.py --dataset folder --data-root D:/datasets/my_xray
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.ml.training import TrainConfig, save_checkpoint, set_seed, train_network  # noqa: E402
from app.ml.xray.features import extract_features, feature_vector  # noqa: E402
from app.ml.xray.model import XRAY_CLASSES, XrayCNN  # noqa: E402
from app.ml.xray.preprocess import build_model_input, preprocess_xray_array  # noqa: E402

logger = get_logger("train_xray")


def build_synthetic(per_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the synthetic corpus into (images, features, labels)."""
    from app.data.synthetic import build_xray_corpus

    corpus = build_xray_corpus(per_class=per_class)
    images: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for index, (image, label, name) in enumerate(corpus):
        pre = preprocess_xray_array(image)
        features = extract_features(pre)
        images.append(build_model_input(pre.grayscale))
        feature_rows.append(feature_vector(features))
        labels.append(label)
        if (index + 1) % 50 == 0:
            logger.info("prepared %d/%d samples", index + 1, len(corpus))

    return np.stack(images), np.stack(feature_rows), np.asarray(labels, dtype=np.int64)


def build_from_dataset(
    name: str, root: Path, limit: int | None, per_class_limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Preprocess a public benchmark.

    Returns ``(images, features, labels, groups)``.  ``groups`` identifies the
    patient so the split cannot leak one patient into both sides.
    """
    from app.data import xray_datasets

    if name == "nih":
        dataset = xray_datasets.load_nih_chestxray14(root, limit=limit)
    elif name == "benchmark":
        dataset = xray_datasets.load_xray_benchmark(root, limit_per_class=per_class_limit)
    else:
        dataset = xray_datasets.load_folder_dataset(root)

    images: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for index in range(len(dataset)):
        pre = preprocess_xray_array(dataset.images[index][0] * 255.0)
        features = extract_features(pre)
        images.append(build_model_input(pre.grayscale))
        feature_rows.append(feature_vector(features))
        labels.append(int(dataset.labels[index]))
        if dataset.groups:
            groups.append(dataset.groups[index])
        if (index + 1) % 200 == 0:
            logger.info("prepared %d/%d samples", index + 1, len(dataset))

    return (
        np.stack(images),
        np.stack(feature_rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=object) if groups else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the MedDiag chest X-ray model.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "folder", "nih", "benchmark"],
        default="synthetic",
        help=(
            "'benchmark' merges the NIH ChestX-ray14 shards with the "
            "tuberculosis and COVID-19 sets (the real-data model)."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=140)
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=None,
        help="Cap the number of real radiographs taken from each class",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Fraction of grouped samples held out as an untouched test partition",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path(settings.model_dir) / "xray_model.pt")
    args = parser.parse_args()

    setup_logging()
    logger.info("Preparing the X-ray training set (%s)...", args.dataset)

    # Seed *before* the network is constructed: the weight initialisation is
    # drawn from the global RNG, so seeding inside the training loop would be
    # too late and every run would start from different weights.
    set_seed(args.seed)

    groups: np.ndarray | None = None
    if args.dataset == "synthetic":
        images, features, labels = build_synthetic(args.per_class)
    else:
        if not args.data_root:
            parser.error("--data-root is required for this dataset")
        images, features, labels, groups = build_from_dataset(
            args.dataset, args.data_root, args.limit, args.per_class_limit
        )

    logger.info("Training set: %d samples, images %s, features %s", len(images), images.shape[1:], features.shape[1])
    if groups is not None:
        logger.info("Grouped split over %d source patients/subjects", len(set(groups.tolist())))

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
    network = XrayCNN(num_classes=len(XRAY_CLASSES))
    metrics, history = train_network(
        network, images, features, labels, XRAY_CLASSES, config, groups=groups
    )

    logger.info("Best validation metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    save_checkpoint(
        network,
        args.output,
        class_names=XRAY_CLASSES,
        architecture="XrayCNN (4-stage) + clinical-feature fusion head",
        metrics=metrics,
        metadata={
            "dataset": args.dataset,
            "input_shape": [1, int(images.shape[2]), int(images.shape[3])],
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
