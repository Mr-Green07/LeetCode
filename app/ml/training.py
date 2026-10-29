"""Shared training loop for all three modalities.

Keeps one implementation of the epoch loop, early stopping, metric
computation and checkpoint writing so the ECG, EEG and X-ray scripts only
differ in how they build their tensors.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass
class TrainConfig:
    """Hyper-parameters shared by every training run."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    validation_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: int = 42
    device: str = "auto"
    class_weights: bool = True
    label_smoothing: float = 0.05
    checkpoint_name: str = "model.pt"

    def resolve_device(self) -> str:
        if not TORCH_AVAILABLE:  # pragma: no cover
            return "cpu"
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable - using CPU")
            return "cpu"
        return self.device


def set_seed(seed: int) -> None:
    """Make a run reproducible."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def stratified_split(
    labels: np.ndarray, validation_ratio: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/validation while keeping the class balance."""
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * (1.0 - validation_ratio)))
        train_idx.extend(indices[:cut].tolist())
        val_idx.extend(indices[cut:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, n_classes: int) -> Dict[str, float]:
    """Accuracy, macro F1, per-class F1 and one-vs-rest ROC AUC."""
    metrics: Dict[str, float] = {}
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    metrics["accuracy"] = accuracy

    present_f1_scores: List[float] = []
    for label in range(n_classes):
        true_positive = float(np.sum((y_pred == label) & (y_true == label)))
        false_positive = float(np.sum((y_pred == label) & (y_true != label)))
        false_negative = float(np.sum((y_pred != label) & (y_true == label)))
        precision = true_positive / max(true_positive + false_positive, 1e-9)
        recall = true_positive / max(true_positive + false_negative, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        metrics[f"f1_{label}"] = f1
        # A class that does not occur in the evaluation set has an undefined
        # F1.  Averaging those zeros into the macro score would halve it for a
        # binary task that is scored against a four-class head, so the macro
        # figure only averages over the classes that are actually present -
        # the same convention the benchmark harness uses.
        if np.any(y_true == label):
            present_f1_scores.append(f1)
    metrics["f1_macro"] = float(np.mean(present_f1_scores)) if present_f1_scores else 0.0

    # macro one-vs-rest ROC AUC
    aucs: List[float] = []
    for label in range(n_classes):
        positives = y_true == label
        if positives.sum() == 0 or positives.sum() == y_true.size:
            continue
        scores = probabilities[:, label]
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = float(positives.sum())
        n_neg = float(len(scores) - n_pos)
        auc = (ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs.append(auc)
    metrics["auc"] = float(np.mean(aucs)) if aucs else 0.0
    return metrics


def train_network(
    network: Any,
    signals: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    config: TrainConfig,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Train a network and return ``(metrics, history)``.

    ``signals`` and ``features`` are the two input streams; the network is
    called as ``network(signals, features)``.

    ``groups`` identifies the recording or subject each sample came from.
    When supplied, the train/validation split is performed on the group so no
    subject is seen in both sides - a window-level split would leak near
    duplicates of the same recording and overstate the reported score.

    When ``config.test_ratio`` is greater than zero the grouped split becomes a
    three-way split: the **test** partition is carved out first and is never
    looked at during training, early stopping or model selection.  The metrics
    for that partition are returned under ``test_*`` keys so a reported number
    can come from data that influenced nothing.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training")

    set_seed(config.seed)
    device = config.resolve_device()

    test_idx: Optional[np.ndarray] = None
    if groups is not None:
        from app.data.splitting import grouped_split_indices, grouped_split_indices_3way

        if config.test_ratio > 0:
            train_idx, val_idx, test_idx = grouped_split_indices_3way(
                np.asarray(labels),
                np.asarray(groups),
                config.validation_ratio,
                config.test_ratio,
                config.seed,
            )
        else:
            train_idx, val_idx = grouped_split_indices(
                np.asarray(labels), np.asarray(groups), config.validation_ratio, config.seed
            )
    else:
        train_idx, val_idx = stratified_split(labels, config.validation_ratio, config.seed)
    signal_tensor = torch.from_numpy(np.asarray(signals, dtype=np.float32))
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    train_dataset = TensorDataset(signal_tensor[train_idx], feature_tensor[train_idx], label_tensor[train_idx])
    val_dataset = TensorDataset(signal_tensor[val_idx], feature_tensor[val_idx], label_tensor[val_idx])

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    network.to(device)
    best_metrics: Dict[str, float] = {}
    best_state = None
    best_epoch = -1
    best_score = (-1.0, float("inf"))
    patience_left = config.patience
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        started = time.perf_counter()
        network.train()
        train_loss = 0.0
        train_batches = 0
        for batch_signals, batch_features, batch_labels in train_loader:
            batch_signals = batch_signals.to(device)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_signals, batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        val_metrics = evaluate(network, val_loader, device, len(class_names))
        scheduler.step(val_metrics.get("accuracy", 0.0))
        elapsed = time.perf_counter() - started

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss / max(train_batches, 1),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "val_f1_macro": val_metrics.get("f1_macro", 0.0),
                "val_auc": val_metrics.get("auc", 0.0),
                "seconds": elapsed,
            }
        )
        logger.info(
            "epoch %2d/%d  loss=%.4f  val_acc=%.3f  val_f1=%.3f  auc=%.3f  (%.1fs)",
            epoch + 1,
            config.epochs,
            history[-1]["train_loss"],
            history[-1]["val_accuracy"],
            history[-1]["val_f1_macro"],
            history[-1]["val_auc"],
            elapsed,
        )

        # Rank by accuracy first and by validation loss second.  Accuracy alone
        # is not enough: once a model reaches 1.000 accuracy it can never
        # improve, so the saved checkpoint would stay stuck on the first epoch
        # that saturated it - which is often the least confident one.
        score = (val_metrics.get("accuracy", 0.0), -val_metrics.get("loss", float("inf")))
        if score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            best_epoch = epoch + 1
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()

    best_metrics["best_epoch"] = float(best_epoch)

    split_sizes = {
        "train_size": float(len(train_idx)),
        "validation_size": float(len(val_idx)),
        "test_size": float(len(test_idx)) if test_idx is not None else 0.0,
    }
    best_metrics.update(split_sizes)

    if test_idx is not None and len(test_idx) > 0:
        test_dataset = TensorDataset(
            signal_tensor[test_idx], feature_tensor[test_idx], label_tensor[test_idx]
        )
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        test_metrics = evaluate(network, test_loader, device, len(class_names))
        logger.info(
            "Untouched test partition: acc=%.4f f1=%.4f auc=%.4f (%d samples)",
            test_metrics.get("accuracy", 0.0),
            test_metrics.get("f1_macro", 0.0),
            test_metrics.get("auc", 0.0),
            len(test_idx),
        )
        for key, value in test_metrics.items():
            best_metrics[f"test_{key}"] = value

    return best_metrics, {"history": history, "class_names": list(class_names), "config": asdict(config)}


def evaluate(network: Any, loader: Any, device: str, n_classes: int) -> Dict[str, float]:
    """Evaluate a network on a data loader."""
    network.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    total_loss = 0.0
    batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_signals, batch_features, batch_labels in loader:
            batch_signals = batch_signals.to(device)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = network(batch_signals, batch_features)
            total_loss += float(criterion(logits, batch_labels).item())
            batches += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)
            y_pred.append(probs.argmax(axis=1))
            y_true.append(batch_labels.cpu().numpy())

    if not y_true:
        return {"accuracy": 0.0, "f1_macro": 0.0, "auc": 0.0, "loss": 0.0}

    metrics = compute_metrics(
        np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(probabilities), n_classes
    )
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics


def save_checkpoint(
    network: Any,
    path: Path,
    class_names: Sequence[str],
    architecture: str,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a checkpoint plus a human-readable metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": list(class_names),
        "architecture": architecture,
        "state_dict": network.state_dict(),
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "metadata": metadata or {},
    }
    torch.save(payload, str(path))

    sidecar = path.with_suffix(".metrics.json")
    sidecar.write_text(json.dumps(payload["metrics"], indent=2), encoding="utf-8")
    logger.info("Checkpoint written to %s", path)
    return path
