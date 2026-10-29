"""Shared model plumbing.

Every modality exposes the same contract:

* load a checkpoint if one exists, otherwise fall back to a deterministic
  heuristic scorer so the API is always usable;
* expose ``predict_proba`` returning a class-probability vector;
* report metadata (architecture, classes, metrics) for the dashboard.

The heuristic fallback is clearly reported through ``model_source`` so a
deployment can never mistake it for a trained network.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:  # torch is optional at import time so the API still boots without it
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on torch-less installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def resolve_device() -> str:
    """Pick the torch device honouring the ``MEDDIAG_DEVICE`` setting."""
    if not TORCH_AVAILABLE:
        return "cpu"
    choice = settings.device
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return choice


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class ModalityModel(ABC):
    """Base class shared by the ECG, EEG and X-ray models."""

    modality: str = "base"
    classes: Sequence[str] = ()
    architecture: str = "unknown"
    input_description: str = ""

    def __init__(self, checkpoint_name: str, *, prefer: Sequence[str] = ()) -> None:
        """Create the model and load its checkpoint.

        ``prefer`` lists checkpoint file names that are tried **before** the
        default one.  It exists so a model trained on real public data can
        take precedence over a synthetic one while the synthetic checkpoint
        stays on disk for the benchmark.  The first file that exists wins;
        when none exists the model stays on the heuristic baseline.
        """
        self.device = resolve_device()
        self.default_checkpoint_name = checkpoint_name
        self.default_checkpoint_path: Path = Path(settings.model_dir) / checkpoint_name
        self.checkpoint_path: Optional[Path] = None
        for name in [*prefer, checkpoint_name]:
            candidate = Path(settings.model_dir) / name
            if candidate.exists():
                self.checkpoint_path = candidate
                break
        if self.checkpoint_path is None:
            self.checkpoint_path = self.default_checkpoint_path
        self.network: Any = None
        self.metrics: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_checkpoint()

    # ------------------------------------------------------------ lifecycle
    @property
    def is_trained(self) -> bool:
        """True when a real neural checkpoint backs this model."""
        return self.network is not None

    @property
    def source(self) -> str:
        return "trained" if self.is_trained else "heuristic_baseline"

    def _load_checkpoint(self) -> None:
        """Load a checkpoint if present; otherwise stay on the heuristic path."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable - %s runs on the heuristic baseline", self.modality)
            return
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            logger.info("No checkpoint at %s - %s runs on the heuristic baseline", self.checkpoint_path, self.modality)
            return
        try:
            state = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
            self._build_and_load(state)
            logger.info("Loaded %s checkpoint %s", self.modality, self.checkpoint_path)
        except Exception as exc:  # pragma: no cover - corrupted checkpoint
            logger.error("Failed to load checkpoint %s: %s", self.checkpoint_path, exc)
            self.network = None

    def _build_and_load(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError

    def save_checkpoint(self, path: Optional[Path] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Persist weights (and metadata) for later inference."""
        if self.network is None:
            raise RuntimeError(f"{self.modality} model has no trainable network")
        target = Path(path or self.default_checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classes": list(self.classes),
            "architecture": self.architecture,
            "state_dict": self.network.state_dict(),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, str(target))
        return target

    # ------------------------------------------------------------ inference
    @abstractmethod
    def predict_proba(self, samples: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Return a probability vector of shape ``(len(self.classes),)``.

        ``context`` carries modality-specific inputs such as the sampling rate.
        """

    def predict(self, samples: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Return a label -> probability mapping."""
        proba = np.asarray(self.predict_proba(samples, context), dtype=np.float64).reshape(-1)
        total = proba.sum()
        if total <= 0:
            proba = np.full(len(self.classes), 1.0 / len(self.classes))
        else:
            proba = proba / total
        return {label: float(p) for label, p in zip(self.classes, proba)}

    # ------------------------------------------------------------- metadata
    @property
    def parameter_count(self) -> int:
        if self.network is None:
            return 0
        return int(sum(p.numel() for p in self.network.parameters()))

    def info(self) -> Dict[str, Any]:
        """Metadata payload used by ``/api/v1/models``."""
        return {
            "modality": self.modality,
            "architecture": self.architecture,
            "version": self.metadata.get("version", "1.0.0"),
            "source": self.source,
            "classes": list(self.classes),
            "input_shape": list(self.metadata.get("input_shape", [])),
            "parameters": self.parameter_count,
            "checkpoint_path": str(self.checkpoint_path) if self.is_trained else None,
            "trained": self.is_trained,
            "accuracy": self.metrics.get("accuracy"),
            "auc": self.metrics.get("auc"),
            "metrics": {k: float(v) for k, v in self.metrics.items() if isinstance(v, (int, float))},
        }

    @staticmethod
    def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
        """Dump a sidecar metadata JSON next to a checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
