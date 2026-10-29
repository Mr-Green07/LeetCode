"""Central model registry.

Models are instantiated lazily (a single ECG checkpoint can take seconds to
load) and then cached for the lifetime of the process.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from app.config import Modality
from app.core.logging import get_logger
from app.ml.base import ModalityModel
from app.ml.ecg.inference import ECGModel
from app.ml.eeg.inference import EEGModel
from app.ml.xray.inference import XrayModel

logger = get_logger(__name__)

_FACTORIES = {
    "ecg": ECGModel,
    "eeg": EEGModel,
    "xray": XrayModel,
}


class ModelRegistry:
    """Thread-safe lazy registry of the per-modality models."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, ModalityModel] = {}

    def get(self, modality: Modality) -> ModalityModel:
        """Return (and cache) the model for ``modality``."""
        key = str(modality)
        if key not in _FACTORIES:
            raise ValueError(f"Unknown modality: {modality}")
        with self._lock:
            if key not in self._models:
                logger.info("Loading %s model...", key)
                self._models[key] = _FACTORIES[key]()
            return self._models[key]

    def load_all(self) -> None:
        """Eagerly instantiate every model (used at startup)."""
        for modality in _FACTORIES:
            try:
                self.get(modality)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load %s model: %s", modality, exc)

    def info(self) -> Dict[str, dict]:
        """Metadata for every registered model."""
        result: Dict[str, dict] = {}
        for modality in _FACTORIES:
            try:
                result[modality] = self.get(modality).info()  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive
                result[modality] = {"modality": modality, "error": str(exc)}
        return result

    def available_modalities(self) -> List[str]:
        return list(_FACTORIES.keys())


registry = ModelRegistry()
