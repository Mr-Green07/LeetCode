"""EEG inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.logging import get_logger
from app.ml.base import TORCH_AVAILABLE, ModalityModel, softmax
from app.ml.eeg.features import (
    FEATURE_ORDER,
    FeatureValue,
    extract_features,
    feature_importance_hint,
    feature_vector,
)
from app.ml.eeg.model import CLASS_DESCRIPTIONS, CLASS_SEVERITY, EEG_CLASSES, EEGNet
from app.ml.eeg.preprocess import PreprocessedEEG, build_model_input, preprocess_eeg

logger = get_logger(__name__)


@dataclass
class EEGAnalysis:
    """Everything the API needs to describe one EEG analysis."""

    probabilities: Dict[str, float]
    label: str
    confidence: float
    severity: str
    description: str
    features: Dict[str, FeatureValue]
    preprocessed: PreprocessedEEG
    quality: Dict[str, float]
    class_relevance: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class EEGModel(ModalityModel):
    """EEG diagnostic model (EEGNet + clinical-feature fusion)."""

    modality = "eeg"
    classes = EEG_CLASSES
    architecture = "EEGNet + clinical-feature fusion head"

    def __init__(self) -> None:
        super().__init__("eeg_model.pt")
        self.metadata.setdefault("input_shape", [1, 1, 2560])
        self.metadata.setdefault("version", "1.0.0")

    # ------------------------------------------------------------- lifecycle
    def _build_and_load(self, state: Dict[str, Any]) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load the EEG checkpoint")
        import torch

        self.classes = list(state.get("classes", list(self.classes)))
        self.metrics = {k: float(v) for k, v in (state.get("metrics") or {}).items() if isinstance(v, (int, float))}
        self.metadata.update(state.get("metadata") or {})

        network = EEGNet(num_classes=len(self.classes), n_features=len(FEATURE_ORDER))
        network.load_state_dict(state["state_dict"])
        network.eval()
        self.network = network.to(self.device)

    # ------------------------------------------------------------- inference
    def predict_proba(self, samples: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        context = context or {}
        fs = float(context.get("sampling_rate", 256.0))
        pre = preprocess_eeg(samples, fs, channel_names=context.get("channel_names"))
        features = extract_features(pre)
        return self._network_proba(pre, features)

    def _network_proba(self, pre: PreprocessedEEG, features: Dict[str, FeatureValue]) -> np.ndarray:
        if self.network is None:
            return self._heuristic_proba(features, pre.quality.get("sqi", 1.0))
        import torch

        window = build_model_input(pre.clean)
        fvec = feature_vector(features)

        self.network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            feat_tensor = torch.from_numpy(fvec).float().unsqueeze(0).to(self.device)
            logits = self.network(tensor, feat_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        return proba

    # ------------------------------------------------------------- heuristic
    def _heuristic_proba(self, features: Dict[str, FeatureValue], sqi: float) -> np.ndarray:
        """Clinical-rule scoring used until a checkpoint is trained."""
        def value(name: str, default: float = 0.0) -> float:
            raw = features[name].value if name in features else default
            return default if raw is None or not np.isfinite(raw) else float(raw)

        swi = value("spike_wave_index")
        entropy = value("spectral_entropy")
        alpha = value("alpha_power")
        beta = value("beta_power")
        theta = value("theta_power")
        theta_beta = value("theta_beta_ratio")
        artifact = value("artifact_ratio")

        # epilepsy: discrete spike-and-wave discharges plus a more organised
        # (lower entropy) background during the discharge
        epilepsy = 3.0 * max(0.0, swi - 0.4) / 1.0 + 2.2 * max(0.0, 0.78 - entropy) / 0.2
        # parkinsonism: beta dominance and a raised beta/alpha ratio
        beta_alpha = beta / max(alpha, 1e-6)
        parkinson = 2.6 * max(0.0, beta - 60.0) / 50.0 + 2.2 * max(0.0, beta_alpha - 1.5) / 4.0
        # depressive pattern: raised theta and a high theta/beta ratio
        depression = 2.8 * max(0.0, theta_beta - 1.5) / 2.0 + 1.8 * max(0.0, theta - 70.0) / 70.0

        normal = 2.6 - 0.25 * (epilepsy + parkinson + depression)

        reliability = float(np.clip(sqi, 0.1, 1.0))
        epilepsy, parkinson, depression = (
            epilepsy * reliability,
            parkinson * reliability,
            depression * reliability,
        )
        if artifact > 20.0:
            normal += 1.0

        scores = np.array([normal, epilepsy, parkinson, depression], dtype=np.float64)
        return softmax(scores)

    # -------------------------------------------------------------- analysis
    def analyze(
        self,
        samples,
        sampling_rate: float,
        *,
        channel_names: Optional[List[str]] = None,
        unit: str = "auto",
    ) -> EEGAnalysis:
        """Run the full EEG analysis pipeline."""
        pre = preprocess_eeg(samples, sampling_rate, channel_names=channel_names, unit=unit)
        features = extract_features(pre)
        proba = np.asarray(self._network_proba(pre, features), dtype=np.float64).reshape(-1)
        if proba.size != len(self.classes):
            proba = np.resize(proba, len(self.classes))
        proba = proba / max(proba.sum(), 1e-12)

        best = int(np.argmax(proba))
        label = self.classes[best]
        relevance = self._class_relevance(features)

        return EEGAnalysis(
            probabilities={cls: float(p) for cls, p in zip(self.classes, proba)},
            label=label,
            confidence=float(proba[best]),
            severity=CLASS_SEVERITY.get(label, "normal"),
            description=CLASS_DESCRIPTIONS.get(label, ""),
            features=features,
            preprocessed=pre,
            quality=pre.quality,
            class_relevance=relevance,
            recommendations=self._recommendations(label, features, relevance),
        )

    # --------------------------------------------------------- explainability
    def _class_relevance(self, features: Dict[str, FeatureValue]) -> Dict[str, float]:
        scored = {name: value for name, value, _ in feature_importance_hint(features)}
        total = sum(scored.values()) or 1.0

        def share(names: List[str]) -> float:
            return float(min(1.0, sum(scored.get(n, 0.0) for n in names) / total))

        relevance = {
            "EPILEPSY": round(share(["spike_wave_index", "gamma_power", "spectral_entropy"]), 4),
            "PARKINSON": round(share(["alpha_power", "beta_power", "signal_variance"]), 4),
            "DEPRESSION": round(share(["theta_beta_ratio", "alpha_asymmetry", "relative_alpha"]), 4),
        }
        relevance["NORMAL"] = round(1.0 - min(0.95, sum(relevance.values())), 4)
        return relevance

    def _recommendations(self, label: str, features: Dict[str, FeatureValue], relevance: Dict[str, float]) -> List[str]:
        advice: List[str] = []
        if label == "NORMAL":
            advice.append("No neurological abnormality pattern detected in this recording.")
            advice.append("Continue routine monitoring; repeat EEG if clinical symptoms evolve.")
        elif label == "EPILEPSY":
            advice.append("Epileptiform activity detected - review with neurology and consider video-EEG monitoring.")
            advice.append("Correlate with clinical seizure semiology; this is not a seizure detector and may miss focal events.")
        elif label == "PARKINSON":
            advice.append("Reduced alpha with dominant beta activity is consistent with a parkinsonian pattern.")
            advice.append("Recommend clinical neurological examination and dopaminergic imaging where indicated.")
        elif label == "DEPRESSION":
            advice.append("Elevated theta/beta ratio suggests a depressive EEG pattern.")
            advice.append("Use only as supportive evidence alongside a validated clinical mood assessment.")

        if features["artifact_ratio"].value > 15:
            advice.append("Artefact load is high - repeat the recording with better electrode impedance before acting.")
        if features["signal_quality"].value < 0.6:
            advice.append("Signal quality is limited - findings should be treated as indicative only.")
        advice.append("Findings must be interpreted by a qualified clinician against the full patient record.")
        return advice
