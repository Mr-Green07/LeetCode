"""ECG inference orchestration.

Combines conditioning, feature extraction, the neural network and the
heuristic baseline into one analysis object the API layer can render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.logging import get_logger
from app.ml.base import TORCH_AVAILABLE, ModalityModel, softmax
from app.ml.ecg.features import (
    FEATURE_ORDER,
    FeatureValue,
    extract_features,
    feature_importance_hint,
    feature_vector,
)
from app.ml.ecg.model import CLASS_DESCRIPTIONS, CLASS_SEVERITY, ECG_CLASSES, ECGNet
from app.ml.ecg.preprocess import PreprocessedECG, build_model_input, preprocess_ecg

logger = get_logger(__name__)

ANALYSIS_LEAD_PREFERENCE = ("ii", "mlii", "lead ii", "lead2", "i", "ml")


@dataclass
class ECGAnalysis:
    """Everything the API needs to describe one ECG analysis."""

    probabilities: Dict[str, float]
    label: str
    confidence: float
    severity: str
    description: str
    features: Dict[str, FeatureValue]
    preprocessed: PreprocessedECG
    analysis_lead_index: int
    quality: Dict[str, float]
    heuristic_scores: Dict[str, float] = field(default_factory=dict)
    class_relevance: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ECGModel(ModalityModel):
    """ECG diagnostic model (ResNet1D + clinical-feature fusion)."""

    modality = "ecg"
    classes = ECG_CLASSES
    architecture = "ResNet1D + clinical-feature fusion head"

    def __init__(self) -> None:
        super().__init__("ecg_model.pt")
        self.metadata.setdefault("input_shape", [1, 2500])
        self.metadata.setdefault("version", "1.0.0")

    # ------------------------------------------------------------- lifecycle
    def _build_and_load(self, state: Dict[str, Any]) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load the ECG checkpoint")
        import torch

        classes = state.get("classes", list(self.classes))
        self.classes = list(classes)
        self.metrics = {k: float(v) for k, v in (state.get("metrics") or {}).items() if isinstance(v, (int, float))}
        self.metadata.update(state.get("metadata") or {})

        network = ECGNet(num_classes=len(self.classes), n_features=len(FEATURE_ORDER))
        network.load_state_dict(state["state_dict"])
        network.eval()
        self.network = network.to(self.device)

    # ------------------------------------------------------------- inference
    def predict_proba(self, samples: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        context = context or {}
        fs = float(context.get("sampling_rate", 500.0))
        lead_names = context.get("lead_names")
        pre = preprocess_ecg(samples, fs, lead_names=lead_names)
        features = extract_features(pre)
        return self._network_proba(pre, features)

    def _network_proba(self, pre: PreprocessedECG, features: Dict[str, FeatureValue]) -> np.ndarray:
        """Run the trained network (or the heuristic fallback)."""
        if self.network is None:
            return self._heuristic_proba(features, pre.quality.get("sqi", 1.0))
        import torch

        lead_idx = self._pick_analysis_lead(pre)
        window = build_model_input(pre.window[lead_idx])
        fvec = feature_vector(features)

        self.network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            feat_tensor = torch.from_numpy(fvec).float().unsqueeze(0).to(self.device)
            logits = self.network(tensor, feat_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        return proba

    @staticmethod
    def _pick_analysis_lead(pre: PreprocessedECG) -> int:
        """Choose the lead that carries the diagnostic information."""
        for wanted in ANALYSIS_LEAD_PREFERENCE:
            for idx, name in enumerate(pre.lead_names):
                if name.strip().lower() == wanted:
                    return idx
        return 0

    # ------------------------------------------------------------- heuristic
    def _heuristic_proba(self, features: Dict[str, FeatureValue], sqi: float) -> np.ndarray:
        """Rule-based scorer used until a checkpoint is trained.

        Scores are clinically-motivated log-odds contributions, not a substitute
        for a validated model - the API always reports ``heuristic_baseline``.
        """
        def value(name: str, default: float = 0.0) -> float:
            raw = features[name].value if name in features else default
            return default if raw is None or not np.isfinite(raw) else float(raw)

        rr_irregularity = value("rr_irregularity")
        pnn50 = value("pnn50")
        rr_cv = value("rr_cv")
        pvc = value("pvc_fraction")
        st_dev = value("st_deviation")
        heart_rate = value("heart_rate", 70.0)
        r_amp = value("r_amplitude", 1.0)
        p_amp = abs(value("p_wave_amplitude", 0.1))

        # atrial fibrillation needs an irregular rhythm AND absent P waves;
        # the P-wave factor keeps the irregular rhythm of frequent ectopy
        # from being reported as AFIB
        p_presence = float(np.clip(p_amp / 0.06, 0.0, 1.0))
        afib_raw = (
            2.6 * max(0.0, rr_irregularity - 3.5) / 10.0
            + 1.8 * max(0.0, pnn50 - 12.0) / 40.0
            + 1.2 * max(0.0, rr_cv - 6.0) / 20.0
        )
        afib = afib_raw * (1.0 - 0.8 * p_presence)

        pvc_score = 3.6 * max(0.0, pvc - 5.0) / 25.0
        mi = 3.4 * max(0.0, -0.03 - st_dev) / 0.05 + 0.8 * max(0.0, 1.0 - r_amp) / 0.6
        stde = 3.6 * max(0.0, st_dev - 0.03) / 0.05
        brady = 0.8 * max(0.0, 55.0 - heart_rate) / 25.0
        tachy = 0.6 * max(0.0, heart_rate - 105.0) / 60.0

        # poor signal suppresses confident abnormality calls
        reliability = float(np.clip(sqi, 0.1, 1.0))
        afib, mi, stde, pvc_score = (
            afib * reliability,
            mi * reliability,
            stde * reliability,
            pvc_score * reliability,
        )

        normal = 2.4 - 0.25 * (afib + mi + stde + pvc_score) + brady * 0.3 + tachy * 0.3
        if not reliability > 0.6:
            normal += 1.2

        scores = np.array([normal, afib, mi, stde, pvc_score], dtype=np.float64)
        return softmax(scores)

    # -------------------------------------------------------------- analysis
    def analyze(
        self,
        samples,
        sampling_rate: float,
        *,
        lead_names: Optional[List[str]] = None,
        unit: str = "auto",
    ) -> ECGAnalysis:
        """Run the full ECG analysis pipeline."""
        pre = preprocess_ecg(samples, sampling_rate, lead_names=lead_names, unit=unit)
        features = extract_features(pre)
        proba = self._network_proba(pre, features)
        proba = np.asarray(proba, dtype=np.float64).reshape(-1)
        if proba.size != len(self.classes):
            proba = np.resize(proba, len(self.classes))
        proba = proba / max(proba.sum(), 1e-12)

        best = int(np.argmax(proba))
        label = self.classes[best]
        confidence = float(proba[best])
        lead_idx = self._pick_analysis_lead(pre)

        relevance = self._class_relevance(features)
        recommendations = self._recommendations(label, features, relevance)

        return ECGAnalysis(
            probabilities={cls: float(p) for cls, p in zip(self.classes, proba)},
            label=label,
            confidence=confidence,
            severity=CLASS_SEVERITY.get(label, "normal"),
            description=CLASS_DESCRIPTIONS.get(label, ""),
            features=features,
            preprocessed=pre,
            analysis_lead_index=lead_idx,
            quality=pre.quality,
            heuristic_scores={cls: float(p) for cls, p in zip(self.classes, proba)},
            class_relevance=relevance,
            recommendations=recommendations,
        )

    # --------------------------------------------------------- explainability
    def _class_relevance(self, features: Dict[str, FeatureValue]) -> Dict[str, float]:
        """Per-class relevance derived from feature deviation profiles."""
        relevance: Dict[str, float] = {}
        scored = dict((name, value) for name, value, _ in feature_importance_hint(features))
        total = sum(scored.values()) or 1.0

        def share(names: List[str]) -> float:
            return float(min(1.0, sum(scored.get(n, 0.0) for n in names) / total))

        relevance["AFIB"] = round(share(["rr_irregularity", "pnn50", "rr_cv", "rr_sd"]), 4)
        relevance["MI"] = round(share(["st_deviation", "r_amplitude", "t_amplitude"]), 4)
        relevance["STDE"] = round(share(["st_deviation"]), 4)
        relevance["PVC"] = round(share(["pvc_fraction", "rr_sd"]), 4)
        relevance["NORM"] = round(1.0 - min(0.95, sum(v for k, v in relevance.items() if k != "NORM")), 4)
        return relevance

    def _recommendations(
        self, label: str, features: Dict[str, FeatureValue], relevance: Dict[str, float]
    ) -> List[str]:
        """Guidance text derived from the model output and the findings."""
        advice: List[str] = []
        if label == "NORM":
            advice.append("No significant abnormality detected in this recording.")
            advice.append("Correlate with symptoms; repeat ECG if symptoms persist or change.")
        elif label == "AFIB":
            advice.append("Rhythm is irregularly irregular - confirm atrial fibrillation with a 12-lead ECG.")
            advice.append("Check rate control, thromboembolic risk (CHA2DS2-VASc) and anticoagulation eligibility.")
        elif label == "MI":
            advice.append("ST depression suggests ischaemia - obtain serial ECG and troponin, and review urgently.")
            advice.append("Escalate to a cardiology team if symptoms are active or the ST changes are dynamic.")
        elif label == "STDE":
            advice.append("ST elevation raises concern for acute myocardial injury - urgent cardiology review is advised.")
            advice.append("Repeat the ECG and compare with prior recordings to confirm the finding.")
        elif label == "PVC":
            advice.append("Ectopic ventricular activity detected - assess burden, electrolytes and ventricular function.")

        if features["signal_quality"].value < 0.6:
            advice.append("Signal quality is limited - repeat the recording with better electrode contact before acting.")
        if features["heart_rate"].value and features["heart_rate"].value > 110:
            advice.append("Tachycardia present - correlate with haemodynamic status and clinical context.")
        if features["heart_rate"].value and features["heart_rate"].value < 50:
            advice.append("Bradycardia present - assess for conduction disease or medication effect.")
        advice.append("Findings must be interpreted by a qualified clinician against the full patient record.")
        return advice
