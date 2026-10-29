"""Chest X-ray inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.logging import get_logger
from app.ml.base import TORCH_AVAILABLE, ModalityModel, softmax
from app.ml.xray.features import (
    FEATURE_ORDER,
    FeatureValue,
    extract_features,
    feature_importance_hint,
    feature_vector,
)
from app.ml.xray.model import CLASS_DESCRIPTIONS, CLASS_SEVERITY, XRAY_CLASSES, XrayCNN
from app.ml.xray.preprocess import PreprocessedXray, build_model_input, preprocess_xray

logger = get_logger(__name__)


@dataclass
class XrayAnalysis:
    """Everything the API needs to describe one radiograph analysis."""

    probabilities: Dict[str, float]
    label: str
    confidence: float
    severity: str
    description: str
    features: Dict[str, FeatureValue]
    preprocessed: PreprocessedXray
    quality: Dict[str, float]
    class_relevance: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class XrayModel(ModalityModel):
    """Chest radiograph model (2D CNN + clinical-feature fusion)."""

    modality = "xray"
    classes = XRAY_CLASSES
    architecture = "XrayCNN (4-stage) + clinical-feature fusion head"

    def __init__(self) -> None:
        # The checkpoint trained on real public radiographs (NIH ChestX-ray14
        # plus the tuberculosis and COVID-19 collections) takes precedence over
        # the synthetic one whenever it is present, because the synthetic model
        # has never seen a real radiograph and maps every one of them onto the
        # same synthetic template.
        super().__init__("xray_model.pt", prefer=("xray_model_nih.pt",))
        self.metadata.setdefault("input_shape", [1, 224, 224])
        self.metadata.setdefault("version", "1.0.0")

    # ------------------------------------------------------------- lifecycle
    def _build_and_load(self, state: Dict[str, Any]) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load the X-ray checkpoint")
        import torch

        self.classes = list(state.get("classes", list(self.classes)))
        self.metrics = {k: float(v) for k, v in (state.get("metrics") or {}).items() if isinstance(v, (int, float))}
        self.metadata.update(state.get("metadata") or {})

        network = XrayCNN(num_classes=len(self.classes), n_features=len(FEATURE_ORDER))
        network.load_state_dict(state["state_dict"])
        network.eval()
        self.network = network.to(self.device)

    # ------------------------------------------------------------- inference
    def predict_proba(self, samples: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        context = context or {}
        data = context.get("bytes")
        if data is None:
            raise ValueError("X-ray inference requires the raw image bytes")
        pre = preprocess_xray(data)
        features = extract_features(pre)
        return self._network_proba(pre, features)

    def _network_proba(self, pre: PreprocessedXray, features: Dict[str, FeatureValue]) -> np.ndarray:
        if self.network is None:
            return self._heuristic_proba(features, pre.quality.get("sqi", 1.0))
        import torch

        image = build_model_input(pre.grayscale)
        fvec = feature_vector(features)

        self.network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
            feat_tensor = torch.from_numpy(fvec).float().unsqueeze(0).to(self.device)
            logits = self.network(tensor, feat_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        return proba

    # ------------------------------------------------------------- heuristic
    def _heuristic_proba(self, features: Dict[str, FeatureValue], sqi: float) -> np.ndarray:
        """Image-statistics scoring used until a checkpoint is trained."""
        def value(name: str, default: float = 0.0) -> float:
            raw = features[name].value if name in features else default
            return default if raw is None or not np.isfinite(raw) else float(raw)

        apical = value("apical_opacity")
        perihilar = value("perihilar_opacity")
        perihilar_peak = value("perihilar_peak")
        peripheral = value("peripheral_infiltration")
        effusion = value("costophrenic_blunting")
        symmetry = value("symmetry_score")
        opacity = value("lung_opacity")

        pneumonia = min(2.0, 2.6 * max(0.0, perihilar - 1.35)) + 1.0 * max(0.0, 0.86 - symmetry) / 0.2
        consolidation = 3.0 * max(0.0, perihilar - 2.25) / 0.5 + 1.5 * max(0.0, 0.82 - symmetry) / 0.1
        effusion_score = 2.8 * max(0.0, effusion - 0.5) / 1.0
        tuberculosis = 2.8 * max(0.0, apical - 0.6) / 1.0
        covid = 3.0 * max(0.0, peripheral - 0.55) / 0.45 + 1.4 * max(0.0, 0.92 - symmetry) / 0.2

        normal = 1.6 - 0.25 * (pneumonia + consolidation + effusion_score + tuberculosis + covid)
        if opacity < 2.0 and apical < 0.4 and effusion < 0.4 and peripheral < 0.5:
            normal += 1.2

        reliability = float(np.clip(sqi, 0.1, 1.0))
        pneumonia, consolidation, effusion_score, tuberculosis, covid = (
            pneumonia * reliability,
            consolidation * reliability,
            effusion_score * reliability,
            tuberculosis * reliability,
            covid * reliability,
        )

        scores = np.array(
            [normal, pneumonia, consolidation, effusion_score, tuberculosis, covid], dtype=np.float64
        )
        return softmax(scores)

    # -------------------------------------------------------------- analysis
    def analyze(self, data: bytes) -> XrayAnalysis:
        """Run the full chest X-ray analysis pipeline."""
        pre = preprocess_xray(data)
        features = extract_features(pre)
        proba = np.asarray(self._network_proba(pre, features), dtype=np.float64).reshape(-1)
        if proba.size != len(self.classes):
            proba = np.resize(proba, len(self.classes))
        proba = proba / max(proba.sum(), 1e-12)

        best = int(np.argmax(proba))
        label = self.classes[best]
        relevance = self._class_relevance(features)
        warnings = self._warnings(pre, features)

        return XrayAnalysis(
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
            warnings=warnings,
        )

    # --------------------------------------------------------- explainability
    def _class_relevance(self, features: Dict[str, FeatureValue]) -> Dict[str, float]:
        scored = {name: value for name, value, _ in feature_importance_hint(features)}
        total = sum(scored.values()) or 1.0

        def share(names: List[str]) -> float:
            return float(min(1.0, sum(scored.get(n, 0.0) for n in names) / total))

        relevance = {
            "PNEUMONIA": round(share(["lung_opacity", "texture_entropy"]), 4),
            "CONSOLIDATION": round(share(["lung_opacity", "edge_density"]), 4),
            "EFFUSION": round(share(["costophrenic_blunting"]), 4),
            "TUBERCULOSIS": round(share(["apical_opacity", "hilar_prominence"]), 4),
            "COVID19": round(share(["peripheral_infiltration", "symmetry_score", "texture_entropy"]), 4),
        }
        relevance["NORMAL"] = round(1.0 - min(0.95, sum(relevance.values())), 4)
        return relevance

    def _warnings(self, pre: PreprocessedXray, features: Dict[str, FeatureValue]) -> List[str]:
        """Warnings about the quality or suitability of the image itself."""
        warnings: List[str] = []
        if not pre.is_chest_xray_like:
            warnings.append("The uploaded image does not look like a frontal chest radiograph - verify the projection and framing.")
        if features["image_quality"].value < 0.5:
            warnings.append("Radiograph quality is low (blur or low contrast) - repeat the exposure if possible.")
        if pre.original_size[0] < 400 or pre.original_size[1] < 400:
            warnings.append("Low-resolution upload - a DICOM or higher-resolution image will improve the analysis.")
        return warnings

    def _recommendations(self, label: str, features: Dict[str, FeatureValue], relevance: Dict[str, float]) -> List[str]:
        advice: List[str] = []
        if label == "NORMAL":
            advice.append("No acute cardiopulmonary abnormality identified on this radiograph.")
            advice.append("Correlate with the clinical presentation; lateral views may reveal additional findings.")
        elif label == "PNEUMONIA":
            advice.append("Findings are compatible with pneumonia - correlate with fever, cough and inflammatory markers.")
            advice.append("Consider a follow-up radiograph to assess response to treatment.")
        elif label == "CONSOLIDATION":
            advice.append("Air-space consolidation identified - evaluate for bacterial infection, aspiration or malignancy.")
            advice.append("Compare with prior imaging to determine whether the finding is new.")
        elif label == "EFFUSION":
            advice.append("Features suggest pleural effusion - ultrasound is the recommended confirmatory step.")
            advice.append("Assess for associated heart failure, infection or malignancy.")
        elif label == "TUBERCULOSIS":
            advice.append("Upper-zone changes raise suspicion for pulmonary tuberculosis - obtain sputum testing and notify public health as indicated.")
            advice.append("Radiology and infectious-disease review is advised before treatment decisions.")
        elif label == "COVID19":
            advice.append("Peripheral bilateral opacities are compatible with COVID-19 pneumonia - confirm with PCR/antigen testing.")
            advice.append("Correlate with oxygen saturation and clinical severity; consider repeat imaging if deterioration occurs.")

        if features["symmetry_score"].value < 0.7:
            advice.append("Marked left/right asymmetry - review for technical rotation before acting on the findings.")
        advice.append("A radiologist must confirm every finding reported by this decision-support tool.")
        return advice
