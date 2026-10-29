"""Diagnosis orchestration.

This is the single place where a modality-specific analysis is turned into the
uniform :class:`~app.schemas.DiagnosisResponse` payload that the API returns.
It also owns the explainability step and the report hand-off.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import Modality, settings
from app.core.audit import log_event, record_analysis
from app.core.logging import get_logger
from app.core.storage import analysis_dir, new_id, save_analysis_payload
from app.ml.registry import registry
from app.ml.xai import gradcam, plots
from app.schemas import (
    ClassScore,
    ClinicalFinding,
    DiagnosisResponse,
    Explainability,
    FeatureContribution,
    PatientInfo,
    Prediction,
    QualityMetrics,
)

logger = get_logger(__name__)


class DiagnosisService:
    """Runs analyses and assembles the standard response payload."""

    # ------------------------------------------------------------------ run
    def analyze_ecg(
        self,
        samples,
        sampling_rate: float,
        patient: PatientInfo,
        *,
        lead_names: Optional[List[str]] = None,
        unit: str = "auto",
    ) -> DiagnosisResponse:
        model = registry.get("ecg")
        analysis = model.analyze(samples, sampling_rate, lead_names=lead_names, unit=unit)
        lead_idx = analysis.analysis_lead_index
        signal = analysis.preprocessed.clean[lead_idx]
        window = analysis.preprocessed.window[lead_idx]

        explainability = self._signal_explainability(
            model=model,
            modality="ecg",
            analysis=analysis,
            signal=signal,
            window_signal=window,
            window=window,
            feature_vector_builder=_ecg_feature_vector,
            target_class=list(analysis.probabilities).index(analysis.label),
            r_peaks=analysis.preprocessed.r_peaks,
        )
        return self._finalize(
            modality="ecg",
            analysis=analysis,
            patient=patient,
            explainability=explainability,
            model=model,
        )

    def analyze_eeg(
        self,
        samples,
        sampling_rate: float,
        patient: PatientInfo,
        *,
        channel_names: Optional[List[str]] = None,
        unit: str = "auto",
    ) -> DiagnosisResponse:
        model = registry.get("eeg")
        analysis = model.analyze(samples, sampling_rate, channel_names=channel_names, unit=unit)
        signal = analysis.preprocessed.clean[0]
        window = analysis.preprocessed.window[0]

        explainability = self._signal_explainability(
            model=model,
            modality="eeg",
            analysis=analysis,
            signal=signal,
            window_signal=window,
            window=window,
            feature_vector_builder=_eeg_feature_vector,
            target_class=list(analysis.probabilities).index(analysis.label),
            r_peaks=None,
        )
        return self._finalize(
            modality="eeg",
            analysis=analysis,
            patient=patient,
            explainability=explainability,
            model=model,
        )

    def analyze_xray(self, data: bytes, patient: PatientInfo) -> DiagnosisResponse:
        model = registry.get("xray")
        analysis = model.analyze(data)
        explainability = self._xray_explainability(model, analysis)
        response = self._finalize(
            modality="xray",
            analysis=analysis,
            patient=patient,
            explainability=explainability,
            model=model,
        )
        if getattr(analysis, "warnings", None):
            response.recommendations = list(analysis.warnings) + list(response.recommendations)
        return response

    # ---------------------------------------------------------- finalisation
    def _finalize(
        self,
        *,
        modality: Modality,
        analysis: Any,
        patient: PatientInfo,
        explainability: Explainability,
        model: Any,
    ) -> DiagnosisResponse:
        """Build the response, persist it and write the audit row."""
        analysis_id = new_id(prefix=f"{modality}-")
        probabilities = dict(analysis.probabilities)
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

        prediction = Prediction(
            label=analysis.label,
            confidence=round(float(analysis.confidence), 4),
            severity=analysis.severity,
            description=analysis.description,
            class_probabilities={k: round(float(v), 4) for k, v in probabilities.items()},
        )

        differential = [
            ClassScore(label=label, probability=round(float(prob), 4), severity=_severity_for(label, modality))
            for label, prob in ordered[1:4]
        ]

        findings = [
            ClinicalFinding(
                name=feature.name.replace("_", " ").title(),
                value=feature.display,
                unit=feature.unit,
                reference_range=f"{feature.reference_low:g} - {feature.reference_high:g} {feature.unit}",
                status=feature.status,
                explanation=feature.explanation,
            )
            for feature in analysis.features.values()
        ]

        quality = QualityMetrics(
            signal_quality_index=round(float(analysis.quality.get("sqi", 0.0)), 4),
            usable=float(analysis.quality.get("sqi", 0.0)) >= 0.35,
            issues=[] if float(analysis.quality.get("sqi", 0.0)) >= 0.35 else ["Input signal quality is below the usable threshold."],
            snr_db=round(float(analysis.quality.get("snr_db", 0.0)), 2),
            clipped_samples=int(round(float(analysis.quality.get("clipped", 0.0)) * 1000)),
        )

        response = DiagnosisResponse(
            analysis_id=analysis_id,
            modality=modality,
            patient=patient,
            prediction=prediction,
            differential=differential,
            findings=findings,
            explainability=explainability,
            recommendations=list(analysis.recommendations),
            quality=quality,
            model_version=str(model.metadata.get("version", "1.0.0")),
            model_source=model.source,
            report_url=f"/api/v1/reports/analysis/{analysis_id}",
            created_at=datetime.now(timezone.utc),
        )

        payload = response.model_dump(mode="json")
        try:
            save_analysis_payload(analysis_id, payload)
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("Could not persist analysis %s: %s", analysis_id, exc)

        record_analysis(
            analysis_id=analysis_id,
            modality=modality,
            patient_ref=patient.patient_id,
            predicted=analysis.label,
            confidence=float(analysis.confidence),
            severity=str(analysis.severity),
            model_version=str(model.metadata.get("version", "1.0.0")),
            payload={
                "recommendations": response.recommendations,
                "findings": [f.model_dump(mode="json") for f in findings],
                "top_features": [f.model_dump(mode="json") for f in explainability.top_features],
            },
        )
        log_event("analysis_completed", analysis_id=analysis_id, detail=f"{modality}:{analysis.label}")
        return response

    # --------------------------------------------------------- explainability
    def _signal_explainability(
        self,
        *,
        model: Any,
        modality: str,
        analysis: Any,
        signal: np.ndarray,
        window_signal: np.ndarray,
        window: np.ndarray,
        feature_vector_builder: Any,
        target_class: Optional[int],
        r_peaks: Optional[np.ndarray],
    ) -> Explainability:
        """Attribution for ECG / EEG: saliency over time + feature contributions."""
        methods: List[str] = []
        top_features: List[FeatureContribution] = []
        saliency_png: Optional[str] = None
        waveform_png: Optional[str] = None
        attribution_method = "heuristic feature deviation"

        feature_names = _feature_order_for(modality)
        ranked = _rank_features(analysis.features)
        top_features = [
            FeatureContribution(
                name=name.replace("_", " ").title(),
                contribution=round(float(score), 4),
                direction="increases_risk" if score > 0 else "neutral",
                detail=f"Deviation from the expected range for {name.replace('_', ' ')}.",
            )
            for name, score in ranked
        ][: settings.xai_max_features]

        if settings.xai_enabled and model.is_trained and target_class is not None:
            try:
                import torch

                fvec = feature_vector_builder(analysis.features)
                feature_tensor = torch.from_numpy(np.asarray(fvec, dtype=np.float32)).unsqueeze(0).to(model.device)
                if modality == "ecg":
                    input_tensor = torch.from_numpy(np.asarray(window, dtype=np.float32)).reshape(1, 1, -1).to(model.device)
                    cam = gradcam.grad_cam_for_ecg(
                        model.network, window[None, :], fvec, target_class, model.device
                    )
                else:
                    input_tensor = torch.from_numpy(np.asarray(window, dtype=np.float32)).reshape(1, 1, 1, -1).to(model.device)
                    cam = gradcam.grad_cam_for_eeg(
                        model.network, window[None, None, :], fvec, target_class, model.device
                    )
                if cam.size > 1:
                    saliency_png = plots.plot_saliency(
                        window_signal,
                        cam,
                        analysis.preprocessed.sampling_rate,
                        title=f"{modality.upper()} attribution - {analysis.label}",
                    )
                    methods.append("Grad-CAM")
                    attribution_method = "Grad-CAM over the final convolutional block"

                # Per-feature attribution straight from the trained network: how
                # much the target-class score drops when each clinical feature is
                # reset to its baseline.  This replaces the heuristic deviation
                # ranking whenever a checkpoint is available.
                attributions = gradcam.feature_attributions(
                    model.network, input_tensor, feature_tensor, target_class
                )
                network_features = _features_from_attributions(attributions, feature_names)
                if network_features:
                    top_features = network_features[: settings.xai_max_features]
                    methods.append("Feature attribution (trained network)")
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Grad-CAM failed for %s: %s", modality, exc)

        if not saliency_png:
            saliency_png = plots.plot_signal(
                signal,
                analysis.preprocessed.sampling_rate,
                r_peaks=r_peaks,
                title=f"{modality.upper()} conditioned signal - {analysis.label}",
            )
        waveform_png = plots.plot_signal(
            signal,
            analysis.preprocessed.sampling_rate,
            r_peaks=r_peaks,
            title=f"{modality.upper()} recording",
            color="#8b93a7",
        )

        summary = _explain_summary(modality, analysis, top_features)

        return Explainability(
            methods=methods or ["Feature deviation analysis"],
            summary=summary,
            top_features=top_features,
            class_relevance={k: round(float(v), 4) for k, v in analysis.class_relevance.items()},
            saliency_png_base64=saliency_png,
            overlay_png_base64=None,
            waveform_png_base64=waveform_png,
            attribution_method=attribution_method,
        )

    def _xray_explainability(self, model: Any, analysis: Any) -> Explainability:
        """Attribution for chest radiographs: Grad-CAM overlay + feature bars."""
        methods: List[str] = ["Feature deviation analysis"]
        overlay_png: Optional[str] = None
        attribution_method = "Image-statistics deviation"

        ranked = _rank_features(analysis.features)
        top_features = [
            FeatureContribution(
                name=name.replace("_", " ").title(),
                contribution=round(float(score), 4),
                direction="increases_risk" if score > 0 else "neutral",
                detail=f"Deviation from the expected range for {name.replace('_', ' ')}.",
            )
            for name, score in ranked
        ][: settings.xai_max_features]

        if settings.xai_enabled and model.is_trained:
            try:
                import torch

                from app.ml.xray.features import FEATURE_ORDER as XRAY_FEATURE_ORDER
                from app.ml.xray.features import feature_vector as xray_feature_vector
                from app.ml.xray.preprocess import build_model_input

                target_class = list(analysis.probabilities).index(analysis.label)
                fvec = xray_feature_vector(analysis.features)
                image_tensor = build_model_input(analysis.preprocessed.grayscale)
                cam = gradcam.grad_cam_for_xray(
                    model.network,
                    image_tensor,
                    fvec,
                    target_class,
                    model.device,
                )
                if cam.size > 1:
                    overlay_png = plots.plot_overlay(
                        analysis.preprocessed.grayscale, cam, title=f"Grad-CAM - {analysis.label}"
                    )
                    methods = ["Grad-CAM"]
                    attribution_method = "Grad-CAM over the final convolutional block"

                attributions = gradcam.feature_attributions(
                    model.network,
                    torch.from_numpy(np.asarray(image_tensor, dtype=np.float32))
                    .unsqueeze(0)
                    .to(model.device),
                    torch.from_numpy(np.asarray(fvec, dtype=np.float32)).unsqueeze(0).to(model.device),
                    target_class,
                )
                network_features = _features_from_attributions(attributions, list(XRAY_FEATURE_ORDER))
                if network_features:
                    top_features = network_features[: settings.xai_max_features]
                    methods.append("Feature attribution (trained network)")
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Grad-CAM failed for xray: %s", exc)

        summary = _explain_summary("xray", analysis, top_features)
        return Explainability(
            methods=methods,
            summary=summary,
            top_features=top_features,
            class_relevance={k: round(float(v), 4) for k, v in analysis.class_relevance.items()},
            saliency_png_base64=plots.plot_image(
                analysis.preprocessed.grayscale, title=f"Radiograph - {analysis.label}"
            ),
            overlay_png_base64=overlay_png,
            waveform_png_base64=plots.plot_class_probabilities(
                analysis.probabilities, title="Radiograph class probabilities"
            ),
            attribution_method=attribution_method,
        )

    # ------------------------------------------------------------------ misc
    def list_analyses(self, **kwargs) -> List[Dict[str, Any]]:
        from app.core.audit import list_analyses as _list

        return _list(**kwargs)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rank_features(features: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Rank features by how far they deviate from their reference band."""
    scored: List[Tuple[str, float]] = []
    for name, feature in features.items():
        value = getattr(feature, "value", None)
        if value is None or not np.isfinite(value):
            continue
        low = getattr(feature, "reference_low", 0.0)
        high = getattr(feature, "reference_high", 0.0)
        if value < low:
            severity = (low - value) / max(abs(low), 1e-6)
        elif value > high:
            severity = (value - high) / max(abs(high), 1e-6)
        else:
            severity = 0.0
        if severity > 0:
            scored.append((name, float(severity)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _feature_order_for(modality: str) -> List[str]:
    """Ordered clinical feature names for a modality (matches ``feature_vector``)."""
    if modality == "ecg":
        from app.ml.ecg.features import FEATURE_ORDER as order
    elif modality == "eeg":
        from app.ml.eeg.features import FEATURE_ORDER as order
    else:
        from app.ml.xray.features import FEATURE_ORDER as order
    return list(order)


def _features_from_attributions(
    attributions: np.ndarray, feature_names: List[str]
) -> List[FeatureContribution]:
    """Turn per-feature network attributions into ranked explanations.

    ``attributions[i]`` is how much the target-class score drops when feature
    ``i`` is reset to its baseline, so a positive value means the feature
    supports the prediction and a negative value means it argues against it.
    """
    values = np.asarray(attributions, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []

    names = list(feature_names)
    if len(names) < values.size:
        names = names + [f"feature_{index}" for index in range(len(names), values.size)]
    names = names[: values.size]

    order = np.argsort(np.abs(values))[::-1]
    contributions: List[FeatureContribution] = []
    for index in order:
        value = float(values[index])
        contributions.append(
            FeatureContribution(
                name=str(names[index]).replace("_", " ").title(),
                contribution=round(value, 4),
                direction="increases_risk" if value >= 0 else "decreases_risk",
                detail=(
                    "Resetting this feature to its baseline lowers the score for the predicted class."
                    if value >= 0
                    else "Resetting this feature to its baseline raises the score for the predicted class."
                ),
            )
        )
    return contributions


def _explain_summary(modality: str, analysis: Any, top_features: List[FeatureContribution]) -> str:
    """Human-readable explanation of why the model decided what it did."""
    if not top_features:
        return (
            f"All measured {modality.upper()} features fall inside their expected reference ranges, "
            f"which supports the '{analysis.label}' result."
        )
    drivers = ", ".join(feature.name for feature in top_features[:3])
    return (
        f"The prediction '{analysis.label}' was driven mainly by {drivers}. "
        f"The model assigned {analysis.confidence:.1%} confidence to this class."
    )


def _ecg_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    from app.ml.ecg.features import feature_vector as _vec

    return _vec(features)


def _eeg_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    from app.ml.eeg.features import feature_vector as _vec

    return _vec(features)


def _severity_for(label: str, modality: str) -> str:
    from app.ml.ecg.model import CLASS_SEVERITY as ecg_sev
    from app.ml.eeg.model import CLASS_SEVERITY as eeg_sev
    from app.ml.xray.model import CLASS_SEVERITY as xray_sev

    table = {"ecg": ecg_sev, "eeg": eeg_sev, "xray": xray_sev}.get(modality, {})
    return table.get(label, "normal")


diagnosis_service = DiagnosisService()
