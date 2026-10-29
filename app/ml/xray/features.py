"""Interpretable chest X-ray features.

Radiologists describe findings by *region* ("apical opacity", "costophrenic
blunting", "peripheral ground-glass"), so the features here do the same: each
lung zone is measured against the mean brightness of the lung fields, which
makes the measurements independent of overall exposure.

The measurements feed the heuristic baseline scorer, the clinical findings
list and the explainability ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from app.ml.xray.preprocess import PreprocessedXray

FEATURE_REFERENCE: Dict[str, Tuple[str, float, float, str]] = {
    "mean_intensity": ("level", 20.0, 120.0, "Average pixel brightness of the radiograph."),
    "lung_opacity": ("%", 0.0, 25.0, "Fraction of the lung fields showing increased density (consolidation / infiltration)."),
    "apical_opacity": ("index", 0.0, 0.60, "Brightness of the lung apices relative to the lung fields - raised in apical tuberculosis."),
    "perihilar_opacity": ("index", 0.0, 1.30, "Brightness of the perihilar zones relative to the lung fields - raised with central consolidation."),
    "perihilar_peak": ("index", 0.0, 2.0, "Peak density of the perihilar zone - air-space consolidation is denser than patchy pneumonia."),
    "peripheral_infiltration": ("index", 0.0, 0.55, "Brightness of the peripheral lower zones relative to the lung fields - typical of COVID-19 pneumonia."),
    "costophrenic_blunting": ("index", 0.0, 0.50, "Brightness of the costophrenic angles relative to the lung fields - indicates pleural effusion."),
    "symmetry_score": ("index", 0.75, 1.0, "Left/right similarity of the lung fields - asymmetry suggests unilateral disease."),
    "texture_entropy": ("bits", 4.0, 7.2, "Entropy of the intensity distribution - higher values suggest heterogeneous lung texture."),
    "edge_density": ("%", 0.5, 12.0, "Density of structural edges - raised with interstitial or focal pathology."),
    "image_quality": ("index", 0.5, 1.0, "Technical quality of the radiograph (sharpness and contrast)."),
}

FEATURE_ORDER: List[str] = list(FEATURE_REFERENCE.keys())


@dataclass
class FeatureValue:
    """One measured image feature with its clinical interpretation."""

    name: str
    value: float
    unit: str
    reference_low: float
    reference_high: float
    status: str
    explanation: str
    display: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "value": self.display,
            "unit": self.unit,
            "reference_range": f"{self.reference_low:g} - {self.reference_high:g} {self.unit}",
            "status": self.status,
            "explanation": self.explanation,
        }


def _status(value: float, low: float, high: float, tolerance: float = 0.15) -> str:
    if low <= value <= high:
        return "normal"
    span = max(high - low, 1e-6)
    margin = span * tolerance
    if (value < low and value >= low - margin) or (value > high and value <= high + margin):
        return "borderline"
    return "abnormal"


def _normalised_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    return xx / max(width - 1, 1), yy / max(height - 1, 1)


def _lung_mask(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Approximate the two lung fields as the darker lateral regions."""
    vertical = np.clip(1.0 - np.maximum(0.0, (ny - 0.74) / 0.24), 0.0, 1.0)
    left = nx < 0.40
    right = nx > 0.60
    return (vertical * (left | right)).astype(np.float64)


def _region_mean(gray: np.ndarray, mask: np.ndarray) -> float:
    """Mean brightness of the masked pixels (0.0 when the region is empty)."""
    if mask.size == 0:
        return 0.0
    selected = gray[mask > 0.5]
    return float(np.mean(selected)) if selected.size else 0.0


def extract_features(pre: PreprocessedXray) -> Dict[str, FeatureValue]:
    """Compute the full chest X-ray feature set with clinical status flags."""
    gray = pre.grayscale.astype(np.float64)
    height, width = gray.shape
    nx, ny = _normalised_grid(height, width)

    lung_mask = _lung_mask(nx, ny)
    lung_values = gray[lung_mask > 0.5]
    if lung_values.size == 0:
        lung_values = gray.ravel()
    lung_mean = float(np.mean(lung_values))
    lung_std = float(np.std(lung_values)) or 1.0

    mean_intensity = float(gray.mean())

    # --- zones, expressed relative to the mean lung brightness.  A high
    # percentile inside the zone is used so a focal opacity is not averaged
    # away by the surrounding aerated lung.
    def zone(nx_mask: np.ndarray, ny_mask: np.ndarray, percentile: float = 85.0) -> float:
        mask = lung_mask * nx_mask * ny_mask
        values = gray[mask > 0.5]
        if values.size < 20:
            return 0.0
        return float((np.percentile(values, percentile) - lung_mean) / max(lung_mean, 1e-6))

    lateral = (nx < 0.42) | (nx > 0.58)
    apical = zone(lateral, (ny > 0.12) & (ny < 0.30))
    perihilar = zone(((nx > 0.28) & (nx < 0.40)) | ((nx > 0.60) & (nx < 0.72)), (ny >= 0.30) & (ny < 0.60))

    # the densest part of the perihilar zone separates air-space consolidation
    # (dense) from patchy pneumonia (less dense at the same mean level)
    perihilar_mask = lung_mask * (
        (((nx > 0.28) & (nx < 0.40)) | ((nx > 0.60) & (nx < 0.72))) & ((ny >= 0.30) & (ny < 0.60))
    )
    perihilar_values = gray[perihilar_mask > 0.5]
    if perihilar_values.size >= 20:
        perihilar_peak = float((np.percentile(perihilar_values, 99) - lung_mean) / max(lung_mean, 1e-6))
    else:
        perihilar_peak = 0.0

    peripheral = zone((nx < 0.34) | (nx > 0.66), (ny >= 0.55) & (ny < 0.88))

    # costophrenic angles: the bottom lateral corners
    angle = lung_mask * ((ny > 0.78) & ((nx < 0.30) | (nx > 0.70)))
    angle_values = gray[angle > 0.5]
    if angle_values.size >= 20:
        angle_value = float(np.percentile(angle_values, 75))
        costophrenic = float((angle_value - lung_mean) / max(lung_mean, 1e-6))
    else:
        costophrenic = 0.0

    # --- dense fraction inside the lungs
    lung_opacity = float(np.mean(lung_values > (lung_mean + 2.5 * lung_std)) * 100.0)

    # --- left/right symmetry of the lung fields
    left_field = gray * lung_mask * (nx < 0.5)
    right_field = gray * lung_mask * (nx >= 0.5)
    left_mean = _region_mean(gray, lung_mask * (nx < 0.5))
    right_mean = _region_mean(gray, lung_mask * (nx >= 0.5))
    if left_mean > 0 and right_mean > 0:
        symmetry = float(np.clip(1.0 - abs(left_mean - right_mean) / max(lung_mean, 1e-6), 0.0, 1.0))
    else:
        symmetry = 1.0

    # --- texture and edges
    hist = np.histogram(gray, bins=64, range=(0, 256))[0].astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gy ** 2 + gx ** 2)
    edge_density = float(np.mean(grad > 12.0) * 100.0)

    quality = float(pre.quality.get("sqi", 0.0))

    values: Dict[str, float] = {
        "mean_intensity": mean_intensity,
        "lung_opacity": lung_opacity,
        "apical_opacity": max(apical, 0.0),
        "perihilar_opacity": max(perihilar, 0.0),
        "perihilar_peak": max(perihilar_peak, 0.0),
        "peripheral_infiltration": max(peripheral, 0.0),
        "costophrenic_blunting": max(costophrenic, 0.0),
        "symmetry_score": symmetry,
        "texture_entropy": entropy,
        "edge_density": edge_density,
        "image_quality": quality,
    }

    features: Dict[str, FeatureValue] = {}
    for name in FEATURE_ORDER:
        value = float(values[name])
        unit, low, high, explanation = FEATURE_REFERENCE[name]
        status = _status(value, low, high)
        digits = 2 if abs(value) < 10 else 1
        features[name] = FeatureValue(
            name=name,
            value=value,
            unit=unit,
            reference_low=low,
            reference_high=high,
            status=status,
            explanation=explanation,
            display=f"{value:.{digits}f}",
        )
    return features


def feature_vector(features: Dict[str, FeatureValue]) -> np.ndarray:
    """Pack the feature dict into the fixed numeric vector used by the model."""
    raw = np.array([features[name].value for name in FEATURE_ORDER], dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    scales = np.array(
        [120.0, 25.0, 0.60, 1.30, 2.0, 0.55, 0.50, 1.0, 7.2, 12.0, 1.0],
        dtype=np.float32,
    )
    return raw / scales


def feature_importance_hint(features: Dict[str, FeatureValue]) -> List[Tuple[str, float, str]]:
    """Rank features by deviation from their reference band."""
    scored: List[Tuple[str, float, str]] = []
    for name, feature in features.items():
        value = feature.value
        if not np.isfinite(value):
            continue
        low, high = feature.reference_low, feature.reference_high
        if value < low:
            severity = (low - value) / max(abs(low), 1e-6)
            direction = "increases_risk"
        elif value > high:
            severity = (value - high) / max(abs(high), 1e-6)
            direction = "increases_risk"
        else:
            severity, direction = 0.0, "neutral"
        if severity == 0:
            continue
        scored.append((name, float(severity), direction))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
