"""Interpretable EEG features.

Spectral band powers, entropy and the asymmetry / ratio metrics that underpin
the neurological screening rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sps

from app.ml.eeg.preprocess import PreprocessedEEG

# name -> (unit, low, high, explanation)
FEATURE_REFERENCE: Dict[str, Tuple[str, float, float, str]] = {
    "delta_power": ("uV2", 0.0, 400.0, "Delta band (0.5-4 Hz) power - dominant in deep sleep and frontal dysfunction."),
    "theta_power": ("uV2", 0.0, 120.0, "Theta band (4-8 Hz) power - elevated in drowsiness and depressive states."),
    "alpha_power": ("uV2", 5.0, 250.0, "Alpha band (8-13 Hz) power - posterior dominant rhythm; reduced in neurological disease."),
    "beta_power": ("uV2", 0.0, 100.0, "Beta band (13-30 Hz) power - increased with cortical activation and in Parkinsonism."),
    "gamma_power": ("uV2", 0.0, 60.0, "Gamma band (30-45 Hz) power - associated with epileptogenic activity."),
    "relative_theta": ("%", 0.0, 35.0, "Theta share of total in-band power."),
    "relative_alpha": ("%", 5.0, 60.0, "Alpha share of total in-band power."),
    "theta_beta_ratio": ("ratio", 0.4, 2.0, "Asendorfer theta/beta ratio - a screening marker of depressive states."),
    "alpha_asymmetry": ("index", -0.3, 0.3, "Frontal alpha asymmetry - reduced left-right asymmetry is associated with mood symptoms."),
    "spectral_entropy": ("bits", 0.0, 0.90, "Spectral entropy - higher values indicate less organised, more pathological rhythms."),
    "spike_wave_index": ("%", 0.0, 0.8, "Fraction of the recording occupied by high-amplitude spike-and-wave discharges."),
    "hjorth_mobility": ("ratio", 0.1, 0.6, "Hjorth mobility - variance of the derivative over variance of the signal."),
    "hjorth_complexity": ("ratio", 1.0, 2.2, "Hjorth complexity - the fractal dimension of the signal."),
    "signal_variance": ("uV2", 1.0, 2000.0, "Variance of the EEG signal in the analysis window."),
    "artifact_ratio": ("%", 0.0, 10.0, "Proportion of the window contaminated by movement or eye-blink artefacts."),
    "signal_quality": ("index", 0.6, 1.0, "Signal quality index derived from spectral cleanliness and artefact load."),
}

FEATURE_ORDER: List[str] = list(FEATURE_REFERENCE.keys())


@dataclass
class FeatureValue:
    """One measured EEG feature with its clinical interpretation."""

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


def band_powers(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Absolute power of the five clinical EEG bands."""
    if signal.size < 64:
        return {name: 0.0 for name in ("delta", "theta", "alpha", "beta", "gamma")}
    freqs, psd = sps.welch(signal, fs=fs, nperseg=min(512, signal.size), noverlap=min(256, signal.size // 2))
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    powers: Dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        powers[name] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def spectral_entropy(powers: Dict[str, float]) -> float:
    """Normalised Shannon entropy of the band-power distribution."""
    values = np.array([max(v, 0.0) for v in powers.values()], dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    entropy = float(-np.sum(probs * np.log2(probs)))
    return float(entropy / np.log2(probs.size))


def hjorth_parameters(signal: np.ndarray) -> Tuple[float, float]:
    """Hjorth activity/mobility/comobility -> (mobility, complexity)."""
    if signal.size < 8:
        return 0.0, 0.0
    var0 = float(np.var(signal))
    d1 = np.diff(signal)
    var1 = float(np.var(d1))
    d2 = np.diff(d1)
    var2 = float(np.var(d2))
    if var0 <= 0 or var1 <= 0:
        return 0.0, 0.0
    mobility = float(np.sqrt(var1 / var0))
    complexity = float(np.sqrt(var2 / var1) / mobility) if mobility > 0 else 0.0
    return mobility, complexity


def spike_wave_index(signal: np.ndarray, fs: float) -> float:
    """Proportion of the window containing sharp, high-amplitude transients."""
    if signal.size < 32:
        return 0.0
    robust_std = float(np.median(np.abs(signal - np.median(signal))) * 1.4826) or 1.0
    threshold = 3.5 * robust_std
    mask = np.abs(signal) > threshold
    # ignore isolated single samples - require a short run
    kernel = max(3, int(0.01 * fs))
    smoothed = np.convolve(mask.astype(np.float64), np.ones(kernel) / kernel, mode="same")
    flagged = smoothed > 0.6
    return float(np.mean(flagged) * 100.0)


def alpha_asymmetry(signal: np.ndarray, fs: float) -> float:
    """Left-right alpha asymmetry proxy from the channel ordering.

    With a single channel the value is 0 (no asymmetry can be computed); with
    multiple channels the first half is treated as left, the second as right.
    """
    if signal.ndim != 2 or signal.shape[0] < 2:
        return 0.0
    half = signal.shape[0] // 2
    left = signal[:half]
    right = signal[half : half * 2] if half >= 1 else signal[:1]
    freqs, psd_l = sps.welch(left.mean(axis=0), fs=fs, nperseg=min(512, left.shape[1]))
    _, psd_r = sps.welch(right.mean(axis=0), fs=fs, nperseg=min(512, right.shape[1]))
    mask = (freqs >= 8.0) & (freqs <= 13.0)
    if not mask.any():
        return 0.0
    a_left = float(np.trapezoid(psd_l[mask], freqs[mask]))
    a_right = float(np.trapezoid(psd_r[mask], freqs[mask]))
    denom = a_left + a_right
    if denom <= 0:
        return 0.0
    return float(np.clip(np.log(a_right / a_left), -1.0, 1.0))


def extract_features(pre: PreprocessedEEG) -> Dict[str, FeatureValue]:
    """Compute the full EEG feature set with clinical status flags."""
    signal = pre.clean
    fs = pre.sampling_rate
    primary = signal[0] if signal.ndim == 2 else signal

    powers = band_powers(primary, fs)
    total_power = sum(powers.values()) or 1e-9
    rel_theta = powers["theta"] / total_power * 100.0
    rel_alpha = powers["alpha"] / total_power * 100.0
    theta_beta = powers["theta"] / (powers["beta"] + 1e-9)
    entropy = spectral_entropy(powers)
    mobility, complexity = hjorth_parameters(primary)
    swi = spike_wave_index(primary, fs)
    asymmetry = alpha_asymmetry(signal, fs)
    variance = float(np.var(primary))
    artifact_ratio = float(pre.artifact_ratio * 100.0)
    sqi = float(pre.quality.get("sqi", 0.0))

    values: Dict[str, float] = {
        "delta_power": powers["delta"],
        "theta_power": powers["theta"],
        "alpha_power": powers["alpha"],
        "beta_power": powers["beta"],
        "gamma_power": powers["gamma"],
        "relative_theta": rel_theta,
        "relative_alpha": rel_alpha,
        "theta_beta_ratio": theta_beta,
        "alpha_asymmetry": asymmetry,
        "spectral_entropy": entropy,
        "spike_wave_index": swi,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "signal_variance": variance,
        "artifact_ratio": artifact_ratio,
        "signal_quality": sqi,
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
        [400.0, 300.0, 250.0, 120.0, 60.0, 35.0, 60.0, 6.0, 0.3, 0.85, 5.0, 0.6, 2.2, 2000.0, 10.0, 1.0],
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
