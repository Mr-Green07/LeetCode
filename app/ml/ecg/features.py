"""Interpretable ECG features.

These are the numbers a cardiologist actually looks at (rate, intervals,
repolarisation, ectopy).  They serve three purposes:

1. they are returned to the clinician as ``findings``;
2. they feed the heuristic baseline scorer;
3. they are concatenated with the CNN embedding so the network can learn
   residual signal that hand-crafted features miss.

Interval and ST measurements are taken from a **median beat template** and
referenced to the isoelectric TP segment, which is how automated ECG systems
avoid the P wave contaminating the ST baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from app.ml.ecg.preprocess import PreprocessedECG, detect_r_peaks

# Fixed ordering of the numeric feature vector handed to the model.
FEATURE_ORDER: List[str] = [
    "heart_rate",
    "rr_mean",
    "rr_sd",
    "rr_cv",
    "rmssd",
    "pnn50",
    "lf_hf",
    "qrs_duration",
    "qt_interval",
    "pr_interval",
    "st_deviation",
    "t_amplitude",
    "r_amplitude",
    "pvc_fraction",
    "rr_irregularity",
    "signal_quality",
    "baseline_wander",
    "p_wave_amplitude",
]

# name -> (unit, low, high, explanation)
FEATURE_REFERENCE: Dict[str, Tuple[str, float, float, str]] = {
    "heart_rate": ("bpm", 60.0, 100.0, "Resting heart rate derived from R-peak intervals."),
    "rr_mean": ("ms", 800.0, 1000.0, "Mean R-R interval; the reciprocal of heart rate."),
    "rr_sd": ("ms", 0.0, 100.0, "R-R variability; elevated values suggest arrhythmia."),
    "rr_cv": ("%", 0.0, 8.0, "Coefficient of variation of R-R intervals (normalised irregularity)."),
    "rmssd": ("ms", 0.0, 50.0, "Root mean square of successive R-R differences (short-term HRV)."),
    "pnn50": ("%", 0.0, 20.0, "Percentage of successive R-R intervals differing by more than 50 ms."),
    "lf_hf": ("ratio", 0.5, 5.0, "Low-frequency to high-frequency power ratio of the HRV spectrum."),
    "qrs_duration": ("ms", 60.0, 130.0, "QRS complex duration; widening indicates bundle branch block or ventricular origin."),
    "qt_interval": ("ms", 250.0, 420.0, "Ventricular depolarisation and repolarisation duration."),
    "pr_interval": ("ms", 120.0, 200.0, "Atrial to ventricular conduction time."),
    "st_deviation": ("mV", -0.05, 0.05, "ST segment deviation from the isoelectric line at J+60 ms."),
    "t_amplitude": ("mV", -0.2, 0.2, "T-wave amplitude on the primary lead."),
    "r_amplitude": ("mV", 0.2, 3.0, "R-wave amplitude; low values can indicate myocardial ischaemia."),
    "pvc_fraction": ("%", 0.0, 5.0, "Percentage of premature beats with a wide QRS complex."),
    "rr_irregularity": ("%", 0.0, 10.0, "Normalised irregularity of the R-R series."),
    "signal_quality": ("index", 0.6, 1.0, "Signal quality index derived from the noise-to-signal ratio."),
    "baseline_wander": ("mV", 0.0, 0.15, "Low-frequency baseline drift amplitude."),
    "p_wave_amplitude": ("mV", 0.05, 0.35, "P-wave amplitude - absent P waves are the hallmark of atrial fibrillation."),
}

# Scaling constants so every feature reaches the network on a comparable scale.
FEATURE_SCALES: Dict[str, float] = {
    "heart_rate": 100.0,
    "rr_mean": 1000.0,
    "rr_sd": 100.0,
    "rr_cv": 10.0,
    "rmssd": 50.0,
    "pnn50": 20.0,
    "lf_hf": 5.0,
    "qrs_duration": 100.0,
    "qt_interval": 440.0,
    "pr_interval": 200.0,
    "st_deviation": 0.1,
    "t_amplitude": 0.2,
    "r_amplitude": 3.0,
    "pvc_fraction": 5.0,
    "rr_irregularity": 10.0,
    "signal_quality": 1.0,
    "baseline_wander": 0.15,
    "p_wave_amplitude": 0.2,
}


@dataclass
class FeatureValue:
    """One measured feature with its clinical interpretation."""

    name: str
    value: float
    unit: str
    reference_low: float
    reference_high: float
    status: str  # normal | borderline | abnormal
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
    """Classify a value against its reference band."""
    if low <= value <= high:
        return "normal"
    span = max(high - low, 1e-6)
    margin = span * tolerance
    if (value < low and value >= low - margin) or (value > high and value <= high + margin):
        return "borderline"
    return "abnormal"


def _hrv_from_rr(rr_intervals: np.ndarray, fs: float) -> Dict[str, float]:
    """Time and frequency domain HRV metrics from an R-R series."""
    if rr_intervals.size < 3:
        return {"rmssd": float("nan"), "pnn50": float("nan"), "lf_hf": float("nan"), "sdnn": float("nan")}
    diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.mean(np.abs(diffs) > 0.05) * 100.0)
    sdnn = float(np.std(rr_intervals, ddof=1)) if rr_intervals.size > 1 else 0.0

    lf_hf = float("nan")
    if rr_intervals.size >= 8:
        # interpolate onto a uniform 4 Hz series before the FFT
        interp_time = np.arange(rr_intervals.size, dtype=np.float64)
        uniform_fs = 4.0
        uniform_t = np.arange(interp_time[-1], 1.0 / uniform_fs, 1.0 / uniform_fs)
        if uniform_t.size >= 8:
            series = np.interp(uniform_t, interp_time, rr_intervals)
            freqs, psd = sps.welch(series - series.mean(), fs=uniform_fs, nperseg=min(len(series), 64))
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            lf = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
            hf = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
            if hf > 1e-12:
                lf_hf = lf / hf
    return {"rmssd": rmssd, "pnn50": pnn50, "lf_hf": lf_hf, "sdnn": sdnn}


def _median_beat_template(
    signal: np.ndarray, r_peaks: np.ndarray, fs: float, pre_s: float = 0.30, post_s: float = 0.50
) -> Tuple[Optional[np.ndarray], int]:
    """Build the median beat template aligned on the R peak.

    Returns ``(template, r_index)`` where ``r_index`` is the sample index of the
    R peak inside the template.
    """
    pre = int(pre_s * fs)
    post = int(post_s * fs)
    beats: List[np.ndarray] = []
    for r in r_peaks:
        start = int(r) - pre
        stop = int(r) + post
        if start < 0 or stop > signal.size:
            continue
        beats.append(signal[start:stop])
    if not beats:
        return None, pre
    width = min(beat.size for beat in beats)
    stack = np.stack([beat[:width] for beat in beats])
    return np.median(stack, axis=0), pre


def _envelope(signal: np.ndarray, fs: float, window_s: float = 0.040) -> np.ndarray:
    """Local-maximum envelope of a signal.

    Taking the running maximum of ``|signal|`` merges the R and S deflections
    into one QRS complex, so a threshold crossing is not thrown off by the
    zero crossing between them.  The envelope widens the complex by the
    window length, so :func:`_envelope_width_ms` subtracts that back out.
    """
    from scipy.ndimage import maximum_filter1d

    size = max(3, int(window_s * fs) | 1)
    return maximum_filter1d(np.abs(signal), size=size)


ENVELOPE_WINDOW_S = 0.020


def _envelope_width_ms(onset: int, offset: int, fs: float) -> float:
    """Convert an envelope span into a QRS duration, removing the envelope bias."""
    span = (offset - onset) / fs - ENVELOPE_WINDOW_S
    return float(max(span, 0.0) * 1000.0)


def _qrs_bounds(
    template: np.ndarray, r_index: int, fs: float, threshold_ratio: float = 0.15
) -> Tuple[int, int]:
    """Find the QRS onset and offset indices around the R peak.

    The local-maximum envelope is used so the R and S deflections count as one
    complex, and the search is bounded to +/-90 ms so an ST displacement can
    never stretch the measured QRS.
    """
    r_value = abs(float(template[r_index]))
    if r_value < 1e-6:
        return r_index, r_index
    threshold = r_value * threshold_ratio
    envelope = _envelope(template, fs)

    window = int(0.09 * fs)
    low = max(0, r_index - window)
    segment = envelope[low : r_index + 1]
    below = np.where(segment <= threshold)[0]
    onset = low + (int(below[-1]) + 1) if below.size else low

    high = min(envelope.size, r_index + window + 1)
    segment = envelope[r_index:high]
    above = np.where(segment > threshold)[0]
    offset = r_index + int(above[-1]) if above.size else r_index

    return min(onset, r_index), max(offset, r_index)


def _st_index_placeholder(size: int, j_point: int, fs: float) -> int:
    """Fallback ST sample index when the segment is too short to median."""
    return min(size - 1, j_point + int(0.06 * fs))


def _j_point(template: np.ndarray, r_index: int, fs: float, threshold_ratio: float = 0.15) -> int:
    """End of the QRS complex (the J point) used to anchor the ST measurement.

    The search is bounded to R+60 ms because the QRS never extends further in a
    supraventricular beat, which keeps an ST elevation from being absorbed
    into the QRS and shifting the ST window into the T wave.
    """
    r_value = abs(float(template[r_index]))
    if r_value < 1e-6:
        return r_index
    threshold = r_value * threshold_ratio

    window = int(0.06 * fs)
    low = max(0, r_index - int(0.02 * fs))
    high = min(template.size, r_index + window + 1)

    segment = np.abs(template[low:high])
    above = np.where(segment > threshold)[0]
    if above.size:
        return low + int(above[-1])
    return r_index


def _beat_windows(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> Dict[str, float]:
    """Median QRS duration, QT, PR, ST deviation and T amplitude."""
    empty = {
        "qrs_duration": float("nan"),
        "qt_interval": float("nan"),
        "pr_interval": float("nan"),
        "st_deviation": float("nan"),
        "t_amplitude": float("nan"),
        "r_amplitude": float("nan"),
        "p_wave_amplitude": float("nan"),
    }
    if r_peaks.size == 0:
        return empty

    template, r_index = _median_beat_template(signal, r_peaks, fs)
    if template is None:
        return empty

    # isoelectric baseline: the flat TP segment before the P wave
    baseline_start = max(0, r_index - int(0.26 * fs))
    baseline_end = max(baseline_start + 1, r_index - int(0.17 * fs))
    if baseline_end > template.size:
        baseline_end = template.size
    baseline = float(np.median(template[baseline_start:baseline_end])) if baseline_end > baseline_start else 0.0
    # re-centre the template on the isoelectric line so every measurement that
    # follows is referenced to the same zero
    template = template - baseline

    onset, offset = _qrs_bounds(template, r_index, fs)
    qrs_duration = _envelope_width_ms(onset, offset, fs)

    # ST measurement: median over the ST segment (J+30 ms .. J+80 ms),
    # referenced to the isoelectric baseline.  Averaging the segment rather
    # than sampling one point makes the measurement robust to noise.
    j_point = _j_point(template, r_index, fs)
    st_start = min(template.size - 1, j_point + int(0.03 * fs))
    st_end = min(template.size, j_point + int(0.08 * fs))
    if st_end > st_start + 2:
        st_deviation = float(np.median(template[st_start:st_end]))
    else:
        st_deviation = float(template[_st_index_placeholder(template.size, j_point, fs)])

    # T wave: largest deflection between 140 ms and 400 ms after R
    t_start = min(template.size - 1, r_index + int(0.14 * fs))
    t_end = min(template.size, r_index + int(0.40 * fs))
    if t_end > t_start + 4:
        t_region = template[t_start:t_end]
        t_amplitude = float(t_region[np.argmax(np.abs(t_region))])
    else:
        t_amplitude = float("nan")

    # QT: from QRS onset to the end of the T wave.  The T end is found by
    # walking out from the T peak until the trace returns near the baseline.
    t_end_index = t_end
    if t_end > t_start + 4:
        region = template[t_start:t_end]
        t_peak = t_start + int(np.argmax(np.abs(region)))
        t_peak_value = abs(float(template[t_peak]))
        limit = max(0.1 * t_peak_value, 1e-3)
        end = t_peak
        while end < t_end - 1 and abs(float(template[end])) > limit:
            end += 1
        t_end_index = min(t_end, end)
    qt_interval = (t_end_index - onset) / fs * 1000.0

    # PR interval: P-wave onset to QRS onset
    p_search_start = max(0, r_index - int(0.22 * fs))
    p_search_end = max(p_search_start + 1, r_index - int(0.09 * fs))
    pr_interval = float("nan")
    p_wave_amplitude = float("nan")
    if p_search_end > p_search_start:
        p_region = template[p_search_start:p_search_end]
        p_peak = p_search_start + int(np.argmax(np.abs(p_region)))
        p_wave_amplitude = float(template[p_peak])
        # a P wave well below the noise floor is treated as absent (e.g. AFIB)
        if abs(p_wave_amplitude) > 0.02:
            p_onset = max(0, p_peak - int(0.05 * fs))
            pr_interval = (onset - p_onset) / fs * 1000.0

    return {
        "qrs_duration": float(qrs_duration),
        "qt_interval": float(qt_interval),
        "pr_interval": float(pr_interval),
        "st_deviation": float(st_deviation),
        "t_amplitude": float(t_amplitude),
        "r_amplitude": float(abs(template[r_index])),
        "p_wave_amplitude": float(p_wave_amplitude),
    }


def _qrs_widths(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> List[float]:
    """QRS width (ms) of every detected beat."""
    widths: List[float] = []
    for r in r_peaks:
        if r < 1 or r >= signal.size - 1:
            widths.append(float("nan"))
            continue
        r_value = abs(float(signal[r]))
        threshold = max(r_value * 0.15, 1e-5)
        left = int(r)
        while left > 0 and abs(signal[left]) > threshold:
            left -= 1
        right = int(r)
        while right < signal.size - 1 and abs(signal[right]) > threshold:
            right += 1
        widths.append((right - left) / fs * 1000.0)
    return widths


def _qrs_widths_robust(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> List[float]:
    """QRS width (ms) per beat using the local-maximum envelope."""
    envelope = _envelope(signal, fs)
    widths: List[float] = []
    window = int(0.09 * fs)
    for r in r_peaks:
        r = int(r)
        if r < 1 or r >= signal.size - 1:
            widths.append(float("nan"))
            continue
        r_value = abs(float(signal[r]))
        threshold = max(r_value * 0.15, 1e-5)
        low = max(0, r - window)
        below = np.where(envelope[low : r + 1] <= threshold)[0]
        onset = low + (int(below[-1]) + 1) if below.size else low
        high = min(envelope.size, r + window + 1)
        above = np.where(envelope[r:high] > threshold)[0]
        offset = r + int(above[-1]) if above.size else r
        widths.append(_envelope_width_ms(onset, offset, fs))
    return widths


def _ectopic_beats(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> float:
    """Percentage of premature beats with a wide, atypical QRS (PVC burden).

    A beat is ectopic when it is premature (R-R well below the median) **and**
    has a wider-than-normal QRS **and** an R wave whose amplitude differs
    from the prevailing morphology.  The third condition keeps the irregular
    rhythm of atrial fibrillation from being reported as ectopy.
    """
    if r_peaks.size < 4:
        return 0.0
    rr = np.diff(r_peaks) / fs
    rr_median = float(np.median(rr))
    if rr_median <= 0:
        return 0.0

    widths = _qrs_widths_robust(signal, r_peaks, fs)
    valid = [w for w in widths if np.isfinite(w)]
    if not valid:
        return 0.0
    width_median = float(np.median(valid))

    amplitudes: List[float] = []
    for r in r_peaks:
        index = int(r)
        if 0 < index < signal.size - 1:
            amplitudes.append(abs(float(signal[index])))
        else:
            amplitudes.append(float("nan"))
    valid_amplitudes = [a for a in amplitudes if np.isfinite(a)]
    if len(valid_amplitudes) < 3:
        return 0.0
    amplitude_median = float(np.median(valid_amplitudes)) or 1e-6

    ectopic = 0
    premature = 0
    for index in range(1, len(r_peaks) - 1):
        if rr[index] >= 0.80 * rr_median:
            continue
        premature += 1
        width = widths[index + 1] if index + 1 < len(widths) else float("nan")
        amplitude = amplitudes[index] if index < len(amplitudes) else float("nan")
        wide = np.isfinite(width) and width > 1.25 * width_median
        amplitude = amplitudes[index + 1] if index + 1 < len(amplitudes) else float("nan")
        atypical = np.isfinite(amplitude) and abs(amplitude - amplitude_median) > 0.25 * amplitude_median
        if wide and atypical:
            ectopic += 1
    if premature == 0:
        return 0.0
    return float(ectopic / max(premature, 1) * 100.0)


def extract_features(pre: PreprocessedECG) -> Dict[str, FeatureValue]:
    """Compute the full ECG feature set with clinical status flags."""
    signal = pre.clean[0] if pre.clean.ndim == 2 else pre.clean
    fs = pre.sampling_rate
    r_peaks = pre.r_peaks if pre.r_peaks.size else detect_r_peaks(signal, fs)

    rr_intervals = np.diff(r_peaks) / fs if r_peaks.size >= 2 else np.array([])
    rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.5)] if rr_intervals.size else np.array([])

    heart_rate = float(60.0 / np.mean(rr_valid)) if rr_valid.size else float("nan")
    rr_mean = float(np.mean(rr_valid) * 1000.0) if rr_valid.size else float("nan")
    rr_sd = float(np.std(rr_valid, ddof=1) * 1000.0) if rr_valid.size > 1 else float("nan")
    rr_cv = (
        float(np.std(rr_valid) / np.mean(rr_valid) * 100.0)
        if rr_valid.size > 1 and np.mean(rr_valid) > 0
        else float("nan")
    )

    hrv = (
        _hrv_from_rr(rr_valid, fs)
        if rr_valid.size
        else {"rmssd": float("nan"), "pnn50": float("nan"), "lf_hf": float("nan")}
    )

    beats = _beat_windows(signal, r_peaks, fs)
    pvc_fraction = _ectopic_beats(signal, r_peaks, fs)

    # Irregularity: mean absolute successive difference relative to the mean
    if rr_valid.size >= 3:
        rr_irregularity = float(np.mean(np.abs(np.diff(rr_valid))) / np.mean(rr_valid) * 100.0)
    else:
        rr_irregularity = 0.0

    if signal.size > 401:
        baseline_wander = float(
            np.std(signal - sps.savgol_filter(signal, min(signal.size // 2 * 2 - 1, 401), 3))
        )
    else:
        baseline_wander = 0.0
    sqi = float(pre.quality.get("sqi", 0.0))

    values: Dict[str, float] = {
        "heart_rate": heart_rate,
        "rr_mean": rr_mean,
        "rr_sd": rr_sd,
        "rr_cv": rr_cv,
        "rmssd": hrv["rmssd"],
        "pnn50": hrv["pnn50"],
        "lf_hf": hrv["lf_hf"],
        "qrs_duration": beats["qrs_duration"],
        "qt_interval": beats["qt_interval"],
        "pr_interval": beats["pr_interval"],
        "st_deviation": beats["st_deviation"],
        "t_amplitude": beats["t_amplitude"],
        "r_amplitude": beats["r_amplitude"],
        "pvc_fraction": pvc_fraction,
        "rr_irregularity": rr_irregularity,
        "signal_quality": sqi,
        "baseline_wander": baseline_wander,
        "p_wave_amplitude": beats["p_wave_amplitude"],
    }

    features: Dict[str, FeatureValue] = {}
    for name in FEATURE_ORDER:
        value = float(values.get(name, float("nan")))
        unit, low, high, explanation = FEATURE_REFERENCE[name]
        if not np.isfinite(value):
            features[name] = FeatureValue(
                name=name,
                value=float("nan"),
                unit=unit,
                reference_low=low,
                reference_high=high,
                status="normal",
                explanation=f"{explanation} Not measurable in this recording.",
                display="n/a",
            )
            continue
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
    scales = np.array([FEATURE_SCALES[name] for name in FEATURE_ORDER], dtype=np.float32)
    return raw / scales


def feature_importance_hint(features: Dict[str, FeatureValue]) -> List[Tuple[str, float, str]]:
    """Rank features by how far they stray from their reference band.

    Used by the heuristic baseline and as an explainability prior.
    """
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
            severity = 0.0
            direction = "neutral"
        if severity == 0:
            continue
        scored.append((name, float(severity), direction))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
