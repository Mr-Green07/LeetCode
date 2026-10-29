"""ECG signal conditioning.

Turns a raw ECG recording (single- or multi-lead) into a clean, fixed-length
window that the network and the feature extractor can both consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from app.core.logging import get_logger

logger = get_logger(__name__)

TARGET_FS = 500.0
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SECONDS)
SQUARE_BINS = 12


@dataclass
class PreprocessedECG:
    """Result of the ECG conditioning pipeline."""

    window: np.ndarray
    """Shape ``(n_leads, WINDOW_SAMPLES)`` - the model input."""

    clean: np.ndarray
    """Filtered signal for the same window, in physical units (mV)."""

    sampling_rate: float
    r_peaks: np.ndarray
    heart_rate: Optional[float]
    lead_names: List[str] = field(default_factory=list)
    quality: Dict[str, float] = field(default_factory=dict)
    source_leads: int = 1

    @property
    def primary(self) -> np.ndarray:
        """The lead used for feature extraction and visualisation."""
        return self.clean[0] if self.clean.ndim == 2 else self.clean


# --------------------------------------------------------------------------- #
# Filtering primitives
# --------------------------------------------------------------------------- #
def _as_float_array(samples) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def bandpass(samples: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass with automatic NaN interpolation."""
    arr = _as_float_array(samples)
    if arr.size < 16:
        return arr
    nyq = fs / 2.0
    high = min(high, nyq * 0.98)
    low = max(low, 0.01)
    if high <= low:
        return arr
    b, a = sps.butter(order, [low / nyq, high / nyq], btype="band")
    return sps.filtfilt(b, a, arr)


def remove_baseline_wander(samples: np.ndarray, fs: float, cutoff: float = 0.05) -> np.ndarray:
    """High-pass baseline removal.

    The 0.05 Hz corner follows the AHA/ANSI ECG standard for ST-segment
    measurement: a 0.5 Hz high-pass distorts the sustained ST plateau enough
    to bias the ST level by more than the deviation we are trying to detect.
    """
    arr = _as_float_array(samples)
    if arr.size < 16:
        return arr
    b, a = sps.butter(2, cutoff / (fs / 2.0), btype="high")
    return sps.filtfilt(b, a, arr)


def notch_filter(samples: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove mains interference at ``freq`` Hz (skipped when not in band)."""
    arr = _as_float_array(samples)
    if arr.size < 32 or freq >= fs / 2.0:
        return arr
    b, a = sps.iirnotch(freq, Q=q, fs=fs)
    return sps.filtfilt(b, a, arr)


def resample_signal(samples: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Polyphase resampling to a fixed target rate."""
    arr = _as_float_array(samples)
    if abs(orig_fs - target_fs) < 1e-6:
        return arr
    gcd = np.gcd(int(round(orig_fs)), int(round(target_fs)))
    up = int(round(target_fs / gcd))
    down = int(round(orig_fs / gcd))
    return sps.resample_poly(arr, up, down)


def zscore(samples: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance normalisation."""
    arr = _as_float_array(samples)
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def to_millivolts(samples: np.ndarray, unit: str = "auto") -> np.ndarray:
    """Best-effort unit harmonisation to millivolts."""
    arr = _as_float_array(samples)
    unit = (unit or "auto").lower()
    if unit in ("uv", "µv", "microvolt", "microvolts"):
        return arr / 1000.0
    if unit in ("mv", "millivolt", "millivolts"):
        return arr
    # auto: a millivolt ECG usually peaks above 0.5; a microvolt signal stays small
    peak = float(np.percentile(np.abs(arr), 99)) if arr.size else 0.0
    if 0 < peak < 0.2:
        return arr / 1000.0
    return arr


# --------------------------------------------------------------------------- #
# Beat detection (Pan-Tompkins flavoured)
# --------------------------------------------------------------------------- #
def detect_r_peaks(signal: np.ndarray, fs: float, min_distance_s: float = 0.25) -> np.ndarray:
    """Detect R-peak sample indices.

    Classic Pan-Tompkins pipeline with an adaptive threshold:

    band-pass 5-15 Hz -> derivative -> squaring -> moving-window integration
    -> running signal/noise peak estimates (SPKI / NPKI) -> adaptive threshold
    -> refractory-period peak picking.

    The adaptive threshold matters clinically: a wide, ectopic QRS spreads its
    energy over more samples than a narrow sinus beat, so a single global
    threshold silently drops premature ventricular complexes.
    """
    arr = _as_float_array(signal)
    if arr.size < int(fs):
        return np.array([], dtype=int)

    filtered = bandpass(arr, fs, 5.0, 15.0, order=2)
    derivative = np.gradient(filtered)
    squared = derivative ** 2
    window = max(int(round(0.12 * fs)), 1)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    if integrated.size == 0 or integrated.max() <= 0:
        return np.array([], dtype=int)

    refractory = int(round(min_distance_s * fs))
    spki = float(np.percentile(integrated, 98))
    npki = float(np.percentile(integrated, 50))

    # a light smoothing is used to refine each candidate onto the true R peak
    refine_window = max(int(round(0.02 * fs)), 1)
    smoothed = np.convolve(arr, np.ones(refine_window) / refine_window, mode="same")
    refine_radius = int(round(0.045 * fs))

    peaks: List[int] = []
    block = int(round(1.0 * fs))
    size = integrated.size
    start = 0

    while start < size:
        end = min(size, start + block)
        chunk = integrated[start:end]
        threshold = npki + 0.25 * max(spki - npki, 0.0)

        above = np.where(chunk > threshold)[0]
        if above.size:
            groups = np.split(above, np.where(np.diff(above) != 1)[0] + 1)
            accepted: List[int] = []
            for group in groups:
                candidate = start + int(group[int(np.argmax(chunk[group]))])
                low = max(0, candidate - refine_radius)
                high = min(smoothed.size, candidate + refine_radius + 1)
                refined = low + int(np.argmax(np.abs(smoothed[low:high])))
                if peaks and refined - peaks[-1] < refractory:
                    if integrated[refined] > integrated[peaks[-1]]:
                        peaks[-1] = refined
                    continue
                peaks.append(refined)
                accepted.append(refined)
            if accepted:
                spki = 0.875 * spki + 0.125 * float(np.max(integrated[accepted]))
            else:
                npki = 0.875 * npki + 0.125 * float(np.max(chunk[above]))
        elif chunk.size:
            npki = 0.875 * npki + 0.125 * float(np.max(chunk))

        start = end

    return np.array(sorted(peaks), dtype=int)


def segment_beats(signal: np.ndarray, r_peaks: np.ndarray, fs: float, before: float = 0.25, after: float = 0.45) -> List[np.ndarray]:
    """Cut individual beats around each R peak."""
    beats: List[np.ndarray] = []
    pre = int(before * fs)
    post = int(after * fs)
    for r in r_peaks:
        start = int(r) - pre
        stop = int(r) + post
        if start < 0 or stop > len(signal):
            continue
        beats.append(signal[start:stop])
    return beats


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _quality_of(clean: np.ndarray, raw: np.ndarray) -> Dict[str, float]:
    """Signal-quality metrics.

    The noise floor is estimated from the quietest tenth of the recording (the
    isoelectric baseline between beats) rather than from the sample-to-sample
    difference, which would count the QRS itself as noise and penalise a
    perfectly clean recording.
    """
    if clean.size == 0:
        return {"sqi": 0.0, "snr_db": 0.0, "clipped": 0.0}

    rms = float(np.sqrt(np.mean(clean ** 2)))
    peak = float(np.max(np.abs(clean)))
    crest = peak / max(rms, 1e-9)

    ordered = np.sort(np.abs(clean))
    baseline = ordered[: max(1, ordered.size // 10)]
    noise = float(np.std(baseline)) or 1e-6
    snr_db = float(20 * np.log10(max(rms, 1e-9) / noise))

    sqi = float(np.clip((crest - 1.5) / 3.0, 0.0, 1.0)) * float(np.clip(snr_db / 25.0, 0.0, 1.0))

    if raw.size:
        span = float(np.max(raw) - np.min(raw)) or 1.0
        clipped = float(np.mean((raw <= np.min(raw) + 0.01 * span) | (raw >= np.max(raw) - 0.01 * span)))
        sqi *= float(np.clip(1.0 - clipped * 2.0, 0.05, 1.0))
    else:
        clipped = 0.0

    return {"sqi": sqi, "snr_db": snr_db, "clipped": clipped}


def preprocess_ecg(
    samples,
    sampling_rate: float,
    *,
    lead_names: Optional[List[str]] = None,
    unit: str = "auto",
) -> PreprocessedECG:
    """Full ECG conditioning pipeline.

    Accepts a 1-D single-lead signal or a 2-D ``(leads, samples)`` array and
    returns a fixed ``(n_leads, WINDOW_SAMPLES)`` window plus diagnostic info.
    """
    raw = np.asarray(samples, dtype=np.float64)
    was_2d = raw.ndim == 2
    if was_2d:
        leads = raw if raw.shape[0] <= raw.shape[1] else raw.T
    else:
        leads = raw.reshape(1, -1)

    if leads.shape[1] < 16:
        raise ValueError("ECG signal is too short to analyse")

    fs = float(sampling_rate)
    processed_leads: List[np.ndarray] = []
    quality_acc: List[Dict[str, float]] = []

    for row in leads:
        row = to_millivolts(_as_float_array(row), unit)
        row = remove_baseline_wander(row, fs)
        row = bandpass(row, fs, 0.05, min(40.0, fs / 2.0 * 0.95), order=2)
        if fs >= 100:
            row = notch_filter(row, fs, 50.0)
        row = resample_signal(row, fs, TARGET_FS)
        processed_leads.append(row)
        quality_acc.append(_quality_of(row, row))

    stacked = np.vstack(processed_leads)

    # align every lead on the longest recording
    min_len = min(row.size for row in processed_leads)
    stacked = stacked[:, :min_len]

    # The full conditioned recording drives rhythm analysis; the fixed-length
    # window is only the network input.
    clean = stacked.astype(np.float64)

    start = _select_window(stacked[0], TARGET_FS, WINDOW_SAMPLES)
    window = stacked[:, start : start + WINDOW_SAMPLES]
    if window.shape[1] < WINDOW_SAMPLES:  # recording shorter than 5 s -> pad
        pad = WINDOW_SAMPLES - window.shape[1]
        window = np.pad(window, ((0, 0), (0, pad)), mode="edge" if window.shape[1] else "constant")

    r_peaks = detect_r_peaks(clean[0], TARGET_FS)
    heart_rate = None
    if r_peaks.size >= 3:
        rr = np.diff(r_peaks) / TARGET_FS
        rr_valid = rr[(rr > 0.3) & (rr < 2.5)]
        if rr_valid.size:
            heart_rate = float(60.0 / np.mean(rr_valid))

    quality = {
        "sqi": float(np.mean([q["sqi"] for q in quality_acc])),
        "snr_db": float(np.mean([q["snr_db"] for q in quality_acc])),
        "clipped": float(np.mean([q["clipped"] for q in quality_acc])),
    }

    names = list(lead_names) if lead_names and len(lead_names) == stacked.shape[0] else []
    if not names:
        names = [f"Lead {i + 1}" for i in range(stacked.shape[0])]

    return PreprocessedECG(
        window=window.astype(np.float32),
        clean=clean,
        sampling_rate=TARGET_FS,
        r_peaks=r_peaks,
        heart_rate=heart_rate,
        lead_names=names,
        quality=quality,
        source_leads=stacked.shape[0],
    )


def _select_window(signal: np.ndarray, fs: float, length: int) -> int:
    """Choose the window offset with the strongest energy (most informative)."""
    if signal.size <= length:
        return 0
    step = length // 4
    best_offset, best_energy = 0, -np.inf
    for offset in range(0, signal.size - length + 1, step):
        chunk = signal[offset : offset + length]
        energy = float(np.var(chunk))
        if energy > best_energy:
            best_energy, best_offset = energy, offset
    return best_offset


def build_model_input(window: np.ndarray) -> np.ndarray:
    """Normalise one ECG channel into the ``(channels, length)`` network layout.

    The result is the per-sample input for the 1-D CNN; a batch therefore has
    the shape ``(n_samples, 1, 2500)``.
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)[:WINDOW_SAMPLES]
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros((1, arr.size), dtype=np.float32)
    return ((arr - arr.mean()) / std).reshape(1, -1)


def summarize_signal(pre: PreprocessedECG, max_points: int = 1200) -> List[Tuple[float, float]]:
    """Down-sample a lead for dashboard plotting."""
    signal = pre.clean[0] if pre.clean.ndim == 2 else pre.clean
    if signal.size == 0:
        return []
    step = max(1, signal.size // max_points)
    return [(float(i) / pre.sampling_rate, float(v)) for i, v in enumerate(signal[::step])]
