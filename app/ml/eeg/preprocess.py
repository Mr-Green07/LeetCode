"""EEG signal conditioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as sps

from app.core.logging import get_logger

logger = get_logger(__name__)

TARGET_FS = 256.0
WINDOW_SECONDS = 10.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SECONDS)


@dataclass
class PreprocessedEEG:
    """Result of the EEG conditioning pipeline."""

    window: np.ndarray
    """Shape ``(n_channels, WINDOW_SAMPLES)``."""

    clean: np.ndarray
    """Filtered, re-referenced signal in microvolts."""

    sampling_rate: float
    channel_names: List[str] = field(default_factory=list)
    quality: Dict[str, float] = field(default_factory=dict)
    source_channels: int = 1
    artifact_ratio: float = 0.0

    @property
    def primary(self) -> np.ndarray:
        return self.clean[0] if self.clean.ndim == 2 else self.clean


def _as_float_array(samples) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def bandpass(samples: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (0.5-45 Hz clinical default)."""
    arr = _as_float_array(samples)
    if arr.size < 32:
        return arr
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    low = max(low, 0.05)
    if high <= low:
        return arr
    b, a = sps.butter(order, [low / nyq, high / nyq], btype="band")
    return sps.filtfilt(b, a, arr)


def notch_filter(samples: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove 50/60 Hz mains interference."""
    arr = _as_float_array(samples)
    if arr.size < 64 or freq >= fs / 2.0:
        return arr
    b, a = sps.iirnotch(freq, Q=q, fs=fs)
    return sps.filtfilt(b, a, arr)


def resample_signal(samples: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Polyphase resampling."""
    arr = _as_float_array(samples)
    if abs(orig_fs - target_fs) < 1e-6:
        return arr
    gcd = np.gcd(int(round(orig_fs)), int(round(target_fs)))
    up = int(round(target_fs / gcd))
    down = int(round(orig_fs / gcd))
    return sps.resample_poly(arr, up, down)


def average_reference(signal: np.ndarray) -> np.ndarray:
    """Re-reference all channels to their average (removes common-mode noise)."""
    if signal.ndim == 2 and signal.shape[0] > 1:
        return signal - signal.mean(axis=0, keepdims=True)
    return signal


def detect_artifacts(signal: np.ndarray, fs: float) -> float:
    """Fraction of the recording contaminated by movement / eye-blink artefacts.

    The comparison is made between local RMS windows rather than against the
    raw amplitude, otherwise a strong rhythmic background (for example a
    posterior alpha rhythm) would be mistaken for an artefact.
    """
    if signal.size < 10:
        return 0.0
    window = max(int(0.25 * fs), 4)
    rms = np.sqrt(np.convolve(signal ** 2, np.ones(window) / window, mode="same"))
    typical = float(np.median(rms))
    if typical <= 0:
        return 0.0
    outlier = rms > 8.0 * typical
    flat = rms < 0.02 * typical
    return float(np.clip(np.mean(outlier | flat), 0.0, 1.0))


def signal_quality(signal: np.ndarray, fs: float, artifact_ratio: float) -> Dict[str, float]:
    """Combine crest factor, noise floor and artefact load into a quality index."""
    if signal.size < 16:
        return {"sqi": 0.0, "snr_db": 0.0}
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / max(rms, 1e-9)

    ordered = np.sort(np.abs(signal))
    baseline = ordered[: max(1, ordered.size // 10)]
    noise = float(np.std(baseline)) or 1e-6
    snr_db = float(20 * np.log10(max(rms, 1e-9) / noise))

    sqi = (
        float(np.clip((crest - 1.5) / 3.0, 0.0, 1.0))
        * float(np.clip(snr_db / 25.0, 0.0, 1.0))
        * float(np.clip(1.0 - artifact_ratio, 0.0, 1.0))
    )
    return {"sqi": sqi, "snr_db": snr_db}


def preprocess_eeg(
    samples,
    sampling_rate: float,
    *,
    channel_names: Optional[List[str]] = None,
    unit: str = "auto",
) -> PreprocessedEEG:
    """Full EEG conditioning pipeline.

    Accepts a 1-D single-channel signal or a 2-D ``(channels, samples)`` array.
    """
    raw = np.asarray(samples, dtype=np.float64)
    if raw.ndim == 2:
        channels = raw if raw.shape[0] <= raw.shape[1] else raw.T
    else:
        channels = raw.reshape(1, -1)

    if channels.shape[1] < 64:
        raise ValueError("EEG signal is too short to analyse")

    fs = float(sampling_rate)
    cleaned: List[np.ndarray] = []
    for row in channels:
        row = _as_float_array(row)
        if (unit or "auto").lower() in ("uv", "µv", "microvolt") or (unit or "auto").lower() == "auto":
            peak = float(np.percentile(np.abs(row), 99)) if row.size else 0.0
            if peak > 200:  # almost certainly millivolts
                row = row * 1000.0
        row = bandpass(row, fs, 0.5, min(45.0, fs / 2.0 * 0.95))
        if fs >= 100:
            row = notch_filter(row, fs, 50.0)
        row = resample_signal(row, fs, TARGET_FS)
        cleaned.append(row)

    stacked = np.vstack(cleaned)
    min_len = min(row.size for row in cleaned)
    stacked = stacked[:, :min_len]
    stacked = average_reference(stacked)

    # pick the channel with the clearest signal
    best_idx = 0
    best_score = -np.inf
    for idx, row in enumerate(stacked):
        freqs, psd = sps.welch(row, fs=TARGET_FS, nperseg=min(256, row.size))
        band = float(np.trapezoid(psd[(freqs >= 1) & (freqs <= 40)], freqs[(freqs >= 1) & (freqs <= 40)]))
        score = band / (float(np.var(row)) + 1e-9)
        if score > best_score:
            best_score, best_idx = score, idx

    channel = stacked[best_idx]
    artifact_ratio = detect_artifacts(channel, TARGET_FS)
    quality = signal_quality(channel, TARGET_FS, artifact_ratio)

    # 10 second window, preferring a low-artefact stretch
    start = _select_window(channel, TARGET_FS, WINDOW_SAMPLES)
    window = stacked[:, start : start + WINDOW_SAMPLES]
    if window.shape[1] < WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - window.shape[1]
        window = np.pad(window, ((0, 0), (0, pad)), mode="edge" if window.shape[1] else "constant")

    names = list(channel_names) if channel_names and len(channel_names) == stacked.shape[0] else []
    if not names:
        names = [f"Ch{i + 1}" for i in range(stacked.shape[0])]

    return PreprocessedEEG(
        window=window.astype(np.float32),
        clean=window.astype(np.float64),
        sampling_rate=TARGET_FS,
        channel_names=names,
        quality=quality,
        source_channels=stacked.shape[0],
        artifact_ratio=artifact_ratio,
    )


def _select_window(signal: np.ndarray, fs: float, length: int) -> int:
    """Pick the 10 s window with the lowest artefact load."""
    if signal.size <= length:
        return 0
    step = length // 4
    best_offset, best_score = 0, np.inf
    for offset in range(0, signal.size - length + 1, step):
        chunk = signal[offset : offset + length]
        robust_std = float(np.median(np.abs(chunk - np.median(chunk))) * 1.4826) or 1.0
        score = float(np.mean(np.abs(chunk) > 4.0 * robust_std))
        if score < best_score:
            best_score, best_offset = score, offset
    return best_offset


def build_model_input(window: np.ndarray) -> np.ndarray:
    """Normalise one EEG channel into the ``(channels, height, length)`` layout.

    A batch therefore has the shape ``(n_samples, 1, 1, 2560)``.
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)[:WINDOW_SAMPLES]
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros((1, 1, arr.size), dtype=np.float32)
    return ((arr - arr.mean()) / std).reshape(1, 1, -1)
