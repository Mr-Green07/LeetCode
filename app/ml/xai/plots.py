"""Rendering helpers for the explainability visuals.

Every chart is produced with matplotlib's Agg backend and returned as PNG
bytes so it can be inlined into JSON responses or embedded into the PDF
report.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from app.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DPI = 120
FIGSIZE = (10.0, 3.4)


def _to_base64(buffer: io.BytesIO) -> str:
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_signal(
    signal: Sequence[float],
    sampling_rate: float,
    *,
    r_peaks: Optional[np.ndarray] = None,
    title: str = "Conditioned signal",
    y_label: str = "Amplitude",
    color: str = "#2f6fed",
) -> str:
    """Render a 1-D signal as a PNG (base64)."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        values = np.zeros(1)

    time = np.arange(values.size) / max(sampling_rate, 1e-6)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(time, values, color=color, linewidth=1.0)
    if r_peaks is not None and len(r_peaks):
        peaks = np.asarray(r_peaks)
        ax.scatter(peaks / max(sampling_rate, 1e-6), values[peaks], s=14, color="#e5484d", zorder=3, label="R peak")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)


def plot_saliency(
    signal: Sequence[float],
    saliency: Sequence[float],
    sampling_rate: float,
    *,
    title: str = "Attribution map",
) -> str:
    """Render a signal with its saliency map underneath."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    weights = np.asarray(saliency, dtype=np.float64).reshape(-1)
    if weights.size != values.size:
        weights = np.resize(weights, values.size)

    time = np.arange(values.size) / max(sampling_rate, 1e-6)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, dpi=DPI, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(time, values, color="#2f6fed", linewidth=1.0)
    ax1.set_title(title, fontsize=11, loc="left")
    ax1.set_ylabel("Signal", fontsize=9)
    ax1.grid(alpha=0.25, linewidth=0.6)

    limit = float(np.max(np.abs(weights))) or 1.0
    ax2.fill_between(time, 0, weights, color="#f5a524", alpha=0.85)
    ax2.axhline(0, color="#8b93a7", linewidth=0.7)
    ax2.set_ylim(-limit, limit)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Importance", fontsize=9)
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)


def plot_overlay(grayscale: np.ndarray, cam: np.ndarray, *, title: str = "Grad-CAM localisation") -> str:
    """Overlay a Grad-CAM map on a greyscale radiograph."""
    image = np.asarray(grayscale, dtype=np.float64)
    heat = np.asarray(cam, dtype=np.float64)
    if image.ndim == 3:
        image = image[0]
    if heat.shape != image.shape:
        try:
            from PIL import Image

            resized = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), Image.BILINEAR
            )
            heat = np.asarray(resized, dtype=np.float64) / 255.0
        except Exception:  # pragma: no cover
            heat = np.resize(heat, image.shape)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=DPI)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    heat_map = ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=11, loc="left")
    ax.axis("off")
    fig.colorbar(heat_map, ax=ax, fraction=0.046, pad=0.04).set_label("Class relevance", fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)


def plot_image(grayscale: np.ndarray, *, title: str = "Radiograph") -> str:
    """Render the preprocessed radiograph as a PNG (base64)."""
    image = np.asarray(grayscale, dtype=np.float64)
    if image.ndim == 3:
        image = image[0]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=DPI)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10, loc="left")
    ax.axis("off")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)


def plot_class_probabilities(probabilities: Dict[str, float], *, title: str = "Class probabilities") -> str:
    """Horizontal bar chart of the class probability distribution."""
    items = sorted(probabilities.items(), key=lambda item: item[1])
    if not items:
        items = [("unknown", 0.0)]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(items) + 1.0), dpi=DPI)
    colors = ["#2f6fed" if value < max(values) else "#e5484d" for value in values]
    ax.barh(labels, values, color=colors, height=0.6)
    for index, value in enumerate(values):
        ax.text(min(value + 0.015, 1.0), index, f"{value:.1%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Probability", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)


def plot_feature_contributions(contributions: Iterable[Tuple[str, float]]) -> str:
    """Diverging bar chart of feature contributions."""
    items = list(contributions)
    if not items:
        items = [("no features", 0.0)]
    items = items[:12]
    names = [item[0] for item in items][::-1]
    values = [float(item[1]) for item in items][::-1]

    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(names) + 1.0), dpi=DPI)
    colors = ["#e5484d" if value > 0 else "#2f6fed" for value in values]
    ax.barh(names, values, color=colors, height=0.6)
    ax.axvline(0, color="#8b93a7", linewidth=0.8)
    ax.set_title("Feature contributions to the prediction", fontsize=11, loc="left")
    ax.set_xlabel("Contribution", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)


def plot_band_spectrum(frequencies: Sequence[float], powers: Sequence[float], *, title: str = "Power spectral density") -> str:
    """Plot a power spectrum (used for the EEG band-power view)."""
    freqs = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    values = np.asarray(powers, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.fill_between(freqs, 0, values, color="#2f6fed", alpha=0.7)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Power", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return _to_base64(buffer)
