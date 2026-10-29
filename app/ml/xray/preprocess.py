"""Chest X-ray image conditioning.

Loads the uploaded image, normalises it to a single greyscale tensor and
computes a small set of image-level measurements that both the heuristic
baseline and the explainability layer use.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from app.core.errors import UnsupportedFormatError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

IMAGE_SIZE = 224


@dataclass
class PreprocessedXray:
    """Result of the X-ray conditioning pipeline."""

    array: np.ndarray
    """Shape ``(1, IMAGE_SIZE, IMAGE_SIZE)`` float32 in ``[0, 1]``."""

    grayscale: np.ndarray
    """Raw greyscale view at ``IMAGE_SIZE`` used for overlays."""

    original_size: Tuple[int, int]
    quality: Dict[str, float] = field(default_factory=dict)
    is_chest_xray_like: bool = True
    source_format: str = "PNG"

    @property
    def height(self) -> int:
        return self.array.shape[1]

    @property
    def width(self) -> int:
        return self.array.shape[2]


def decode_image(data: bytes) -> "PIL.Image.Image":  # noqa: F821 - lazy import
    """Decode raw bytes into a PIL image (PNG / JPEG / DICOM)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image handling") from exc

    if not data:
        raise ValidationError("Uploaded image is empty")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        # Fall back to DICOM if Pillow cannot decode it
        try:
            import pydicom  # type: ignore

            dataset = pydicom.dcmread(io.BytesIO(data))
            array = dataset.pixel_array.astype(np.float32)
            if array.ndim == 3:
                array = array.max(axis=-1)
            if array.max() > 1:
                array = array / 255.0
            image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
            return image
        except Exception:
            raise UnsupportedFormatError(
                "Unsupported image format. Upload a PNG, JPEG or DICOM chest radiograph."
            ) from exc


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on a uint8 image."""
    img = np.asarray(image, dtype=np.uint8)
    height, width = img.shape
    rows, cols = tile_grid
    th, tw = max(height // rows, 1), max(width // cols, 1)
    out = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            y0, y1 = i * th, (i + 1) * th if i < rows - 1 else height
            x0, x1 = j * tw, (j + 1) * tw if j < cols - 1 else width
            tile = img[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            limit = max(1.0, clip_limit * tile.size / 256.0)
            excess = float(np.sum(hist - np.clip(hist, 0.0, limit)))
            hist = np.clip(hist, 0.0, limit)
            if excess > 0:
                hist += excess / 256.0
            cdf = hist.cumsum()
            cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1.0)
            lut = np.clip(cdf * 255.0, 0, 255).astype(np.uint8)
            out[y0:y1, x0:x1] = lut[tile]
    return out


def to_grayscale(image) -> np.ndarray:
    """Convert a PIL image to a uint8 greyscale array."""
    from PIL import Image

    gray = image.convert("L")
    return np.asarray(gray, dtype=np.uint8)


def resize_image(gray: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bilinear resize of a greyscale array to ``size x size``."""
    try:
        from PIL import Image

        pil = Image.fromarray(gray.astype(np.uint8), mode="L")
        return np.asarray(pil.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    except Exception:  # pragma: no cover
        height, width = gray.shape
        ys = np.linspace(0, height - 1, size).astype(int)
        xs = np.linspace(0, width - 1, size).astype(int)
        return gray[np.ix_(ys, xs)]


def normalize(gray: np.ndarray) -> np.ndarray:
    """Scale to ``[0, 1]`` and build the ``(1, H, W)`` network input."""
    arr = np.asarray(gray, dtype=np.float32)
    minimum, maximum = float(arr.min()), float(arr.max())
    if maximum - minimum < 1e-6:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - minimum) / (maximum - minimum)
    return arr[None, :, :]


def looks_like_chest_xray(gray: np.ndarray) -> bool:
    """Cheap sanity check that the upload resembles a frontal chest radiograph."""
    height, width = gray.shape
    if height == 0 or width == 0:
        return False
    aspect = width / max(height, 1)
    if aspect < 0.45 or aspect > 2.2:
        return False
    # chest films are mostly dark (lungs) with a bright central column (mediastinum)
    centre = gray[:, int(width * 0.4) : int(width * 0.6)]
    sides = np.concatenate([gray[:, : int(width * 0.25)], gray[:, int(width * 0.75) :]], axis=1)
    if centre.size == 0 or sides.size == 0:
        return False
    return float(centre.mean()) >= float(sides.mean()) * 0.9


def image_quality(gray: np.ndarray) -> Dict[str, float]:
    """Sharpness and contrast measures for the uploaded radiograph."""
    values = np.asarray(gray, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values ** 2)))
    peak = float(np.max(np.abs(values)))
    crest = peak / max(rms, 1e-9)

    gy, gx = np.gradient(values)
    laplacian_variance = float(np.var(gy + gx))
    contrast = float(np.std(values))

    # a radiograph has a wide dynamic range (dark lungs, bright mediastinum)
    sharpness = float(np.clip(laplacian_variance / (contrast ** 2 + 1e-9) * 60.0, 0.0, 1.0))
    sqi = float(np.clip((crest - 1.5) / 2.0, 0.0, 1.0)) * float(np.clip(contrast / 25.0, 0.0, 1.0))
    return {"sqi": sqi, "sharpness": sharpness, "contrast": contrast}


def preprocess_xray(data: bytes, *, size: int = IMAGE_SIZE) -> PreprocessedXray:
    """Full X-ray conditioning pipeline from raw upload bytes."""
    image = decode_image(data)
    original_size = (int(image.size[0]), int(image.size[1]))
    source_format = (image.format or "PNG").upper()

    gray_full = to_grayscale(image)
    gray = resize_image(gray_full, size)
    enhanced = clahe(gray)
    array = normalize(enhanced)

    quality = image_quality(gray)
    quality["is_chest_xray_like"] = 1.0 if looks_like_chest_xray(gray) else 0.0

    return PreprocessedXray(
        array=array.astype(np.float32),
        grayscale=enhanced,
        original_size=original_size,
        quality=quality,
        is_chest_xray_like=bool(quality["is_chest_xray_like"]),
        source_format=source_format,
    )


def preprocess_xray_array(gray: np.ndarray, *, size: int = IMAGE_SIZE) -> PreprocessedXray:
    """Condition an in-memory greyscale array (used by the training scripts)."""
    source = np.asarray(gray)
    if source.ndim == 3:
        source = source[0]
    resized = resize_image(source.astype(np.uint8), size)
    enhanced = clahe(resized)
    array = normalize(enhanced)
    quality = image_quality(resized)
    quality["is_chest_xray_like"] = 1.0 if looks_like_chest_xray(resized) else 0.0
    return PreprocessedXray(
        array=array.astype(np.float32),
        grayscale=enhanced,
        original_size=(int(source.shape[1]), int(source.shape[0])),
        quality=quality,
        is_chest_xray_like=bool(quality["is_chest_xray_like"]),
        source_format="ARRAY",
    )


def build_model_input(gray: np.ndarray) -> np.ndarray:
    """Normalise a greyscale view into the ``(1, H, W)`` network layout."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return (arr / 255.0)[None, :, :]
