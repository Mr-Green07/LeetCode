"""Parsers for uploaded biosignal files.

Supported containers:

* ``.csv``  - one column of samples, or multiple columns (one per channel);
* ``.json`` - ``{"samples": [...], "sampling_rate": 500, "channel_names": [...]}``;
* ``.txt``  - whitespace / comma separated values;
* ``.mat``  - MATLAB v5 file (scipy) exposing an array plus an optional
  ``fs`` variable.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.errors import UnsupportedFormatError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedSignal:
    """A parsed biosignal upload."""

    samples: np.ndarray
    sampling_rate: float
    channel_names: List[str] = field(default_factory=list)
    source_format: str = "unknown"

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0]) if self.samples.ndim == 2 else 1


def _to_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _guess_sampling_rate(header: Optional[str], default: float) -> float:
    """Look for an explicit sampling rate in a CSV header row."""
    if not header:
        return default
    match = re.search(r"(?:fs|sampling[_\s-]*rate|sample[_\s-]*rate)\s*[:=]\s*(\d+(?:\.\d+)?)", header, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def parse_csv(data: bytes, default_rate: float) -> ParsedSignal:
    """Parse a CSV/TSV signal file.

    The first non-numeric line is treated as a header when it carries a
    sampling-rate declaration.  Subsequent columns are treated as channels.
    """
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive
        raise UnsupportedFormatError("Could not decode the uploaded file as text.") from exc

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValidationError("The uploaded file is empty.")

    header: Optional[str] = None
    start = 0
    first_cells = [cell.strip() for cell in re.split(r"[,\t;]", lines[0])]
    if not all(_is_number(cell) for cell in first_cells if cell):
        header = lines[0]
        start = 1

    rows: List[List[float]] = []
    for line in lines[start:]:
        cells = [cell.strip() for cell in re.split(r"[,\t;]", line) if cell.strip() != ""]
        numeric: List[float] = []
        for cell in cells:
            if _is_number(cell):
                numeric.append(float(cell))
        if numeric:
            rows.append(numeric)

    if not rows:
        raise ValidationError("No numeric samples were found in the uploaded file.")

    width = max(len(row) for row in rows)
    rows = [row for row in rows if len(row) == width]
    array = np.asarray(rows, dtype=np.float64).T  # (channels, samples)

    # a leading integer column that looks like an index is dropped
    if array.shape[0] > 1:
        first = array[0]
        if np.allclose(first, np.arange(first.size)) or np.allclose(first, np.arange(1, first.size + 1)):
            array = array[1:]

    names = [f"Ch{i + 1}" for i in range(array.shape[0])]
    return ParsedSignal(
        samples=array,
        sampling_rate=_guess_sampling_rate(header, default_rate),
        channel_names=names,
        source_format="CSV",
    )


def parse_json(data: bytes, default_rate: float) -> ParsedSignal:
    """Parse a JSON signal document."""
    try:
        document = json.loads(data.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise UnsupportedFormatError("The uploaded file is not valid JSON.") from exc

    if not isinstance(document, dict):
        raise ValidationError("The JSON document must be an object.")

    samples = document.get("samples") or document.get("signal") or document.get("data")
    if samples is None:
        raise ValidationError("The JSON document must contain a 'samples' array.")

    array = _to_array(samples)
    names = document.get("channel_names") or document.get("lead_names") or []
    if not isinstance(names, list):
        names = []
    if not names:
        names = [f"Ch{i + 1}" for i in range(array.shape[0] if array.ndim == 2 else 1)]

    rate = float(
        document.get("sampling_rate")
        or document.get("fs")
        or document.get("sample_rate")
        or default_rate
    )
    return ParsedSignal(samples=array, sampling_rate=rate, channel_names=names, source_format="JSON")


def parse_txt(data: bytes, default_rate: float) -> ParsedSignal:
    """Parse a whitespace-separated text file."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        raise UnsupportedFormatError("Could not decode the uploaded file as text.") from exc

    rows: List[List[float]] = []
    for line in text.splitlines():
        cells = line.replace(",", " ").split()
        if not cells:
            continue
        try:
            rows.append([float(cell) for cell in cells])
        except ValueError:
            continue

    if not rows:
        raise ValidationError("No numeric samples were found in the uploaded file.")

    width = max(len(row) for row in rows)
    rows = [row for row in rows if len(row) == width]
    array = np.asarray(rows, dtype=np.float64).T
    names = [f"Ch{i + 1}" for i in range(array.shape[0])]
    return ParsedSignal(samples=array, sampling_rate=default_rate, channel_names=names, source_format="TXT")


def parse_mat(data: bytes, default_rate: float) -> ParsedSignal:
    """Parse a MATLAB v5 file."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedFormatError("MATLAB file support requires scipy.") from exc

    try:
        document = loadmat(io.BytesIO(data))
    except Exception as exc:
        raise UnsupportedFormatError("Could not read the uploaded MATLAB file.") from exc

    matrix = None
    for key in ("val", "data", "ecg", "eeg", "signal", "X"):
        if key in document:
            candidate = document[key]
            if isinstance(candidate, np.ndarray) and candidate.size > 1:
                matrix = candidate
                break
    if matrix is None:
        arrays = [v for v in document.values() if isinstance(v, np.ndarray) and v.size > 1]
        if not arrays:
            raise ValidationError("No usable array was found in the MATLAB file.")
        matrix = max(arrays, key=lambda item: item.size)

    array = _to_array(matrix)
    rate = default_rate
    for key in ("fs", "sampling_rate", "Fs"):
        if key in document:
            try:
                rate = float(np.asarray(document[key]).reshape(-1)[0])
            except Exception:  # pragma: no cover
                pass
    names = [f"Ch{i + 1}" for i in range(array.shape[0] if array.ndim == 2 else 1)]
    return ParsedSignal(samples=array, sampling_rate=rate, channel_names=names, source_format="MAT")


def parse_signal_upload(filename: str, data: bytes, default_rate: float) -> ParsedSignal:
    """Dispatch to the parser matching the file extension."""
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return parse_csv(data, default_rate)
    if name.endswith(".json"):
        return parse_json(data, default_rate)
    if name.endswith(".txt") or name.endswith(".dat"):
        return parse_txt(data, default_rate)
    if name.endswith(".mat"):
        return parse_mat(data, default_rate)
    # fall back on content sniffing
    stripped = data.lstrip()[:1]
    if stripped in (b"{", b"["):
        return parse_json(data, default_rate)
    if name.endswith(".hea"):
        raise UnsupportedFormatError(
            "WFDB records need both the .dat and .hea files - upload a CSV or JSON export instead."
        )
    return parse_txt(data, default_rate)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
