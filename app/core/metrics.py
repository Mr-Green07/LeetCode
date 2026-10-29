"""Lightweight in-process metrics for the operations dashboard."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List

from app.core.logging import get_logger

logger = get_logger(__name__)


class MetricsRegistry:
    """Thread-safe counters and latency histograms.

    Deliberately dependency-free: a production deployment would swap this for
    Prometheus, but the interface stays identical.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._errors: Dict[str, int] = defaultdict(int)

    # -------------------------------------------------------------- recording
    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            bucket = self._latencies[name]
            bucket.append(seconds)
            if len(bucket) > 1000:  # keep memory bounded
                del bucket[: len(bucket) // 2]

    def record_error(self, name: str) -> None:
        with self._lock:
            self._errors[name] += 1

    # ---------------------------------------------------------------- reading
    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            latency_summary = {}
            for key, values in self._latencies.items():
                if not values:
                    continue
                ordered = sorted(values)
                latency_summary[key] = {
                    "count": len(ordered),
                    "avg_ms": round(sum(ordered) / len(ordered) * 1000, 2),
                    "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1 if len(ordered) > 1 else 0] * 1000, 2),
                    "max_ms": round(ordered[-1] * 1000, 2),
                }
            return {
                "uptime_seconds": round(time.time() - self._started, 2),
                "counters": dict(self._counters),
                "errors": dict(self._errors),
                "latency": latency_summary,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()
            self._errors.clear()
            self._started = time.time()


metrics = MetricsRegistry()
