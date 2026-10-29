"""Evaluate the trained checkpoints on freshly seeded synthetic data.

The training corpus uses seed bases 2024/2025/2026; this script uses a
completely different seed range so the samples are genuinely unseen.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.synthetic import build_ecg_corpus, build_eeg_corpus, build_xray_corpus
from app.ml.registry import registry


def evaluate(name: str, corpus, modality: str) -> tuple[int, int, list[str]]:
    model = registry.get(modality)
    correct = 0
    total = 0
    errors: list[str] = []
    for signal, label, _ in corpus:
        expected = model.classes[label]
        if modality == "xray":
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(signal, mode="L").save(buffer, format="PNG")
            analysis = model.analyze(buffer.getvalue())
        else:
            analysis = model.analyze(signal, 500.0 if modality == "ecg" else 256.0)
        predicted = analysis.label
        total += 1
        if predicted == expected:
            correct += 1
        elif len(errors) < 6:
            errors.append(f"{expected} -> {predicted}")
    print(f"{name:6s} source={model.source:20s} {correct}/{total} = {correct / total:6.1%}")
    for line in errors:
        print(f"        miss: {line}")
    return correct, total, errors


def main() -> int:
    print("Evaluating trained checkpoints on unseen seeds\n")
    ecg = build_ecg_corpus(per_class=12, duration_s=10.0, seed=90210)
    eeg = build_eeg_corpus(per_class=12, duration_s=10.0, seed=90211)
    xray = build_xray_corpus(per_class=10, seed=90212)

    total_correct = 0
    total_count = 0
    for name, corpus, modality in (
        ("ECG", ecg, "ecg"),
        ("EEG", eeg, "eeg"),
        ("X-ray", xray, "xray"),
    ):
        correct, count, _ = evaluate(name, corpus, modality)
        total_correct += correct
        total_count += count

    print(f"\nOVERALL {total_correct}/{total_count} = {total_correct / total_count:.1%}")
    return 0 if total_correct == total_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
