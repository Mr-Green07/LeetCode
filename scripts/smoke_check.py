"""Ad-hoc smoke test used during development (not part of the pytest suite)."""

from __future__ import annotations

import sys
import traceback

sys.path.insert(0, r"D:\minimaxc\meddiag")

SEED_BASE = 20260101

from app.data.synthetic import synth_ecg, synth_eeg, synth_xray  # noqa: E402
from app.ml.registry import registry  # noqa: E402
from app.schemas import PatientInfo  # noqa: E402
from app.services.diagnosis import diagnosis_service  # noqa: E402


def main() -> int:
    patient = PatientInfo(patient_id="PT-SMOKE")
    failures = 0

    print("== ECG ==")
    for label in ["NORM", "AFIB", "MI", "STDE", "PVC"]:
        try:
            signal, fs = synth_ecg(label, duration_s=10, seed=SEED_BASE + sum(ord(c) for c in label))
            result = diagnosis_service.analyze_ecg(signal, fs, patient)
            probs = {k: round(v, 3) for k, v in result.prediction.class_probabilities.items()}
            ok = "OK " if result.prediction.label == label else "MISS"
            print(f"  {ok} {label:5s} -> {result.prediction.label:5s} conf={result.prediction.confidence:.3f} {probs}")
        except Exception:
            failures += 1
            traceback.print_exc()

    print("== EEG ==")
    for label in ["NORMAL", "EPILEPSY", "PARKINSON", "DEPRESSION"]:
        try:
            signal, fs = synth_eeg(label, duration_s=10, seed=SEED_BASE + sum(ord(c) for c in label))
            result = diagnosis_service.analyze_eeg(signal, fs, patient)
            probs = {k: round(v, 3) for k, v in result.prediction.class_probabilities.items()}
            ok = "OK " if result.prediction.label == label else "MISS"
            print(f"  {ok} {label:11s} -> {result.prediction.label:11s} conf={result.prediction.confidence:.3f} {probs}")
        except Exception:
            failures += 1
            traceback.print_exc()

    print("== X-RAY ==")
    for label in ["NORMAL", "PNEUMONIA", "CONSOLIDATION", "EFFUSION", "TUBERCULOSIS", "COVID19"]:
        try:
            import io

            from PIL import Image

            image = synth_xray(label, seed=SEED_BASE + sum(ord(c) for c in label))
            buffer = io.BytesIO()
            Image.fromarray(image.astype("uint8"), mode="L").save(buffer, format="PNG")
            result = diagnosis_service.analyze_xray(buffer.getvalue(), patient)
            probs = {k: round(v, 3) for k, v in result.prediction.class_probabilities.items()}
            ok = "OK " if result.prediction.label == label else "MISS"
            print(f"  {ok} {label:13s} -> {result.prediction.label:13s} conf={result.prediction.confidence:.3f} {probs}")
        except Exception:
            failures += 1
            traceback.print_exc()

    print(f"\nfailures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
