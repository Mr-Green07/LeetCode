"""Write a demo corpus of synthetic recordings to ``data/demo``.

The corpus mirrors the layout the dataset adapters expect, so it can also be
used to train on::

    python scripts/train_ecg.py --dataset folder --data-root data/demo/ecg
    python scripts/train_eeg.py --dataset folder --data-root data/demo/eeg
    python scripts/train_xray.py --dataset folder --data-root data/demo/xray
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.data.synthetic import write_demo_dataset  # noqa: E402

logger = get_logger("make_demo_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic demo corpus.")
    parser.add_argument("--output", type=Path, default=Path(settings.data_dir) / "demo")
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()

    setup_logging()
    created = write_demo_dataset(args.output, per_class=args.per_class)
    logger.info("ECG files : %d", len(created["ecg"]))
    logger.info("EEG files : %d", len(created["eeg"]))
    logger.info("X-ray files: %d", len(created["xray"]))
    logger.info("Demo corpus written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
