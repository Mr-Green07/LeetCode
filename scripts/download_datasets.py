"""Download the public benchmark datasets used for real training.

The datasets themselves are large and are **not** bundled with the project.
This script fetches them into ``data/datasets`` and prints the training
command for each one.

Examples
--------
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --dataset mitbih
    python scripts/download_datasets.py --dataset bonn --limit 100
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("download_datasets")

# PhysioNet serves every MIT-BIH record as three individual files.
MITBIH_RECORDS = [
    "100", "101", "103", "104", "105", "106", "108", "109", "111", "112",
    "113", "114", "115", "116", "118", "119", "121", "122", "123", "124",
    "200", "201", "203", "205", "207", "208", "209", "210", "212", "213",
    "214", "215", "217", "219", "220", "223", "230", "231", "232", "233",
    "234",
]
MITBIH_BASE = "https://physionet.org/files/mitdb/1.0.0"
MITBIH_EXTENSIONS = ("dat", "hea", "atr")

# The Bonn EEG database is distributed as 100 single-channel text files per
# set.  The original university host has moved, so the files are pulled from
# a redistribution mirror that preserves the original data unmodified.
BONN_BASE = "https://raw.githubusercontent.com/NeuralPsy/Bonn-EEG-Dataset/main"
BONN_SETS = ["Z", "O", "N", "F", "S"]

DATASETS = {
    "mitbih": {
        "name": "MIT-BIH Arrhythmia Database (PhysioNet)",
        "size": "~120 MB",
        "train": "python scripts/train_ecg.py --dataset mitbih --data-root data/datasets/mitdb",
        "note": "48 two-hour 2-lead records at 360 Hz with beat annotations.",
    },
    "bonn": {
        "name": "Bonn University EEG Database",
        "size": "~10 MB",
        "train": "python scripts/train_eeg.py --dataset bonn --data-root data/datasets/bonn",
        "note": "500 single-channel EEG recordings in 5 sets for epileptiform / seizure screening.",
    },
    "ptbxl": {
        "name": "PTB-XL (PhysioNet)",
        "url": "https://physionet.org/files/ptb-xl/1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
        "size": "~2 GB",
        "train": "python scripts/train_ecg.py --dataset ptbxl --data-root data/datasets/ptb-xl",
        "note": "21 837 12-lead 10-second records at 500 Hz with SCP-ECG diagnostic labels.",
    },
    "chestxray": {
        "name": "NIH ChestX-ray14 / COVID-19 Radiography",
        "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "size": "~45 GB",
        "train": "python scripts/train_xray.py --dataset nih --data-root data/datasets/chestxray14",
        "note": "112 120 frontal chest radiographs with 14 disease labels.",
    },
}


def _fetch(url: str, destination: Path, *, retries: int = 3) -> bool:
    """Stream one file to disk, retrying transient network errors."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MedDiag/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise OSError("empty response")
            destination.write_bytes(payload)
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
    return False


def download_archive(url: str, destination: Path) -> bool:
    """Download and extract a single zip archive."""
    archive = destination.with_suffix(".zip")
    if not _fetch(url, archive):
        return False
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    except zipfile.BadZipFile as exc:
        logger.error("%s is not a valid zip archive: %s", archive.name, exc)
        return False
    finally:
        archive.unlink(missing_ok=True)
    return True


def download_mitbih(target: Path, limit: int | None = None) -> bool:
    """Fetch the MIT-BIH records one file at a time."""
    records = MITBIH_RECORDS[:limit] if limit else MITBIH_RECORDS
    jobs = []
    for record in records:
        for extension in MITBIH_EXTENSIONS:
            name = f"{record}.{extension}"
            jobs.append((f"{MITBIH_BASE}/{name}", target / name))

    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for url, destination in pool.map(lambda job: _fetch(*job), jobs):
            if not url:
                failures += 1
    logger.info("MIT-BIH: %d files requested, %d failed", len(jobs), failures)
    return failures == 0


def download_bonn(target: Path, limit: int | None = None) -> bool:
    """Fetch the Bonn EEG text files set by set."""
    per_set = limit or 100
    jobs = []
    for set_name in BONN_SETS:
        for index in range(1, per_set + 1):
            name = f"{set_name}{index:03d}.txt"
            jobs.append((f"{BONN_BASE}/Set_{set_name}/{name}", target / set_name / name))

    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ok, _ in pool.map(lambda job: (_fetch(*job), job[0]), jobs):
            if not ok:
                failures += 1
    logger.info("Bonn: %d files requested, %d failed", len(jobs), failures)
    return failures == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public benchmark datasets.")
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="list")
    parser.add_argument("--output", type=Path, default=Path(settings.data_dir) / "datasets")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of files per dataset")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if args.dataset == "list":
        print("\nAvailable datasets:\n")
        for key, meta in DATASETS.items():
            print(f"  {key:10s} {meta['name']}")
            print(f"             {meta['note']}")
            print(f"             size {meta['size']}")
            print(f"             train: {meta['train']}\n")
        return 0

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failures = 0
    for key in keys:
        meta = DATASETS[key]
        target = args.output / key
        if key == "mitbih":
            ok = download_mitbih(target, args.limit)
        elif key == "bonn":
            ok = download_bonn(target, args.limit)
        else:
            ok = download_archive(meta["url"], target)
        if not ok:
            failures += 1
            continue
        logger.info("%s ready in %s", meta["name"], target)
        print(f"  next: {meta['train']}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
