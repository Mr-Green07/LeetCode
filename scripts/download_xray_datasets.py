"""Download the real chest X-ray benchmarks used by the X-ray model.

Three public HuggingFace datasets are fetched anonymously (no token needed):

``timm/nih-chest-xray-14``
    The NIH Clinical Center Chest X-ray dataset (ChestX-ray14) in sharded
    parquet form, with the official 14 finding labels per image.  Only the
    first few shards are downloaded - the full dataset is 21 GB and a few
    thousand real radiographs are enough to train the model.

``realsudarshan/chest-xray-tb-pneumonia``
    A folder-structured set of normal, pneumonia and tuberculosis radiographs.

``hawking32/covid_chestXray``
    A folder-structured set of normal, pneumonia and COVID-19 radiographs.

Usage
-----
    python scripts/download_xray_datasets.py --dataset nih
    python scripts/download_xray_datasets.py --dataset tb
    python scripts/download_xray_datasets.py --dataset covid
    python scripts/download_xray_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("download_xray")

HF_RESOLVE = "https://huggingface.co/datasets"
HF_API = "https://huggingface.co/api/datasets"

NIH_REPO = "timm/nih-chest-xray-14"
NIH_TRAIN_SHARDS = 4
NIH_TEST_SHARDS = 1

TB_REPO = "realsudarshan/chest-xray-tb-pneumonia"
COVID_REPO = "hawking32/covid_chestXray"

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"


def _download(url: str, target: Path, client: httpx.Client) -> bool:
    """Stream one file to disk.  Returns True on success."""
    if target.exists() and target.stat().st_size > 0:
        logger.info("already present: %s", target.name)
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                logger.warning("%s -> HTTP %s", url, response.status_code)
                return False
            total = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    total += len(chunk)
        temporary.replace(target)
        logger.info("downloaded %s (%.1f MB)", target.name, total / 1e6)
        return True
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("failed %s: %s", target.name, exc)
        if temporary.exists():
            temporary.unlink()
        return False


def _repo_files(repo: str, client: httpx.Client) -> list[dict]:
    """List every file of a repository, following the API pagination."""
    url = f"{HF_API}/{repo}/tree/main?recursive=1"
    entries: list[dict] = []
    while url:
        response = client.get(url)
        response.raise_for_status()
        entries.extend(response.json())
        url = ""
        link = response.headers.get("link") or response.headers.get("Link") or ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
                break
    return entries


def download_nih(root: Path, client: httpx.Client) -> int:
    """Download a few shards of the NIH ChestX-ray14 parquet dataset."""
    target_dir = root / "nih_cxr"
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index in range(NIH_TRAIN_SHARDS):
        name = f"data/train-{index:05d}-of-00032.parquet"
        if _download(f"{HF_RESOLVE}/{NIH_REPO}/resolve/main/{name}", target_dir / name, client):
            count += 1
    for index in range(NIH_TEST_SHARDS):
        name = f"data/test-{index:05d}-of-00010.parquet"
        if _download(f"{HF_RESOLVE}/{NIH_REPO}/resolve/main/{name}", target_dir / name, client):
            count += 1
    return count


def download_folder_repo(repo: str, root: Path, client: httpx.Client, limit: int = 0) -> int:
    """Download every file of a folder-structured repository."""
    entries = _repo_files(repo, client)
    files = [
        entry
        for entry in entries
        if entry.get("type") == "file"
        and entry["path"].lower().endswith((".png", ".jpg", ".jpeg"))
        and not entry["path"].startswith("valid")
    ]
    if limit:
        files = files[:limit]
    logger.info("%s: %d images", repo, len(files))

    def fetch(entry: dict) -> bool:
        return _download(
            f"{HF_RESOLVE}/{repo}/resolve/main/{entry['path']}",
            root / entry["path"],
            client,
        )

    downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ok in pool.map(fetch, files):
            if ok:
                downloaded += 1
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Download real chest X-ray benchmarks.")
    parser.add_argument(
        "--dataset",
        choices=["nih", "tb", "covid", "all"],
        default="all",
    )
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    setup_logging()
    args.root.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "meddiag/1.0"}
    with httpx.Client(timeout=args.timeout, follow_redirects=True, headers=headers) as client:
        if args.dataset in ("nih", "all"):
            logger.info("=== NIH ChestX-ray14 ===")
            download_nih(args.root, client)
        if args.dataset in ("tb", "all"):
            logger.info("=== tuberculosis / pneumonia ===")
            download_folder_repo(TB_REPO, args.root / "tb_pneumonia", client)
        if args.dataset in ("covid", "all"):
            logger.info("=== COVID-19 ===")
            download_folder_repo(COVID_REPO, args.root / "covid_cxr", client)

    logger.info("done - data root: %s", args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
