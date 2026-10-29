"""Fill the generated part of TESTING.md section 5 from the benchmark artefacts.

The tables are copied verbatim from ``results/results.md`` and
``results/summary.csv`` so the document can never drift from the numbers the
harness actually produced.  The markers in TESTING.md are::

    <!-- RESULTS-SUMMARY -->
    ...generated content...
    <!-- /RESULTS-SUMMARY -->
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TESTING = ROOT / "TESTING.md"

START = "<!-- RESULTS-SUMMARY -->"
END = "<!-- /RESULTS-SUMMARY -->"

SYNTHETIC = {"synthetic-ecg", "synthetic-eeg", "synthetic-xray"}


def load_rows():
    with (RESULTS / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary_table(rows):
    lines = [
        "### 5.1 Headline results",
        "",
        "| Evaluation set | Modality | Model | Source | Partition | n | Groups | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Macro-AUC | Latency ms |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {d} | {m} | {model} | {src} | {part} | {n} | {g} | {acc} | {mp} | {mr} | {mf} | {wf} | {auc} | {lat} |".format(
                d=row["dataset"],
                m=row["modality"],
                model=row["model"],
                src=row["model_source"],
                part=row["partition"],
                n=row["n_samples"],
                g=row["n_groups"],
                acc=f"{float(row['accuracy']):.4f}",
                mp=f"{float(row['macro_precision']):.4f}",
                mr=f"{float(row['macro_recall']):.4f}",
                mf=f"{float(row['macro_f1']):.4f}",
                wf=f"{float(row['weighted_f1']):.4f}",
                auc=f"{float(row['macro_auc']):.4f}",
                lat=f"{float(row['mean_latency_ms']):.1f}",
            )
        )
    lines.append("")
    return lines


def per_dataset_blocks(markdown: str, rows):
    """Extract the per-class + confusion blocks of the real evaluation sets."""
    lines = ["### 5.2 Per-class results and confusion matrices", ""]

    sections = {}
    current = None
    buffer: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("### "):
            if current:
                sections[current] = buffer
            current = line[4:].strip()
            buffer = []
        elif current:
            buffer.append(line)
    if current:
        sections[current] = buffer

    order = []
    for row in rows:
        key = (row["dataset"], row["model"])
        if key in order:
            continue
        order.append(key)

    for dataset, model in order:
        header = f"{dataset} - {model}"
        if header not in sections:
            continue
        block = [line for line in sections[header]]
        while block and not block[0].strip():
            block.pop(0)
        lines.append(f"**{header}**")
        lines.append("")
        lines.extend(block)
        lines.append("")
    return lines


def main() -> int:
    if not (RESULTS / "summary.csv").exists():
        print("results/summary.csv not found - run scripts/run_benchmark.py first")
        return 1

    rows = load_rows()
    markdown = (RESULTS / "results.md").read_text(encoding="utf-8")

    body = summary_table(rows)
    body.append("")
    body.extend(per_dataset_blocks(markdown, rows))

    text = TESTING.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("markers not found in TESTING.md")
        return 1

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    TESTING.write_text(head + START + "\n" + "\n".join(body) + "\n" + END + tail, encoding="utf-8")
    print(f"TESTING.md section 5 filled with {len(rows)} result rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
