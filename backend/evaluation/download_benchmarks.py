#!/usr/bin/env python3
"""Download the raw source files used to build the golden benchmark.

This downloads real benchmark data only. It never creates synthetic questions
or answers; a failed download is reported as a failure. Use
``evaluation.build_golden_dataset`` afterwards to produce the deterministic
per-benchmark evaluation sets.

Usage:
    python -m evaluation.download_benchmarks --dataset all
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_benchmarks")

SOURCES: Dict[str, Dict[str, str]] = {
    "financebench": {
        "url": (
            "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/"
            "financebench_merged.jsonl"
        ),
        "filename": "financebench_merged.jsonl",
        "licence": "CC-BY-NC-4.0",
    },
    "cuad": {
        "url": (
            "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/"
            "CUAD_v1/CUAD_v1.json"
        ),
        "filename": "CUAD_v1.json",
        "licence": "CC-BY-4.0",
    },
}


def download_file(url: str, destination: Path, token: Optional[str] = None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    logger.info("Downloading %s", url)
    with requests.get(url, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(temporary, "wb") as handle:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    logger.info("Saved %s (%d bytes)", destination, destination.stat().st_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download raw benchmark source files")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=sorted(SOURCES) + ["all"],
        default="all",
        help="Benchmark to download",
    )
    parser.add_argument("--output", "-o", default="data/benchmarks")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    names = sorted(SOURCES) if args.dataset == "all" else [args.dataset]
    failures = 0
    for name in names:
        source = SOURCES[name]
        destination = Path(args.output) / name / source["filename"]
        if destination.exists() and not args.force:
            logger.info("Already present: %s", destination)
            continue
        try:
            download_file(source["url"], destination, args.hf_token)
        except Exception as exc:
            failures += 1
            logger.error("Failed to download %s: %s", name, exc)

    if failures:
        logger.error("%d benchmark download(s) failed; no synthetic data was substituted.", failures)
        return 1
    logger.info("Raw benchmark files are ready under %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
