#!/usr/bin/env python3
"""Download a LegalBench-RAG source into the gitignored benchmark cache.

LegalBench-RAG is the legal retrieval benchmark relevant to TenderSync.
LegalBench (without the RAG suffix) is a legal-reasoning benchmark and is not a
substitute for document retrieval evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from evaluation.benchmark_sources import LEGALBENCH_RAG_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="cuad",
        choices=["cuad", "contractnli", "maud", "privacy_qa"],
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required") from exc

    sources = ["cuad", "contractnli", "maud", "privacy_qa"] if args.all else [args.source]
    patterns = ["README.md", "LICENSE", "SHA256SUMS"]
    for source in sources:
        patterns.extend([f"benchmarks/{source}.json", f"corpus/{source}/*"])
    LEGALBENCH_RAG_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="awinml/legalbench-rag",
        repo_type="dataset",
        local_dir=str(LEGALBENCH_RAG_DIR),
        allow_patterns=patterns,
        force_download=args.force,
    )
    print(f"LegalBench-RAG: {path}")
    print(f"Sources: {', '.join(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
