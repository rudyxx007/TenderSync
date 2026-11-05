#!/usr/bin/env python3
"""Compatibility wrapper for the canonical RAG evaluation runner.

The old ``run_benchmark`` command implemented a second evaluation path with a
different wrapper, no tenant scoping, and no seeding check. Benchmark results
must come from the same retrieval/generation path as the application, so this
command now delegates to ``evaluation.run_evaluation``.

Prerequisite: seed the dataset's documents first with
``evaluation.ingest_golden``. Evaluating an empty tenant is what produced the
old all-zero reports.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_benchmark")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a benchmark through the canonical RAG evaluation pipeline"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["financebench", "cuad", "austender"],
        help="Benchmark to evaluate; exactly one source per run",
    )
    parser.add_argument("--dataset", "-d", required=True, help="Seeded dataset JSON file")
    parser.add_argument("--output", "-o", required=True, help="Report JSON path")
    parser.add_argument("--tenant-id", "-t", required=True, help="Evaluation organisation UUID")
    parser.add_argument("--samples", "-n", type=int, default=0, help="0 = entire dataset")
    parser.add_argument("--embedding-model", default="voyage-4-large")
    parser.add_argument("--reranker-model", default="voyage-rerank-2")
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["hybrid", "bm25_only", "vector_only"],
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Accepted for compatibility; the canonical runner computes all metrics",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.metrics:
        logger.warning("--metrics is ignored; the canonical runner computes the full metric set.")

    command = [
        sys.executable,
        "-m",
        "evaluation.run_evaluation",
        "--benchmark",
        args.benchmark,
        "--dataset",
        args.dataset,
        "--output",
        args.output,
        "--tenant-id",
        args.tenant_id,
        "--samples",
        str(args.samples),
        "--embedding-model",
        args.embedding_model,
        "--reranker",
        args.reranker_model,
        "--retrieval-mode",
        args.retrieval_mode,
    ]
    if args.verbose:
        command.append("--verbose")

    logger.info("Delegating to the canonical evaluation runner.")
    return subprocess.run(command, cwd=BACKEND_DIR).returncode


if __name__ == "__main__":
    raise SystemExit(main())
