#!/usr/bin/env python3
"""Run the TenderSync RAG evaluation harness.

Examples
--------
One benchmark at a time, real pipeline:

    python -m evaluation.run_evaluation \
        --benchmark financebench \
        --dataset evaluation/datasets/financebench.json \
        --output evaluation/results/financebench/eval_report.json \
        --tenant-id <org-uuid> \
        --verbose

A mixed-source dataset is rejected; FinanceBench and CUAD are never scored
in the same run.

CI release gate:

    python -m evaluation.run_evaluation ... --fail-on-thresholds

Before evaluating a benchmark dataset, seed its source documents into the
tenant with ``python -m evaluation.ingest_golden`` so retrieval has something
to search. Evaluating against an empty tenant is what produced the previous
all-zero report; this runner now says so explicitly.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BACKEND_DIR / ".env")
except ImportError:
    pass

import rag_service
from evaluation.check_thresholds import check_thresholds, load_thresholds
from evaluation.pipeline_wrapper import PipelineConfigurationError, RAGPipelineWrapper
from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGASPipelineEvaluator,
    load_golden_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("evaluation")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the TenderSync RAG pipeline")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["financebench", "cuad", "austender"],
        help="Benchmark to evaluate; exactly one source per run",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Benchmark dataset (defaults to evaluation/datasets/<benchmark>.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Report path (defaults to evaluation/results/<benchmark>/eval_report.json)",
    )
    parser.add_argument("--tenant-id", "-t", required=True, help="Organisation UUID to evaluate")
    parser.add_argument(
        "--samples", "-n", type=int, default=0, help="Number of samples (0 = the entire dataset)"
    )
    parser.add_argument("--category", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "voyage-4-large"))
    parser.add_argument("--reranker", default=os.getenv("RERANKER_MODEL", "voyage-rerank-2"))
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["hybrid", "bm25_only", "vector_only"],
    )
    parser.add_argument("--api-base", default=None, help="Evaluate a running FastAPI service")
    parser.add_argument("--document-id", default=None, help="Force retrieval to one document")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Exit non-zero when any required metric misses its minimum (use in CI)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _dataset_metadata(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {
                key: payload[key]
                for key in ("description", "version", "source_datasets", "licences", "generated")
                if key in payload
            }
    except Exception:
        pass
    return {}


def main() -> int:
    args = parse_args()
    dataset_path = (
        Path(args.dataset)
        if args.dataset
        else Path(__file__).parent / "datasets" / f"{args.benchmark}.json"
    )
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return 2

    dataset = load_golden_dataset(str(dataset_path))
    if not dataset:
        logger.error("Dataset contains no usable samples: %s", dataset_path)
        return 2

    sources = sorted(
        {
            str((sample.metadata or {}).get("source_dataset") or "unknown")
            for sample in dataset
        }
    )
    if sources != [args.benchmark]:
        logger.error(
            "Refusing to evaluate benchmark %r with dataset sources %s. "
            "Run each benchmark separately with its own single-source dataset.",
            args.benchmark,
            sources,
        )
        return 2

    if args.category:
        dataset = [s for s in dataset if s.category == args.category]
    if args.difficulty:
        dataset = [s for s in dataset if s.difficulty == args.difficulty]
    if args.samples > 0 and args.samples < len(dataset):
        random.Random(args.seed).shuffle(dataset)
        dataset = dataset[: args.samples]

    if args.document_id:
        for sample in dataset:
            sample.document_id = args.document_id

    logger.info("Benchmark: %s", args.benchmark)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Samples: %d%s", len(dataset), "" if args.samples else " (entire benchmark)")
    logger.info("Tenant: %s", args.tenant_id)
    if not all(sample.document_id for sample in dataset):
        logger.warning(
            "Some samples have no document_id. Retrieval will search the whole "
            "tenant, so results may mix documents. Run evaluation.ingest_golden "
            "to create a seeded dataset."
        )

    wrapper = RAGPipelineWrapper(
        api_base=args.api_base,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        tenant_id=args.tenant_id,
    )

    try:
        embedder = wrapper.embedder
    except PipelineConfigurationError as exc:
        logger.error("%s", exc)
        return 2

    evaluator = RAGASEvaluator(embedding_service=embedder)
    pipeline_evaluator = RAGASPipelineEvaluator(evaluator=evaluator, rag_pipeline=wrapper)

    logger.info("Starting evaluation...")
    started = time.monotonic()
    result = pipeline_evaluator.evaluate_dataset(
        dataset=dataset,
        tenant_id=args.tenant_id,
        run_rag_pipeline=True,
        verbose=args.verbose,
    )

    report = {
        "benchmark": args.benchmark,
        "dataset": str(dataset_path),
        "dataset_metadata": _dataset_metadata(dataset_path),
        "tenant_id": args.tenant_id,
        "config": {
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker,
            "retrieval_mode": args.retrieval_mode,
            "top_k": args.top_k,
            "samples_requested": args.samples or len(dataset),
        },
        "total_samples": result.total_samples,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "errors": result.errors,
        "metrics": result.metrics,
        "per_category": result.per_category,
        "per_difficulty": result.per_difficulty,
        "samples": result.samples,
        "timestamp": result.timestamp,
    }

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "results" / args.benchmark / "eval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 64)
    print("RAG EVALUATION SUMMARY")
    print("=" * 64)
    print(f"Samples: {result.total_samples}    Errors: {result.errors}    Report: {output_path}")
    print("-" * 64)
    for metric_name, values in sorted(result.metrics.items()):
        print(
            f"  {metric_name:<30} {values['mean'] * 100:6.2f}%"
            f"  (min {values['min'] * 100:5.1f}%, max {values['max'] * 100:5.1f}%, n={values['n']})"
        )

    retrieved_total = sum(
        sample.get("num_retrieved_chunks", 0) for sample in result.samples
    )
    if retrieved_total == 0:
        print(
            "\nDIAGNOSTIC: retrieval returned zero chunks for every sample.\n"
            "The metrics above are not a measurement of RAG quality. Seed the\n"
            "tenant first:\n"
            "  python -m evaluation.ingest_golden --dataset <dataset> "
            f"--tenant-id {args.tenant_id} --user-id <uuid>\n"
        )
    elif result.errors:
        print(f"\nDIAGNOSTIC: {result.errors} sample(s) raised pipeline errors. See the report.")

    thresholds_path = Path(__file__).parent / "config" / "metrics_thresholds.json"
    if thresholds_path.exists():
        check = check_thresholds(report, load_thresholds(str(thresholds_path)))
        print("-" * 64)
        print("Release gate")
        print("-" * 64)
        for item in check["results"]:
            actual = "missing" if item["actual"] is None else f"{item['actual']:.2%}"
            status = "PASS" if item["passed"] else "FAIL"
            print(f"  {status:<4} {item['metric']:<30} {actual:>9}  (min {item['threshold']:.0%})")

        if not check["passed"]:
            logger.warning("%d metric(s) below the release threshold", check["failed_metrics"])
            if args.fail_on_thresholds:
                wrapper.close()
                return 1
        else:
            print("\nAll gated metrics passed.")

    wrapper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
