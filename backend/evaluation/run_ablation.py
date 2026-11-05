#!/usr/bin/env python3
"""Ablation study over the real TenderSync RAG pipeline.

Each configuration invokes ``evaluation.run_evaluation`` as a subprocess, so
the ablation measures the same retrieval/generation code path as production
and the main evaluation. The previous version instantiated a retriever without
a database client, queried with the wrong tenant, and derived metric names from
a docstring; it could not produce a valid comparison.

Usage:
    python -m evaluation.run_ablation \
        --benchmark financebench \
        --dataset evaluation/seeded/financebench.json \
        --tenant-id <org-uuid> \
        --output evaluation/results/financebench/ablation_report.json

The dataset must already be seeded with ``evaluation.ingest_golden``.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ablation")


DEFAULT_CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "name": "hybrid_rerank",
        "description": "Hybrid BM25 + vector with RRF and cross-encoder reranking (baseline)",
        "retrieval_mode": "hybrid",
        "reranker": "voyage-rerank-2",
    },
    {
        "name": "hybrid_no_rerank",
        "description": "Hybrid retrieval with RRF only, no cross-encoder reranking",
        "retrieval_mode": "hybrid",
        "reranker": "none",
    },
    {
        "name": "vector_only",
        "description": "Dense pgvector retrieval only",
        "retrieval_mode": "vector_only",
        "reranker": "none",
    },
    {
        "name": "bm25_only",
        "description": "Postgres full-text BM25 retrieval only",
        "retrieval_mode": "bm25_only",
        "reranker": "none",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a real RAG ablation study")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["financebench", "cuad", "austender"],
        help="Benchmark to ablate; exactly one source per run",
    )
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--tenant-id", "-t", required=True)
    parser.add_argument("--samples", "-n", type=int, default=0, help="0 = entire dataset")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "voyage-4-large"))
    parser.add_argument("--baseline", default="hybrid_rerank")
    parser.add_argument("--configs", default=None, help="Optional JSON list of configuration overrides")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_configs(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_CONFIGURATIONS
    with open(path, encoding="utf-8") as handle:
        configs = json.load(handle)
    if not isinstance(configs, list) or not configs:
        raise ValueError("--configs must contain a non-empty JSON list")
    required = {"name", "retrieval_mode", "reranker"}
    for config in configs:
        missing = required - set(config)
        if missing:
            raise ValueError(f"Configuration {config.get('name', '?')} missing {sorted(missing)}")
    return configs


def run_configuration(
    config: Dict[str, Any],
    benchmark: str,
    dataset_path: Path,
    tenant_id: str,
    samples: int,
    embedding_model: str,
    run_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    report_path = run_dir / f"{config['name']}.json"
    command = [
        sys.executable,
        "-m",
        "evaluation.run_evaluation",
        "--benchmark",
        benchmark,
        "--dataset",
        str(dataset_path),
        "--output",
        str(report_path),
        "--tenant-id",
        tenant_id,
        "--samples",
        str(samples),
        "--embedding-model",
        embedding_model,
        "--reranker",
        config["reranker"],
        "--retrieval-mode",
        config["retrieval_mode"],
    ]
    if verbose:
        command.append("--verbose")

    logger.info("Running %s: %s", config["name"], config["description"])
    completed = subprocess.run(command, cwd=BACKEND_DIR)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Configuration {config['name']} failed with exit code {completed.returncode}. "
            f"See {report_path}."
        )

    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    if not report.get("total_samples"):
        raise RuntimeError(f"Configuration {config['name']} evaluated zero samples")
    if sum(s.get("num_retrieved_chunks", 0) for s in report.get("samples", [])) == 0:
        raise RuntimeError(
            f"Configuration {config['name']} retrieved zero chunks for every sample. "
            "Seed the tenant with evaluation.ingest_golden first."
        )

    return {
        "name": config["name"],
        "description": config["description"],
        "retrieval_mode": config["retrieval_mode"],
        "reranker_model": config["reranker"],
        "metrics": report["metrics"],
        "num_samples": report["total_samples"],
        "elapsed_seconds": report.get("elapsed_seconds"),
        "errors": report.get("errors", 0),
    }


def compare(results: List[Dict[str, Any]], baseline_name: str) -> Dict[str, Any]:
    baseline = next((r for r in results if r["name"] == baseline_name), None)
    if baseline is None:
        raise ValueError(f"Baseline configuration not found: {baseline_name}")

    comparison = {
        "baseline": baseline_name,
        "baseline_metrics": {k: v["mean"] for k, v in baseline["metrics"].items()},
        "configs": [],
    }

    for result in results:
        if result["name"] == baseline_name:
            continue
        deltas = {}
        for metric, baseline_values in baseline["metrics"].items():
            actual = result["metrics"].get(metric, {}).get("mean", 0.0)
            base = baseline_values.get("mean", 0.0)
            deltas[metric] = {
                "actual": actual,
                "baseline": base,
                "delta": actual - base,
                "delta_pct": ((actual - base) / base * 100) if base else None,
            }
        comparison["configs"].append(
            {
                "name": result["name"],
                "description": result["description"],
                "metrics": {k: v["mean"] for k, v in result["metrics"].items()},
                "deltas": deltas,
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
    return comparison


def print_summary(report: Dict[str, Any]) -> None:
    results = report["configurations"]
    metric_names = sorted({name for result in results for name in result["metrics"]})
    print("\n" + "=" * 88)
    print("RAG ABLATION STUDY")
    print("=" * 88)
    print(f"Samples: {report['num_samples']}    Baseline: {report['baseline']}")
    print("-" * 88)
    header = f"{'Configuration':<22}" + "".join(f"{name:>15}" for name in metric_names)
    print(header)
    print("-" * len(header))
    for result in results:
        row = f"{result['name']:<22}"
        for name in metric_names:
            value = result["metrics"].get(name, {}).get("mean")
            row += f"{(f'{value * 100:.2f}%' if value is not None else 'n/a'):>15}"
        print(row)
    print("=" * 88)


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return 2
    if not os.getenv("SUPABASE_URL") or not (
        os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    ):
        logger.error("SUPABASE_URL and SUPABASE_SECRET_KEY are required to run an ablation.")
        return 2

    try:
        configs = load_configs(args.configs)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_path.parent / "ablation_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for config in configs:
        try:
            results.append(
                run_configuration(
                    config=config,
                    benchmark=args.benchmark,
                    dataset_path=dataset_path,
                    tenant_id=args.tenant_id,
                    samples=args.samples,
                    embedding_model=args.embedding_model,
                    run_dir=run_dir,
                    verbose=args.verbose,
                )
            )
        except Exception as exc:
            logger.error("Ablation configuration failed: %s", exc)
            return 1

    report = {
        "description": "Ablation study over the real TenderSync hybrid RAG pipeline",
        "benchmark": args.benchmark,
        "dataset": str(dataset_path),
        "num_samples": results[0]["num_samples"],
        "baseline": args.baseline,
        "embedding_model": args.embedding_model,
        "timestamp": datetime.now().isoformat(),
        "configurations": results,
        "comparison": compare(results, args.baseline),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print_summary(report)
    logger.info("Ablation report written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
