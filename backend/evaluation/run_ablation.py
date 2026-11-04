#!/usr/bin/env python3
"""
Ablation Study Runner - Compares different RAG configurations.

Runs the same evaluation dataset through multiple configurations
to measure the impact of each component.

Usage:
    python -m evaluation.run_ablation \
        --dataset evaluation/golden_dataset.json \
        --output evaluation/results/ablation_report.json \
        --samples 50
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    load_golden_dataset,
    EvaluationSample,
    EvaluationResult,
)
from evaluation.pipeline_wrapper import RAGPipelineWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default ablation configurations
DEFAULT_CONFIGURATIONS = [
    {
        "name": "full_hybrid",
        "description": "Full hybrid retrieval + reranking (baseline)",
        "embedding_model": "voyage-4-large",
        "reranker_model": "voyage-rerank-2",
        "retrieval_mode": "hybrid",
        "top_k": 50,
        "rerank_top_k": 8,
    },
    {
        "name": "no_reranker",
        "description": "Hybrid retrieval without reranking",
        "embedding_model": "voyage-4-large",
        "reranker_model": "none",
        "retrieval_mode": "hybrid",
        "top_k": 8,
        "rerank_top_k": 8,
    },
    {
        "name": "bm25_only",
        "description": "BM25 keyword search only",
        "embedding_model": "voyage-4-large",
        "reranker_model": "none",
        "retrieval_mode": "bm25_only",
        "top_k": 8,
        "rerank_top_k": 8,
    },
    {
        "name": "vector_only",
        "description": "Vector similarity search only",
        "embedding_model": "voyage-4-large",
        "reranker_model": "none",
        "retrieval_mode": "vector_only",
        "top_k": 8,
        "rerank_top_k": 8,
    },
    {
        "name": "finance_embedding",
        "description": "Finance-tuned embedding model",
        "embedding_model": "voyage-finance-2",
        "reranker_model": "voyage-rerank-2",
        "retrieval_mode": "hybrid",
        "top_k": 50,
        "rerank_top_k": 8,
    },
    {
        "name": "law_embedding",
        "description": "Law-tuned embedding model",
        "embedding_model": "voyage-law-2",
        "reranker_model": "voyage-rerank-2",
        "retrieval_mode": "hybrid",
        "top_k": 50,
        "rerank_top_k": 8,
    },
    {
        "name": "bge_reranker",
        "description": "BGE reranker instead of Voyage",
        "embedding_model": "voyage-4-large",
        "reranker_model": "bge-reranker-v2-m3",
        "retrieval_mode": "hybrid",
        "top_k": 50,
        "rerank_top_k": 8,
    },
    {
        "name": "contextual_off",
        "description": "Without contextual retrieval (no context prepended)",
        "embedding_model": "voyage-4-large",
        "reranker_model": "voyage-rerank-2",
        "retrieval_mode": "hybrid",
        "top_k": 50,
        "rerank_top_k": 8,
        "contextual_retrieval": False,
    },
]


def run_single_config(
    config: Dict[str, Any],
    dataset: List[EvaluationSample],
    tenant_id: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation with a single configuration.
    
    Returns dict with config name, metrics, and latency stats.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running configuration: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Initialize pipeline with this config
    pipeline = RAGPipelineWrapper(
        embedding_model=config["embedding_model"],
        reranker_model=config["reranker_model"],
        retrieval_mode=config["retrieval_mode"],
        top_k=config["top_k"],
        rerank_top_k=config.get("rerank_top_k", 8),
    )
    
    # Initialize evaluator
    evaluator = RAGASEvaluator()
    
    # Run evaluation on each sample
    results: List[EvaluationResult] = []
    total_latency = 0
    
    for i, sample in enumerate(dataset):
        if verbose:
            logger.info(f"  [{i+1}/{len(dataset)}] {sample.id}: {sample.query[:50]}...")
        
        sample_start = time.time()
        
        # Query pipeline
        pipeline_result = pipeline.query(
            query=sample.query,
            tenant_id=tenant_id,
        )
        
        # Evaluate
        eval_result = evaluator.evaluate_sample(
            sample=sample,
            generated_answer=pipeline_result["answer"],
            retrieved_chunks=pipeline_result["retrieved_chunks"],
            ground_truth_chunks=sample.relevant_chunks,
        )
        
        results.append(eval_result)
        total_latency += (time.time() - sample_start) * 1000
    
    # Aggregate metrics
    metrics = {}
    for metric_name in evaluator._compute_context_recall.__doc__.split() or ["context_recall", "context_precision", "faithfulness", "answer_relevance"]:
        # Get all values for this metric
        values = [r.metrics.get(metric_name, 0) for r in results]
        if values:
            metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    
    # Also collect compliance metrics if present
    for sample, result in zip(dataset, results):
        for key in sample.metadata or {}:
            if key not in metrics:
                metrics[key] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                }
    
    elapsed = time.time() - start_time
    
    return {
        "config": config,
        "metrics": metrics,
        "num_samples": len(dataset),
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / len(dataset) if dataset else 0,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


def compute_ablation_comparison(
    all_results: List[Dict[str, Any]],
    baseline_name: str = "full_hybrid",
) -> Dict[str, Any]:
    """
    Compute comparison against baseline.
    
    Shows delta for each metric for each configuration vs baseline.
    """
    # Find baseline
    baseline = None
    for result in all_results:
        if result["config"]["name"] == baseline_name:
            baseline = result
            break
    
    if baseline is None:
        logger.warning(f"Baseline '{baseline_name}' not found in results")
        return {"error": f"Baseline '{baseline_name}' not found"}
    
    baseline_metrics = baseline.get("metrics", {})
    
    comparison = {
        "baseline": baseline_name,
        "baseline_metrics": {
            k: v["mean"] for k, v in baseline_metrics.items()
        },
        "configs": [],
    }
    
    for result in all_results:
        config_name = result["config"]["name"]
        if config_name == baseline_name:
            continue
        
        config_metrics = result.get("metrics", {})
        deltas = {}
        
        for metric_name, baseline_value in baseline_metrics.items():
            actual_value = config_metrics.get(metric_name, {}).get("mean", 0)
            deltas[metric_name] = {
                "delta": actual_value - baseline_value,
                "delta_pct": ((actual_value - baseline_value) / baseline_value * 100) if baseline_value else 0,
                "actual": actual_value,
                "baseline": baseline_value,
                "improved": actual_value > baseline_value,
            }
        
        comparison["configs"].append({
            "name": config_name,
            "description": result["config"]["description"],
            "metrics": {
                k: v["mean"] for k, v in config_metrics.items()
            },
            "deltas": deltas,
            "avg_latency_ms": result.get("avg_latency_ms", 0),
            "total_latency_ms": result.get("total_latency_ms", 0),
        })
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study comparing RAG configurations"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to golden dataset JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path for output ablation report"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=0,
        help="Number of samples (0=all)"
    )
    parser.add_argument(
        "--configs", "-c",
        type=str,
        default=None,
        help="Path to custom configs JSON (overrides defaults)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="full_hybrid",
        help="Baseline configuration name for comparison"
    )
    parser.add_argument(
        "--tenant-id", "-t",
        type=str,
        default="test-tenant",
        help="Tenant ID for retrieval"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Continue if a config fails"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    
    dataset = load_golden_dataset(str(dataset_path))
    logger.info(f"Loaded {len(dataset)} samples")
    
    if args.samples > 0 and args.samples < len(dataset):
        import random
        random.seed(42)
        dataset = random.sample(dataset, args.samples)
        logger.info(f"Sampled to {len(dataset)} samples")
    
    # Load configurations
    if args.configs and Path(args.configs).exists():
        with open(args.configs) as f:
            configs = json.load(f)
        logger.info(f"Loaded {len(configs)} custom configurations")
    else:
        configs = DEFAULT_CONFIGURATIONS
        logger.info(f"Using {len(configs)} default configurations")
    
    # Run ablation
    all_results = []
    
    for config in configs:
        try:
            result = run_single_config(
                config=config,
                dataset=dataset,
                tenant_id=args.tenant_id,
                verbose=args.verbose,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Config '{config['name']}' failed: {e}")
            if not args.skip_failed:
                sys.exit(1)
    
    # Compute comparison
    comparison = compute_ablation_comparison(
        all_results, 
        baseline_name=args.baseline
    )
    
    # Build final report
    report = {
        "description": "Ablation study comparing RAG pipeline configurations",
        "dataset": str(dataset_path),
        "num_samples": len(dataset),
        "baseline": args.baseline,
        "timestamp": datetime.now().isoformat(),
        "configurations": [
            {
                "name": r["config"]["name"],
                "description": r["config"]["description"],
                "embedding_model": r["config"]["embedding_model"],
                "reranker_model": r["config"]["reranker_model"],
                "retrieval_mode": r["config"]["retrieval_mode"],
                "metrics": r["metrics"],
                "latency_ms": r["avg_latency_ms"],
            }
            for r in all_results
        ],
        "comparison": comparison,
        "configuration_results": all_results,
    }
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Ablation report saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(dataset)}")
    print(f"Baseline: {args.baseline}")
    print()
    
    print("Metrics by configuration:")
    print("-" * 80)
    
    if all_results:
        first = all_results[0]
        metric_names = list(first.get("metrics", {}).keys())
        
        # Header
        header = f"{'Config':<20}"
        for m in metric_names:
            header += f" {m:>12}"
        header += f" {'Latency':>10}"
        print(header)
        print("-" * len(header))
        
        # Rows
        for result in all_results:
            row = f"{result['config']['name']:<20}"
            for m in metric_names:
                val = result.get("metrics", {}).get(m, {}).get("mean", 0)
                row += f" {val*100:>11.2f}%"
            row += f" {result.get('avg_latency_ms', 0):>8.1f}ms"
            print(row)
        
        print("-" * len(header))
        print()
        
        # Comparison
        if comparison.get("configs"):
            print(f"Comparison vs baseline '{args.baseline}':")
            print("-" * 80)
            
            for config_comparison in comparison["configs"]:
                print(f"\n  {config_comparison['name']}:")
                print(f"    {config_comparison['description']}")
                for metric_name, delta_data in config_comparison["deltas"].items():
                    sign = "+" if delta_data["improved"] else ""
                    delta_pct = delta_data["delta_pct"]
                    print(f"    {metric_name}: {sign}{delta_pct:.2f}% "
                          f"(Δ={delta_data['delta']*100:+.2f}pp, "
                          f"actual={delta_data['actual']*100:.2f}%, "
                          f"baseline={delta_data['baseline']*100:.2f}%)")
            
            print()
    
    print("=" * 80)
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
