#!/usr/bin/env python3
"""
Benchmark Runner - Evaluates on standard benchmark datasets.

Supports FinanceBench, CUAD, and AusTender RFP benchmarks.
Runs evaluation and outputs standardized benchmark reports.

Usage:
    python -m evaluation.run_benchmark \
        --dataset data/benchmarks/financebench \
        --output evaluation/results/financebench_benchmark.json \
        --metrics context_recall context_precision faithfulness
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
    EvaluationSample,
)
from evaluation.pipeline_wrapper import RAGPipelineWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Supported benchmark datasets and their metrics
BENCHMARK_CONFIGS = {
    "financebench": {
        "description": "FinanceBench - Financial document QA benchmark",
        "source": "https://github.com/potsawee/FinanceBench",
        "recommended_metrics": [
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_similarity",
        ],
        "priority_metrics": ["faithfulness", "context_recall"],
    },
    "cuad": {
        "description": "CUAD - Contract Understanding Atticus Dataset",
        "source": "https://www.atticusprojectai.org/cuad",
        "recommended_metrics": [
            "context_recall",
            "context_precision",
            "faithfulness",
            "compliance_score_accuracy",
        ],
        "priority_metrics": ["context_recall", "faithfulness"],
    },
    "austender": {
        "description": "AusTender RFPs - Australian Government tender documents",
        "source": "https://www.tenders.gov.au",
        "recommended_metrics": [
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_relevance",
            "compliance_score_accuracy",
        ],
        "priority_metrics": ["context_recall", "compliance_score_accuracy"],
    },
}


def load_benchmark_dataset(dataset_path: str) -> List[EvaluationSample]:
    """
    Load benchmark dataset from directory.
    
    Expected structure:
        dataset_path/
            ├── metadata.json
            └── samples/
                ├── sample_001.json
                ├── sample_002.json
                └── ...
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    samples = []
    
    # Try to load metadata
    metadata_file = dataset_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        logger.info(f"Dataset: {metadata.get('description', dataset_path)}")
        logger.info(f"  Source: {metadata.get('source', 'unknown')}")
        logger.info(f"  Version: {metadata.get('version', 'unknown')}")
    
    # Load samples
    samples_dir = dataset_dir / "samples"
    if samples_dir.exists():
        for sample_file in sorted(samples_dir.glob("*.json")):
            with open(sample_file) as f:
                sample_data = json.load(f)
            samples.append(EvaluationSample(
                id=sample_data["id"],
                query=sample_data["query"],
                ground_truth=sample_data["ground_truth"],
                relevant_chunks=sample_data.get("relevant_chunks", []),
                expected_compliance_score=sample_data.get("expected_compliance_score"),
                expected_budget=sample_data.get("expected_budget"),
                expected_deadline=sample_data.get("expected_deadline"),
                expected_tender_id=sample_data.get("expected_tender_id"),
                category=sample_data.get("category", "general"),
                difficulty=sample_data.get("difficulty", "medium"),
                metadata=sample_data.get("metadata", {}),
            ))
    else:
        # Try loading single JSON file with samples array
        for json_file in dataset_dir.glob("*.json"):
            if json_file.name == "metadata.json":
                continue
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    samples.append(EvaluationSample(
                        id=item["id"],
                        query=item["query"],
                        ground_truth=item["ground_truth"],
                        relevant_chunks=item.get("relevant_chunks", []),
                        category=item.get("category", "general"),
                        difficulty=item.get("difficulty", "medium"),
                    ))
            elif isinstance(data, dict) and "samples" in data:
                for item in data["samples"]:
                    samples.append(EvaluationSample(
                        id=item["id"],
                        query=item["query"],
                        ground_truth=item["ground_truth"],
                        relevant_chunks=item.get("relevant_chunks", []),
                        category=item.get("category", "general"),
                        difficulty=item.get("difficulty", "medium"),
                    ))
    
    return samples


def compute_metric(
    metric_name: str,
    sample: EvaluationSample,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """Compute a single metric for a sample."""
    evaluator = RAGASEvaluator()
    
    if metric_name == "context_recall":
        return evaluator._compute_context_recall(
            retrieved_chunks, sample.relevant_chunks
        )
    elif metric_name == "context_precision":
        return evaluator._compute_context_precision(
            retrieved_chunks, sample.relevant_chunks
        )
    elif metric_name == "faithfulness":
        return evaluator._compute_faithfulness(answer, retrieved_chunks)
    elif metric_name == "answer_relevance":
        return evaluator._compute_answer_relevance(sample.query, answer)
    elif metric_name == "compliance_score_accuracy":
        if sample.expected_compliance_score is not None:
            return evaluator._compute_compliance_accuracy(
                answer, sample.expected_compliance_score, retrieved_chunks
            )
        return 0.0
    else:
        # Default: return 0 for unknown metrics
        logger.warning(f"Unknown metric: {metric_name}")
        return 0.0


def run_benchmark(
    dataset_path: str,
    output_path: str,
    metrics: List[str],
    tenant_id: str = "test-tenant",
    embedding_model: str = "voyage-4-large",
    reranker_model: str = "voyage-rerank-2",
    retrieval_mode: str = "hybrid",
    samples_limit: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run benchmark evaluation.
    
    Returns benchmark report dict.
    """
    logger.info(f"Loading benchmark dataset: {dataset_path}")
    samples = load_benchmark_dataset(dataset_path)
    
    logger.info(f"Loaded {len(samples)} samples")
    
    if samples_limit > 0 and samples_limit < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, samples_limit)
        logger.info(f"Limited to {len(samples)} samples")
    
    # Initialize pipeline and evaluator
    pipeline = RAGPipelineWrapper(
        api_base=None,  # Direct mode
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        retrieval_mode=retrieval_mode,
    )
    
    evaluator = RAGASEvaluator()
    
    # Run evaluation
    results = {
        "dataset": Path(dataset_path).name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "retrieval_mode": retrieval_mode,
        },
        "num_samples": len(samples),
        "samples": [],
        "metrics": {},
    }
    
    total_latency = 0
    
    for i, sample in enumerate(samples):
        if verbose:
            logger.info(f"[{i+1}/{len(samples)}] {sample.id}")
        
        sample_start = time.time()
        
        # Query pipeline
        pipeline_result = pipeline.query(
            query=sample.query,
            tenant_id=tenant_id,
        )
        
        answer = pipeline_result.get("answer", "")
        retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
        
        sample_latency = (time.time() - sample_start) * 1000
        total_latency += sample_latency
        
        # Compute each metric
        sample_metrics = {}
        for metric_name in metrics:
            try:
                value = compute_metric(
                    metric_name, sample, answer, retrieved_chunks
                )
                sample_metrics[metric_name] = value
            except Exception as e:
                logger.warning(f"Metric {metric_name} failed for {sample.id}: {e}")
                sample_metrics[metric_name] = 0.0
        
        sample_result = {
            "id": sample.id,
            "query": sample.query,
            "ground_truth": sample.ground_truth,
            "answer": answer,
            "metrics": sample_metrics,
            "latency_ms": sample_latency,
            "num_retrieved_chunks": len(retrieved_chunks),
        }
        
        results["samples"].append(sample_result)
        
        if verbose:
            print(f"  Metrics: {', '.join(f'{k}={v:.2%}' for k, v in sample_metrics.items())}")
    
    # Aggregate metrics
    metrics_aggregate = {}
    for metric_name in metrics:
        values = [s["metrics"].get(metric_name, 0) for s in results["samples"]]
        if values:
            metrics_aggregate[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "n": len(values),
            }
    
    results["metrics"] = metrics_aggregate
    results["total_latency_ms"] = total_latency
    results["avg_latency_ms"] = total_latency / len(samples) if samples else 0
    
    # Determine pass/fail based on priority metrics
    priority_metrics = BENCHMARK_CONFIGS.get(
        Path(dataset_path).name, {}
    ).get("priority_metrics", metrics)
    
    all_passed = True
    pass_fail_details = {}
    
    for metric_name in priority_metrics:
        if metric_name in metrics_aggregate:
            mean_val = metrics_aggregate[metric_name]["mean"]
            # Simple threshold: >= 0.7 considered passing
            passed = mean_val >= 0.7
            pass_fail_details[metric_name] = {
                "passed": passed,
                "value": mean_val,
                "threshold": 0.7,
            }
            if not passed:
                all_passed = False
    
    results["overall_passed"] = all_passed
    results["pass_fail_details"] = pass_fail_details
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on standard datasets"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to benchmark dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path for output benchmark report JSON"
    )
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        nargs="+",
        default=["context_recall", "context_precision", "faithfulness", "answer_relevance"],
        help="Metrics to compute (default: context_recall context_precision faithfulness answer_relevance)"
    )
    parser.add_argument(
        "--tenant-id", "-t",
        type=str,
        default="test-tenant",
        help="Tenant ID for retrieval"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="voyage-4-large",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="voyage-rerank-2",
        help="Reranker model to use"
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "bm25_only", "vector_only"],
        help="Retrieval mode"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Limit number of samples (0=all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    
    # Run benchmark
    logger.info("Starting benchmark evaluation...")
    
    results = run_benchmark(
        dataset_path=str(dataset_path),
        output_path=args.output,
        metrics=args.metrics,
        tenant_id=args.tenant_id,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        retrieval_mode=args.retrieval_mode,
        samples_limit=args.samples if args.samples else 0,
        verbose=args.verbose,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Dataset: {results['dataset']}")
    print(f"Source: {BENCHMARK_CONFIGS.get(results['dataset'], {}).get('source', 'unknown')}")
    print(f"Samples: {results['num_samples']}")
    print(f"Config: {results['config']['embedding_model']} + {results['config']['reranker_model']} ({results['config']['retrieval_mode']})")
    print()
    
    print("Aggregated Metrics:")
    print("-" * 70)
    for metric_name, metric_data in results.get("metrics", {}).items():
        print(f"  {metric_name}:")
        print(f"    Mean:   {metric_data['mean']*100:.2f}%")
        print(f"    Std:    {metric_data['std']*100:.2f}%")
        print(f"    Range:  [{metric_data['min']*100:.2f}% - {metric_data['max']*100:.2f}%]")
        print(f"    Median: {metric_data['median']*100:.2f}%")
    
    print()
    print("Latency:")
    print(f"  Total:   {results['total_latency_ms']:.1f}ms")
    print(f"  Average: {results['avg_latency_ms']:.1f}ms per sample")
    print()
    
    # Pass/Fail
    print("Pass/Fail Summary:")
    print("-" * 70)
    if results.get("pass_fail_details"):
        for metric_name, details in results["pass_fail_details"].items():
            status = "✅ PASS" if details["passed"] else "❌ FAIL"
            print(f"  {status} {metric_name}: {details['value']*100:.2f}% (threshold: {details['threshold']*100:.0f}%)")
    
    overall = "✅ PASSED" if results.get("overall_passed") else "❌ FAILED"
    print(f"\nOverall: {overall}")
    print("=" * 70)


if __name__ == "__main__":
    main()
