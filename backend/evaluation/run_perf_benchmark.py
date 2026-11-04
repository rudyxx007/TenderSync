#!/usr/bin/env python3
"""
Performance Benchmark Runner - Measures latency and throughput.

Runs micro-benchmarks on RAG pipeline components to track
performance characteristics over time.

Usage:
    python -m evaluation.run_perf_benchmark \
        --output evaluation/results/perf_benchmark.json \
        --iterations 100
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import statistics
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    unit: str = "ms"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerfBenchmarkReport:
    """Complete performance benchmark report."""
    run_id: str
    timestamp: str
    system_info: Dict[str, Any]
    benchmarks: List[BenchmarkResult]
    overall_summary: Dict[str, Any]


def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure execution time of a function.
    
    Returns (result, elapsed_ms).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def run_benchmark(
    name: str,
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    *args,
    **kwargs,
) -> BenchmarkResult:
    """
    Run a benchmark on a function.
    
    Args:
        name: Benchmark name
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations (not counted)
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        BenchmarkResult with statistics
    """
    times = []
    
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Actual benchmark
    for _ in range(iterations):
        _, elapsed = measure_execution_time(func, *args, **kwargs)
        times.append(elapsed)
    
    # Compute statistics
    result = BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=sum(times),
        mean_time_ms=statistics.mean(times),
        median_time_ms=statistics.median(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        p50_time_ms=sorted(times)[int(len(times) * 0.5)],
        p95_time_ms=sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times),
        p99_time_ms=sorted(times)[int(len(times) * 0.99)] if len(times) > 100 else max(times),
    )
    
    return result


def get_system_info() -> Dict[str, Any]:
    """Get current system information."""
    process = psutil.Process()
    
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_used_gb": process.memory_info().rss / (1024**3),
        "python_version": sys.version,
        "platform": sys.platform,
    }


def benchmark_embedding(
    embedding_service,
    queries: List[str],
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark embedding service."""
    
    def embed_all():
        for query in queries:
            embedding_service.embed_query(query)
    
    return run_benchmark(
        name="embedding_generation",
        func=embed_all,
        iterations=iterations // len(queries),
    )


def benchmark_hybrid_retrieval(
    retriever,
    query: str,
    embedding: List[float],
    tenant_id: str,
    iterations: int = 50,
) -> BenchmarkResult:
    """Benchmark hybrid retrieval."""
    
    def retrieve():
        retriever.retrieve(
            query=query,
            query_embedding=embedding,
            tenant_id=tenant_id,
            top_k=10,
        )
    
    return run_benchmark(
        name="hybrid_retrieval",
        func=retrieve,
        iterations=iterations,
    )


def benchmark_reranking(
    reranker,
    query: str,
    documents: List[str],
    iterations: int = 50,
) -> BenchmarkResult:
    """Benchmark reranking."""
    
    def rerank():
        reranker.rerank(
            query=query,
            documents=documents,
        )
    
    return run_benchmark(
        name="reranking",
        func=rerank,
        iterations=iterations,
    )


def benchmark_end_to_end(
    pipeline,
    queries: List[str],
    tenant_id: str,
    iterations: int = 20,
) -> BenchmarkResult:
    """Benchmark end-to-end query processing."""
    
    def process_query():
        for query in queries:
            pipeline.query(query, tenant_id=tenant_id)
    
    return run_benchmark(
        name="end_to_end_query",
        func=process_query,
        iterations=iterations // len(queries),
    )


def run_full_perf_benchmark(
    output_path: str,
    iterations: int = 100,
    embedding_model: str = "voyage-4-large",
    tenant_id: str = "test-tenant",
) -> PerfBenchmarkReport:
    """
    Run full performance benchmark suite.
    
    Returns PerfBenchmarkReport.
    """
    logger.info("Starting performance benchmarks...")
    
    system_info = get_system_info()
    benchmarks = []
    run_id = f"perf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Import services
    try:
        from backend.services.voyage_embeddings import VoyageEmbeddingService, VoyageReranker
        from backend.services.hybrid_retriever import HybridRetriever
        from backend.services.pipeline_adapter import RAGPipeline
        
        # Initialize services
        logger.info("Initializing services...")
        embedder = VoyageEmbeddingService(model=embedding_model)
        reranker = VoyageReranker(model="voyage-rerank-2")
        
        # Test queries
        test_queries = [
            "What are the mandatory certification requirements?",
            "What is the contract value and payment schedule?",
            "When is the submission deadline?",
            "What insurance coverage is required?",
        ]
        
        # Test documents for reranking
        test_documents = [
            "The supplier must hold ISO 9001:2015 certification",
            "ISO 27001:2013 is required for information security",
            "ISO 14001:2015 environmental certification is mandatory",
            "All certifications must be valid through contract period",
        ]
        
        # Benchmark 1: Embedding generation
        logger.info("Benchmarking embedding generation...")
        embed_result = run_benchmark(
            name="embedding_single_query",
            func=lambda: embedder.embed_query(test_queries[0]),
            iterations=iterations,
        )
        benchmarks.append(embed_result)
        
        # Benchmark 2: Batch embedding
        logger.info("Benchmarking batch embedding...")
        batch_result = run_benchmark(
            name="embedding_batch",
            func=lambda: [embedder.embed_query(q) for q in test_queries],
            iterations=max(1, iterations // 10),
        )
        benchmarks.append(batch_result)
        
        # Benchmark 3: Reranking
        logger.info("Benchmarking reranking...")
        rerank_result = run_benchmark(
            name="reranking_documents",
            func=lambda: reranker.rerank(
                query=test_queries[0],
                documents=test_documents,
            ),
            iterations=max(1, iterations // 10),
        )
        benchmarks.append(rerank_result)
        
        # Benchmark 4: End-to-end pipeline (if available)
        logger.info("Benchmarking end-to-end pipeline...")
        try:
            pipeline = RAGPipeline(
                embedding_model=embedding_model,
                reranker_model="voyage-rerank-2",
            )
            
            pipeline_result = run_benchmark(
                name="end_to_end_query",
                func=lambda: pipeline.query(test_queries[0], tenant_id=tenant_id),
                iterations=max(1, iterations // 20),
            )
            benchmarks.append(pipeline_result)
        except Exception as e:
            logger.warning(f"End-to-end benchmark skipped: {e}")
        
        # Cleanup
        del embedder, reranker
        
    except ImportError as e:
        logger.warning(f"Could not import services for benchmarking: {e}")
        logger.info("Running lightweight benchmarks instead...")
        
        # Simple function benchmarks
        def dummy_embedding(text: str) -> List[float]:
            """Simulate embedding with hash-based vector."""
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            return [hash_val / (2**32)] * 1024
        
        import hashlib
        
        emb_result = run_benchmark(
            name="dummy_embedding",
            func=lambda: dummy_embedding("test query"),
            iterations=iterations,
        )
        benchmarks.append(emb_result)
    
    # Compute overall summary
    total_time = sum(b.total_time_ms for b in benchmarks)
    avg_latency = statistics.mean([b.mean_time_ms for b in benchmarks]) if benchmarks else 0
    
    overall_summary = {
        "total_benchmarks": len(benchmarks),
        "total_time_ms": total_time,
        "avg_mean_latency_ms": avg_latency,
        "embedding_latency_ms": next((b.mean_time_ms for b in benchmarks if "embedding" in b.name.lower()), None),
        "reranking_latency_ms": next((b.mean_time_ms for b in benchmarks if "rerank" in b.name.lower()), None),
        "end_to_end_latency_ms": next((b.mean_time_ms for b in benchmarks if "end_to_end" in b.name.lower()), None),
    }
    
    return PerfBenchmarkReport(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        benchmarks=benchmarks,
        overall_summary=overall_summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks on RAG pipeline"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path for output benchmark JSON"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="voyage-4-large",
        help="Embedding model for benchmarks"
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="test-tenant",
        help="Tenant ID for retrieval tests"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with fewer iterations"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.iterations = 10
        logger.info("Quick mode: reduced iterations")
    
    # Run benchmarks
    report = run_full_perf_benchmark(
        output_path=args.output,
        iterations=args.iterations,
        embedding_model=args.embedding_model,
        tenant_id=args.tenant_id,
    )
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "system_info": report.system_info,
        "benchmarks": [
            {
                "name": b.name,
                "iterations": b.iterations,
                "mean_ms": round(b.mean_time_ms, 3),
                "median_ms": round(b.median_time_ms, 3),
                "min_ms": round(b.min_time_ms, 3),
                "max_ms": round(b.max_time_ms, 3),
                "std_dev_ms": round(b.std_dev_ms, 3),
                "p95_ms": round(b.p95_time_ms, 3),
                "p99_ms": round(b.p99_time_ms, 3),
            }
            for b in report.benchmarks
        ],
        "overall_summary": report.overall_summary,
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Performance report saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Run ID: {report.run_id}")
    print(f"Timestamp: {report.timestamp}")
    print()
    print("System Info:")
    for key, value in report.system_info.items():
        print(f"  {key}: {value}")
    print()
    print("Benchmark Results:")
    print("-" * 60)
    print(f"{'Benchmark':<30} {'Mean':>10} {'P95':>10} {'P99':>10} {'Iterations':>10}")
    print("-" * 60)
    
    for b in report.benchmarks:
        print(
            f"{b.name:<30} "
            f"{b.mean_time_ms:>8.2f}ms "
            f"{b.p95_time_ms:>8.2f}ms "
            f"{b.p99_time_ms:>8.2f}ms "
            f"{b.iterations:>10}"
        )
    
    print("-" * 60)
    print()
    print("Overall Summary:")
    for key, value in report.overall_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}ms" if "latency" in key or "time" in key else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
