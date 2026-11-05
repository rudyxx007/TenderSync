#!/usr/bin/env python3
"""A/B benchmark runner for chunking and embedding choices.

This runner deliberately keeps FinanceBench and CUAD separate. It evaluates
retrieval locally, so it can compare chunkers and embedding models without
seeding a tenant or calling Groq. Generation metrics still belong in
``run_evaluation``.

Examples:
    # Structural chunking comparison (no embedding model required)
    python -m evaluation.ab_benchmark --mode chunking \
        --benchmark cuad --variants section_aware,markdown_contextual,docling_hybrid

    # Embedding comparison under one fixed chunker
    python -m evaluation.ab_benchmark --mode embedding \
        --benchmark cuad --chunking markdown_contextual \
        --embedders qwen3-0.6b,qwen3-4b,fin-e5

FinanceBench samples are evidence-only; they have no full ``document_text``
and are reported as skipped for chunking/retrieval A/B. CUAD samples contain
the full contract text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from evaluation.ragas_evaluator import EvaluationSample, RAGASEvaluator, load_golden_dataset
from services.contextual_chunker import Chunk, ChunkingStrategy, ContextualChunker
from services.local_embeddings import LocalEmbeddingUnavailable, build_local_embedder

CHUNKING_VARIANTS = ("section_aware", "markdown_contextual", "docling_hybrid")
EMBEDDER_VARIANTS = ("qwen3-0.6b", "qwen3-4b", "fin-e5")


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", (text or "").lower()))


def _token_coverage(needle: str, haystack: str) -> float:
    needle_tokens = set(_normalise(needle).split())
    if not needle_tokens:
        return 0.0
    haystack_tokens = set(_normalise(haystack).split())
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def _evidence_hit(ground_truth: str, chunks: Sequence[Chunk]) -> bool:
    needle = _normalise(ground_truth)
    if not needle:
        return False
    for chunk in chunks:
        haystack = _normalise(chunk.content)
        if needle in haystack or _token_coverage(ground_truth, chunk.content) >= 0.85:
            return True
    return False


def _document_text(sample: EvaluationSample) -> str:
    metadata = sample.metadata or {}
    return str(metadata.get("document_text") or metadata.get("text") or "")


def _chunks_for_variant(
    sample: EvaluationSample,
    variant: str,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
) -> List[Chunk]:
    text = _document_text(sample)
    if not text.strip():
        return []

    if variant == "docling_hybrid":
        from services.docling_chunker import (
            DoclingHybridChunker,
            docling_document_from_markdown,
        )

        document = docling_document_from_markdown(text, f"{sample.id}.md")
        chunks = DoclingHybridChunker(max_tokens=max_tokens).chunk(
            document,
            document_id=sample.id,
            document_title=str((sample.metadata or {}).get("source_document") or sample.id),
        )
    else:
        strategy = (
            ChunkingStrategy.MARKDOWN_CONTEXTUAL
            if variant == "markdown_contextual"
            else ChunkingStrategy.PARENT_CHILD
        )
        chunks = ContextualChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).chunk_with_context(
            document_text=text,
            document_id=sample.id,
            document_title=str((sample.metadata or {}).get("source_document") or sample.id),
            chunking_strategy=strategy,
        )
    return [chunk for chunk in chunks if not chunk.metadata.get("is_parent")]


def _chunk_record(sample: EvaluationSample, chunks: List[Chunk]) -> Dict[str, Any]:
    relevant = sample.relevant_chunks or []
    hits = sum(1 for text in relevant if _evidence_hit(text, chunks))
    lengths = [len(chunk.content) for chunk in chunks]
    return {
        "sample_id": sample.id,
        "chunk_count": len(chunks),
        "characters": sum(lengths),
        "mean_chunk_chars": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "table_chunks": sum(1 for chunk in chunks if "|" in chunk.content),
        "contextual_chunks": sum(1 for chunk in chunks if chunk.contextual_prefix),
        "evidence_total": len(relevant),
        "evidence_hits": hits,
        "evidence_coverage": (hits / len(relevant)) if relevant else None,
    }


def _aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [record for record in records if record["evidence_coverage"] is not None]
    return {
        "samples": len(records),
        "samples_with_evidence": len(scored),
        "mean_chunk_count": round(
            statistics.mean(record["chunk_count"] for record in records), 2
        )
        if records
        else 0.0,
        "mean_chunk_chars": round(
            statistics.mean(record["mean_chunk_chars"] for record in records), 2
        )
        if records
        else 0.0,
        "max_chunk_chars": max((record["max_chunk_chars"] for record in records), default=0),
        "table_chunks": sum(record["table_chunks"] for record in records),
        "contextual_chunks": sum(record["contextual_chunks"] for record in records),
        "evidence_coverage": round(
            statistics.mean(record["evidence_coverage"] for record in scored), 4
        )
        if scored
        else None,
    }


def _run_chunking(
    samples: Sequence[EvaluationSample],
    variants: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for variant in variants:
        records: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        for sample in samples:
            if not _document_text(sample).strip():
                continue
            try:
                chunks = _chunks_for_variant(
                    sample, variant, chunk_size, chunk_overlap, max_tokens
                )
                records.append(_chunk_record(sample, chunks))
            except Exception as exc:
                errors.append({"sample_id": sample.id, "error": str(exc)})
        results[variant] = {
            "variant": variant,
            "metrics": _aggregate(records),
            "samples": records,
            "errors": errors,
        }
    return results


def _embedding_retrieval(
    sample: EvaluationSample,
    chunks: Sequence[Chunk],
    embedder,
    top_k: int,
) -> Dict[str, Any]:
    if not chunks:
        return {}
    import numpy as np

    vectors = np.asarray(
        embedder.embed_documents([chunk.contextualized_content for chunk in chunks]),
        dtype=float,
    )
    query = np.asarray(embedder.embed_query(sample.query), dtype=float)
    scores = vectors @ query
    indexes = np.argsort(scores)[::-1][:top_k]
    retrieved = [
        {"content": chunks[index].content, "score": float(scores[index])}
        for index in indexes
    ]
    evaluator = RAGASEvaluator()
    return {
        "context_recall": evaluator._compute_context_recall(retrieved, sample.relevant_chunks),
        "context_precision": evaluator._compute_context_precision(
            retrieved, sample.relevant_chunks
        ),
        "ndcg": evaluator._compute_ndcg(retrieved, sample.relevant_chunks),
        "retrieved": retrieved,
    }


def _run_embeddings(
    samples: Sequence[EvaluationSample],
    chunking: str,
    embedders: Sequence[str],
    dimension: Optional[int],
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
) -> Dict[str, Any]:
    prepared: List[Tuple[EvaluationSample, List[Chunk]]] = []
    for sample in samples:
        if not _document_text(sample).strip():
            continue
        try:
            prepared.append(
                (
                    sample,
                    _chunks_for_variant(
                        sample, chunking, chunk_size, chunk_overlap, max_tokens
                    ),
                )
            )
        except Exception:
            continue

    results: Dict[str, Any] = {}
    for name in embedders:
        try:
            embedder = build_local_embedder(name, dimension=dimension)
        except LocalEmbeddingUnavailable as exc:
            results[name] = {"variant": name, "error": str(exc), "samples": 0}
            continue

        records: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        for sample, chunks in prepared:
            try:
                metrics = _embedding_retrieval(sample, chunks, embedder, top_k)
                if metrics:
                    records.append({"sample_id": sample.id, **metrics})
            except Exception as exc:
                errors.append({"sample_id": sample.id, "error": str(exc)})

        metric_names = ("context_recall", "context_precision", "ndcg")
        summary = {
            metric: round(
                statistics.mean(record[metric] for record in records), 4
            )
            if records
            else None
            for metric in metric_names
        }
        results[name] = {
            "variant": name,
            "model_id": getattr(embedder, "model_id", name),
            "dimension": getattr(embedder, "dimension", dimension),
            "metrics": summary,
            "samples": records,
            "errors": errors,
        }
    return results


def _winner(results: Dict[str, Any], metric: str) -> Optional[str]:
    scored = [
        (name, result.get("metrics", {}).get(metric))
        for name, result in results.items()
        if result.get("metrics", {}).get(metric) is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B chunking and embedding benchmark")
    parser.add_argument("--benchmark", required=True, choices=["financebench", "cuad"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["chunking", "embedding"])
    parser.add_argument("--variants", default=",".join(CHUNKING_VARIANTS))
    parser.add_argument("--chunking", default="markdown_contextual", choices=CHUNKING_VARIANTS)
    parser.add_argument("--embedders", default=",".join(EMBEDDER_VARIANTS))
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = (
        Path(args.dataset)
        if args.dataset
        else Path(__file__).parent / "datasets" / f"{args.benchmark}.json"
    )
    samples = load_golden_dataset(str(dataset_path))
    sources = sorted({str((sample.metadata or {}).get("source_dataset") or "unknown") for sample in samples})
    if sources != [args.benchmark]:
        print(
            f"Refusing to evaluate benchmark {args.benchmark!r} with sources {sources}",
            file=sys.stderr,
        )
        return 2
    if args.samples > 0:
        samples = samples[: args.samples]

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = set(variants) - set(CHUNKING_VARIANTS)
    if unknown:
        print(f"Unknown chunking variants: {sorted(unknown)}", file=sys.stderr)
        return 2

    if args.mode == "chunking":
        results = _run_chunking(
            samples,
            variants,
            args.chunk_size,
            args.chunk_overlap,
            args.max_tokens,
        )
        winner = _winner(results, "evidence_coverage")
        metric = "evidence_coverage"
    else:
        embedders = [item.strip() for item in args.embedders.split(",") if item.strip()]
        unknown = set(embedders) - set(EMBEDDER_VARIANTS)
        if unknown:
            print(f"Unknown embedders: {sorted(unknown)}", file=sys.stderr)
            return 2
        results = _run_embeddings(
            samples,
            args.chunking,
            embedders,
            args.dimension,
            args.top_k,
            args.chunk_size,
            args.chunk_overlap,
            args.max_tokens,
        )
        winner = _winner(results, "context_recall")
        metric = "context_recall"

    report = {
        "description": "Separate-benchmark RAG chunking/embedding A/B",
        "benchmark": args.benchmark,
        "dataset": str(dataset_path),
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "samples": len(samples),
            "chunking": args.chunking if args.mode == "embedding" else None,
            "dimension": args.dimension if args.mode == "embedding" else None,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "max_tokens": args.max_tokens,
            "note": (
                "FinanceBench has evidence-only samples and no full document_text; "
                "chunking/retrieval comparisons need CUAD full contracts."
                if args.benchmark == "financebench"
                else "CUAD samples contain full contract text."
            ),
        },
        "winner_metric": metric,
        "winner": winner if len(samples) >= 10 else None,
        "winner_note": (
            None
            if len(samples) >= 10
            else "winner suppressed for smoke runs with fewer than 10 samples"
        ),
        "variants": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"{args.mode.title()} A/B: {args.benchmark}")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name:<20} ERROR: {result['error']}")
        else:
            values = result.get("metrics", {})
            print(f"  {name:<20} {values}")
    print(
        f"Winner by {metric}: "
        f"{report['winner'] or 'not determined (need at least 10 samples)'}"
    )
    print(f"Report: {output_path}")

    errors = [result for result in results.values() if result.get("error")]
    if args.strict and errors:
        return 1
    return 1 if errors and len(errors) == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
