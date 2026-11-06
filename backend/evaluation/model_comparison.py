#!/usr/bin/env python3
"""Run real per-benchmark chunker, embedding and reranker comparisons.

Every run is scoped to exactly one benchmark. The runner never pools CUAD,
FinanceBench or LegalBench-RAG scores. It uses one Hugging Face tokenizer and
one token budget for all chunkers, so the structural comparison is not
confounded by different chunk sizes.

Examples:
    python -m evaluation.model_comparison --benchmark cuad --mode chunking
    python -m evaluation.model_comparison --benchmark financebench --mode embedding
    python -m evaluation.model_comparison --benchmark cuad --mode rerank

FinanceBench needs its source PDFs parsed first:
    python -m evaluation.prepare_financebench
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from evaluation.ab_benchmark import _evidence_hit, _normalise  # noqa: E402
from evaluation.benchmark_sources import (  # noqa: E402
    load_financebench_samples,
    load_legalbench_rag_samples,
    markdown_with_detected_headings,
)
from evaluation.ragas_evaluator import (  # noqa: E402
    EvaluationSample,
    RAGASEvaluator,
    load_golden_dataset,
)
from services.contextual_chunker import (  # noqa: E402
    Chunk,
    ChunkingStrategy,
    ContextualChunker,
)
from services.local_embeddings import (  # noqa: E402
    EMBEDDER_VARIANTS,
    LocalEmbeddingUnavailable,
    build_local_embedder,
)
from services.local_rerankers import (  # noqa: E402
    RERANKER_VARIANTS,
    LocalRerankerUnavailable,
    build_reranker,
)
from services.token_budget import HuggingFaceCounter  # noqa: E402

CHUNKING_VARIANTS = ("section_aware", "markdown_contextual", "docling_hybrid")


def _document_text(sample: EvaluationSample) -> str:
    return str((sample.metadata or {}).get("document_text") or "").strip()


def _is_plain_legal_text(benchmark: str) -> bool:
    return benchmark in {"cuad", "legalbench_rag"}


def _prepare_text(sample: EvaluationSample, benchmark: str) -> str:
    text = _document_text(sample)
    if not text:
        return ""
    if _is_plain_legal_text(benchmark):
        return markdown_with_detected_headings(text)
    return text


def _load_samples(
    benchmark: str,
    dataset: Optional[str],
    samples: int,
) -> Tuple[List[EvaluationSample], Dict[str, Any]]:
    if benchmark == "financebench":
        loaded, info = load_financebench_samples(
            Path(dataset) if dataset else None,
            require_documents=True,
        )
    elif benchmark == "cuad":
        path = Path(dataset) if dataset else (
            BACKEND_DIR / "evaluation" / "datasets" / "cuad.json"
        )
        loaded = load_golden_dataset(str(path))
        info = {"dataset": str(path), "total_samples": len(loaded)}
    elif benchmark == "legalbench_rag":
        loaded, info = load_legalbench_rag_samples(source="cuad", mini=True)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    sources = sorted(
        {str((sample.metadata or {}).get("source_dataset") or "unknown") for sample in loaded}
    )
    if benchmark == "legalbench_rag":
        if sources != ["legalbench_rag"]:
            raise ValueError(f"Mixed LegalBench-RAG sources: {sources}")
    elif sources != [benchmark]:
        raise ValueError(
            f"Refusing to evaluate {benchmark!r} with mixed sources {sources}"
        )

    if samples > 0:
        loaded = loaded[:samples]
    info["samples_selected"] = len(loaded)
    info["sources"] = sources
    return loaded, info


def _chunks_for(
    sample: EvaluationSample,
    benchmark: str,
    variant: str,
    counter: HuggingFaceCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    text = _prepare_text(sample, benchmark)
    if not text:
        return []
    title = str((sample.metadata or {}).get("source_document") or sample.id)

    if variant == "docling_hybrid":
        from services.docling_chunker import (
            DoclingHybridChunker,
            docling_document_from_markdown,
        )

        document = docling_document_from_markdown(text, f"{sample.id}.md")
        chunks = DoclingHybridChunker(
            max_tokens=max_tokens,
            tokenizer=counter.tokenizer,
        ).chunk(document, document_id=sample.id, document_title=title)
    else:
        strategy = (
            ChunkingStrategy.MARKDOWN_CONTEXTUAL
            if variant == "markdown_contextual"
            else ChunkingStrategy.PARENT_CHILD
        )
        chunks = ContextualChunker(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            token_counter=counter.count,
        ).chunk_with_context(
            document_text=text,
            document_id=sample.id,
            document_title=title,
            chunking_strategy=strategy,
        )

    children = [chunk for chunk in chunks if not chunk.metadata.get("is_parent")]
    for chunk in children:
        chunk.metadata["tokens"] = counter.count(chunk.content)
    return children


def _chunk_metrics(
    sample: EvaluationSample,
    chunks: Sequence[Chunk],
    counter: HuggingFaceCounter,
) -> Dict[str, Any]:
    relevant = sample.relevant_chunks or []
    hits = sum(1 for text in relevant if _evidence_hit(text, chunks))
    full_spans = 0
    for evidence in relevant:
        normalized = _normalise(evidence)
        if normalized and any(normalized in _normalise(chunk.content) for chunk in chunks):
            full_spans += 1
    tokens = [int(chunk.metadata.get("tokens", counter.count(chunk.content))) for chunk in chunks]
    chars = [len(chunk.content) for chunk in chunks]
    return {
        "sample_id": sample.id,
        "chunk_count": len(chunks),
        "mean_tokens": round(statistics.mean(tokens), 2) if tokens else 0.0,
        "max_tokens": max(tokens) if tokens else 0,
        "mean_chars": round(statistics.mean(chars), 2) if chars else 0.0,
        "table_chunks": sum(1 for chunk in chunks if "|" in chunk.content),
        "contextual_chunks": sum(1 for chunk in chunks if chunk.contextual_prefix),
        "evidence_total": len(relevant),
        "evidence_hits": hits,
        "evidence_coverage": (hits / len(relevant)) if relevant else None,
        "full_span_containment": (full_spans / len(relevant)) if relevant else None,
    }


def _aggregate(records: Sequence[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "samples": len(records),
        "samples_with_evidence": sum(
            1
            for record in records
            if any(record.get(name) is not None for name in metric_names)
        ),
    }
    for name in metric_names:
        values = [record[name] for record in records if record.get(name) is not None]
        if not values:
            summary[name] = None
            continue
        summary[name] = round(statistics.mean(values), 4)
        summary[f"{name}_min"] = round(min(values), 4)
        summary[f"{name}_max"] = round(max(values), 4)
    for name in ("chunk_count", "mean_tokens", "max_tokens", "mean_chars"):
        values = [record.get(name, 0) for record in records]
        if values:
            summary[name] = round(statistics.mean(values), 2)
    return summary


def _run_chunking(
    samples: Sequence[EvaluationSample],
    benchmark: str,
    variants: Sequence[str],
    counter: HuggingFaceCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for variant in variants:
        records: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        started = time.monotonic()
        for sample in samples:
            try:
                chunks = _chunks_for(
                    sample, benchmark, variant, counter, max_tokens, overlap_tokens
                )
                records.append(_chunk_metrics(sample, chunks, counter))
            except Exception as exc:
                errors.append({"sample_id": sample.id, "error": str(exc)})
        results[variant] = {
            "variant": variant,
            "metrics": _aggregate(
                records,
                ("evidence_coverage", "full_span_containment"),
            ),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "samples": records,
            "errors": errors,
        }
    return results


def _retrieval_metrics(
    evaluator: RAGASEvaluator,
    sample: EvaluationSample,
    retrieved: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, float]:
    relevant = sample.relevant_chunks or []
    flags = [
        evaluator._chunk_matches_any(item.get("content", ""), relevant)
        for item in retrieved
    ]
    first = next((rank for rank, flag in enumerate(flags, start=1) if flag), None)
    return {
        "context_recall": evaluator._compute_context_recall(list(retrieved), relevant),
        "context_precision": evaluator._compute_context_precision(list(retrieved), relevant),
        "ndcg": evaluator._compute_ndcg(list(retrieved), relevant),
        "mrr": (1.0 / first) if first else 0.0,
        "hit_at_k": 1.0 if first else 0.0,
        "retrieved": len(retrieved),
        "_top_k": top_k,
    }


def _rank_chunks(
    sample: EvaluationSample,
    chunks: Sequence[Chunk],
    embedder: Any,
    top_k: int,
) -> List[Tuple[Chunk, float]]:
    if not chunks:
        return []
    vectors = np.asarray(
        embedder.embed_documents([chunk.contextualized_content for chunk in chunks]),
        dtype=float,
    )
    query = np.asarray(embedder.embed_query(sample.query), dtype=float)
    scores = vectors @ query
    indexes = np.argsort(scores)[::-1][:top_k]
    return [(chunks[int(index)], float(scores[int(index)])) for index in indexes]


def _run_embeddings(
    samples: Sequence[EvaluationSample],
    benchmark: str,
    chunking: str,
    embedders: Sequence[str],
    counter: HuggingFaceCounter,
    max_tokens: int,
    overlap_tokens: int,
    dimension: int,
    top_k: int,
    quantization: str,
    max_length: int,
    batch_size: int,
) -> Dict[str, Any]:
    prepared: List[Tuple[EvaluationSample, List[Chunk]]] = []
    for sample in samples:
        try:
            chunks = _chunks_for(
                sample, benchmark, chunking, counter, max_tokens, overlap_tokens
            )
            if chunks:
                prepared.append((sample, chunks))
        except Exception:
            continue

    evaluator = RAGASEvaluator()
    results: Dict[str, Any] = {}
    for name in embedders:
        started = time.monotonic()
        try:
            embedder = build_local_embedder(
                name,
                dimension=dimension,
                quantization=quantization,
                max_length=max_length,
                batch_size=batch_size,
            )
        except LocalEmbeddingUnavailable as exc:
            results[name] = {
                "variant": name,
                "error": str(exc),
                "metrics": {},
                "samples": [],
            }
            continue

        records: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        try:
            for sample, chunks in prepared:
                try:
                    ranked = _rank_chunks(sample, chunks, embedder, top_k)
                    retrieved = [
                        {"content": chunk.content, "score": score}
                        for chunk, score in ranked
                    ]
                    metrics = _retrieval_metrics(evaluator, sample, retrieved, top_k)
                    records.append({"sample_id": sample.id, **metrics})
                except Exception as exc:
                    errors.append({"sample_id": sample.id, "error": str(exc)})
        finally:
            embedder.close()

        results[name] = {
            "variant": name,
            "model_id": getattr(embedder, "model_id", name),
            "dimension": getattr(embedder, "dimension", dimension),
            "quantization": getattr(embedder, "quantization", None),
            "metrics": _aggregate(
                records,
                ("context_recall", "context_precision", "ndcg", "mrr", "hit_at_k"),
            ),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "samples": records,
            "errors": errors,
        }
    return results


def _run_rerankers(
    samples: Sequence[EvaluationSample],
    benchmark: str,
    chunking: str,
    rerankers: Sequence[str],
    first_stage_embedder: str,
    counter: HuggingFaceCounter,
    max_tokens: int,
    overlap_tokens: int,
    dimension: int,
    first_stage_k: int,
    top_k: int,
    quantization: str,
    max_length: int,
    batch_size: int,
) -> Dict[str, Any]:
    evaluator = RAGASEvaluator()
    first_stage: Dict[str, Any] = {}
    started = time.monotonic()
    try:
        embedder = build_local_embedder(
            first_stage_embedder,
            dimension=dimension,
            quantization=quantization,
            max_length=max_length,
            batch_size=batch_size,
        )
    except LocalEmbeddingUnavailable as exc:
        return {
            "error": str(exc),
            "first_stage_embedder": first_stage_embedder,
            "results": {},
        }

    try:
        for sample in samples:
            chunks = _chunks_for(
                sample, benchmark, chunking, counter, max_tokens, overlap_tokens
            )
            if not chunks:
                continue
            ranked = _rank_chunks(sample, chunks, embedder, first_stage_k)
            first_stage[sample.id] = {
                "sample": sample,
                "candidates": [chunk for chunk, _ in ranked],
                "baseline_scores": [score for _, score in ranked],
            }
    finally:
        embedder.close()

    baseline_records = []
    for item in first_stage.values():
        sample = item["sample"]
        retrieved = [
            {"content": chunk.content, "score": score}
            for chunk, score in zip(item["candidates"][:top_k], item["baseline_scores"][:top_k])
        ]
        baseline_records.append(
            {"sample_id": sample.id, **_retrieval_metrics(evaluator, sample, retrieved, top_k)}
        )
    results: Dict[str, Any] = {
        "first_stage_baseline": {
            "variant": "first_stage_baseline",
            "metrics": _aggregate(
                baseline_records,
                ("context_recall", "context_precision", "ndcg", "mrr", "hit_at_k"),
            ),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "samples": baseline_records,
            "errors": [],
        }
    }

    for name in rerankers:
        reranker_started = time.monotonic()
        try:
            reranker = build_reranker(
                name,
                quantization=quantization,
                max_length=max_length,
                batch_size=batch_size,
            )
        except LocalRerankerUnavailable as exc:
            results[name] = {"variant": name, "error": str(exc), "samples": []}
            continue

        records: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        try:
            for item in first_stage.values():
                sample = item["sample"]
                candidates = item["candidates"]
                try:
                    scores = reranker.score(
                        sample.query, [chunk.content for chunk in candidates]
                    )
                    order = sorted(
                        range(len(candidates)), key=lambda index: scores[index], reverse=True
                    )[:top_k]
                    retrieved = [
                        {"content": candidates[index].content, "score": scores[index]}
                        for index in order
                    ]
                    records.append(
                        {
                            "sample_id": sample.id,
                            **_retrieval_metrics(evaluator, sample, retrieved, top_k),
                        }
                    )
                except Exception as exc:
                    errors.append({"sample_id": sample.id, "error": str(exc)})
        finally:
            reranker.close()

        results[name] = {
            "variant": name,
            "model_id": getattr(reranker, "model_id", name),
            "quantization": getattr(reranker, "quantization", None),
            "metrics": _aggregate(
                records,
                ("context_recall", "context_precision", "ndcg", "mrr", "hit_at_k"),
            ),
            "elapsed_seconds": round(time.monotonic() - reranker_started, 2),
            "samples": records,
            "errors": errors,
        }
    return {
        "first_stage_embedder": first_stage_embedder,
        "first_stage_k": first_stage_k,
        "results": results,
    }


def _winner(results: Dict[str, Any], metric: str, minimum_samples: int = 10) -> Optional[str]:
    scored = [
        (name, result.get("metrics", {}).get(metric))
        for name, result in results.items()
        if result.get("metrics", {}).get(metric) is not None
    ]
    if not scored:
        return None
    samples = max(
        (result.get("metrics", {}).get("samples", 0) for result in results.values()),
        default=0,
    )
    if samples < minimum_samples:
        return None
    return max(scored, key=lambda item: item[1])[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-benchmark model comparison")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["cuad", "financebench", "legalbench_rag"],
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["chunking", "embedding", "rerank"],
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--variants", default=",".join(CHUNKING_VARIANTS))
    parser.add_argument("--chunking", default="markdown_contextual", choices=CHUNKING_VARIANTS)
    parser.add_argument("--embedders", default=",".join(EMBEDDER_VARIANTS))
    parser.add_argument("--rerankers", default=",".join(RERANKER_VARIANTS))
    parser.add_argument("--first-stage-embedder", default="qwen3-0.6b")
    parser.add_argument("--first-stage-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--quantization", default="auto", choices=["auto", "bf16", "4bit"])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--overlap-tokens", type=int, default=300)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, info = _load_samples(args.benchmark, args.dataset, args.samples)
    if not samples:
        print(f"No usable samples for {args.benchmark}", file=sys.stderr)
        return 2

    counter = HuggingFaceCounter(args.tokenizer)
    report: Dict[str, Any] = {
        "description": "Per-benchmark RAG model comparison; no cross-benchmark pooling",
        "benchmark": args.benchmark,
        "mode": args.mode,
        "dataset_info": info,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "samples": len(samples),
            "tokenizer": args.tokenizer,
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "top_k": args.top_k,
            "dimension": args.dimension if args.mode != "chunking" else None,
            "quantization": args.quantization if args.mode != "chunking" else None,
        },
        "notes": [],
    }

    if args.mode == "chunking":
        variants = [item.strip() for item in args.variants.split(",") if item.strip()]
        unknown = set(variants) - set(CHUNKING_VARIANTS)
        if unknown:
            print(f"Unknown chunking variants: {sorted(unknown)}", file=sys.stderr)
            return 2
        results = _run_chunking(
            samples,
            args.benchmark,
            variants,
            counter,
            args.max_tokens,
            args.overlap_tokens,
        )
        report["results"] = results
        report["winner_metric"] = "evidence_coverage"
        report["winner"] = _winner(results, "evidence_coverage")
        report["notes"].append(
            "Structural evidence coverage only; run --mode embedding for retrieval quality."
        )
        if args.benchmark == "financebench":
            report["notes"].append(
                "FinanceBench uses Docling-parsed filing Markdown; samples whose PDF "
                "could not be downloaded/parsed are excluded."
            )
    elif args.mode == "embedding":
        embedders = [item.strip() for item in args.embedders.split(",") if item.strip()]
        unknown = set(embedders) - set(EMBEDDER_VARIANTS)
        if unknown:
            print(f"Unknown embedders: {sorted(unknown)}", file=sys.stderr)
            return 2
        results = _run_embeddings(
            samples,
            args.benchmark,
            args.chunking,
            embedders,
            counter,
            args.max_tokens,
            args.overlap_tokens,
            args.dimension,
            args.top_k,
            args.quantization,
            args.max_length,
            args.batch_size,
        )
        report["results"] = results
        report["winner_metric"] = "context_recall"
        report["winner"] = _winner(results, "context_recall")
        report["notes"].append(
            "Qwen3-4B is evaluated at 1024 dimensions via Matryoshka truncation; "
            "its native output is 2560."
        )
    else:
        rerankers = [item.strip() for item in args.rerankers.split(",") if item.strip()]
        unknown = set(rerankers) - set(RERANKER_VARIANTS)
        if unknown:
            print(f"Unknown rerankers: {sorted(unknown)}", file=sys.stderr)
            return 2
        result = _run_rerankers(
            samples,
            args.benchmark,
            args.chunking,
            rerankers,
            args.first_stage_embedder,
            counter,
            args.max_tokens,
            args.overlap_tokens,
            args.dimension,
            args.first_stage_k,
            args.top_k,
            args.quantization,
            args.max_length,
            args.batch_size,
        )
        results = result.get("results", {})
        report["results"] = results
        report["rerank_config"] = {
            "first_stage_embedder": result.get("first_stage_embedder"),
            "first_stage_k": result.get("first_stage_k"),
        }
        report["winner_metric"] = "ndcg"
        report["winner"] = _winner(results, "ndcg")
        if result.get("error"):
            report["error"] = result["error"]
        report["notes"].append(
            "All rerankers receive the identical first-stage candidate set; "
            "the baseline is the first-stage order before reranking."
        )

    output = (
        Path(args.output)
        if args.output
        else BACKEND_DIR
        / "evaluation"
        / "results"
        / "model_comparison"
        / f"{args.benchmark}_{args.mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{args.mode.title()} comparison: {args.benchmark}")
    for name, result in report["results"].items():
        if result.get("error"):
            print(f"  {name:<28} ERROR: {result['error']}")
        else:
            print(f"  {name:<28} {result.get('metrics')}")
    print(f"Winner: {report.get('winner') or 'not determined'}")
    print(f"Report: {output}")

    errors = [
        result
        for result in report["results"].values()
        if result.get("error") or result.get("errors")
    ]
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
