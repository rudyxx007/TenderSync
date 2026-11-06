#!/usr/bin/env python3
"""Run the required RAGAS + TruLens audit on one benchmark at a time.

This is an offline audit, not the production Postgres/RLS path: it uses the
same Markdown-contextual chunker and a deterministic lexical retriever, then
calls Groq to generate a cited answer from those chunks. Keeping retrieval
local makes the audit reproducible without a seeded tenant while still
exercising the real RAGAS and TruLens packages.

Examples:
    python -m evaluation.run_audit --benchmark cuad --samples 3
    python -m evaluation.run_audit --benchmark legalbench_rag --samples 3

Artifacts:
    evaluation/results/audit/<benchmark>_ragas.json
    evaluation/results/audit/<benchmark>_trulens.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BACKEND_DIR / ".env")
except ImportError:
    pass

from evaluation.benchmark_sources import (  # noqa: E402
    load_financebench_samples,
    load_legalbench_rag_samples,
)
from evaluation.ragas_evaluator import (  # noqa: E402
    EvaluationSample,
    load_golden_dataset,
)
from evaluation.ragas_optional import run_ragas  # noqa: E402
from services.contextual_chunker import ContextualChunker  # noqa: E402
from services.token_budget import TiktokenCounter  # noqa: E402


def _load(benchmark: str, samples: int) -> Tuple[List[EvaluationSample], Dict[str, Any]]:
    if benchmark == "cuad":
        path = BACKEND_DIR / "evaluation" / "datasets" / "cuad.json"
        loaded = load_golden_dataset(str(path))
        info = {"dataset": str(path), "total_samples": len(loaded)}
    elif benchmark == "financebench":
        loaded, info = load_financebench_samples(require_documents=True)
    elif benchmark == "legalbench_rag":
        loaded, info = load_legalbench_rag_samples(source="cuad", mini=True)
    else:  # pragma: no cover
        raise ValueError(benchmark)
    if samples:
        loaded = loaded[:samples]
    info["samples_selected"] = len(loaded)
    return loaded, info


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", (text or "").lower())
        if len(token) > 1
    }


def _retrieve(query: str, chunks: Sequence[Any], top_k: int = 5) -> List[Dict[str, Any]]:
    q = _tokens(query)
    scored = []
    for chunk in chunks:
        text = chunk.content
        c = _tokens(text)
        if not q or not c:
            continue
        overlap = len(q & c) / (len(q) ** 0.5 * len(c) ** 0.5)
        # Legal queries often contain exact clause terms; reward exact phrase hits.
        phrase_bonus = 0.15 if query.lower().strip() and query.lower().strip() in text.lower() else 0.0
        scored.append((overlap + phrase_bonus, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"content": chunk.content, "score": round(score, 6), "chunk_id": getattr(chunk, "id", None)}
        for score, chunk in scored[:top_k]
    ]


def _generate(query: str, retrieved: Sequence[Dict[str, Any]]) -> str:
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is required for the audit generation step.")
    context = "\n\n".join(
        f"[{i}] {item['content']}" for i, item in enumerate(retrieved, start=1)
    )
    prompt = (
        "Answer the question using only the numbered evidence. "
        "Every factual sentence must carry an inline citation such as [1]. "
        "If the evidence is insufficient, reply exactly: "
        "Not found in the provided documents.\n\n"
        f"QUESTION:\n{query}\n\nEVIDENCE:\n{context}"
    )
    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {
                "role": "system",
                "content": "You are a contract analyst. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        reasoning_effort="low",
    )
    answer = (completion.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError(
            f"Groq returned no content (finish_reason={completion.choices[0].finish_reason})"
        )
    return answer


def _citation_support(output: Any) -> float:
    """Deterministic TruLens feedback: citation-resolvable factual sentences."""
    if not isinstance(output, dict):
        return 0.0
    answer = str(output.get("answer") or "")
    contexts = output.get("contexts") or []
    if not answer or not contexts:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        markers = [int(x) for x in re.findall(r"\[(\d+)\]", sentence)]
        if not markers:
            continue
        if all(1 <= marker <= len(contexts) for marker in markers):
            supported += 1
    return supported / len(sentences)


class AuditPipeline:
    def __init__(self, samples: Sequence[EvaluationSample], max_tokens: int = 512):
        self.samples = list(samples)
        self.chunker = ContextualChunker(
            chunk_size=max_tokens,
            chunk_overlap=64,
            token_counter=TiktokenCounter().count,
        )

    def query(self, query: str, sample_id: str) -> Dict[str, Any]:
        sample = next(item for item in self.samples if item.id == sample_id)
        text = str((sample.metadata or {}).get("document_text") or "")
        chunks = self.chunker.chunk_with_context(
            document_text=text,
            document_id=sample.id,
            document_title=str((sample.metadata or {}).get("source_document") or sample.id),
        )
        # The Markdown-contextual run uses child chunks; parent rows are retrieval
        # context and are not counted as separate generator passages here.
        children = [chunk for chunk in chunks if not chunk.metadata.get("is_parent")]
        retrieved = _retrieve(query, children, top_k=5)
        answer = _generate(query, retrieved)
        return {
            "answer": answer,
            "contexts": [item["content"] for item in retrieved],
            "retrieved": retrieved,
            "sample_id": sample.id,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["cuad", "financebench", "legalbench_rag"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output-dir", default=str(BACKEND_DIR / "evaluation" / "results" / "audit"))
    args = parser.parse_args()

    samples, info = _load(args.benchmark, args.samples)
    if not samples:
        print(f"No samples for {args.benchmark}", file=sys.stderr)
        return 2

    pipeline = AuditPipeline(samples)
    ragas_rows: List[Dict[str, Any]] = []
    sampled: List[Dict[str, Any]] = []
    for sample in samples:
        result = pipeline.query(sample.query, sample.id)
        reference = sample.ground_truth or (
            sample.relevant_chunks[0] if sample.relevant_chunks else ""
        )
        row = {
            "sample_id": sample.id,
            "query": sample.query,
            "ground_truth": reference,
            "generated_answer": result["answer"],
            "retrieved_contexts": result["contexts"],
            "retrieved": result["retrieved"],
        }
        sampled.append(row)
        ragas_rows.append(
            {
                "user_input": sample.query,
                "retrieved_contexts": result["contexts"],
                "response": result["answer"],
                "ground_truth": reference,
            }
        )

    ragas = run_ragas(ragas_rows)
    ragas["benchmark"] = args.benchmark
    ragas["dataset_info"] = info
    ragas["generation_model"] = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    ragas["note"] = (
        "Offline audit path: Markdown-contextual chunker plus deterministic lexical "
        "retrieval. It is not the production Postgres/RLS hybrid query."
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ragas_path = out_dir / f"{args.benchmark}_ragas.json"
    ragas_path.write_text(json.dumps({"summary": ragas, "samples": sampled}, indent=2), encoding="utf-8")

    # TruLens records the same query path and a deterministic citation-support
    # feedback. The feedback is intentionally not an LLM judge: it tests whether
    # the answer's markers resolve to passages actually supplied to the model.
    trulens_summary: Dict[str, Any] = {"status": "not_run"}
    try:
        from evaluation.trulens_optional import TruLensTracer
        from trulens.core import Metric, TruSession

        metric = Metric(
            implementation=_citation_support,
            name="Citation Support",
            higher_is_better=True,
        ).on_response()
        session = TruSession()
        tracer = TruLensTracer(
            pipeline,
            app_name=f"TenderSync Audit {args.benchmark}",
            app_version=datetime.now().strftime("%Y%m%d%H%M"),
            feedbacks=[metric],
        )
        started = time.monotonic()
        with tracer:
            for sample in samples:
                tracer.pipeline.query(sample.query, sample.id)
        records, _ = session.get_records_and_feedback()
        trulens_summary = {
            "status": "executed",
            "app_name": f"TenderSync Audit {args.benchmark}",
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "records": int(len(records)),
            "feedback": "Citation Support",
            "feedback_note": "Deterministic marker-to-context check; not an LLM judge.",
        }
    except Exception as exc:
        trulens_summary = {"status": "failed", "error": str(exc)}

    trulens_path = out_dir / f"{args.benchmark}_trulens.json"
    trulens_path.write_text(
        json.dumps({"summary": trulens_summary, "samples": sampled}, indent=2),
        encoding="utf-8",
    )

    print(f"Audit: {args.benchmark} ({len(samples)} samples)")
    print("RAGAS:", json.dumps(ragas["metrics"], indent=2))
    print("TruLens:", trulens_summary)
    print(f"RAGAS report: {ragas_path}")
    print(f"TruLens report: {trulens_path}")
    return 0 if trulens_summary.get("status") == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
