"""Benchmark source adapters for the model-comparison harness.

FinanceBench and CUAD are never blended. FinanceBench uses its released PDFs
converted to Markdown; CUAD uses the contracts already embedded in the golden
dataset. LegalBench-RAG is a separate legal retrieval benchmark and is loaded
only when explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from evaluation.ragas_evaluator import EvaluationSample, load_golden_dataset

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "benchmarks"
FINANCEBENCH_DIR = DATA_DIR / "financebench"
LEGALBENCH_RAG_DIR = DATA_DIR / "legalbench-rag"


def financebench_markdown_dir() -> Path:
    return FINANCEBENCH_DIR / "markdown"


def load_financebench_samples(
    dataset_path: Optional[Path] = None,
    markdown_dir: Optional[Path] = None,
    require_documents: bool = True,
) -> tuple[List[EvaluationSample], Dict[str, Any]]:
    """Load FinanceBench samples and attach the matching parsed filing text."""
    dataset_path = dataset_path or (
        BACKEND_DIR / "evaluation" / "datasets" / "financebench.json"
    )
    samples = load_golden_dataset(str(dataset_path))
    markdown_dir = markdown_dir or financebench_markdown_dir()

    loaded: List[EvaluationSample] = []
    missing: List[str] = []
    for sample in samples:
        source = str((sample.metadata or {}).get("source_document") or "")
        path = markdown_dir / f"{source}.md"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                sample.metadata["document_text"] = text
                sample.metadata["document_source_path"] = str(path)
                loaded.append(sample)
                continue
        missing.append(source)

    info = {
        "dataset": str(dataset_path),
        "markdown_dir": str(markdown_dir),
        "total_samples": len(samples),
        "samples_with_documents": len(loaded),
        "missing_documents": sorted(set(missing)),
    }
    if require_documents and not loaded:
        raise FileNotFoundError(
            "No FinanceBench Markdown documents found. Run "
            "`python -m evaluation.prepare_financebench` first."
        )
    return loaded, info


def _legalbench_mini(tests: Sequence[Dict[str, Any]], n: int = 194) -> List[Dict[str, Any]]:
    """Reproduce the LegalBench-RAG mini selection from its README.

    The reference implementation seeds the RNG with the first snippet path,
    takes ``random.random()`` as the sort key, and keeps the first ``n``.
    """
    def key(test: Dict[str, Any]) -> float:
        random.seed(test["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def load_legalbench_rag_samples(
    source: str = "cuad",
    corpus_root: Optional[Path] = None,
    benchmark_root: Optional[Path] = None,
    mini: bool = True,
    limit: int = 0,
) -> tuple[List[EvaluationSample], Dict[str, Any]]:
    """Load a LegalBench-RAG source without mixing it with another benchmark."""
    corpus_root = corpus_root or (LEGALBENCH_RAG_DIR / "corpus")
    benchmark_root = benchmark_root or (LEGALBENCH_RAG_DIR / "benchmarks")
    benchmark_path = benchmark_root / f"{source}.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"LegalBench-RAG benchmark not found: {benchmark_path}. "
            "Run `python -m evaluation.prepare_legalbench_rag --source "
            f"{source}` first."
        )

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tests = payload.get("tests") or []
    if mini:
        tests = _legalbench_mini(tests, n=194)
    if limit > 0:
        tests = tests[:limit]

    samples: List[EvaluationSample] = []
    missing_files: List[str] = []
    for index, test in enumerate(tests):
        snippets = test.get("snippets") or []
        if not snippets:
            continue
        file_path = snippets[0].get("file_path") or ""
        document_path = corpus_root / file_path
        if not document_path.exists():
            missing_files.append(file_path)
            continue
        document_text = document_path.read_text(encoding="utf-8")
        evidence = []
        for snippet in snippets:
            answer = snippet.get("answer")
            if isinstance(answer, str) and answer.strip():
                evidence.append(answer)
        samples.append(
            EvaluationSample(
                id=f"legalbench_rag_{source}_{index:04d}",
                query=str(test.get("query") or ""),
                ground_truth=evidence[0] if evidence else "",
                relevant_chunks=evidence,
                category=source,
                difficulty="hard",
                metadata={
                    "source_dataset": "legalbench_rag",
                    "source_group": source,
                    "source_document": Path(file_path).stem,
                    "document_text": document_text,
                    "file_path": file_path,
                    "licence": "per-source; see LegalBench-RAG LICENSE",
                },
            )
        )

    info = {
        "benchmark": f"legalbench_rag:{source}",
        "benchmark_path": str(benchmark_path),
        "corpus_root": str(corpus_root),
        "mini": mini,
        "total_tests": len(tests),
        "samples_with_documents": len(samples),
        "documents": len({s.metadata["file_path"] for s in samples}),
        "missing_files": sorted(set(missing_files))[:20],
    }
    return samples, info


def stable_group_id(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def markdown_with_detected_headings(text: str, max_heading_chars: int = 140) -> str:
    """Add Markdown heading markers to numbered/uppercase legal headings.

    CUAD and parts of LegalBench-RAG are plain text exports, not Markdown. This
    normalisation gives both Markdown-aware chunkers the same structural input.
    It does not invent content; it only prefixes existing heading-like lines.
    """
    from services.contextual_chunker import ContextualChunker

    chunker = ContextualChunker()
    output: List[str] = []
    for line in (text or "").splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or len(stripped) > max_heading_chars:
            output.append(line)
            continue
        if re.match(r"^#{1,6}\s+", stripped):
            output.append(line)
            continue
        match, level = chunker._match_heading(stripped)
        if match and level:
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"{indent}{'#' * min(6, max(1, level))} {stripped}\n")
        else:
            output.append(line)
    return "".join(output)
