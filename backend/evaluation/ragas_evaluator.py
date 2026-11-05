"""RAGAS-style evaluation for TenderSync.

These are deterministic, reference-based implementations of the metrics the
RAGAS project popularised: context recall, context precision, faithfulness,
answer relevance, answer similarity and NDCG@10. No LLM judge is required, so
CI is reproducible and free. An optional embedding service improves the
semantic components; the lexical fallbacks still work offline.

The previous module was named "RAGAS" but did not use ragas, and its
``_extract_compliance_score`` called ``match.group(2)`` on a pattern with one
capturing group (raising ``IndexError`` for every compliance sample). It also
never populated per-category results and never computed NDCG, despite the CI
threshold file requiring both.
"""

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "and",
    "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "it", "its", "their", "there", "which", "who", "whom", "what", "when",
    "where", "why", "how", "does", "do", "did", "has", "have", "had",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "shall", "not", "no", "any", "all", "each", "per", "into", "under",
}


class EvaluationMetric(Enum):
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    ANSWER_SIMILARITY = "answer_similarity"
    NDCG = "ndcg"
    COMPLIANCE_SCORE_ACCURACY = "compliance_score_accuracy"
    BUDGET_EXTRACTION_ACCURACY = "budget_extraction_accuracy"
    CITATION_VERIFICATION_RATE = "citation_verification_rate"


@dataclass
class EvaluationSample:
    id: str
    query: str
    ground_truth: str
    relevant_chunks: List[str] = field(default_factory=list)
    expected_compliance_score: Optional[float] = None
    expected_budget: Optional[str] = None
    expected_deadline: Optional[str] = None
    expected_tender_id: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    sample_id: str
    query: str
    generated_answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_time_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    category: str = "general"
    difficulty: str = "medium"
    ground_truth: str = ""
    source_dataset: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "generated_answer": self.generated_answer,
            "metrics": self.metrics,
            "evaluation_time_ms": self.evaluation_time_ms,
            "timestamp": self.timestamp,
            "category": self.category,
            "difficulty": self.difficulty,
            "source_dataset": self.source_dataset,
            "error": self.error,
            "num_retrieved_chunks": len(self.retrieved_chunks),
        }


@dataclass
class AggregateEvaluationResult:
    total_samples: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_category: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    per_difficulty: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    ablation_results: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    errors: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "errors": self.errors,
            "metrics": self.metrics,
            "per_category": self.per_category,
            "per_difficulty": self.per_difficulty,
            "ablation_results": self.ablation_results,
            "samples": self.samples,
            "timestamp": self.timestamp,
        }


# ----------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------
def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*", (text or "").lower())


def _content_tokens(text: str) -> set:
    return {t for t in _tokens(text) if t not in STOPWORDS and len(t) > 1}


def _numbers(text: str) -> List[str]:
    """Normalised numeric literals: 2,500,000 / 2500000 / $1577.00 -> comparable."""
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")
    normalised = []
    for value in found:
        value = value.replace(",", "")
        try:
            normalised.append(f"{float(value):.10g}")
        except ValueError:
            normalised.append(value)
    return normalised


def _coverage(needle: str, haystack: str) -> float:
    """Fraction of needle's content tokens present in haystack."""
    needle_tokens = _content_tokens(needle)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & _content_tokens(haystack)) / len(needle_tokens)


def _numeric_supported(text: str, context: str) -> bool:
    numbers = _numbers(text)
    if not numbers:
        return True
    context_numbers = set(_numbers(context))
    return all(number in context_numbers for number in numbers)


class RAGASEvaluator:
    """Deterministic reference metrics for a RAG pipeline."""

    def __init__(
        self,
        llm_client=None,  # reserved for optional LLM judging
        embedding_service=None,
        citation_verifier=None,
        context_match_threshold: float = 0.80,
    ):
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.citation_verifier = citation_verifier
        self.context_match_threshold = context_match_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def evaluate_sample(
        self,
        sample: EvaluationSample,
        generated_answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: Optional[List[str]] = None,
    ) -> EvaluationResult:
        start = time.time()
        ground_truth_chunks = ground_truth_chunks or sample.relevant_chunks

        metrics: Dict[str, float] = {
            "context_recall": self._compute_context_recall(retrieved_chunks, ground_truth_chunks),
            "context_precision": self._compute_context_precision(retrieved_chunks, ground_truth_chunks),
            "ndcg": self._compute_ndcg(retrieved_chunks, ground_truth_chunks),
            "faithfulness": self._compute_faithfulness(generated_answer, retrieved_chunks),
            "answer_relevance": self._compute_answer_relevance(sample.query, generated_answer),
        }

        if sample.ground_truth:
            metrics["answer_similarity"] = self._compute_answer_similarity(
                generated_answer, sample.ground_truth
            )

        if sample.expected_compliance_score is not None:
            metrics["compliance_score_accuracy"] = self._compute_compliance_accuracy(
                generated_answer, sample.expected_compliance_score
            )
        if sample.expected_budget is not None:
            metrics["budget_extraction_accuracy"] = self._compute_budget_accuracy(
                generated_answer, sample.expected_budget
            )

        if self.citation_verifier and generated_answer and retrieved_chunks:
            try:
                verification = self.citation_verifier.verify_response(
                    generated_answer, retrieved_chunks, response_type="evaluation"
                )
                metrics["citation_verification_rate"] = float(
                    verification.get("verification_score", 0.0)
                )
            except Exception:
                pass

        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            ground_truth=sample.ground_truth,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
            evaluation_time_ms=(time.time() - start) * 1000,
            category=sample.category,
            difficulty=sample.difficulty,
            source_dataset=(sample.metadata or {}).get("source_dataset"),
        )

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------
    def _chunks_are_equivalent(self, ground_truth: str, retrieved: str) -> bool:
        if not ground_truth or not retrieved:
            return False
        gt_norm = " ".join(_tokens(ground_truth))
        ret_norm = " ".join(_tokens(retrieved))
        if gt_norm and gt_norm in ret_norm:
            return True
        if ret_norm and ret_norm in gt_norm:
            return True

        coverage = max(
            _coverage(ground_truth, retrieved),
            _coverage(retrieved, ground_truth) if len(_tokens(retrieved)) < 60 else 0.0,
        )
        if coverage >= self.context_match_threshold and _numeric_supported(ground_truth, retrieved):
            return True

        # Short numeric answers (e.g. "$1577.00") need only numeric support.
        gt_content = _content_tokens(ground_truth)
        if len(gt_content) <= 3 and _numbers(ground_truth):
            return _numeric_supported(ground_truth, retrieved)
        return False

    def _chunk_matches_any(self, ground_truth: str, retrieved_texts: Sequence[str]) -> bool:
        return any(self._chunks_are_equivalent(ground_truth, text) for text in retrieved_texts)

    def _compute_context_recall(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: List[str],
    ) -> float:
        if not ground_truth_chunks:
            return 1.0
        if not retrieved_chunks:
            return 0.0
        retrieved_texts = [chunk.get("content", "") for chunk in retrieved_chunks]
        matched = sum(
            1 for gt in ground_truth_chunks if self._chunk_matches_any(gt, retrieved_texts)
        )
        return matched / len(ground_truth_chunks)

    def _compute_context_precision(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: List[str],
        k: int = 10,
        k_values: Optional[Sequence[int]] = None,
    ) -> float:
        """RAGAS context precision: mean precision@rank of relevant items.

        ``k_values`` is accepted for callers written against the older
        multi-cutoff API; the returned value uses the largest requested cutoff.
        """
        if k_values:
            k = max(k_values)
        if not ground_truth_chunks or not retrieved_chunks:
            return 0.0

        cutoff = min(k, len(retrieved_chunks))
        flags = [
            self._chunk_matches_any(
                chunk.get("content", ""),
                ground_truth_chunks,
            )
            for chunk in retrieved_chunks[:cutoff]
        ]
        if not any(flags):
            return 0.0

        cumulative = 0
        precision_sum = 0.0
        for rank, relevant in enumerate(flags, start=1):
            if relevant:
                cumulative += 1
                precision_sum += cumulative / rank
        return precision_sum / max(1, min(len(ground_truth_chunks), cutoff))

    def _compute_ndcg(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: List[str],
        k: int = 10,
    ) -> float:
        if not ground_truth_chunks or not retrieved_chunks:
            return 0.0

        relevances = [
            1 if self._chunk_matches_any(chunk.get("content", ""), ground_truth_chunks) else 0
            for chunk in retrieved_chunks[:k]
        ]
        dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
        ideal_hits = min(len(ground_truth_chunks), k)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg else 0.0

    # ------------------------------------------------------------------
    # Generation metrics
    # ------------------------------------------------------------------
    def _compute_faithfulness(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> float:
        """Sentence-level support: numbers must match, content must be covered."""
        if not answer or answer.strip().startswith("Not found in the provided document"):
            return 0.0
        if not retrieved_chunks:
            return 0.0

        chunk_texts = [chunk.get("content", "") for chunk in retrieved_chunks if chunk.get("content")]
        if not chunk_texts:
            return 0.0
        combined = "\n".join(chunk_texts)

        sentences = self._extract_sentences(answer)
        if not sentences:
            return 0.0

        scored = []
        for sentence in sentences:
            # Remove citation markers before matching.
            clean = re.sub(r"\[\d+\]", "", sentence).strip()
            if not clean:
                continue
            if not _numeric_supported(clean, combined):
                scored.append(0.0)
                continue
            best = max(_coverage(clean, chunk) for chunk in chunk_texts)
            if best >= 0.85:
                scored.append(1.0)
            elif best >= 0.60:
                scored.append(0.6)
            elif best >= 0.40:
                scored.append(0.3)
            else:
                scored.append(0.0)

        return float(np.mean(scored)) if scored else 0.0

    def _compute_answer_relevance(self, query: str, answer: str) -> float:
        if not answer or answer.strip().startswith("Not found in the provided document"):
            return 0.0

        lexical = _coverage(query, answer)

        if self.embedding_service:
            try:
                query_embedding = self.embedding_service.embed_query(query)
                answer_embedding = self.embedding_service.embed_single(answer)
                cosine = max(0.0, self._cosine_similarity(query_embedding, answer_embedding))
                return 0.5 * lexical + 0.5 * cosine
            except Exception:
                pass

        return lexical

    def _compute_answer_similarity(self, answer: str, ground_truth: str) -> float:
        if not answer or not ground_truth:
            return 0.0
        answer_tokens = set(_tokens(answer))
        truth_tokens = set(_tokens(ground_truth))
        if not answer_tokens or not truth_tokens:
            return 0.0

        overlap = len(answer_tokens & truth_tokens)
        precision = overlap / len(answer_tokens)
        recall = overlap / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        numeric_ok = _numeric_supported(ground_truth, answer) and _numeric_supported(answer, ground_truth)
        if not numeric_ok:
            f1 *= 0.5
        return float(f1)

    # ------------------------------------------------------------------
    # Extraction metrics
    # ------------------------------------------------------------------
    def _compute_compliance_accuracy(
        self,
        answer: str,
        expected_score: float,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        extracted = self._extract_compliance_score(answer)
        if extracted is None:
            return 0.0
        error = abs(extracted - float(expected_score))
        return max(0.0, 1.0 - error / 100.0)

    def _compute_budget_accuracy(self, answer: str, expected_budget: str) -> float:
        extracted = self._extract_budget(answer)
        expected = self._normalize_budget(expected_budget)
        actual = self._normalize_budget(extracted) if extracted else 0.0
        if not expected or not actual:
            return 0.0
        relative_error = abs(actual - expected) / expected
        return max(0.0, 1.0 - relative_error)

    def _extract_compliance_score(self, text: str) -> Optional[float]:
        if not text:
            return None
        patterns = [
            r"compliance\s*(?:score|level|rate)?\s*(?:of|is|:)?\s*(?:approximately\s*)?(\d+(?:\.\d+)?)\s*(?:%|percent)",
            r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:compliant|compliance|conformance)",
            r"(?:score|rating)\s*(?:of|is|:)\s*(\d+(?:\.\d+)?)\s*(?:/\s*100|out\s+of\s+100)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_budget(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(
            r"(?:budget|value|cost|funding|contract\s+value)\s*(?:of|is|:)?\s*"
            r"((?:AUD|USD|GBP|EUR|NZD)?\s*[\$£€₹]?\s*\d[\d,.]*\s*(?:million|billion|thousand|m|bn|k)?)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        match = re.search(
            r"((?:AUD|USD|GBP|EUR|NZD)?\s*[\$£€₹]\s*\d[\d,.]*\s*(?:million|billion|thousand|m|bn|k)?)",
            text,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    def _normalize_budget(self, budget: Optional[str]) -> float:
        if not budget:
            return 0.0

        text = budget.lower().replace(",", "").replace(" ", "")
        text = re.sub(r"(aud|usd|gbp|eur|nzd)", "", text)
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return 0.0

        value = float(match.group(1))
        if "billion" in text or re.search(r"\d+b(n)?$", text):
            value *= 1_000_000_000
        elif "million" in text or re.search(r"\d+m$", text):
            value *= 1_000_000
        elif "thousand" in text or re.search(r"\d+k$", text):
            value *= 1_000
        return value

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]

    @staticmethod
    def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        v1 = np.asarray(vec1, dtype=float)
        v2 = np.asarray(vec2, dtype=float)
        if v1.size == 0 or v2.size == 0 or v1.shape != v2.shape:
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


class RAGASPipelineEvaluator:
    """Runs a RAG pipeline over a dataset and aggregates the metrics."""

    def __init__(self, evaluator: RAGASEvaluator, rag_pipeline=None):
        self.evaluator = evaluator
        self.rag_pipeline = rag_pipeline

    def evaluate_dataset(
        self,
        dataset: List[EvaluationSample],
        tenant_id: str,
        run_rag_pipeline: bool = True,
        verbose: bool = False,
    ) -> AggregateEvaluationResult:
        if run_rag_pipeline and not self.rag_pipeline:
            raise ValueError("rag_pipeline is required when run_rag_pipeline=True")

        results: List[EvaluationResult] = []
        delay = float(os.getenv("EVAL_SAMPLE_DELAY_SECONDS", "0"))
        total = len(dataset)

        for index, sample in enumerate(dataset, start=1):
            if verbose:
                print(f"[{index}/{total}] {sample.id}")

            if not run_rag_pipeline:
                results.append(
                    self.evaluator.evaluate_sample(sample, "", [], sample.relevant_chunks)
                )
                continue

            start = time.time()
            error = None
            answer = ""
            retrieved_chunks: List[Dict[str, Any]] = []
            try:
                pipeline_result = self.rag_pipeline.query(
                    sample.query,
                    tenant_id,
                    document_id=sample.document_id,
                )
                answer = pipeline_result.get("answer", "")
                retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
            except Exception as exc:
                error = str(exc)
                if verbose:
                    print(f"    ERROR: {error}")

            result = self.evaluator.evaluate_sample(
                sample, answer, retrieved_chunks, sample.relevant_chunks
            )
            result.error = error
            result.evaluation_time_ms = (time.time() - start) * 1000
            results.append(result)

            if delay > 0 and index < total:
                time.sleep(delay)

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[EvaluationResult]) -> AggregateEvaluationResult:
        if not results:
            return AggregateEvaluationResult(total_samples=0)

        metric_names = sorted({name for result in results for name in result.metrics})

        def summarise(subset: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
            summary: Dict[str, Dict[str, float]] = {}
            for metric in metric_names:
                values = [r.metrics[metric] for r in subset if metric in r.metrics]
                if not values:
                    continue
                array = np.asarray(values, dtype=float)
                summary[metric] = {
                    "mean": float(array.mean()),
                    "std": float(array.std()),
                    "min": float(array.min()),
                    "max": float(array.max()),
                    "n": int(array.size),
                }
            return summary

        per_category = {
            category: summarise([r for r in results if r.category == category])
            for category in sorted({r.category for r in results})
        }
        per_difficulty = {
            difficulty: summarise([r for r in results if r.difficulty == difficulty])
            for difficulty in sorted({r.difficulty for r in results})
        }

        return AggregateEvaluationResult(
            total_samples=len(results),
            metrics=summarise(results),
            per_category=per_category,
            per_difficulty=per_difficulty,
            samples=[result.to_dict() for result in results],
            errors=sum(1 for result in results if result.error),
        )

    def export_results(self, result: AggregateEvaluationResult, output_path: str, format: str = "json") -> None:
        if format != "json":
            raise ValueError("Only JSON export is implemented")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------
def load_golden_dataset(path: str) -> List[EvaluationSample]:
    """Load a golden dataset from JSON.

    Accepts either a top-level list or an object with ``samples``/``data``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("samples", payload.get("data", [payload]))
    if not isinstance(payload, list):
        raise ValueError("Expected a list or an object with a 'samples' key")

    samples: List[EvaluationSample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        relevant = item.get("relevant_chunks", item.get("contexts", item.get("evidence", [])))
        if isinstance(relevant, str):
            relevant = [relevant]
        relevant = [chunk for chunk in (relevant or []) if isinstance(chunk, str)]

        samples.append(
            EvaluationSample(
                id=str(item.get("id", item.get("_id", f"sample_{index:04d}"))),
                query=item.get("query", item.get("question", "")),
                ground_truth=item.get(
                    "ground_truth", item.get("answer", item.get("expected_answer", ""))
                ),
                relevant_chunks=relevant,
                expected_compliance_score=item.get("expected_compliance_score"),
                expected_budget=item.get("expected_budget"),
                expected_deadline=item.get("expected_deadline"),
                expected_tender_id=item.get("expected_tender_id"),
                category=item.get("category", "general"),
                difficulty=item.get("difficulty", "medium"),
                document_id=item.get("document_id"),
                metadata=item.get("metadata", {}),
            )
        )
    return samples


def create_sample_subset(
    dataset: List[EvaluationSample],
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[EvaluationSample]:
    filtered = dataset
    if category:
        filtered = [s for s in filtered if s.category == category]
    if difficulty:
        filtered = [s for s in filtered if s.difficulty == difficulty]
    if n_samples and len(filtered) > n_samples:
        random.Random(seed).shuffle(filtered := list(filtered))
        filtered = filtered[:n_samples]
    return filtered
