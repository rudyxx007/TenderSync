"""
RAGAS Evaluation Framework for TenderSync Enterprise RAG Pipeline.

Automated evaluation using RAGAS metrics:
- Context Recall: Measures how much relevant context was retrieved
- Context Precision: Measures ranking quality of retrieved context
- Faithfulness: Measures hallucination/factual accuracy
- Answer Relevance: Measures how relevant the answer is to the question

Plus custom TenderSync metrics:
- Compliance Score Accuracy
- Citation Verification Rate
- Numeric Extraction Accuracy

References:
- https://docs.ragas.io/
- https://www.trulens.org/
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np


class EvaluationMetric(Enum):
    """Supported evaluation metrics."""
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RELEVANCE = "context_relevance"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    ANSWER_SIMILARITY = "answer_similarity"
    COMPLIANCE_SCORE_ACCURACY = "compliance_score_accuracy"
    CITATION_VERIFICATION_RATE = "citation_verification_rate"
    NUMERIC_EXTRACTION_ACCURACY = "numeric_extraction_accuracy"
    NDCG = "ndcg"


@dataclass
class EvaluationSample:
    """A single evaluation sample (query + expected result)."""
    id: str
    query: str
    ground_truth: str  # Expected answer
    relevant_chunks: List[str]  # Expected relevant context
    expected_compliance_score: Optional[float] = None
    expected_budget: Optional[str] = None
    expected_deadline: Optional[str] = None
    expected_tender_id: Optional[str] = None
    category: str = "general"  # 'compliance', 'budget', 'deadline', 'general'
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    sample_id: str
    query: str
    generated_answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_time_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "generated_answer": self.generated_answer,
            "metrics": self.metrics,
            "evaluation_time_ms": self.evaluation_time_ms,
            "timestamp": self.timestamp,
            "num_retrieved_chunks": len(self.retrieved_chunks),
        }


@dataclass
class AggregateEvaluationResult:
    """Aggregated results across all samples."""
    total_samples: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ablation_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "metrics": self.metrics,
            "per_category": self.per_category,
            "per_difficulty": self.per_difficulty,
            "ablation_results": self.ablation_results,
            "timestamp": self.timestamp,
        }


class RAGASEvaluator:
    """
    Evaluator implementing RAGAS metrics for RAG pipeline evaluation.
    
    Uses both LLM-based evaluation and deterministic metrics where possible.
    For LLM-based metrics, requires an LLM client (e.g., Groq).
    
    References:
    - RAGAS documentation: https://docs.ragas.io/
    - Metrics explained:
      - Context Recall: % of relevant context retrieved
      - Context Precision: Quality of ranking (relevant items at top)
      - Faithfulness: How factually accurate the answer is
      - Answer Relevance: How relevant the answer is to the query
    """
    
    def __init__(
        self,
        llm_client=None,  # Optional LLM client for evaluation
        embedding_service=None,  # For semantic similarity metrics
        citation_verifier=None,  # For citation verification
    ):
        """
        Initialize evaluator.
        
        Args:
            llm_client: Optional LLM client for LLM-based metrics
            embedding_service: Optional embedding service for similarity
            citation_verifier: Optional citation verifier for compliance checks
        """
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.citation_verifier = citation_verifier
    
    def evaluate_sample(
        self,
        sample: EvaluationSample,
        generated_answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            sample: The evaluation sample
            generated_answer: The RAG pipeline's answer
            retrieved_chunks: Chunks retrieved by the pipeline
            ground_truth_chunks: Expected relevant chunks
            
        Returns:
            EvaluationResult with all metrics
        """
        start_time = time.time()
        
        ground_truth_chunks = ground_truth_chunks or sample.relevant_chunks
        
        metrics = {}
        
        # Context Recall: How many relevant chunks were retrieved?
        metrics["context_recall"] = self._compute_context_recall(
            retrieved_chunks, ground_truth_chunks
        )
        
        # Context Precision: Are relevant chunks ranked higher?
        metrics["context_precision"] = self._compute_context_precision(
            retrieved_chunks, ground_truth_chunks
        )
        
        # Faithfulness: Is the answer entailed by retrieved context?
        metrics["faithfulness"] = self._compute_faithfulness(
            generated_answer, retrieved_chunks
        )
        
        # Answer Relevance: Is the answer relevant to the query?
        metrics["answer_relevance"] = self._compute_answer_relevance(
            sample.query, generated_answer
        )
        
        # Custom metrics
        if sample.expected_compliance_score is not None:
            metrics["compliance_score_accuracy"] = self._compute_compliance_accuracy(
                generated_answer, sample.expected_compliance_score, retrieved_chunks
            )
        
        if sample.expected_budget is not None:
            metrics["budget_extraction_accuracy"] = self._compute_budget_accuracy(
                generated_answer, sample.expected_budget
            )
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return EvaluationResult(
            sample_id=sample.id,
            query=sample.query,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
            evaluation_time_ms=evaluation_time,
        )
    
    def _compute_context_recall(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: List[str],
    ) -> float:
        """
        Compute context recall: what fraction of relevant context was retrieved?
        
        Context Recall = |{retrieved chunks} ∩ {relevant chunks}| / |{relevant chunks}|
        
        Uses semantic similarity for fuzzy matching.
        """
        if not ground_truth_chunks:
            return 1.0  # No ground truth, assume perfect
        
        if not retrieved_chunks:
            return 0.0
        
        retrieved_texts = [chunk.get("content", "") for chunk in retrieved_chunks]
        
        matched = 0
        for gt_chunk in ground_truth_chunks:
            if self._chunk_matches_any(gt_chunk, retrieved_texts):
                matched += 1
        
        return matched / len(ground_truth_chunks)
    
    def _compute_context_precision(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_chunks: List[str],
    ) -> float:
        """
        Compute context precision: are relevant chunks ranked at the top?
        
          Precision@k = (number of relevant chunks in top k) / k
        
        We compute this at multiple k values and average.
        """
        if not ground_truth_chunks or not retrieved_chunks:
            return 0.0
        
        retrieved_texts = [chunk.get("content", "") for chunk in retrieved_chunks]
        
        # Compute precision at different cutoffs
        k_values = [1, 2, 3, 5, 10]
        precision_scores = []
        
        for k in k_values:
            if k > len(retrieved_chunks):
                k = len(retrieved_chunks)
            
            top_k_texts = retrieved_texts[:k]
            relevant_count = sum(
                1 for gt in ground_truth_chunks
                if self._chunk_matches_any(gt, top_k_texts)
            )
            
            precision_scores.append(relevant_count / k)
        
        # Average precision at different k
        return np.mean(precision_scores) if precision_scores else 0.0
    
    def _compute_faithfulness(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Compute faithfulness: how factually accurate is the answer?
        
        This measures whether the answer's claims are supported by
        the retrieved context (no hallucination).
        
        Uses the citation verifier for NLI-based checking.
        """
        if not answer or not retrieved_chunks:
            return 0.0
        
        if self.citation_verifier:
            verification = self.citation_verifier.verify_response(
                answer, retrieved_chunks, None, "general"
            )
            return verification["verification_score"]
        
        # Fallback: simple heuristic
        # Check if answer contains only information from chunks
        chunk_text = " ".join([c.get("content", "") for c in retrieved_chunks]).lower()
        
        # Extract sentences from answer
        sentences = self._extract_sentences(answer)
        
        if not sentences:
            return 0.0
        
        supported_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if sentence_lower in chunk_text:
                supported_count += 1
            else:
                # Check if key facts from sentence are in context
                # Extract numbers and key entities
                import re
                numbers_in_sentence = re.findall(r'\d+[\d,.]*\s*(?:million|billion|thousand)?', sentence_lower)
                numbers_in_chunk = re.findall(r'\d+[\d,.]*\s*(?:million|billion|thousand)?', chunk_text)
                
                # Check if sentence numbers match chunk numbers
                sentence_nums = set(n.strip() for n in numbers_in_sentence)
                chunk_nums = set(n.strip() for n in numbers_in_chunk)
                
                if sentence_nums and not sentence_nums.issubset(chunk_nums):
                    # Numbers don't match - not supported
                    continue
                
                # Word overlap check
                if self._is_supported_by_context(sentence, chunk_text):
                    supported_count += 1
        
        return supported_count / len(sentences) if sentences else 0.0
    
    def _compute_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Compute answer relevance: how relevant is the answer to the query?
        
        This can use:
        1. LLM-based evaluation (best)
        2. Semantic similarity between query and answer
        """
        if not answer:
            return 0.0
        
        if self.llm_client:
            # Use LLM to evaluate relevance
            return self._llm_evaluate_relevance(query, answer)
        
        # Fallback: semantic similarity
        if self.embedding_service:
            try:
                query_emb = self.embedding_service.embed_query(query)
                answer_emb = self.embedding_service.embed_single(answer)
                similarity = self._cosine_similarity(query_emb, answer_emb)
                return max(0, similarity)  # Ensure non-negative
            except Exception:
                pass
        
        # Simple keyword overlap as last resort
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & answer_words) / len(query_words)
        return overlap
    
    def _compute_compliance_accuracy(
        self,
        answer: str,
        expected_score: float,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> float:
        """Compute accuracy of compliance score extraction."""
        extracted_score = self._extract_compliance_score(answer)
        
        if extracted_score is None:
            return 0.0
        
        # Compute accuracy as inverse of absolute error
        error = abs(extracted_score - expected_score)
        accuracy = max(0, 1 - (error / 100))
        
        return accuracy
    
    def _compute_budget_accuracy(
        self,
        answer: str,
        expected_budget: str,
    ) -> float:
        """Compute accuracy of budget extraction."""
        extracted_budget = self._extract_budget(answer)
        expected_normalized = self._normalize_budget(expected_budget)
        extracted_normalized = self._normalize_budget(extracted_budget)
        
        if extracted_normalized is None or expected_normalized == 0:
            return 0.0
        
        error = abs(extracted_normalized - expected_normalized) / expected_normalized
        accuracy = max(0, 1 - error)
        
        return accuracy
    
    def _chunk_matches_any(
        self,
        ground_truth_chunk: str,
        retrieved_texts: List[str],
    ) -> bool:
        """Check if a ground truth chunk matches any retrieved chunk."""
        for retrieved in retrieved_texts:
            if self._chunks_are_equivalent(ground_truth_chunk, retrieved):
                return True
        return False
    
    def _chunks_are_equivalent(self, chunk1: str, chunk2: str) -> bool:
        """Check if two chunks are semantically equivalent."""
        # Exact match
        if chunk1.strip() == chunk2.strip():
            return True
        
        # Semantic similarity using embeddings
        if self.embedding_service:
            try:
                emb1 = self.embedding_service.embed_single(chunk1)
                emb2 = self.embedding_service.embed_single(chunk2)
                sim = self._cosine_similarity(emb1, emb2)
                return sim > 0.85  # High threshold for equivalence
            except Exception:
                pass
        
        # Fallback: high word overlap
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.7
    
    def _is_supported_by_context(self, sentence: str, context: str) -> bool:
        """Check if a sentence is supported by the context."""
        # Check for key entities/facts from sentence in context
        sentence_words = set(sentence.lower().split())
        context_words = set(context.lower().split())
        
        # At least 50% of important words should be in context
        important_words = sentence_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        
        if not important_words:
            return True
        
        overlap = len(important_words & context_words) / len(important_words)
        return overlap > 0.5
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _extract_compliance_score(self, text: str) -> Optional[float]:
        """Extract compliance score from text."""
        import re
        
        # Pattern: "X%" or "X percent" near "compliance"
        match = re.search(r'(?:compliance\s*(?:score|level|rate)?\s*[:\s]*)'
                         r'(?:approximately\s*)?'
                         r'(?:[\"\']?)?(\d+(?:\.\d+)?)(?:[\"\']?)?\s*%',
                         text, re.IGNORECASE)
        if match:
            return float(match.group(2))
        
        # Pattern: "X% compliance"
        match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:compliance|conformance)',
                         text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        return None
    
    def _extract_budget(self, text: str) -> Optional[str]:
        """Extract budget value from text."""
        import re
        
        # Pattern: currency amounts
        match = re.search(r'(?:budget|value|cost|value|funding)\s*[:\s]*'
                         r'[\$£€₹]?\s*([\d,.]+(?:\s*(?:million|billion|thousand))?',
                         text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Just find currency amounts
        match = re.search(r'[\$£€₹]\s*([\d,.]+(?:\s*(?:million|billion|thousand))?',
                         text)
        if match:
            return match.group(1)
        
        return None
    
    def _normalize_budget(self, budget_str: str) -> float:
        """Normalize budget string to numeric value."""
        if not budget_str:
            return 0.0
        
        import re
        text = budget_str.lower().replace(',', '')
        
        multipliers = {
            'million': 1_000_000,
            'billion': 1_000_000_000,
            'thousand': 1_000,
            'k': 1_000,
        }
        
        for word, mult in multipliers.items():
            if word in text:
                match = re.search(r'(\d+(?:\.\d+)?)', text)
                if match:
                    return float(match.group(1)) * mult
        
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        
        return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _llm_evaluate_relevance(self, query: str, answer: str) -> float:
        """Use LLM to evaluate answer relevance."""
        if not self.llm_client:
            return 0.0
        
        prompt = f"""
        Rate the relevance of the following answer to the query on a scale of 0-1.
        
        Query: {query}
        
        Answer: {answer}
        
        Consider:
        - Does the answer address the query?
        - Is the answer focused on the query?
        - Does it provide useful information for the query?
        
        Return a single number between 0 and 1.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            
            content = response.choices[0].message.content.strip()
            score = float(content)
            return max(0, min(1, score))
        except Exception:
            return 0.0


class RAGASPipelineEvaluator:
    """
    Full pipeline evaluator that runs evaluation on a dataset.
    
    Orchestrates the evaluation process and produces aggregate results.
    """
    
    def __init__(
        self,
        evaluator: RAGASEvaluator,
        rag_pipeline=None,  # The RAG pipeline to evaluate
    ):
        self.evaluator = evaluator
        self.rag_pipeline = rag_pipeline
    
    def evaluate_dataset(
        self,
        dataset: List[EvaluationSample],
        tenant_id: str,
        run_rag_pipeline: bool = True,
        verbose: bool = False,
    ) -> AggregateEvaluationResult:
        """
        Evaluate the RAG pipeline on a dataset.
        
        Args:
            dataset: List of evaluation samples
            tenant_id: Tenant ID for retrieval context
            run_rag_pipeline: Whether to run the actual RAG pipeline
            verbose: Print progress
            
        Returns:
            AggregateEvaluationResult with all metrics
        """
        results = []
        
        for i, sample in enumerate(dataset):
            if verbose:
                print(f"Evaluating sample {i+1}/{len(dataset)}: {sample.id}")
            
            # Run RAG pipeline to get answer and retrieved chunks
            if run_rag_pipeline and self.rag_pipeline:
                start_time = time.time()
                pipeline_result = self.rag_pipeline.query(
                    sample.query, tenant_id
                )
                retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
                generated_answer = pipeline_result.get("answer", "")
            else:
                retrieved_chunks = []
                generated_answer = ""
            
            # Evaluate
            eval_result = self.evaluator.evaluate_sample(
                sample,
                generated_answer,
                retrieved_chunks,
                sample.relevant_chunks,
            )
            results.append(eval_result)
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> AggregateEvaluationResult:
        """Aggregate individual results into summary metrics."""
        if not results:
            return AggregateEvaluationResult(total_samples=0)
        
        # Compute average metrics
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        
        for result in results:
            for metric_name, value in result.metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0) + value
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
        
        avg_metrics = {
            name: {"mean": metric_sums[name] / metric_counts[name], "count": metric_counts[name]}
            for name in metric_sums
        }
        
        # Per-category breakdown
        per_category: Dict[str, Dict[str, float]] = {}
        per_difficulty: Dict[str, Dict[str, float]] = {}
        
        # Would need to track categories in results - simplified here
        # In practice, would group by sample.category and sample.difficulty
        
        return AggregateEvaluationResult(
            total_samples=len(results),
            metrics=avg_metrics,
            per_category=per_category,
            per_difficulty=per_difficulty,
        )
    
    def export_results(
        self,
        result: AggregateEvaluationResult,
        output_path: str,
        format: str = "json",
    ):
        """Export evaluation results to file."""
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        elif format == "csv":
            # Would implement CSV export
            pass


def load_golden_dataset(path: str) -> List[EvaluationSample]:
    """Load golden dataset from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        samples.append(EvaluationSample(
            id=item["id"],
            query=item["query"],
            ground_truth=item["ground_truth"],
            relevant_chunks=item.get("relevant_chunks", []),
            expected_compliance_score=item.get("expected_compliance_score"),
            expected_budget=item.get("expected_budget"),
            expected_deadline=item.get("expected_deadline"),
            expected_tender_id=item.get("expected_tender_id"),
            category=item.get("category", "general"),
            difficulty=item.get("difficulty", "medium"),
            metadata=item.get("metadata", {}),
        ))
    
    return samples


def create_sample_subset(
    dataset: List[EvaluationSample],
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[EvaluationSample]:
    """Create a subset of the dataset for ablation studies."""
    filtered = dataset
    
    if category:
        filtered = [s for s in filtered if s.category == category]
    
    if difficulty:
        filtered = [s for s in filtered if s.difficulty == difficulty]
    
    if n_samples and len(filtered) > n_samples:
        random.seed(seed)
        filtered = random.sample(filtered, n_samples)
    
    return filtered
