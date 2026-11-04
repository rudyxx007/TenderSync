"""
Tests for the RAG evaluation framework.

Covers:
- RAGAS evaluator metrics computation
- Golden dataset loading
- Pipeline wrapper functionality
- Threshold checking
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

# Ensure backend is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    EvaluationSample,
    EvaluationResult,
    load_golden_dataset,
    create_sample_subset,
)
from evaluation.check_thresholds import check_thresholds, load_thresholds


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""
    
    def test_sample_creation(self):
        sample = EvaluationSample(
            id="test_001",
            query="What is the contract value?",
            ground_truth="The contract value is AUD 2,500,000.",
            relevant_chunks=["The total contract value is AUD 2,500,000."],
            category="budget",
            difficulty="easy",
        )
        
        assert sample.id == "test_001"
        assert sample.query == "What is the contract value?"
        assert sample.category == "budget"
        assert sample.difficulty == "easy"
    
    def test_sample_with_optional_fields(self):
        sample = EvaluationSample(
            id="test_002",
            query="What certifications are required?",
            ground_truth="ISO 9001 and ISO 27001 required.",
            relevant_chunks=["Must have ISO 9001:2015 and ISO 27001:2013."],
            expected_compliance_score=100.0,
            expected_budget="2500000",
            metadata={"source": "financebench"},
        )
        
        assert sample.expected_compliance_score == 100.0
        assert sample.expected_budget == "2500000"
        assert sample.metadata["source"] == "financebench"


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator metrics."""
    
    @pytest.fixture
    def evaluator(self):
        return RAGASEvaluator()
    
    @pytest.fixture
    def sample(self):
        return EvaluationSample(
            id="test_001",
            query="What is the contract value?",
            ground_truth="The contract is worth AUD 2.5M.",
            relevant_chunks=["The total contract value is AUD 2,500,000."],
        )
    
    def test_context_recall_perfect_match(self, evaluator, sample):
        """Context recall should be 1.0 when all relevant chunks retrieved."""
        retrieved = [
            {"content": "The total contract value is AUD 2,500,000."},
            {"content": "Payment is due within 30 days."},
        ]
        
        result = evaluator._compute_context_recall(retrieved, sample.relevant_chunks)
        assert result == 1.0
    
    def test_context_recall_partial_match(self, evaluator, sample):
        """Context recall should be partial when some chunks missing."""
        retrieved = [
            {"content": "Payment is due within 30 days."},
        ]
        
        result = evaluator._compute_context_recall(retrieved, sample.relevant_chunks)
        assert result == 0.0
    
    def test_context_recall_no_retrieval(self, evaluator, sample):
        """Context recall should be 0.0 when nothing retrieved."""
        retrieved = []
        
        result = evaluator._compute_context_recall(retrieved, sample.relevant_chunks)
        assert result == 0.0
    
    def test_context_recall_no_ground_truth(self, evaluator):
        """Context recall should be 1.0 when no ground truth."""
        retrieved = [{"content": "Some content"}]
        
        result = evaluator._compute_context_recall(retrieved, [])
        assert result == 1.0
    
    def test_context_precision_ranking(self, evaluator):
        """Context precision should reward relevant chunks at top."""
        retrieved = [
            {"content": "The total contract value is AUD 2,500,000."},  # relevant
            {"content": "Payment is due within 30 days."},  # not relevant
            {"content": "The supplier must have ISO 9001 certification."},  # not relevant
        ]
        ground_truth = ["The total contract value is AUD 2,500,000."]
        
        result = evaluator._compute_context_precision(retrieved, ground_truth, k_values=[1, 2, 3])
        # Should be > 0 because relevant chunk is first
        assert result > 0
    
    def test_faithfulness_high(self, evaluator):
        """Faithfulness should be high when answer matches context."""
        answer = "The contract value is AUD 2,500,000."
        chunks = [
            {"content": "The total contract value is AUD 2,500,000."},
        ]
        
        result = evaluator._compute_faithfulness(answer, chunks)
        assert result > 0.5  # Should detect high overlap
    
    def test_faithfulness_low(self, evaluator):
        """Faithfulness should be low when answer contradicts context."""
        answer = "The contract value is AUD 5,000,000."
        chunks = [
            {"content": "The total contract value is AUD 2,500,000."},
        ]
        
        result = evaluator._compute_faithfulness(answer, chunks)
        assert result < 0.5  # Should detect discrepancy
    
    def test_answer_relevance(self, evaluator):
        """Answer relevance should measure query-answer overlap."""
        query = "What is the contract value?"
        answer = "The contract is worth AUD 2,500,000."
        
        result = evaluator._compute_answer_relevance(query, answer)
        assert result > 0  # Should detect relevance
    
    def test_answer_relevance_empty_answer(self, evaluator):
        """Answer relevance should be 0 for empty answer."""
        query = "What is the contract value?"
        answer = ""
        
        result = evaluator._compute_answer_relevance(query, answer)
        assert result == 0.0
    
    def test_evaluate_sample(self, evaluator, sample):
        """Full sample evaluation should return metrics."""
        retrieved = [
            {"content": "The total contract value is AUD 2,500,000."},
        ]
        
        result = evaluator.evaluate_sample(
            sample=sample,
            generated_answer="The contract is worth AUD 2,500,000.",
            retrieved_chunks=retrieved,
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.sample_id == sample.id
        assert "context_recall" in result.metrics
        assert "faithfulness" in result.metrics
        assert "answer_relevance" in result.metrics


class TestGoldenDataset:
    """Tests for golden dataset loading."""
    
    def test_load_sample_dataset(self, tmp_path):
        """Should load a sample dataset."""
        dataset_json = {
            "samples": [
                {
                    "id": "sample_001",
                    "query": "Test query",
                    "ground_truth": "Test answer",
                    "relevant_chunks": ["Relevant chunk"],
                    "category": "general",
                    "difficulty": "easy",
                }
            ]
        }
        
        dataset_file = tmp_path / "test_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_json, f)
        
        samples = load_golden_dataset(str(dataset_file))
        
        assert len(samples) == 1
        assert samples[0].id == "sample_001"
        assert samples[0].query == "Test query"
    
    def test_load_dataset_with_metadata(self, tmp_path):
        """Should handle samples with metadata."""
        dataset_json = {
            "samples": [
                {
                    "id": "sample_001",
                    "query": "Test query",
                    "ground_truth": "Test answer",
                    "relevant_chunks": [],
                    "expected_compliance_score": 85.0,
                    "expected_budget": "1000000",
                    "category": "compliance",
                    "difficulty": "hard",
                    "metadata": {"source": "cuad"},
                }
            ]
        }
        
        dataset_file = tmp_path / "test_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_json, f)
        
        samples = load_golden_dataset(str(dataset_file))
        
        assert samples[0].expected_compliance_score == 85.0
        assert samples[0].expected_budget == "1000000"
        assert samples[0].category == "compliance"
        assert samples[0].difficulty == "hard"
        assert samples[0].metadata["source"] == "cuad"


class TestSampleSubset:
    """Tests for sample subset creation."""
    
    def test_filter_by_category(self):
        """Should filter samples by category."""
        samples = [
            EvaluationSample(id="1", query="q1", ground_truth="a1", category="compliance"),
            EvaluationSample(id="2", query="q2", ground_truth="a2", category="budget"),
            EvaluationSample(id="3", query="q3", ground_truth="a3", category="compliance"),
        ]
        
        filtered = create_sample_subset(samples, category="compliance")
        
        assert len(filtered) == 2
        assert all(s.category == "compliance" for s in filtered)
    
    def test_filter_by_difficulty(self):
        """Should filter samples by difficulty."""
        samples = [
            EvaluationSample(id="1", query="q1", ground_truth="a1", difficulty="easy"),
            EvaluationSample(id="2", query="q2", ground_truth="a2", difficulty="hard"),
            EvaluationSample(id="3", query="q3", ground_truth="a3", difficulty="easy"),
        ]
        
        filtered = create_sample_subset(samples, difficulty="easy")
        
        assert len(filtered) == 2
        assert all(s.difficulty == "easy" for s in filtered)
    
    def test_sample_count(self):
        """Should limit sample count when specified."""
        samples = [
            EvaluationSample(id=str(i), query=f"q{i}", ground_truth=f"a{i}")
            for i in range(100)
        ]
        
        sampled = create_sample_subset(samples, n_samples=10, seed=42)
        
        assert len(sampled) == 10
    
    def test_sample_determinism(self):
        """Same seed should produce same sample."""
        samples = [
            EvaluationSample(id=str(i), query=f"q{i}", ground_truth=f"a{i}")
            for i in range(100)
        ]
        
        sampled1 = create_sample_subset(samples, n_samples=10, seed=42)
        sampled2 = create_sample_subset(samples, n_samples=10, seed=42)
        
        assert [s.id for s in sampled1] == [s.id for s in sampled2]


class TestThresholdChecker:
    """Tests for threshold checking."""
    
    def test_check_passes(self, tmp_path):
        """Should pass when metrics meet thresholds."""
        report = {
            "metrics": {
                "context_recall": {"mean": 0.90},
                "faithfulness": {"mean": 0.92},
            },
            "total_samples": 100,
        }
        
        thresholds = {
            "thresholds": {
                "context_recall": {"min": 0.80, "target": 0.90},
                "faithfulness": {"min": 0.85, "target": 0.95},
            }
        }
        
        result = check_thresholds(report, thresholds)
        
        assert result["passed"] is True
        assert result["failed_metrics"] == 0
    
    def test_check_fails(self, tmp_path):
        """Should fail when metrics below thresholds."""
        report = {
            "metrics": {
                "context_recall": {"mean": 0.70},
                "faithfulness": {"mean": 0.92},
            },
            "total_samples": 100,
        }
        
        thresholds = {
            "thresholds": {
                "context_recall": {"min": 0.80, "target": 0.90},
                "faithfulness": {"min": 0.85, "target": 0.95},
            }
        }
        
        result = check_thresholds(report, thresholds)
        
        assert result["passed"] is False
        assert result["failed_metrics"] >= 1
    
    def test_check_details(self, tmp_path):
        """Should provide details for each metric."""
        report = {
            "metrics": {
                "context_recall": {"mean": 0.75},
            },
            "total_samples": 50,
        }
        
        thresholds = {
            "thresholds": {
                "context_recall": {"min": 0.80, "target": 0.90},
            }
        }
        
        result = check_thresholds(report, thresholds)
        
        assert len(result["results"]) == 1
        check_result = result["results"][0]
        assert check_result["metric"] == "context_recall"
        assert check_result["actual"] == 0.75
        assert check_result["threshold"] == 0.80
        assert check_result["passed"] is False
        assert check_result["gap"] > 0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_result_to_dict(self):
        """Should serialize to dictionary."""
        result = EvaluationResult(
            sample_id="test_001",
            query="Test query",
            generated_answer="Test answer",
            retrieved_chunks=[{"content": "Chunk 1"}],
            metrics={"context_recall": 0.9, "faithfulness": 0.85},
            evaluation_time_ms=150.5,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["sample_id"] == "test_001"
        assert result_dict["query"] == "Test query"
        assert result_dict["metrics"]["context_recall"] == 0.9
        assert result_dict["num_retrieved_chunks"] == 1


class TestPipelineWrapper:
    """Tests for RAGPipelineWrapper."""
    
    def test_wrapper_initialization(self):
        """Should initialize with default values."""
        from evaluation.pipeline_wrapper import RAGPipelineWrapper
        
        wrapper = RAGPipelineWrapper()
        
        assert wrapper.embedding_model == "voyage-4-large"
        assert wrapper.reranker_model == "voyage-rerank-2"
        assert wrapper.retrieval_mode == "hybrid"
        assert wrapper.top_k == 10
    
    def test_wrapper_custom_config(self):
        """Should accept custom configuration."""
        from evaluation.pipeline_wrapper import RAGPipelineWrapper
        
        wrapper = RAGPipelineWrapper(
            embedding_model="voyage-finance-2",
            reranker_model="bge-reranker-v2-m3",
            retrieval_mode="vector_only",
            top_k=20,
        )
        
        assert wrapper.embedding_model == "voyage-finance-2"
        assert wrapper.reranker_model == "bge-reranker-v2-m3"
        assert wrapper.retrieval_mode == "vector_only"
        assert wrapper.top_k == 20
    
    def test_wrapper_health_check_no_api(self):
        """Health check should return True in direct mode."""
        from evaluation.pipeline_wrapper import RAGPipelineWrapper
        
        wrapper = RAGPipelineWrapper(api_base=None)
        
        # In direct mode (no API), health check returns True
        assert wrapper.health_check() is True


class TestSimilarityFunctions(unittest.TestCase):
    """Tests for similarity computation."""
    
    def setUp(self):
        self.evaluator = RAGASEvaluator()
    
    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors should be 1.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        
        sim = self.evaluator._cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 1.0, places=5)
    
    def test_cosine_similarity_opposite(self):
        """Cosine similarity of opposite vectors should be -1.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        
        sim = self.evaluator._cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)
    
    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors should be 0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        
        sim = self.evaluator._cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)
    
    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0 similarity."""
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        
        sim = self.evaluator._cosine_similarity(v1, v2)
        self.assertEqual(sim, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
