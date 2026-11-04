"""
Standalone tests for the RAG evaluation framework.

These tests don't depend on the project's conftest.py fixture,
avoiding dependency issues with langgraph/ormsgpack.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Dict, Any

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
from evaluation.pipeline_wrapper import RAGPipelineWrapper


class TestEvaluationSample(unittest.TestCase):
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
        
        self.assertEqual(sample.id, "test_001")
        self.assertEqual(sample.query, "What is the contract value?")
        self.assertEqual(sample.category, "budget")
        self.assertEqual(sample.difficulty, "easy")
    
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
        
        self.assertEqual(sample.expected_compliance_score, 100.0)
        self.assertEqual(sample.expected_budget, "2500000")
        self.assertEqual(sample.metadata["source"], "financebench")


class TestRAGASEvaluator(unittest.TestCase):
    """Tests for RAGASEvaluator metrics."""
    
    def setUp(self):
        self.evaluator = RAGASEvaluator()
    
    def test_context_recall_perfect_match(self):
        """Context recall should be 1.0 when all relevant chunks retrieved."""
        retrieved = [
            {"content": "The total contract value is AUD 2,500,000."},
            {"content": "Payment is due within 30 days."},
        ]
        ground_truth = ["The total contract value is AUD 2,500,000."]
        
        result = self.evaluator._compute_context_recall(retrieved, ground_truth)
        self.assertEqual(result, 1.0)
    
    def test_context_recall_partial_match(self):
        """Context recall should be partial when some chunks missing."""
        retrieved = [
            {"content": "Payment is due within 30 days."},
        ]
        ground_truth = ["The total contract value is AUD 2,500,000."]
        
        result = self.evaluator._compute_context_recall(retrieved, ground_truth)
        self.assertEqual(result, 0.0)
    
    def test_context_recall_no_retrieval(self):
        """Context recall should be 0.0 when nothing retrieved."""
        retrieved = []
        ground_truth = ["The total contract value is AUD 2,500,000."]
        
        result = self.evaluator._compute_context_recall(retrieved, ground_truth)
        self.assertEqual(result, 0.0)
    
    def test_context_recall_no_ground_truth(self):
        """Context recall should be 1.0 when no ground truth."""
        retrieved = [{"content": "Some content"}]
        ground_truth = []
        
        result = self.evaluator._compute_context_recall(retrieved, ground_truth)
        self.assertEqual(result, 1.0)
    
    def test_context_precision_ranking(self):
        """Context precision should reward relevant chunks at top."""
        retrieved = [
            {"content": "The total contract value is AUD 2,500,000."},  # relevant
            {"content": "Payment is due within 30 days."},  # not relevant
            {"content": "The supplier must have ISO 9001 certification."},  # not relevant
        ]
        ground_truth = ["The total contract value is AUD 2,500,000."]
        
        result = self.evaluator._compute_context_precision(retrieved, ground_truth)
        # Should be > 0 because relevant chunk is first
        self.assertGreater(result, 0)
    
    def test_faithfulness_high(self):
        """Faithfulness should be high when answer matches context."""
        answer = "The contract value is AUD 2,500,000."
        chunks = [
            {"content": "The total contract value is AUD 2,500,000."},
        ]
        
        result = self.evaluator._compute_faithfulness(answer, chunks)
        self.assertGreater(result, 0.5)  # Should detect high overlap
    
    def test_faithfulness_low(self):
        """Faithfulness should be lower when answer contradicts context."""
        answer = "The contract value is AUD 5,000,000."
        chunks = [
            {"content": "The total contract value is AUD 2,500,000."},
        ]
        
        result = self.evaluator._compute_faithfulness(answer, chunks)
        # Note: The simple heuristic may still show some overlap due to shared words
        # We just check it's not perfect (1.0)
        self.assertLess(result, 1.0)
    
    def test_answer_relevance(self):
        """Answer relevance should measure query-answer overlap."""
        query = "What is the contract value?"
        answer = "The contract is worth AUD 2,500,000."
        
        result = self.evaluator._compute_answer_relevance(query, answer)
        self.assertGreater(result, 0)  # Should detect relevance
    
    def test_answer_relevance_empty_answer(self):
        """Answer relevance should be 0 for empty answer."""
        query = "What is the contract value?"
        answer = ""
        
        result = self.evaluator._compute_answer_relevance(query, answer)
        self.assertEqual(result, 0.0)
    
    def test_evaluate_sample(self):
        """Full sample evaluation should return metrics."""
        sample = EvaluationSample(
            id="test_001",
            query="What is the contract value?",
            ground_truth="The contract is worth AUD 2.5M.",
            relevant_chunks=["The total contract value is AUD 2,500,000."],
        )
        
        retrieved = [
            {"content": "The total contract value is AUD 2,500,000."},
        ]
        
        result = self.evaluator.evaluate_sample(
            sample=sample,
            generated_answer="The contract is worth AUD 2,500,000.",
            retrieved_chunks=retrieved,
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.sample_id, sample.id)
        self.assertIn("context_recall", result.metrics)
        self.assertIn("faithfulness", result.metrics)
        self.assertIn("answer_relevance", result.metrics)


class TestGoldenDataset(unittest.TestCase):
    """Tests for golden dataset loading."""
    
    def test_load_sample_dataset(self):
        """Should load a sample dataset."""
        # Load_golden_dataset expects a list at top level
        dataset_json = [
            {
                "id": "sample_001",
                "query": "Test query",
                "ground_truth": "Test answer",
                "relevant_chunks": ["Relevant chunk"],
                "category": "general",
                "difficulty": "easy",
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dataset_json, f)
            temp_path = f.name
        
        try:
            samples = load_golden_dataset(temp_path)
            
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].id, "sample_001")
            self.assertEqual(samples[0].query, "Test query")
        finally:
            os.unlink(temp_path)
    
    def test_load_dataset_with_metadata(self):
        """Should handle samples with metadata."""
        # Load_golden_dataset expects a list at top level
        dataset_json = [
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dataset_json, f)
            temp_path = f.name
        
        try:
            samples = load_golden_dataset(temp_path)
            
            self.assertEqual(samples[0].expected_compliance_score, 85.0)
            self.assertEqual(samples[0].expected_budget, "1000000")
            self.assertEqual(samples[0].category, "compliance")
            self.assertEqual(samples[0].difficulty, "hard")
            self.assertEqual(samples[0].metadata["source"], "cuad")
        finally:
            os.unlink(temp_path)
    
    def test_load_golden_dataset_sample_file(self):
        """Should load the golden dataset sample file from the project."""
        # Load the sample file we created - it has 'samples' wrapper
        sample_path = Path(__file__).parent.parent / "evaluation" / "golden_dataset_sample.json"
        
        if sample_path.exists():
            # For files with 'samples' key, we need to extract the list
            with open(sample_path) as f:
                data = json.load(f)
            samples_list = data.get("samples", data if isinstance(data, list) else [])
            self.assertGreater(len(samples_list), 0)
            self.assertEqual(samples_list[0]["id"], "sample_001")
        else:
            self.skipTest("Golden dataset sample file not found")


class TestSampleSubset(unittest.TestCase):
    """Tests for sample subset creation."""
    
    def test_filter_by_category(self):
        """Should filter samples by category."""
        samples = [
            EvaluationSample(id="1", query="q1", ground_truth="a1", relevant_chunks=["chunk1"], category="compliance"),
            EvaluationSample(id="2", query="q2", ground_truth="a2", relevant_chunks=["chunk2"], category="budget"),
            EvaluationSample(id="3", query="q3", ground_truth="a3", relevant_chunks=["chunk3"], category="compliance"),
        ]
        
        filtered = create_sample_subset(samples, category="compliance")
        
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(s.category == "compliance" for s in filtered))
    
    def test_filter_by_difficulty(self):
        """Should filter samples by difficulty."""
        samples = [
            EvaluationSample(id="1", query="q1", ground_truth="a1", relevant_chunks=["c1"], difficulty="easy"),
            EvaluationSample(id="2", query="q2", ground_truth="a2", relevant_chunks=["c2"], difficulty="hard"),
            EvaluationSample(id="3", query="q3", ground_truth="a3", relevant_chunks=["c3"], difficulty="easy"),
        ]
        
        filtered = create_sample_subset(samples, difficulty="easy")
        
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(s.difficulty == "easy" for s in filtered))
    
    def test_sample_count(self):
        """Should limit sample count when specified."""
        samples = [
            EvaluationSample(id=str(i), query=f"q{i}", ground_truth=f"a{i}", relevant_chunks=[f"chunk{i}"])
            for i in range(100)
        ]
        
        sampled = create_sample_subset(samples, n_samples=10, seed=42)
        
        self.assertEqual(len(sampled), 10)
    
    def test_sample_determinism(self):
        """Same seed should produce same sample."""
        samples = [
            EvaluationSample(id=str(i), query=f"q{i}", ground_truth=f"a{i}", relevant_chunks=[f"chunk{i}"])
            for i in range(100)
        ]
        
        sampled1 = create_sample_subset(samples, n_samples=10, seed=42)
        sampled2 = create_sample_subset(samples, n_samples=10, seed=42)
        
        self.assertEqual([s.id for s in sampled1], [s.id for s in sampled2])


class TestThresholdChecker(unittest.TestCase):
    """Tests for threshold checking."""
    
    def test_check_passes(self):
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
        
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_metrics"], 0)
    
    def test_check_fails(self):
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
        
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(result["failed_metrics"], 1)
    
    def test_check_details(self):
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
        
        self.assertEqual(len(result["results"]), 1)
        check_result = result["results"][0]
        self.assertEqual(check_result["metric"], "context_recall")
        self.assertEqual(check_result["actual"], 0.75)
        self.assertEqual(check_result["threshold"], 0.80)
        self.assertFalse(check_result["passed"])
        self.assertGreater(check_result["gap"], 0)


class TestEvaluationResult(unittest.TestCase):
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
        
        self.assertEqual(result_dict["sample_id"], "test_001")
        self.assertEqual(result_dict["query"], "Test query")
        self.assertEqual(result_dict["metrics"]["context_recall"], 0.9)
        self.assertEqual(result_dict["num_retrieved_chunks"], 1)


class TestPipelineWrapper(unittest.TestCase):
    """Tests for RAGPipelineWrapper."""
    
    def test_wrapper_initialization(self):
        """Should initialize with default values."""
        wrapper = RAGPipelineWrapper()
        
        self.assertEqual(wrapper.embedding_model, "voyage-4-large")
        self.assertEqual(wrapper.reranker_model, "voyage-rerank-2")
        self.assertEqual(wrapper.retrieval_mode, "hybrid")
        self.assertEqual(wrapper.top_k, 10)
    
    def test_wrapper_custom_config(self):
        """Should accept custom configuration."""
        wrapper = RAGPipelineWrapper(
            embedding_model="voyage-finance-2",
            reranker_model="bge-reranker-v2-m3",
            retrieval_mode="vector_only",
            top_k=20,
        )
        
        self.assertEqual(wrapper.embedding_model, "voyage-finance-2")
        self.assertEqual(wrapper.reranker_model, "bge-reranker-v2-m3")
        self.assertEqual(wrapper.retrieval_mode, "vector_only")
        self.assertEqual(wrapper.top_k, 20)
    
    def test_wrapper_health_check_no_api(self):
        """Health check should return True in direct mode."""
        wrapper = RAGPipelineWrapper(api_base=None)
        
        # In direct mode (no API), health check returns True
        self.assertTrue(wrapper.health_check())


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
    unittest.main(verbosity=2)
