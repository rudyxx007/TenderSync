"""
TenderSync RAG Evaluation Framework.

Provides automated evaluation using RAGAS metrics, golden datasets,
ablation studies, and benchmark comparisons.

Metrics:
- Context Recall: % of relevant context retrieved
- Context Precision: Quality of context ranking
- Faithfulness: Factual accuracy (no hallucination)
- Answer Relevance: Relevance to the query
- Compliance Score Accuracy: Accuracy of compliance score extraction
- Citation Verification Rate: Rate of verified citations

References:
- RAGAS: https://docs.ragas.io/
- TruLens: https://www.trulens.org/
"""

from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGASPipelineEvaluator,
    EvaluationSample,
    EvaluationResult,
    AggregateEvaluationResult,
    EvaluationMetric,
    load_golden_dataset,
    create_sample_subset,
)

from evaluation.pipeline_wrapper import (
    RAGPipelineWrapper,
    EvaluationPipeline,
    RAGPipelineResult,
)

__all__ = [
    # Evaluators
    "RAGASEvaluator",
    "RAGASPipelineEvaluator",
    
    # Data classes
    "EvaluationSample",
    "EvaluationResult",
    "AggregateEvaluationResult",
    "EvaluationMetric",
    "RAGPipelineWrapper",
    "EvaluationPipeline",
    "RAGPipelineResult",
    
    # Utilities
    "load_golden_dataset",
    "create_sample_subset",
]

__version__ = "1.0.0"
