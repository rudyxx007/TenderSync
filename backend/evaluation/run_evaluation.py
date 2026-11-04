#!/usr/bin/env python3
"""
RAG Evaluation Runner - Entry point for CLI evaluation execution.

Usage:
    python -m evaluation.run_evaluation \
        --dataset evaluation/golden_dataset.json \
        --output evaluation/results/eval_report.json \
        --tenant-id test-tenant \
        --samples 100 \
        --verbose
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure backend is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment from .env file
try:
    from dotenv import load_dotenv
    # Load .env from the backend directory
    backend_dir = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=backend_dir / '.env')
except ImportError:
    pass  # dotenv not installed, environment must be set externally

# Verify required env vars
REQUIRED_VARS = ['VOYAGE_API_KEY']
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    print(f"ERROR: Missing required environment variables: {missing}")
    sys.exit(1)

from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGASPipelineEvaluator,
    load_golden_dataset,
    EvaluationSample,
)
from evaluation.pipeline_wrapper import RAGPipelineWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RAG evaluation on golden dataset'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to golden dataset JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path for output evaluation report'
    )
    parser.add_argument(
        '--tenant-id', '-t',
        type=str,
        required=True,
        help='Tenant ID for retrieval context'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=0,
        help='Number of samples to evaluate (0=all)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter by category (compliance, budget, deadline, general)'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        default=None,
        help='Filter by difficulty (easy, medium, hard)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='voyage-4-large',
        help='Embedding model to use'
    )
    parser.add_argument(
        '--reranker',
        type=str,
        default='voyage-rerank-2',
        help='Reranker model to use'
    )
    parser.add_argument(
        '--retrieval-mode',
        type=str,
        default='hybrid',
        choices=['hybrid', 'bm25_only', 'vector_only'],
        help='Retrieval mode'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='Base URL for RAG service API'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_golden_dataset(str(dataset_path))
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Filter dataset
    if args.category:
        dataset = [s for s in dataset if s.category == args.category]
        logger.info(f"Filtered to {len(dataset)} samples in category '{args.category}'")
    
    if args.difficulty:
        dataset = [s for s in dataset if s.difficulty == args.difficulty]
        logger.info(f"Filtered to {len(dataset)} samples with difficulty '{args.difficulty}'")
    
    if args.samples > 0 and args.samples < len(dataset):
        import random
        random.seed(42)
        dataset = random.sample(dataset, args.samples)
        logger.info(f"Sampled {len(dataset)} samples for evaluation")
    
    # Initialize RAG pipeline wrapper
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipelineWrapper(
        api_base=args.api_base,
        embedding_model=args.model,
        reranker_model=args.reranker,
        retrieval_mode=args.retrieval_mode,
    )
    
    # Initialize evaluator
    evaluator = RAGASEvaluator()
    pipeline_evaluator = RAGASPipelineEvaluator(
        evaluator=evaluator,
        rag_pipeline=pipeline,
    )
    
    # Run evaluation
    logger.info(f"Starting evaluation on {len(dataset)} samples...")
    result = pipeline_evaluator.evaluate_dataset(
        dataset=dataset,
        tenant_id=args.tenant_id,
        run_rag_pipeline=True,
        verbose=args.verbose,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {result.total_samples}")
    print()
    
    if result.metrics:
        print("Metrics:")
        print("-" * 40)
        for metric_name, metric_data in result.metrics.items():
            mean = metric_data.get("mean", 0) * 100
            print(f"  {metric_name}: {mean:.2f}%")
    
    # Check against thresholds
    thresholds_path = Path(__file__).parent / "config" / "metrics_thresholds.json"
    if thresholds_path.exists():
        print("\n" + "-" * 40)
        print("Threshold Check:")
        print("-" * 40)
        
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        
        all_passed = True
        for metric_name, threshold_data in thresholds.get("thresholds", {}).items():
            if metric_name in result.metrics:
                actual = result.metrics[metric_name].get("mean", 0)
                min_threshold = threshold_data.get("min", 0)
                status = "PASS" if actual >= min_threshold else "FAIL"
                if actual < min_threshold:
                    all_passed = False
                print(f"  {status} {metric_name}: {actual:.2%} (min: {min_threshold:.0%})")
        
        if not all_passed:
            logger.warning("Some metrics below minimum thresholds!")
            sys.exit(1)
    
    print()
    print("=" * 60)
    print("Evaluation PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
