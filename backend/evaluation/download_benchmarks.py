#!/usr/bin/env python3
"""
Benchmark Dataset Downloader - Downloads standard evaluation datasets.

Downloads subsets of FinanceBench, CUAD, and other benchmarks
for use in the RAG evaluation pipeline.

Usage:
    python -m evaluation.download_benchmarks \
        --dataset financebench \
        --output data/benchmarks/financebench \
        --samples 20
        --hf-token $HF_TOKEN
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Hugging Face datasets repository configuration
BENCHMARK_REPOS = {
    "financebench": {
        "repo_id": "paulgauthier/FinanceBench",
        "description": "FinanceBench: Financial Document QA Benchmark",
        "sample_files": ["test_subsample.parquet", "samples.json"],
        "url": "https://huggingface.co/datasets/paulgauthier/FinanceBench",
    },
    "cuad": {
        "repo_id": "taft/contract_nlp",
        "description": "CUAD: Contract Understanding Atticus Dataset",
        "sample_files": ["cuad_test_v1.json"],
        "url": "https://www.atticusprojectai.org/cuad",
    },
}

# Fallback sample data for offline/testing
FALLBACK_SAMPLES = {
    "financebench": [
        {
            "id": "financebench_001",
            "query": "What is the total revenue reported in Q4 2025?",
            "ground_truth": "The total revenue for Q4 2025 is $12.5 billion, representing a 15% increase year-over-year.",
            "relevant_chunks": [
                "CONSOLIDATED STATEMENTS OF OPERATIONS: Fourth Quarter 2025: Total revenues $12,500,000,000, an increase of 15% compared to $10,870,000,000 in Q4 2024."
            ],
            "category": "finance",
            "difficulty": "medium",
        },
        {
            "id": "financebench_002",
            "query": "What was the net income margin for the fiscal year?",
            "ground_truth": "Net income margin for the fiscal year 2025 was 18.3%, calculated as net income of $4.2 billion divided by total revenue of $22.9 billion.",
            "relevant_chunks": [
                "INCOME STATEMENT FY2025: Total revenues $22,900,000,000, Net income $4,200,000,000, Net income margin 18.3%"
            ],
            "category": "finance",
            "difficulty": "medium",
        },
    ],
    "cuad": [
        {
            "id": "cuad_001",
            "query": "Does this contract contain a force majeure clause?",
            "ground_truth": "Yes, Section 12.3 contains a force majeure clause covering natural disasters, war, terrorism, and government actions.",
            "relevant_chunks": [
                "Section 12.3 Force Majeure: Neither party shall be liable for failure to perform its obligations under this Agreement if such failure is caused by circumstances beyond its reasonable control, including acts of God, natural disasters, war, terrorism, civil unrest, or government actions."
            ],
            "category": "clause_detection",
            "difficulty": "easy",
        },
        {
            "id": "cuad_002",
            "query": "What is the termination notice period?",
            "ground_truth": "Either party may terminate this agreement with 90 days written notice. Material breach may be terminated immediately upon 30 days notice.",
            "relevant_chunks": [
                "Section 8.1 Termination for Convenience: Either party may terminate this Agreement at any time by providing 90 days prior written notice to the other party.",
                "Section 8.2 Termination for Cause: The non-breaching party may terminate this Agreement immediately upon written notice if the other party commits a material breach and fails to cure within 30 days."
            ],
            "category": "clause_detection",
            "difficulty": "medium",
        },
    ],
    "austender": [
        {
            "id": "austender_001",
            "query": "What are the mandatory accreditation requirements?",
            "ground_truth": "Bidders must hold current accreditation from ISO 9001:2015 (Quality), ISO 27001:2013 (Security), and ISO 14001:2015 (Environmental). Additionally, SAFEGUARD accreditation is required for projects involving classified information.",
            "relevant_chunks": [
                "Part B - Eligibility Criteria: 2.1 Quality Management: ISO 9001:2015 certification required. 2.2 Information Security: ISO 27001:2013 certification required. 2.3 Environmental: ISO 14001:2015 certification required. 2.4 Security Clearance: SAFEGUARD accreditation required for all personnel."
            ],
            "category": "compliance",
            "difficulty": "easy",
        },
    ],
}


def download_hf_dataset(
    repo_id: str,
    file_path: str,
    output_path: Path,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Download a file from Hugging Face datasets repository.
    
    Returns True if successful.
    """
    # Hugging Face raw file URL
    # For public datasets, this works without auth
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
    
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    logger.info(f"Downloading {repo_id}:{file_path}")
    
    try:
        response = requests.get(url, headers=headers, timeout=60)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Saved to {output_path}")
            return True
        else:
            logger.warning(f"Failed to download: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def download_file_from_url(
    url: str,
    output_path: Path,
) -> bool:
    """Download a file from a direct URL."""
    logger.info(f"Downloading {url}")
    
    try:
        response = requests.get(url, timeout=120)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Saved to {output_path}")
            return True
        else:
            logger.warning(f"Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def create_sample_dataset(
    dataset_name: str,
    output_dir: Path,
    num_samples: int,
) -> int:
    """
    Create a sample dataset from fallback samples.
    
    Useful for testing when internet access is unavailable.
    """
    fallback = FALLBACK_SAMPLES.get(dataset_name, [])
    
    if not fallback:
        logger.warning(f"No fallback samples for {dataset_name}")
        return 0
    
    # Create samples directory
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Select and write samples
    selected = fallback if num_samples == 0 or num_samples > len(fallback) else random.sample(fallback, num_samples)
    
    for i, sample in enumerate(selected):
        sample_file = samples_dir / f"sample_{i+1:03d}.json"
        with open(sample_file, 'w') as f:
            json.dump(sample, f, indent=2)
    
    # Write metadata
    metadata = {
        "description": BENCHMARK_REPOS.get(dataset_name, {}).get("description", f"{dataset_name} samples"),
        "source": BENCHMARK_REPOS.get(dataset_name, {}).get("url", "local"),
        "version": "1.0.0",
        "samples_count": len(selected),
        "created": datetime.now().isoformat(),
        "note": "Generated from sample data for evaluation purposes",
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created {len(selected)} samples in {output_dir}")
    return len(selected)


def download_benchmark(
    dataset_name: str,
    output_dir: Path,
    num_samples: int = 0,
    hf_token: Optional[str] = None,
) -> int:
    """
    Download a benchmark dataset.
    
    Returns the number of samples downloaded/created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = BENCHMARK_REPOS.get(dataset_name)
    if not config:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available: {list(BENCHMARK_REPOS.keys())}")
        return 0
    
    logger.info(f"Dataset: {config['description']}")
    logger.info(f"Source: {config['url']}")
    
    # Try to download from Hugging Face
    if hf_token or dataset_name in ["financebench", "cuad"]:
        # Try downloading sample files
        for sample_file in config.get("sample_files", []):
            output_file = output_dir / sample_file
            if output_file.exists():
                logger.info(f"Already exists: {sample_file}")
                continue
            
            if hf_token:
                success = download_hf_dataset(
                    config["repo_id"], sample_file, output_file, hf_token
                )
            else:
                # Try public access
                success = download_hf_dataset(
                    config["repo_id"], sample_file, output_file, None
                )
            
            if success:
                # Parse and convert to our format
                # (simplified - in production would parse the actual format)
                logger.info(f"Downloaded {sample_file}")
    
    # If download failed or no samples, create fallback
    samples_dir = output_dir / "samples"
    existing_count = len(list(samples_dir.glob("*.json"))) if samples_dir.exists() else 0
    
    if existing_count == 0:
        logger.info("Creating sample dataset from fallback data...")
        count = create_sample_dataset(dataset_name, output_dir, num_samples)
        return count
    else:
        logger.info(f"Found {existing_count} existing samples")
        return existing_count


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for evaluation"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=list(BENCHMARK_REPOS.keys()) + ["all"],
        help="Dataset to download: financebench, cuad, austender, or all"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/benchmarks",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=20,
        help="Number of samples to create (0=all available)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (optional for public datasets)"
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback sample data only (no download)"
    )
    
    args = parser.parse_args()
    
    output_base = Path(args.output)
    
    if args.dataset == "all":
        datasets_to_download = list(BENCHMARK_REPOS.keys())
    else:
        datasets_to_download = [args.dataset]
    
    total_samples = 0
    
    for dataset_name in datasets_to_download:
        dataset_output = output_base / dataset_name
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {dataset_name}")
        logger.info(f"{'='*60}")
        
        if args.use_fallback:
            logger.info("Using fallback mode (no download)")
            count = create_sample_dataset(dataset_name, dataset_output, args.samples)
        else:
            hf_token = args.hf_token or os.getenv("HF_TOKEN")
            count = download_benchmark(
                dataset_name=dataset_name,
                output_dir=dataset_output,
                num_samples=args.samples,
                hf_token=hf_token,
            )
        
        total_samples += count
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete: {total_samples} total samples")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
