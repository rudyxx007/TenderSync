#!/usr/bin/env python3
"""
Experiment Tracker - Logs evaluation results to experiment tracking systems.

Supports Weights & Biases (wandb) and local logging.
Can be extended to support MLflow, Comet, etc.

Usage:
    python -m evaluation.track_experiment \
        --results evaluation/results/financebench/eval_report.json \
        --commit abc123 \
        --wandb-project tender-sync-rag-eval
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> Optional[Dict[str, Any]]:
    """Load evaluation results from JSON file."""
    path = Path(results_path)
    if not path.exists():
        logger.error(f"Results file not found: {results_path}")
        return None
    
    with open(path) as f:
        return json.load(f)


def track_with_wandb(
    results: Dict[str, Any],
    commit_sha: str,
    project: str = "tender-sync-rag-eval",
    run_name: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[list] = None,
) -> bool:
    """
    Track experiment results with Weights & Biases.
    
    Returns True if successful.
    """
    try:
        import wandb
        
        # Initialize wandb run
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=tags or ["rag-evaluation"],
            config={
                "commit_sha": commit_sha,
                "timestamp": results.get("timestamp", datetime.now().isoformat()),
            },
        )
        
        # Log metrics
        metrics = results.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            wandb.log({
                f"metrics/{metric_name}_mean": metric_data.get("mean", 0),
                f"metrics/{metric_name}_std": metric_data.get("std", 0),
                f"metrics/{metric_name}_min": metric_data.get("min", 0),
                f"metrics/{metric_name}_max": metric_data.get("max", 0),
            })
        
        # Log summary
        wandb.summary["overall_passed"] = results.get("overall_passed", False)
        wandb.summary["total_samples"] = results.get("total_samples", 0)
        
        # Log config
        config = results.get("config", {})
        for key, value in config.items():
            wandb.config[key] = value
        
        # Finish run
        run.finish()
        
        logger.info(f"Experiment tracked in wandb: {run.url}")
        return True
        
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        logger.error(f"Failed to track with wandb: {e}")
        return False


def track_locally(
    results: Dict[str, Any],
    commit_sha: str,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Track experiment locally by saving summary files.
    
    Returns path to saved tracking data.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "experiment_logs"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tracking entry
    tracking_entry = {
        "commit_sha": commit_sha,
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "total_samples": results.get("total_samples", 0),
            "metrics": {
                name: {"mean": data.get("mean", 0)}
                for name, data in results.get("metrics", {}).items()
            },
            "overall_passed": results.get("overall_passed", False),
        },
        "config": results.get("config", {}),
    }
    
    # Save entry
    entry_file = output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(entry_file, 'w') as f:
        json.dump(tracking_entry, f, indent=2)
    
    # Update recent experiments index
    index_file = output_dir / "recent_experiments.json"
    recent = []
    if index_file.exists():
        with open(index_file) as f:
            recent = json.load(f)
    
    recent.insert(0, {
        "file": str(entry_file.relative_to(output_dir)),
        "timestamp": tracking_entry["timestamp"],
        "commit": commit_sha,
        "passed": tracking_entry["results_summary"]["overall_passed"],
    })
    
    # Keep only last 100
    recent = recent[:100]
    
    with open(index_file, 'w') as f:
        json.dump(recent, f, indent=2)
    
    logger.info(f"Experiment logged locally: {entry_file}")
    return entry_file


def main():
    parser = argparse.ArgumentParser(
        description="Track experiment results to monitoring systems"
    )
    parser.add_argument(
        "--results", "-r",
        type=str,
        required=True,
        help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--commit", "-c",
        type=str,
        default=None,
        help="Git commit SHA (auto-detected if not provided)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tender-sync-rag-eval",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom run name for wandb"
    )
    parser.add_argument(
        "--local-output",
        type=str,
        default=None,
        help="Directory for local experiment logs"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags for wandb run"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use local tracking (no wandb)"
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    if results is None:
        sys.exit(1)
    
    # Get commit SHA
    commit_sha = args.commit
    if not commit_sha:
        import subprocess
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except Exception:
            commit_sha = "unknown"
    
    logger.info(f"Tracking experiment:")
    logger.info(f"  Commit: {commit_sha}")
    logger.info(f"  Samples: {results.get('total_samples', 'N/A')}")
    logger.info(f"  Metrics: {list(results.get('metrics', {}).keys())}")
    
    # Track locally
    track_locally(results, commit_sha, args.local_output)
    
    # Track with wandb (unless offline)
    if not args.offline:
        wandb_success = track_with_wandb(
            results=results,
            commit_sha=commit_sha,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            tags=args.tags,
        )
        
        if not wandb_success:
            logger.warning(
                "W&B tracking failed. To enable, install wandb and set WANDB_API_KEY."
            )
    
    logger.info("Experiment tracking complete.")


if __name__ == "__main__":
    main()
