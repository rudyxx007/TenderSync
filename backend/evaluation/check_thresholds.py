#!/usr/bin/env python3
"""
Threshold Checker - Validates evaluation results against minimum thresholds.

Used in CI/CD to fail builds when metrics drop below acceptable levels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_thresholds(thresholds_path: str) -> Dict[str, Any]:
    """Load threshold configuration."""
    with open(thresholds_path) as f:
        return json.load(f)


def load_report(report_path: str) -> Dict[str, Any]:
    """Load evaluation report."""
    with open(report_path) as f:
        return json.load(f)


def check_thresholds(
    report: Dict[str, Any],
    thresholds_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check evaluation metrics against thresholds.
    
    Returns dict with:
        - passed: bool
        - results: list of {metric, actual, threshold, passed}
        - summary: overall statistics
    """
    results = []
    all_passed = True
    
    thresholds = thresholds_config.get("thresholds", {})
    report_metrics = report.get("metrics", {})
    
    for metric_name, threshold_data in thresholds.items():
        if metric_name not in report_metrics:
            continue
        
        actual = report_metrics[metric_name].get("mean", 0)
        min_threshold = threshold_data.get("min", 0)
        target = threshold_data.get("target", 1.0)
        
        passed = actual >= min_threshold
        if not passed:
            all_passed = False
        
        results.append({
            "metric": metric_name,
            "actual": actual,
            "threshold": min_threshold,
            "target": target,
            "passed": passed,
            "gap": min_threshold - actual if not passed else 0,
        })
    
    # Check category-specific thresholds
    category_thresholds = thresholds_config.get("category_thresholds", {})
    per_category = report.get("per_category", {})
    
    for category, cat_thresholds in category_thresholds.items():
        if category not in per_category:
            continue
        
        for metric_name, threshold in cat_thresholds.items():
            if metric_name not in per_category[category]:
                continue
            
            actual = per_category[category][metric_name].get("mean", 0)
            passed = actual >= threshold
            
            if not passed:
                all_passed = False
            
            results.append({
                "metric": f"{category}.{metric_name}",
                "actual": actual,
                "threshold": threshold,
                "target": threshold,
                "passed": passed,
                "gap": threshold - actual if not passed else 0,
            })
    
    return {
        "passed": all_passed,
        "results": results,
        "total_metrics": len(results),
        "passed_metrics": sum(1 for r in results if r["passed"]),
        "failed_metrics": sum(1 for r in results if not r["passed"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check evaluation results against thresholds"
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        required=True,
        help="Path to evaluation report JSON"
    )
    parser.add_argument(
        "--thresholds", "-t",
        type=str,
        default="evaluation/config/metrics_thresholds.json",
        help="Path to thresholds config JSON"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Also fail on metrics below target (not just minimum)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write detailed check results"
    )
    
    args = parser.parse_args()
    
    # Load data
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: Report not found: {args.report}")
        sys.exit(1)
    
    thresholds_path = Path(args.thresholds)
    if not thresholds_path.exists():
        thresholds_path = Path(__file__).parent.parent / "config" / "metrics_thresholds.json"
    
    if not thresholds_path.exists():
        print(f"ERROR: Thresholds config not found")
        sys.exit(1)
    
    report = load_report(args.report)
    thresholds_config = load_thresholds(str(thresholds_path))
    
    # Run checks
    check_results = check_thresholds(report, thresholds_config)
    
    # Print results
    print("=" * 60)
    print("THRESHOLD CHECK RESULTS")
    print("=" * 60)
    print()
    
    overall_status = "✅ PASSED" if check_results["passed"] else "❌ FAILED"
    print(f"Overall Status: {overall_status}")
    print()
    
    print(f"Metrics checked: {check_results['total_metrics']}")
    print(f"  Passed: {check_results['passed_metrics']}")
    print(f"  Failed: {check_results['failed_metrics']}")
    print()
    
    print("Detailed Results:")
    print("-" * 60)
    print(f"{'Metric':<35} {'Actual':>8} {'Min':>8} {'Target':>8} {'Status':>8}")
    print("-" * 60)
    
    for result in check_results["results"]:
        status = "✓" if result["passed"] else "✗"
        print(
            f"{result['metric']:<35} "
            f"{result['actual']*100:>7.2f}% "
            f"{result['threshold']*100:>7.2f}% "
            f"{result['target']*100:>7.2f}% "
            f"{status:>8}"
        )
    
    print("-" * 60)
    print()
    
    # Check for warning-level failures (below target but above minimum)
    warnings = [
        r for r in check_results["results"]
        if not args.fail_on_warning 
        and r["actual"] < r["target"] 
        and r["actual"] >= r["threshold"]
    ]
    
    if warnings:
        print("⚠️  Below target (but above minimum):")
        for w in warnings:
            print(f"  - {w['metric']}: {w['actual']*100:.2f}% (target: {w['target']*100:.0f}%)")
        print()
    
    # Output detailed results
    if args.output:
        output = {
            "overall_passed": check_results["passed"],
            "timestamp": report.get("timestamp", ""),
            "total_samples": report.get("total_samples", 0),
            "checks": check_results["results"],
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Detailed results written to: {args.output}")
    
    # Exit with appropriate code
    if check_results["passed"]:
        print("✅ All metrics meet minimum thresholds")
        sys.exit(0)
    else:
        print("❌ Some metrics below minimum thresholds")
        print()
        print("Failed metrics:")
        for result in check_results["results"]:
            if not result["passed"]:
                gap = result["gap"] * 100
                print(f"  - {result['metric']}: {result['actual']*100:.2f}% "
                      f"(need: {result['threshold']*100:.0f}%, gap: {gap:.2f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
