#!/usr/bin/env python3
"""Validate an evaluation report against minimum metric thresholds.

A metric missing from the report now fails when it is required. Previously the
checker silently skipped absent metrics, so an evaluation that produced no
usable measurements could still be reported as passing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REQUIRED = [
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
]


def load_thresholds(thresholds_path: str) -> Dict[str, Any]:
    with open(thresholds_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_report(report_path: str) -> Dict[str, Any]:
    with open(report_path, encoding="utf-8") as handle:
        return json.load(handle)


def check_thresholds(
    report: Dict[str, Any],
    thresholds_config: Dict[str, Any],
    fail_on_missing: bool = True,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    all_passed = True

    thresholds = thresholds_config.get("thresholds", {})
    report_metrics = report.get("metrics", {})

    for metric_name, threshold_data in thresholds.items():
        required = threshold_data.get("required", metric_name in DEFAULT_REQUIRED)
        metric_payload = report_metrics.get(metric_name)

        if metric_payload is None:
            if required and fail_on_missing:
                all_passed = False
                results.append(
                    {
                        "metric": metric_name,
                        "actual": None,
                        "threshold": threshold_data.get("min", 0),
                        "target": threshold_data.get("target", 1.0),
                        "passed": False,
                        "gap": None,
                        "reason": "metric missing from report",
                    }
                )
            continue

        actual = float(metric_payload.get("mean", 0))
        minimum = float(threshold_data.get("min", 0))
        target = float(threshold_data.get("target", 1.0))
        passed = actual >= minimum
        if not passed:
            all_passed = False

        results.append(
            {
                "metric": metric_name,
                "actual": actual,
                "threshold": minimum,
                "target": target,
                "passed": passed,
                "gap": max(0.0, minimum - actual),
                "reason": None if passed else "below minimum",
            }
        )

    # Category-level thresholds (optional configuration).
    per_category = report.get("per_category", {})
    for category, category_thresholds in thresholds_config.get("category_thresholds", {}).items():
        for metric_name, minimum in category_thresholds.items():
            payload = per_category.get(category, {}).get(metric_name)
            if not payload:
                continue
            actual = float(payload.get("mean", 0))
            passed = actual >= float(minimum)
            if not passed:
                all_passed = False
            results.append(
                {
                    "metric": f"{category}.{metric_name}",
                    "actual": actual,
                    "threshold": float(minimum),
                    "target": float(minimum),
                    "passed": passed,
                    "gap": max(0.0, float(minimum) - actual),
                    "reason": None if passed else "category below minimum",
                }
            )

    return {
        "passed": all_passed,
        "results": results,
        "total_metrics": len(results),
        "passed_metrics": sum(1 for item in results if item["passed"]),
        "failed_metrics": sum(1 for item in results if not item["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check evaluation metrics against thresholds")
    parser.add_argument("--report", "-r", required=True)
    parser.add_argument(
        "--thresholds",
        "-t",
        default=str(Path(__file__).parent / "config" / "metrics_thresholds.json"),
    )
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}")
        sys.exit(1)

    thresholds_path = Path(args.thresholds)
    if not thresholds_path.exists():
        print(f"ERROR: thresholds not found: {thresholds_path}")
        sys.exit(1)

    result = check_thresholds(load_report(str(report_path)), load_thresholds(str(thresholds_path)))

    print("\nThreshold check")
    print("-" * 60)
    for item in result["results"]:
        actual = "missing" if item["actual"] is None else f"{item['actual']:.2%}"
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  {status:<4} {item['metric']:<32} {actual:>9}  (min {item['threshold']:.0%})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if not result["passed"]:
        print(
            f"\n{result['failed_metrics']} metric(s) failed. "
            "This run is not acceptable for release."
        )
        sys.exit(1)

    print(f"\nAll {result['total_metrics']} checked metrics passed.")


if __name__ == "__main__":
    main()
