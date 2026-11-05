#!/usr/bin/env python3
"""Validate a golden dataset before it is used for evaluation.

Checks structure, provenance, uniqueness and evidence quality. Run in CI on
every commit; no credentials required.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_SOURCES = {"financebench", "cuad", "austender"}
MIN_SAMPLES = 100
BENCHMARK_MIN_SAMPLES = {"financebench": 50, "cuad": 40, "austender": 1}


def validate(path: Path, benchmark: Optional[str] = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not path.exists():
        return {"valid": False, "errors": [f"dataset not found: {path}"], "warnings": []}

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        samples = payload.get("samples", [])
        declared_sources = {
            str(source).lower() for source in payload.get("source_datasets", [])
        }
        if not payload.get("description"):
            warnings.append("missing top-level description")
    else:
        samples = payload
        declared_sources = set()
        warnings.append("dataset is a bare list; provenance metadata is missing")

    if not isinstance(samples, list):
        return {"valid": False, "errors": ["'samples' must be a list"], "warnings": warnings}

    ids = Counter()
    source_counts = Counter()
    categories = Counter()
    difficulties = Counter()

    for index, sample in enumerate(samples):
        label = sample.get("id", f"index {index}")
        ids[label] += 1

        for field in ("query", "ground_truth"):
            if not sample.get(field) or not str(sample[field]).strip():
                errors.append(f"{label}: empty {field}")

        chunks = sample.get("relevant_chunks") or sample.get("evidence") or []
        if isinstance(chunks, str):
            chunks = [chunks]
        chunks = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
        if not chunks:
            errors.append(f"{label}: no relevant_chunks/evidence")

        source = (
            (sample.get("metadata") or {}).get("source_dataset")
            or sample.get("source_dataset")
            or "unknown"
        )
        source_counts[source] += 1
        categories[sample.get("category", "general")] += 1
        difficulties[sample.get("difficulty", "medium")] += 1

    minimum = BENCHMARK_MIN_SAMPLES.get(benchmark, MIN_SAMPLES) if benchmark else MIN_SAMPLES
    if len(samples) < minimum:
        errors.append(
            f"only {len(samples)} samples; this benchmark requires at least {minimum}"
        )

    duplicates = [sample_id for sample_id, count in ids.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate ids: {duplicates[:10]}")

    if benchmark:
        if set(source_counts) != {benchmark}:
            errors.append(
                f"benchmark {benchmark!r} but dataset contains source(s) "
                f"{sorted(source_counts)}; datasets and runs must be single-source"
            )
        if declared_sources and declared_sources != {benchmark}:
            warnings.append(
                f"declared_sources {sorted(declared_sources)} do not match "
                f"benchmark {benchmark!r}"
            )
    else:
        if declared_sources and not REQUIRED_SOURCES.issubset(declared_sources):
            missing = sorted(REQUIRED_SOURCES - declared_sources)
            warnings.append(
                f"benchmark source(s) not represented in this dataset: {missing} "
                "(AusTender RFP samples require a local manifest)"
            )

        if len(source_counts) < 3:
            warnings.append(
                f"only {len(source_counts)} benchmark source(s) present: {dict(source_counts)}"
            )
        if len(source_counts) > 1:
            warnings.append(
                "mixed-source dataset; evaluation runs must use one benchmark at a time"
            )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "benchmark": benchmark,
        "sample_count": len(samples),
        "sources": dict(source_counts),
        "categories": dict(categories),
        "difficulties": dict(difficulties),
        "declared_sources": sorted(declared_sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a golden dataset")
    parser.add_argument(
        "--dataset", "-d", default="evaluation/datasets/financebench.json"
    )
    parser.add_argument(
        "--benchmark",
        choices=["financebench", "cuad", "austender"],
        default=None,
        help="Validate one benchmark and enforce its single-source/minimum-size rules",
    )
    args = parser.parse_args()

    result = validate(Path(args.dataset), benchmark=args.benchmark)

    print(f"Dataset: {args.dataset}")
    print(f"Benchmark: {args.benchmark or 'combined'}")
    print(f"Samples: {result.get('sample_count', 0)}")
    print(f"Sources: {result.get('sources', {})}")
    print(f"Categories: {result.get('categories', {})}")
    print(f"Difficulties: {result.get('difficulties', {})}")

    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    for error in result["errors"]:
        print(f"ERROR: {error}")

    print("VALID" if result["valid"] else "INVALID")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
