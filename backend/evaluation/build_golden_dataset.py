#!/usr/bin/env python3
"""Build separate per-benchmark RAG evaluation sets from real benchmarks.

Sources
-------
* FinanceBench (PatronusAI/financebench, CC-BY-NC-4.0): 150 financial-document
  Q&A records with human-annotated evidence spans.
* CUAD (The Atticus Project, CC-BY-4.0): 510 contracts with 41 clause-review
  questions each and extractive answers.
* AusTender: optional local manifest of questions authored against Australian
  government tender documents (``--austender-manifest``). The public AusTender
  site does not expose stable full-text RFP URLs, so this is opt-in.

The output is one file per benchmark under ``evaluation/datasets``. There is no
combined evaluation dataset; FinanceBench and CUAD are scored in separate runs.
Run it again to regenerate deterministically:

    python -m evaluation.build_golden_dataset \
        --output-dir evaluation/datasets

Nothing here invents questions or answers. If a source cannot be downloaded,
the builder fails instead of silently substituting synthetic samples.
"""

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

FINANCEBENCH_URL = (
    "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/"
    "financebench_merged.jsonl"
)
CUAD_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/"
    "CUAD_v1/CUAD_v1.json"
)

SOURCES = {
    "financebench": {
        "name": "FinanceBench",
        "url": "https://huggingface.co/datasets/PatronusAI/financebench",
        "licence": "CC-BY-NC-4.0",
        "note": "Evidence spans only (the merged JSONL does not include the 10-K PDFs).",
    },
    "cuad": {
        "name": "CUAD (Contract Understanding Atticus Dataset)",
        "url": "https://www.atticusprojectai.org/cuad",
        "licence": "CC-BY-4.0",
        "note": "Full contract text is stored in metadata.document_text for seeding.",
    },
    "austender": {
        "name": "Australian Government tenders (AusTender)",
        "url": "https://www.tenders.gov.au",
        "licence": "Commonwealth of Australia (public tender notices)",
        "note": "Questions are authored against tender documents supplied via --austender-manifest.",
    },
}


# ----------------------------------------------------------------------
# Download / cache
# ----------------------------------------------------------------------
def cached_download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {url}. Place the file at {destination} and re-run."
        ) from exc
    return destination


def load_financebench(cache_dir: Path) -> List[Dict[str, Any]]:
    path = cached_download(FINANCEBENCH_URL, cache_dir / "financebench" / "financebench_merged.jsonl")
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            evidence = [
                item.get("evidence_text", "")
                for item in (record.get("evidence") or [])
                if item.get("evidence_text")
            ]
            if not record.get("question") or not record.get("answer") or not evidence:
                continue
            record["_evidence"] = evidence
            records.append(record)
    if not records:
        raise RuntimeError("FinanceBench file contained no usable records.")
    return records


def load_cuad(cache_dir: Path) -> List[Dict[str, Any]]:
    path = cached_download(CUAD_URL, cache_dir / "cuad" / "CUAD_v1.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    contracts = payload.get("data") or []
    if not contracts:
        raise RuntimeError("CUAD file contained no contracts.")
    return contracts


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------
def _difficulty_from_reasoning(reasoning: Optional[str]) -> str:
    text = (reasoning or "").lower()
    if "extract" in text or "count" in text:
        return "easy"
    if "logical" in text or "reason" in text or "compare" in text:
        return "hard"
    return "medium"


def financebench_samples(records: Iterable[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record.get("question_type") or "Other"].append(record)

    ordered = []
    index = 0
    while len(ordered) < sum(len(group) for group in groups.values()):
        added = False
        for key in sorted(groups):
            if index < len(groups[key]):
                ordered.append(groups[key][index])
                added = True
        if not added:
            break
        index += 1

    samples = []
    for record in ordered:
        if len(samples) >= count:
            break
        evidence = []
        seen = set()
        for text in record["_evidence"]:
            text = text.strip()
            if text and text[:200] not in seen:
                seen.add(text[:200])
                evidence.append(text[:8000])
        if not evidence:
            continue

        question_type = (record.get("question_type") or "").lower()
        category = "budget" if any(
            token in question_type for token in ("metric", "numerical", "percentage")
        ) else "financial"

        samples.append(
            {
                "id": f"financebench_{record.get('financebench_id', len(samples))}",
                "query": record["question"],
                "ground_truth": str(record["answer"]),
                "relevant_chunks": evidence,
                "category": category,
                "difficulty": _difficulty_from_reasoning(record.get("question_reasoning")),
                "metadata": {
                    "source_dataset": "financebench",
                    "source_document": record.get("doc_name"),
                    "company": record.get("company"),
                    "doc_type": record.get("doc_type"),
                    "question_type": record.get("question_type"),
                    "question_reasoning": record.get("question_reasoning"),
                    "financebench_id": record.get("financebench_id"),
                    "doc_link": record.get("doc_link"),
                    "licence": SOURCES["financebench"]["licence"],
                },
            }
        )
    return samples


def _answer_window(context: str, start: int, length: int, radius: int = 900) -> str:
    left = max(0, start - radius)
    right = min(len(context), start + max(length, 1) + radius)
    if left > 0:
        boundary = context.rfind("\n\n", 0, left)
        if boundary != -1 and left - boundary < 500:
            left = boundary + 2
    boundary = context.find("\n\n", right)
    if boundary != -1 and boundary - right < 500:
        right = boundary
    window = context[left:right].strip()
    return window or context[max(0, start - radius): start + length + radius].strip()


def cuad_samples(contracts: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    candidates = []
    for contract in contracts:
        title = contract.get("title") or "Unknown contract"
        paragraphs = contract.get("paragraphs") or []
        if not paragraphs:
            continue
        paragraph = paragraphs[0]
        context = paragraph.get("context") or ""
        if len(context) < 500:
            continue

        chosen = None
        for qa in paragraph.get("qas") or []:
            answers = [answer for answer in (qa.get("answers") or []) if answer.get("text")]
            if not answers:
                continue
            question = qa.get("question") or ""
            if "Document Name" in question:
                continue
            answer = answers[0]
            text = str(answer["text"]).strip()
            if not (2 <= len(text) <= 500):
                continue
            chosen = (qa, answer)
            break

        if not chosen:
            continue
        qa, answer = chosen
        answer_text = str(answer["text"]).strip()
        start = context.lower().find(answer_text.lower())
        if start == -1:
            start = int(answer.get("answer_start") or 0)
        window = _answer_window(context, start, len(answer_text))
        if not window or answer_text.lower() not in window.lower():
            # Guarantee the annotated answer is present in the evidence span.
            window = context[max(0, start - 400): start + len(answer_text) + 400].strip()
        if not window:
            continue

        candidates.append(
            {
                "id": f"cuad_{qa.get('id', len(candidates))}",
                "query": qa["question"],
                "ground_truth": answer_text,
                "relevant_chunks": [window],
                "category": "clause_detection",
                "difficulty": "hard" if len(context) > 30_000 else "medium" if len(context) > 8_000 else "easy",
                "metadata": {
                    "source_dataset": "cuad",
                    "source_document": title,
                    "document_text": context,
                    "clause_category": qa.get("id"),
                    "context_chars": len(context),
                    "licence": SOURCES["cuad"]["licence"],
                },
            }
        )

    if len(candidates) <= count:
        return candidates

    stride = len(candidates) / count
    return [candidates[min(len(candidates) - 1, int(i * stride))] for i in range(count)]


def load_austender_manifest(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not path.exists():
        raise RuntimeError(f"AusTender manifest not found: {path}")

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    cleaned = []
    for index, sample in enumerate(samples):
        if not sample.get("query") or not sample.get("ground_truth"):
            raise RuntimeError(f"AusTender manifest sample {index} lacks query/ground_truth")
        if not sample.get("relevant_chunks") and not (sample.get("metadata") or {}).get("document_text"):
            raise RuntimeError(f"AusTender manifest sample {index} lacks evidence text")
        sample.setdefault("id", f"austender_{index:03d}")
        sample.setdefault("category", "compliance")
        sample.setdefault("difficulty", "medium")
        sample.setdefault("metadata", {})
        sample["metadata"].setdefault("source_dataset", "austender")
        sample["metadata"].setdefault("licence", SOURCES["austender"]["licence"])
        cleaned.append(sample)
    return cleaned


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TenderSync per-benchmark datasets")
    parser.add_argument("--output-dir", "-o", default="evaluation/datasets")
    parser.add_argument("--cache-dir", default="data/benchmarks")
    parser.add_argument("--financebench-count", type=int, default=55)
    parser.add_argument("--cuad-count", type=int, default=45)
    parser.add_argument("--austender-manifest", default=None)
    parser.add_argument("--total", type=int, default=100)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = BACKEND_DIR / cache_dir

    print("Loading FinanceBench...")
    finance_records = load_financebench(cache_dir)
    print(f"  {len(finance_records)} usable records")

    print("Loading CUAD...")
    contracts = load_cuad(cache_dir)
    print(f"  {len(contracts)} contracts")

    manifest = Path(args.austender_manifest) if args.austender_manifest else None
    if manifest and not manifest.is_absolute():
        manifest = BACKEND_DIR / manifest
    austender = load_austender_manifest(manifest)
    print(f"  {len(austender)} AusTender sample(s)")

    cuad_count = min(args.cuad_count, args.total - len(austender))
    finance_count = max(0, args.total - cuad_count - len(austender))

    samples = []
    samples.extend(financebench_samples(finance_records, finance_count))
    samples.extend(cuad_samples(contracts, cuad_count))
    samples.extend(austender)

    if len(samples) < args.total:
        # Top up with additional FinanceBench records rather than fabricating data.
        extra = financebench_samples(
            [record for record in finance_records], args.total - len(samples) + finance_count
        )
        existing = {sample["id"] for sample in samples}
        for sample in extra:
            if sample["id"] not in existing:
                samples.append(sample)
                existing.add(sample["id"])
            if len(samples) >= args.total:
                break

    samples = samples[: args.total]
    if len(samples) < args.total:
        print(
            f"ERROR: only {len(samples)} real samples available; required {args.total}. "
            "Provide more sources rather than synthetic data.",
            file=sys.stderr,
        )
        return 1

    counts: Dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample["metadata"]["source_dataset"]] += 1

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = BACKEND_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in sorted(counts):
        source_samples = [
            sample
            for sample in samples
            if sample["metadata"]["source_dataset"] == source
        ]
        payload = {
            "description": (
                f"TenderSync {SOURCES[source]['name']} benchmark — "
                f"{len(source_samples)} real queries with human-annotated evidence."
            ),
            "version": "2.0",
            "generated": date.today().isoformat(),
            "source_datasets": [source],
            "counts": {source: len(source_samples)},
            "licences": {source: SOURCES[source]["licence"]},
            "notes": {
                source: SOURCES[source]["note"],
                "seed": (
                    "Run evaluation.ingest_golden before evaluation. This file "
                    "contains exactly one benchmark and must be evaluated alone."
                ),
            },
            "samples": source_samples,
        }
        output = output_dir / f"{source}.json"
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {len(source_samples)} {source} samples to {output}")

    print(f"\nSources: {dict(counts)}")
    print(f"Categories: {dict((c, sum(1 for s in samples if s['category'] == c)) for c in sorted({s['category'] for s in samples}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
