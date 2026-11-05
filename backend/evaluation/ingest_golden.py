#!/usr/bin/env python3
"""Seed a golden dataset's source documents into an evaluation tenant.

Retrieval cannot be measured against an empty database. This reads a golden
dataset, ingests each sample's source text through the same production path as
an uploaded tender (contextual parent/child chunking + embeddings), writes the
resulting ``document_id`` back onto each sample, and saves a seeded copy for
``run_evaluation``.

Source text selection, in order:
1. ``metadata.document_text`` (full contract / RFP text)
2. ``relevant_chunks`` joined together (evidence-only benchmarks such as the
   FinanceBench merged JSONL, which does not ship the 10-K PDFs)

Evidence-only mode is a retrieval sanity harness, not the official full-document
FinanceBench leaderboard. The report records ``seed_mode`` so this is explicit.

Usage:
    python -m evaluation.ingest_golden \
        --dataset evaluation/datasets/financebench.json \
        --output evaluation/seeded/financebench.json \
        --tenant-id <org-uuid> \
        --user-id <user-uuid>
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BACKEND_DIR / ".env")
except ImportError:
    pass

import rag_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_golden")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed golden dataset documents into a tenant")
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Seeded copy for this single benchmark (never combined)",
    )
    parser.add_argument("--tenant-id", "-t", required=True)
    parser.add_argument("--user-id", "-u", required=True, help="UUID of the evaluation user")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even when a document with the same content hash already exists",
    )
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="Group samples sharing metadata.source_document into one ingested document",
    )
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "voyage-4-large"))
    return parser.parse_args()


def source_text(sample: Dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    if metadata.get("document_text"):
        return str(metadata["document_text"])
    chunks = sample.get("relevant_chunks") or []
    if isinstance(chunks, str):
        chunks = [chunks]
    return "\n\n".join(str(chunk) for chunk in chunks)


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return 2

    with open(dataset_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if args.limit:
        samples = samples[: args.limit]

    sources = {
        str((sample.get("metadata") or {}).get("source_dataset") or "unknown")
        for sample in samples
    }
    if len(sources) != 1:
        logger.error(
            "Refusing to seed a mixed-source dataset %s. Seed each benchmark "
            "separately so FinanceBench and CUAD are never evaluated together.",
            sorted(sources),
        )
        return 2

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_SECRET_KEY are required.")
        return 2

    from supabase import create_client

    client = create_client(url, key)
    embedder = rag_service.build_embedder(args.embedding_model)

    # Optionally ingest one document per source (financebench doc_name / CUAD title).
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for index, sample in enumerate(samples):
        metadata = sample.get("metadata") or {}
        key_name = (
            f"{metadata.get('source_dataset', 'dataset')}::{metadata.get('source_document')}"
            if args.group_by_source and metadata.get("source_document")
            else f"sample::{sample.get('id', index)}"
        )
        groups.setdefault(key_name, []).append(sample)

    seeded = 0
    failures = 0
    for group_name, group in groups.items():
        text = "\n\n".join(source_text(sample) for sample in group)
        if not text.strip():
            logger.warning("Skipping %s (no source text)", group_name)
            failures += len(group)
            continue

        safe_name = group_name.replace("::", "_").replace("/", "_").replace("\\", "_")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        filename = f"eval_{safe_name[:80]}_{digest}.txt"

        # Reuse an existing document with the exact same content hash. This
        # makes CI reruns cheap and prevents duplicate evaluation documents.
        # --force is available when the embedding model or schema changed.
        document_id = None
        if not args.force:
            try:
                existing = (
                    client.table("documents")
                    .select("document_id")
                    .eq("org_id", args.tenant_id)
                    .eq("filename", filename)
                    .limit(1)
                    .execute()
                )
                if existing.data:
                    document_id = str(existing.data[0]["document_id"])
            except Exception as exc:
                logger.warning("Could not check existing document %s: %s", filename, exc)

        if document_id:
            logger.info("Reusing document %s for %s", document_id, group_name)
        else:
            try:
                result = rag_service.ingest_document(
                    supabase_client=client,
                    org_id=args.tenant_id,
                    user_id=args.user_id,
                    filename=filename,
                    document_text=text,
                    embedder=embedder,
                )
            except Exception as exc:
                logger.error("Ingestion failed for %s: %s", group_name, exc)
                failures += len(group)
                continue

            document_id = result.document_id
            logger.info(
                "Ingested %s: %d parents, %d children (embedded=%s)",
                group_name,
                result.parent_count,
                result.child_count,
                result.embedded,
            )

        for sample in group:
            sample["document_id"] = document_id
            sample.setdefault("metadata", {})["seed_mode"] = (
                "full_document" if (sample.get("metadata") or {}).get("document_text") else "evidence_only"
            )
            sample["metadata"]["seed_group"] = group_name
        seeded += len(group)

    if isinstance(payload, dict):
        payload["samples"] = samples
        payload["seed"] = {
            "tenant_id": args.tenant_id,
            "user_id": args.user_id,
            "seeded_samples": seeded,
            "failed_samples": failures,
            "group_by_source": args.group_by_source,
            "force": args.force,
            "note": "Evidence-only datasets are a retrieval sanity harness, not a full-document leaderboard.",
        }
        output_payload = payload
    else:
        output_payload = {
            "description": "Seeded golden dataset",
            "seed": {"tenant_id": args.tenant_id, "seeded_samples": seeded},
            "samples": samples,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    logger.info("Seeded %d sample(s), %d failure(s). Wrote %s", seeded, failures, output_path)
    return 0 if seeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
