#!/usr/bin/env python3
"""Download FinanceBench source filings and convert them to Markdown.

The 55-sample FinanceBench dataset shipped in this repository contains evidence
spans only. Retrieval and chunking need the full filings, so this script
downloads the released PDFs and parses them with Docling. Downloaded PDFs and
Markdown are kept under the gitignored ``backend/data/benchmarks`` directory.

Usage:
    python -m evaluation.prepare_financebench
    python -m evaluation.prepare_financebench --limit 5 --force
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from evaluation.benchmark_sources import FINANCEBENCH_DIR
from evaluation.ragas_evaluator import load_golden_dataset


def _target_url(url: str) -> str:
    if "pdfTarget=" not in url:
        return url
    query = parse_qs(urlparse(url).query)
    encoded = query.get("pdfTarget")
    if not encoded:
        return url
    return base64.b64decode(unquote(encoded[0])).decode("utf-8")


def download_pdfs(samples, pdf_dir: Path, force: bool = False, limit: int = 0) -> int:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    links = {}
    for sample in samples:
        metadata = sample.metadata or {}
        source = str(metadata.get("source_document") or "")
        url = str(metadata.get("doc_link") or "")
        if source and url:
            links[source] = url
    names = sorted(links)[:limit] if limit else sorted(links)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/130 Safari/537.36"
        )
    }
    downloaded = 0
    for index, name in enumerate(names, start=1):
        target = pdf_dir / f"{name}.pdf"
        if target.exists() and target.stat().st_size > 10_000 and not force:
            print(f"[{index}/{len(names)}] exists {name}", flush=True)
            continue
        try:
            response = requests.get(
                _target_url(links[name]), headers=headers, timeout=120, allow_redirects=True
            )
            if response.status_code == 200 and response.content[:4] == b"%PDF":
                target.write_bytes(response.content)
                downloaded += 1
                print(f"[{index}/{len(names)}] downloaded {name}", flush=True)
            else:
                print(
                    f"[{index}/{len(names)}] skipped {name}: HTTP {response.status_code}, "
                    f"{response.headers.get('content-type', 'unknown')}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[{index}/{len(names)}] error {name}: {exc}", flush=True)
    return downloaded


def parse_pdfs(
    pdf_dir: Path, markdown_dir: Path, force: bool = False, limit: int = 0
) -> int:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise SystemExit(
            "Docling is required: pip install -r backend/requirements.txt"
        ) from exc

    markdown_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    parsed = 0
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    for index, pdf in enumerate(pdfs, start=1):
        target = markdown_dir / f"{pdf.stem}.md"
        if target.exists() and target.stat().st_size > 100 and not force:
            print(f"[{index}] exists {target.name}", flush=True)
            continue
        try:
            result = converter.convert(str(pdf))
            markdown = result.document.export_to_markdown()
            if not markdown.strip():
                print(f"[{index}] empty {pdf.name}", flush=True)
                continue
            target.write_text(markdown, encoding="utf-8")
            parsed += 1
            print(
                f"[{index}] parsed {pdf.name} -> {len(markdown)} chars",
                flush=True,
            )
        except Exception as exc:
            print(f"[{index}] failed {pdf.name}: {exc}", flush=True)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit unique PDFs")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()

    dataset_path = BACKEND_DIR / "evaluation" / "datasets" / "financebench.json"
    samples = load_golden_dataset(str(dataset_path))
    pdf_dir = FINANCEBENCH_DIR / "pdfs"
    markdown_dir = FINANCEBENCH_DIR / "markdown"

    if not args.parse_only:
        download_pdfs(samples, pdf_dir, force=args.force, limit=args.limit)
    if not args.download_only:
        parse_pdfs(pdf_dir, markdown_dir, force=args.force, limit=args.limit)
    print(f"PDFs: {pdf_dir}")
    print(f"Markdown: {markdown_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
