"""Optional Docling HybridChunker adapter.

The production upload path currently keeps only Docling's Markdown export.
HybridChunker needs the richer ``DoclingDocument`` (layout, headings, table
structure), so this adapter is deliberately separate and lazy. The A/B runner
can convert benchmark Markdown into a DoclingDocument without changing the
production parsing path.

HybridChunker is included in the installed ``docling`` package (not a separate
model download). The first call may download the tokenizer used for token
budgeting.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, List, Optional

from services.contextual_chunker import Chunk


class DoclingChunkerUnavailable(RuntimeError):
    """Raised when Docling or its chunking extra is unavailable."""


def docling_document_from_markdown(text: str, filename: str = "document.md") -> Any:
    """Parse Markdown into a DoclingDocument for HybridChunker.

    This is for benchmark/A-B use. For a real PDF, pass the PDF path directly
    to ``document_from_path`` so layout and table structure are preserved.
    """
    if not text or not text.strip():
        raise ValueError("Cannot build a DoclingDocument from empty text.")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DoclingChunkerUnavailable(
            "Docling is not installed. Install backend requirements or `pip install docling`."
        ) from exc

    suffix = Path(filename).suffix or ".md"
    with tempfile.NamedTemporaryFile(
        "w", suffix=suffix, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
        path = handle.name
    try:
        return DocumentConverter().convert(path).document
    finally:
        Path(path).unlink(missing_ok=True)


def document_from_path(path: str) -> Any:
    """Convert a PDF/image/HTML/Markdown path with Docling."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover
        raise DoclingChunkerUnavailable(
            "Docling is not installed. Install backend requirements or `pip install docling`."
        ) from exc
    return DocumentConverter().convert(path).document


class DoclingHybridChunker:
    """Token-aware Docling chunks with heading and table context."""

    def __init__(
        self,
        max_tokens: int = 512,
        merge_peers: bool = True,
        repeat_table_header: bool = True,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Create a Docling HybridChunker.

        ``tokenizer`` may be a Hugging Face tokenizer (for example the Qwen3
        embedding tokenizer). Supplying it makes Docling and the Markdown
        chunker use the exact same token budget/tokenizer in A/B runs.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        try:
            from docling.chunking import HybridChunker
        except ImportError as exc:  # pragma: no cover
            raise DoclingChunkerUnavailable(
                "Docling HybridChunker is unavailable. Install `docling` with "
                "its chunking dependencies."
            ) from exc

        kwargs = {
            "merge_peers": merge_peers,
            "repeat_table_header": repeat_table_header,
        }
        if tokenizer is not None:
            try:
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )
            except ImportError as exc:  # pragma: no cover
                raise DoclingChunkerUnavailable(
                    "Docling's HuggingFaceTokenizer adapter is unavailable."
                ) from exc
            kwargs["tokenizer"] = HuggingFaceTokenizer(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
        else:
            # Legacy Docling API; emits a deprecation warning on newer releases.
            kwargs["max_tokens"] = max_tokens
        self._chunker = HybridChunker(**kwargs)
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: Any,
        document_id: str,
        document_title: Optional[str] = None,
    ) -> List[Chunk]:
        """Convert Docling ``DocChunk`` objects to TenderSync ``Chunk`` objects."""
        if document is None:
            return []
        if not hasattr(document, "export_to_markdown"):
            raise TypeError("DoclingHybridChunker.chunk expects a DoclingDocument")

        chunks: List[Chunk] = []
        for index, doc_chunk in enumerate(self._chunker.chunk(dl_doc=document)):
            text = (getattr(doc_chunk, "text", "") or "").strip()
            if not text:
                continue
            meta = getattr(doc_chunk, "meta", None)
            headings = list(getattr(meta, "headings", None) or [])
            captions = list(getattr(meta, "captions", None) or [])
            chunk = Chunk(
                content=text,
                chunk_id=f"{document_id}_docling_{index:04d}",
                document_id=document_id,
                section_hierarchy=headings or ([document_title] if document_title else []),
                document_summary=document_title,
                section_summary=" > ".join(headings) if headings else None,
                metadata={
                    "is_parent": False,
                    "strategy": "docling_hybrid",
                    "docling_index": index,
                    "captions": captions,
                    "max_tokens": self.max_tokens,
                },
            )
            chunks.append(chunk)
        return chunks
