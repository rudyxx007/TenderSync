"""Contextual, parent-child chunking for TenderSync.

Small child chunks are embedded (with a document/section context prefix) and
matched; their larger parent section is retrieved alongside them for the LLM.

The previous implementation crashed on any multi-line document: section offset
tracking called ``.start`` on ``str`` objects. It also used the section *number*
as the section title and truncated parent sections. This version parses
headings into a proper hierarchy, preserves Markdown pipe tables, and creates
synthetic parents when a PDF has no detectable headings.

Reference: Anthropic, "Introducing Contextual Retrieval" (2024) and Snowflake's
"Long-Context Isn't All You Need" finance-RAG study (2025). The
``MARKDOWN_CONTEXTUAL`` strategy follows the latter's Markdown/header-aware
configuration; the A/B harness supplies the same tokenizer and token budget to
all chunkers so the comparison is not confounded by chunk size.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SECTION_BASED = "section_based"
    MARKDOWN_CONTEXTUAL = "markdown_contextual"
    PARENT_CHILD = "parent_child"


@dataclass
class Chunk:
    content: str
    chunk_id: str
    parent_id: Optional[str] = None
    document_id: Optional[str] = None
    section_hierarchy: List[str] = field(default_factory=list)
    document_summary: Optional[str] = None
    section_summary: Optional[str] = None
    contextual_prefix: Optional[str] = None
    original_start: int = 0
    original_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def contextualized_content(self) -> str:
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.content}"
        return self.content

    @property
    def full_context(self) -> str:
        parts = []
        if self.document_summary:
            parts.append(f"DOCUMENT SUMMARY: {self.document_summary}")
        if self.section_hierarchy:
            parts.append("SECTION PATH: " + " > ".join(self.section_hierarchy))
        if self.section_summary:
            parts.append(f"SECTION SUMMARY: {self.section_summary}")
        parts.append(f"CHUNK CONTENT: {self.content}")
        return "\n\n".join(parts)


@dataclass
class DocumentStructure:
    document_id: str
    title: Optional[str] = None
    total_pages: int = 0
    total_characters: int = 0
    sections: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None

    def get_section_path(self, position: int) -> List[str]:
        path: List[str] = []
        for section in self.sections:
            if section.get("start", 0) <= position < section.get("end", 10**12):
                path.append(section.get("title", "Untitled"))
                for nested in section.get("subsections", []):
                    if nested.get("start", 0) <= position < nested.get("end", 10**12):
                        path.append(nested.get("title", "Untitled"))
        return path


class ContextualChunker:
    """Split a document into contextual child chunks with parent sections."""

    SECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"^\s*#{1,6}\s+(.+?)\s*$"), "markdown"),
        (re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+([A-Z][^\n]{2,120})$"), "numbered"),
        (re.compile(r"^\s*(?:SECTION|CLAUSE|ARTICLE|APPENDIX|SCHEDULE|PART)\s+([A-Z0-9][^\n]{0,120})$", re.I), "labelled"),
        (re.compile(r"^\s*([A-Z][A-Z0-9 &/,'-]{6,90})\s*$"), "uppercase"),
    ]

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        min_chunk_size: int = 80,
        max_chunk_size: int = 2400,
        preserve_tables: bool = True,
        add_context: bool = True,
        context_generator: Optional[Callable[..., str]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        if chunk_size < min_chunk_size:
            raise ValueError("chunk_size must be >= min_chunk_size")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_tables = preserve_tables
        self.add_context = add_context
        self.context_generator = context_generator
        # When supplied, chunk_size/chunk_overlap are interpreted as tokens
        # rather than characters. This keeps Markdown, section-aware and
        # Docling variants on the same context budget.
        self.token_counter = token_counter

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def chunk_with_context(
        self,
        document_text: str,
        document_id: str,
        document_title: Optional[str] = None,
        document_summary: Optional[str] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.PARENT_CHILD,
        section_hierarchy: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        if not document_text or not document_text.strip():
            return []

        structure = self.parse_structure(document_text, document_id, document_title)
        if section_hierarchy:
            structure.sections = section_hierarchy

        summary = document_summary or self._generate_document_summary(
            document_text, document_title
        )

        if chunking_strategy == ChunkingStrategy.PARENT_CHILD:
            chunks = self._parent_child_chunks(document_text, document_id, structure, summary)
        elif chunking_strategy == ChunkingStrategy.SECTION_BASED:
            chunks = self._section_chunks(document_text, document_id, structure, summary)
        elif chunking_strategy == ChunkingStrategy.MARKDOWN_CONTEXTUAL:
            chunks = self._markdown_contextual_chunks(
                document_text, document_id, structure, summary
            )
        else:
            chunks = [
                self._make_chunk(
                    text=content,
                    chunk_id=f"{document_id}_chunk_{i:04d}",
                    document_id=document_id,
                    start=start,
                    end=end,
                    summary=summary,
                    hierarchy=[document_title] if document_title else [],
                    is_parent=False,
                )
                for i, (start, end, content) in enumerate(
                    self._split_text(document_text, 0, self.chunk_size, self.chunk_overlap),
                    start=1,
                )
                if len(content) >= self.min_chunk_size
            ]

        if self.add_context:
            chunks = self._add_contextual_prefixes(chunks, document_text, summary)
        return chunks

    # ------------------------------------------------------------------
    # Structure parsing
    # ------------------------------------------------------------------
    def parse_structure(
        self, text: str, document_id: str, title: Optional[str]
    ) -> DocumentStructure:
        lines = text.splitlines(keepends=True)
        offsets: List[int] = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line)

        sections: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = []

        for line_no, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 140:
                continue

            match, level = self._match_heading(stripped)
            if not match:
                continue

            start = offsets[line_no]
            section = {
                "title": match,
                "level": level,
                "start": start,
                "end": len(text),
                "subsections": [],
            }

            # Close any section at the same or deeper level. A parent's range
            # must NOT be truncated when its first child starts, otherwise a
            # later sibling (e.g. 4.2 after 4.1) falls outside the parent and the
            # section path loses "4 Mandatory Requirements".
            while stack and stack[-1]["level"] >= level:
                stack.pop()["end"] = start

            if stack:
                stack[-1]["subsections"].append(section)
            else:
                if sections:
                    sections[-1]["end"] = start
                sections.append(section)
            stack.append(section)

        if not sections:
            sections = [{"title": title or "Document", "level": 1, "start": 0, "end": len(text), "subsections": []}]

        return DocumentStructure(
            document_id=document_id,
            title=title,
            total_characters=len(text),
            sections=sections,
        )

    def _match_heading(self, line: str) -> Tuple[Optional[str], int]:
        for pattern, kind in self.SECTION_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue

            if kind == "numbered":
                number, heading = match.group(1), match.group(2).strip()
                level = number.count(".") + 1
                return f"{number} {heading}".strip(), level
            if kind == "markdown":
                level = len(line) - len(line.lstrip("#"))
                return match.group(1).strip(), max(1, level)
            if kind == "labelled":
                # Normalise "SECTION 4.2 Mandatory Certifications"
                return " ".join(match.group(0).strip().split()), 1
            if kind == "uppercase":
                # Avoid treating every SHOUTING sentence as a heading.
                if len(line.split()) <= 12:
                    return " ".join(line.title().split()), 1
        return None, 0

    # ------------------------------------------------------------------
    # Parent-child construction
    # ------------------------------------------------------------------
    def _parent_child_chunks(
        self,
        text: str,
        document_id: str,
        structure: DocumentStructure,
        summary: str,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        section_counter = 0

        for section in structure.sections:
            section_start = section["start"]
            section_end = section["end"]
            section_text = text[section_start:section_end].strip()
            if not section_text:
                continue

            # Include nested subsection titles in the retrieval path.
            hierarchy = self._section_path(structure, section_start)
            title = section.get("title") or "Section"

            section_counter += 1
            parent_id = f"{document_id}_parent_{section_counter:04d}"

            # Parent keeps the whole section (up to a sane generation limit).
            parent_text = section_text[: self.max_chunk_size * 4]
            chunks.append(
                Chunk(
                    content=parent_text,
                    chunk_id=parent_id,
                    document_id=document_id,
                    section_hierarchy=hierarchy,
                    document_summary=summary,
                    section_summary=self._summarize_section(section_text, title),
                    original_start=section_start,
                    original_end=section_end,
                    metadata={"is_parent": True, "section": title},
                )
            )

            for i, (start, end, content) in enumerate(
                self._split_text(
                    section_text,
                    0,
                    self.chunk_size,
                    self.chunk_overlap,
                ),
                start=1,
            ):
                if len(content) < self.min_chunk_size:
                    continue
                # A parent spans several subsections, so each child needs the
                # path for its own absolute offset (e.g. 4 > 4.2), not the path
                # at the parent's start.
                child_hierarchy = self._section_path(structure, section_start + start)
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_id=f"{document_id}_child_{section_counter:04d}_{i:03d}",
                        parent_id=parent_id,
                        document_id=document_id,
                        section_hierarchy=child_hierarchy,
                        document_summary=summary,
                        section_summary=self._summarize_section(section_text, title),
                        original_start=section_start + start,
                        original_end=section_start + end,
                        metadata={
                            "is_parent": False,
                            "section": title,
                            "chunk_number": i,
                        },
                    )
                )

        # Ensure children exist even if every section was tiny.
        if not any(not c.metadata.get("is_parent") for c in chunks):
            chunks.extend(self._synthetic_parent_child(text, document_id, summary))
        return chunks

    def _synthetic_parent_child(self, text: str, document_id: str, summary: str) -> List[Chunk]:
        """Fallback when headings are absent: group children under a parent."""
        chunks: List[Chunk] = []
        children = list(self._split_text(text, 0, self.chunk_size, self.chunk_overlap))
        group_size = max(1, min(4, (self.max_chunk_size * 4) // max(self.chunk_size, 1)))

        for group_no in range(0, len(children), group_size):
            group = children[group_no: group_no + group_size]
            if not group:
                continue
            parent_id = f"{document_id}_parent_syn_{group_no // group_size + 1:04d}"
            parent_start = group[0][0]
            parent_end = group[-1][1]
            parent_text = text[parent_start:parent_end].strip()
            if not parent_text:
                continue

            chunks.append(
                Chunk(
                    content=parent_text,
                    chunk_id=parent_id,
                    document_id=document_id,
                    section_hierarchy=[f"Section {group_no // group_size + 1}"],
                    document_summary=summary,
                    section_summary=self._summarize_section(parent_text, "Section"),
                    original_start=parent_start,
                    original_end=parent_end,
                    metadata={"is_parent": True, "section": "Document body"},
                )
            )
            for i, (start, end, content) in enumerate(group, start=1):
                if len(content) < self.min_chunk_size:
                    continue
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_id=f"{parent_id}_child_{i:03d}",
                        parent_id=parent_id,
                        document_id=document_id,
                        section_hierarchy=[f"Section {group_no // group_size + 1}"],
                        document_summary=summary,
                        section_summary=self._summarize_section(parent_text, "Section"),
                        original_start=start,
                        original_end=end,
                        metadata={"is_parent": False, "chunk_number": i},
                    )
                )
        return chunks

    def _section_chunks(
        self,
        text: str,
        document_id: str,
        structure: DocumentStructure,
        summary: str,
    ) -> List[Chunk]:
        chunks = []
        for i, section in enumerate(structure.sections, start=1):
            content = text[section["start"]:section["end"]].strip()
            if len(content) < self.min_chunk_size:
                continue
            chunks.append(
                Chunk(
                    content=content[: self.max_chunk_size],
                    chunk_id=f"{document_id}_section_{i:04d}",
                    document_id=document_id,
                    section_hierarchy=[section.get("title", "Document")],
                    document_summary=summary,
                    section_summary=self._summarize_section(content, section.get("title", "")),
                    original_start=section["start"],
                    original_end=section["end"],
                    metadata={"is_parent": True, "section": section.get("title")},
                )
            )
        return chunks

    def _markdown_contextual_chunks(
        self,
        text: str,
        document_id: str,
        structure: DocumentStructure,
        summary: str,
    ) -> List[Chunk]:
        """Snowflake-style Markdown/header-aware chunks, without parent rows.

        The method is: preserve the parsed heading hierarchy, split each section
        recursively (paragraph/table/sentence boundaries) under one token
        budget, attach document and section context, and do not create parent
        rows. This mirrors the Markdown-aware configuration used in Snowflake's
        finance-RAG chunking study; it is not a byte-for-byte copy of Snowflake's
        closed-source ``SPLIT_TEXT_MARKDOWN_HEADER`` implementation.
        """
        chunks: List[Chunk] = []
        for section_no, section in enumerate(structure.sections, start=1):
            section_start = section["start"]
            section_end = section["end"]
            section_text = text[section_start:section_end].strip()
            if not section_text:
                continue

            title = section.get("title") or "Section"
            for i, (start, end, content) in enumerate(
                self._split_text(section_text, 0, self.chunk_size, self.chunk_overlap),
                start=1,
            ):
                if len(content) < self.min_chunk_size:
                    continue
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_id=f"{document_id}_md_{section_no:04d}_{i:03d}",
                        document_id=document_id,
                        section_hierarchy=self._section_path(
                            structure, section_start + start
                        ),
                        document_summary=summary,
                        section_summary=self._summarize_section(section_text, title),
                        original_start=section_start + start,
                        original_end=section_start + end,
                        metadata={
                            "is_parent": False,
                            "strategy": "markdown_contextual",
                            "snowflake_markdown_header": True,
                            "section": title,
                            "heading_path": self._section_path(
                                structure, section_start + start
                            ),
                            "chunk_number": i,
                        },
                    )
                )
        return chunks

    def _section_path(self, structure: DocumentStructure, position: int) -> List[str]:
        path: List[str] = []

        def walk(sections: List[Dict[str, Any]]) -> None:
            for section in sections:
                if section.get("start", 0) <= position < section.get("end", 10**12):
                    path.append(section.get("title", "Untitled"))
                    walk(section.get("subsections", []))
                    return

        walk(structure.sections)
        return path or [structure.title or "Document"]

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------
    def _split_text(
        self, text: str, base_offset: int, chunk_size: int, overlap: int
    ) -> List[Tuple[int, int, str]]:
        """Split text on paragraph/sentence boundaries, keeping tables intact.

        If ``token_counter`` is configured, ``chunk_size`` and ``overlap`` are
        token counts. Otherwise this preserves the historical character-based
        production behavior.
        """
        if not text:
            return []
        if self.token_counter is not None:
            return self._split_text_by_tokens(
                text, base_offset, chunk_size, overlap
            )

        pieces: List[Tuple[int, int, str]] = []
        start = 0
        length = len(text)
        step = max(1, chunk_size - overlap)

        while start < length:
            end = min(start + chunk_size, length)

            if end < length:
                end = self._snap_to_boundary(text, start, end)
                end = self._avoid_table_split(text, start, end)

            content = text[start:end].strip()
            if content:
                pieces.append((base_offset + start, base_offset + end, content))

            if end >= length:
                break
            start = max(start + step, end - overlap)
            if start >= length:
                break

        return pieces

    def _split_text_by_tokens(
        self, text: str, base_offset: int, chunk_size: int, overlap: int
    ) -> List[Tuple[int, int, str]]:
        """Token-budgeted split preserving original character offsets.

        A binary search finds the largest prefix that fits the budget, then the
        boundary is snapped to a paragraph/sentence and extended if it would
        split a Markdown pipe table. Tables may exceed the budget rather than be
        cut in half, matching the production chunker's table contract.
        """
        pieces: List[Tuple[int, int, str]] = []
        start = 0
        length = len(text)

        while start < length:
            end = self._max_token_end(text, start, chunk_size)
            if end < length:
                end = self._snap_to_boundary(text, start, end)
                end = self._avoid_table_split(text, start, end)
            if end <= start:
                end = min(length, start + 1)

            content = text[start:end].strip()
            if content:
                pieces.append((base_offset + start, base_offset + end, content))
            if end >= length:
                break

            span_tokens = max(1, int(self.token_counter(text[start:end])))
            chars_per_token = max(1.0, (end - start) / span_tokens)
            overlap_chars = min(
                max(0, int(overlap * chars_per_token)),
                max(0, (end - start) // 2),
            )
            next_start = max(start + 1, end - overlap_chars)
            start = next_start

        return pieces

    def _max_token_end(self, text: str, start: int, budget: int) -> int:
        """Largest end offset whose slice fits the token budget."""
        low = start + 1
        high = len(text)
        best = start + 1
        while low <= high:
            mid = (low + high) // 2
            if self.token_counter is not None and self.token_counter(text[start:mid]) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    @staticmethod
    def _snap_to_boundary(text: str, start: int, end: int) -> int:
        window_start = max(start + 1, end - 200)
        candidates = [
            text.rfind("\n\n", window_start, end),
            text.rfind(". ", window_start, end),
            text.rfind("\n", window_start, end),
        ]
        best = max(candidates)
        if best <= start:
            return end
        return best + (2 if text[best:best + 2] == "\n\n" else 1)

    def _avoid_table_split(self, text: str, start: int, end: int) -> int:
        if not self.preserve_tables:
            return end

        # If the split falls inside a Markdown pipe table, extend to its end.
        line_start = text.rfind("\n", start, end) + 1
        if "|" not in text[line_start:end]:
            return end

        next_newline = text.find("\n", end)
        while next_newline != -1:
            line_end = text.find("\n", next_newline + 1)
            line = text[next_newline + 1: line_end if line_end != -1 else len(text)]
            if "|" not in line:
                break
            next_newline = line_end

        table_end = len(text) if next_newline == -1 else next_newline
        if table_end - start <= self.max_chunk_size * 2:
            return table_end
        return end

    # ------------------------------------------------------------------
    # Context / summaries
    # ------------------------------------------------------------------
    def _generate_document_summary(self, text: str, title: Optional[str]) -> str:
        if callable(self.context_generator):
            try:
                summary = self.context_generator(document=text[:8000], chunk="", mode="document")
                if summary:
                    return str(summary)[:800]
            except Exception:
                pass

        first = re.sub(r"\s+", " ", text[:1200]).strip()
        prefix = f"{title}. " if title else ""
        return (prefix + first)[:800]

    @staticmethod
    def _summarize_section(text: str, title: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", compact[:1200])
        summary = " ".join(sentences[:2])[:500]
        return f"{title}: {summary}" if title else summary

    def _add_contextual_prefixes(
        self,
        chunks: List[Chunk],
        full_document: str,
        document_summary: str,
    ) -> List[Chunk]:
        for chunk in chunks:
            if callable(self.context_generator) and not chunk.metadata.get("is_parent"):
                try:
                    context = self.context_generator(
                        document=full_document[:12000],
                        chunk=chunk.content,
                        mode="chunk",
                    )
                    if context:
                        chunk.contextual_prefix = str(context).strip()
                        continue
                except Exception:
                    pass

            parts = []
            if chunk.section_hierarchy:
                parts.append("Section: " + " > ".join(chunk.section_hierarchy))
            if chunk.document_summary:
                parts.append("Document: " + chunk.document_summary[:200])
            if chunk.metadata.get("section"):
                parts.append(f"Part: {chunk.metadata['section']}")
            chunk.contextual_prefix = "\n".join(parts) if parts else None
        return chunks

    def _make_chunk(self, **kwargs) -> Chunk:
        return Chunk(**kwargs)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def extract_tables_as_chunks(
        self, document_text: str, table_markers: Optional[List[Tuple[int, int, str]]] = None
    ) -> List[Chunk]:
        markers = table_markers or self._detect_tables(document_text)
        chunks = []
        for i, (start, end, title) in enumerate(markers):
            table_text = document_text[start:end]
            if len(table_text) < self.min_chunk_size:
                continue
            chunks.append(
                Chunk(
                    content=table_text,
                    chunk_id=f"table_{i:04d}",
                    section_hierarchy=["Tables", title or f"Table {i + 1}"],
                    metadata={"is_parent": False, "is_table": True, "table_title": title},
                )
            )
        return chunks

    @staticmethod
    def _detect_tables(text: str) -> List[Tuple[int, int, str]]:
        lines = text.splitlines(keepends=True)
        offsets, cursor = [], 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line)

        tables: List[Tuple[int, int, str]] = []
        start_line: Optional[int] = None

        for i, line in enumerate(lines + [""]):
            is_table = "|" in line and line.count("|") >= 2
            if is_table and start_line is None:
                start_line = i
            elif not is_table and start_line is not None:
                end_line = i
                block = "".join(lines[start_line:end_line])
                if "|" in block:
                    tables.append(
                        (
                            offsets[start_line],
                            offsets[end_line] if end_line < len(offsets) else len(text),
                            f"Table {len(tables) + 1}",
                        )
                    )
                start_line = None
        return tables


class ParentChildIndex:
    """In-memory index used by tests and small offline runs."""

    def __init__(self):
        self.parents: Dict[str, Chunk] = {}
        self.children: Dict[str, List[str]] = {}
        self.all_chunks: Dict[str, Chunk] = {}

    def add_chunk(self, chunk: Chunk) -> None:
        self.all_chunks[chunk.chunk_id] = chunk
        if chunk.parent_id:
            self.children.setdefault(chunk.parent_id, []).append(chunk.chunk_id)
        else:
            self.parents[chunk.chunk_id] = chunk

    def get_parent(self, child_id: str) -> Optional[Chunk]:
        chunk = self.all_chunks.get(child_id)
        return self.all_chunks.get(chunk.parent_id) if chunk and chunk.parent_id else None

    def get_children(self, parent_id: str) -> List[Chunk]:
        return [self.all_chunks[cid] for cid in self.children.get(parent_id, []) if cid in self.all_chunks]

    def expand_to_parents(self, chunk_ids: List[str]) -> List[Chunk]:
        seen, parents = set(), []
        for cid in chunk_ids:
            parent = self.get_parent(cid)
            if parent and parent.chunk_id not in seen:
                seen.add(parent.chunk_id)
                parents.append(parent)
        return parents
