"""
Contextual Retrieval Chunker for TenderSync Enterprise RAG Pipeline.

Implements Anthropic's Contextual Retrieval method:
1. Generate document-level summaries
2. Generate section-level summaries  
3. For each chunk, prepend contextual information before embedding
4. Maintain parent-child relationships for hierarchical retrieval

This dramatically improves retrieval accuracy by ensuring chunks
retain context about their position in the document hierarchy.

References:
- https://www.anthropic.com/research/contextual-retrieval
- Reduces retrieval failures by 49% (67% with reranking)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ChunkingStrategy(Enum):
    """Supported chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SECTION_BASED = "section_based"
    PARENT_CHILD = "parent_child"


@dataclass
class Chunk:
    """
    Represents a text chunk with contextual metadata.
    
    Attributes:
        content: The actual text content of the chunk
        chunk_id: Unique identifier for this chunk
        parent_id: ID of parent chunk (for hierarchical retrieval)
        document_id: ID of the source document
        section_hierarchy: List of section titles from document root to this chunk
        document_summary: Summary of the entire document
        section_summary: Summary of the section containing this chunk
        contextual_prefix: Generated context prepended before embedding
        original_start: Character offset in original document
        original_end: Character offset in original document
        metadata: Additional metadata (page numbers, etc.)
    """
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
        """Return content with contextual prefix prepended."""
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.content}"
        return self.content
    
    @property
    def full_context(self) -> str:
        """Return complete context for the chunk."""
        parts = []
        
        if self.document_summary:
            parts.append(f"DOCUMENT SUMMARY: {self.document_summary}")
        
        if self.section_hierarchy:
            hierarchy = " > ".join(self.section_hierarchy)
            parts.append(f"SECTION PATH: {hierarchy}")
        
        if self.section_summary:
            parts.append(f"SECTION SUMMARY: {self.section_summary}")
        
        parts.append(f"CHUNK CONTENT: {self.content}")
        
        return "\n\n".join(parts)


@dataclass
class DocumentStructure:
    """
    Represents the hierarchical structure of a document.
    
    Used for maintaining section relationships in parent-child indexing.
    """
    document_id: str
    title: Optional[str] = None
    total_pages: int = 0
    total_characters: int = 0
    sections: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    
    def get_section_path(self, chunk_start: int) -> List[str]:
        """Get the section hierarchy path for a chunk at given position."""
        path = []
        for section in self.sections:
            if section.get("start", 0) <= chunk_start < section.get("end", float("inf")):
                path.append(section.get("title", "Unknown Section"))
                # Check for nested sections
                for nested in section.get("subsections", []):
                    if nested.get("start", 0) <= chunk_start < nested.get("end", float("inf")):
                        path.append(nested.get("title", "Unknown Subsection"))
        return path


class ContextualChunker:
    """
    Advanced chunker implementing Contextual Retrieval.
    
    Creates chunks with prepended contextual information to improve
    retrieval accuracy. Supports both flat and parent-child chunking.
    
    Example:
        chunker = ContextualChunker()
        chunks = chunker.chunk_with_context(
            document_text=full_pdf_text,
            document_id="doc_123",
            document_title="RFP Tender Document",
            chunking_strategy=ChunkingStrategy.PARENT_CHILD
        )
    """
    
    # Section headers patterns (common in RFP/legal documents)
    SECTION_PATTERNS = [
        # Numbered sections: 1., 1.1, 1.1.1, etc.
        (r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{5,100})$', "numbered"),
        # All-caps headers
        (r'^\s*([A-Z][A-Z\s]{10,80})\s*$', "uppercase"),
        # Markdown headers
        (r'^#{1,6}\s+(.+)$', "markdown"),
        # Common RFP sections
        (r'^\s*(?:1\.\s*)?(?:SCOPE|OFFEROR|BACKGROUND|REQUIREMENTS|DELIVERABLES|SCHEDULE|BUDGET|COMPLY|EVALUATION|CONTACT|SECTION|CLAUSE|ARTICLE|APPENDIX)\s*[.:]?\s*(.+)$', "rfp_header"),
    ]
    
    # Context generation prompt for LLM
    CONTEXT_PROMPT = """\
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document 
for the purposes of improving search retrieval of the chunk. Answer only with the 
succinct context and nothing else.

The context should include:
1. What section this chunk belongs to
2. What this chunk is about in the context of the full document
3. Any relevant identifiers (clause numbers, section titles, etc.)
"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000,
        preserve_tables: bool = True,
        add_context: bool = True,
        context_generator=None,  # Optional LLM client for context generation
    ):
        """
        Initialize ContextualChunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size before forced split
            preserve_tables: Try to keep tables intact in chunks
            add_context: Whether to add contextual prefixes
            context_generator: Optional callable for generating context (LLM)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_tables = preserve_tables
        self.add_context = add_context
        self.context_generator = context_generator
    
    def chunk_with_context(
        self,
        document_text: str,
        document_id: str,
        document_title: Optional[str] = None,
        document_summary: Optional[str] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.PARENT_CHILD,
        section_hierarchy: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        """
        Chunk a document with contextual retrieval preparation.
        
        Args:
            document_text: Full text of the document
            document_id: Unique identifier for the document
            document_title: Title of the document
            document_summary: Pre-computed document summary (or None to auto-generate)
            chunking_strategy: Strategy for creating chunks
            section_hierarchy: Pre-parsed section structure
            
        Returns:
            List of Chunk objects with contextual information
        """
        # Parse document structure if needed
        if chunking_strategy in (ChunkingStrategy.PARENT_CHILD, ChunkingStrategy.SECTION_BASED):
            structure = self._parse_document_structure(
                document_text, document_id, document_title
            )
            if section_hierarchy:
                structure.sections = section_hierarchy
        else:
            structure = None
        
        # Generate document summary if not provided
        if not document_summary and self.context_generator:
            document_summary = self._generate_document_summary(
                document_text, document_title
            )
        elif not document_summary:
            # Simple fallback: use first 500 chars
            document_summary = document_text[:500].replace("\n", " ").strip()
        
        # Create chunks based on strategy
        if chunking_strategy == ChunkingStrategy.PARENT_CHILD:
            chunks = self._create_parent_child_chunks(
                document_text, document_id, structure, document_summary
            )
        elif chunking_strategy == ChunkingStrategy.SECTION_BASED:
            chunks = self._create_section_based_chunks(
                document_text, document_id, structure, document_summary
            )
        elif chunking_strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._create_semantic_chunks(
                document_text, document_id, document_summary
            )
        else:  # FIXED_SIZE
            chunks = self._create_fixed_size_chunks(
                document_text, document_id, document_summary
            )
        
        # Add contextual prefixes
        if self.add_context:
            chunks = self._add_contextual_prefixes(
                chunks, document_text, document_summary
            )
        
        return chunks
    
    def _parse_document_structure(
        self, text: str, document_id: str, title: Optional[str]
    ) -> DocumentStructure:
        """Parse document to extract section hierarchy."""
        sections = []
        current_section = None
        lines = text.split("\n")
        
        for i, line in enumerate(lines):
            line_start = text.find(line) if i == 0 else (
                lines[i-1].start + len(lines[i-1]) + 1
            )
            
            for pattern, ptype in self.SECTION_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    section_title = match.group(1).strip() if ptype == "numbered" else match.group(1).strip()
                    
                    if current_section:
                        current_section["end"] = line_start
                    
                    current_section = {
                        "title": section_title,
                        "level": self._determine_section_level(ptype, match),
                        "start": line_start,
                        "end": len(text),
                        "subsections": [],
                    }
                    sections.append(current_section)
                    break
        
        if current_section:
            current_section["end"] = len(text)
        
        return DocumentStructure(
            document_id=document_id,
            title=title,
            total_pages=0,  # Would need page info from PDF parser
            total_characters=len(text),
            sections=sections,
            summary=None,
        )
    
    def _determine_section_level(self, pattern_type: str, match) -> int:
        """Determine nesting level of a section."""
        if pattern_type == "numbered":
            number = match.group(1)
            return len(number.split("."))
        elif pattern_type == "markdown":
            hashes = match.group(0).count("#")
            return hashes
        return 1
    
    def _create_fixed_size_chunks(
        self, text: str, document_id: str, document_summary: str
    ) -> List[Chunk]:
        """Create fixed-size overlapping chunks."""
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                for delimiter in [". ", "? ", "! ", "\n\n"]:
                    delim_pos = text.find(delimiter, start + self.chunk_size // 2, end + 50)
                    if delim_pos != -1 and delim_pos < end + 50:
                        end = delim_pos + len(delimiter)
                        break
            
            content = text[start:end].strip()
            
            if len(content) >= self.min_chunk_size:
                chunk_num += 1
                chunks.append(Chunk(
                    content=content,
                    chunk_id=f"{document_id}_chunk_{chunk_num:04d}",
                    document_id=document_id,
                    original_start=start,
                    original_end=end,
                    metadata={"chunk_number": chunk_num},
                ))
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _create_parent_child_chunks(
        self,
        text: str,
        document_id: str,
        structure: Optional[DocumentStructure],
        document_summary: str,
    ) -> List[Chunk]:
        """
        Create parent-child chunks for hierarchical retrieval.
        
        Creates small semantic chunks (children) linked to larger
        parent sections for contextual retrieval.
        """
        chunks = []
        chunk_num = 0
        
        if structure and structure.sections:
            # Create chunks per section
            for section in structure.sections:
                section_text = text[section["start"]:section["end"]]
                section_title = section.get("title", "Unknown Section")
                
                # Create parent chunk for the whole section
                if len(section_text) <= self.max_chunk_size * 3:
                    chunk_num += 1
                    parent_id = f"{document_id}_parent_{chunk_num:04d}"
                    
                    parent_chunk = Chunk(
                        content=section_text[:self.max_chunk_size * 2],
                        chunk_id=parent_id,
                        document_id=document_id,
                        parent_id=None,
                        section_hierarchy=[section_title],
                        document_summary=document_summary,
                        section_summary=self._summarize_section(section_text, section_title),
                        original_start=section["start"],
                        original_end=section["end"],
                        metadata={"is_parent": True, "section": section_title},
                    )
                    chunks.append(parent_chunk)
                    
                    # Create child chunks within this section
                    child_chunks = self._create_fixed_size_chunks(
                        section_text, f"{document_id}_child", document_summary
                    )
                    for child in child_chunks:
                        child.parent_id = parent_id
                        child.section_hierarchy = [section_title]
                        child.document_summary = document_summary
                        child.section_summary = parent_chunk.section_summary
                        chunks.append(child)
                else:
                    # Section too large, just create regular chunks
                    section_chunks = self._create_fixed_size_chunks(
                        section_text, document_id, document_summary
                    )
                    for chunk in section_chunks:
                        chunk.section_hierarchy = [section_title]
                        chunk.document_summary = document_summary
                        chunk.section_summary = self._summarize_section(
                            section_text[:5000], section_title
                        )
                    chunks.extend(section_chunks)
        else:
            # No structure, use fixed-size chunking
            chunks = self._create_fixed_size_chunks(text, document_id, document_summary)
        
        return chunks
    
    def _create_section_based_chunks(
        self,
        text: str,
        document_id: str,
        structure: DocumentStructure,
        document_summary: str,
    ) -> List[Chunk]:
        """Create one chunk per section."""
        chunks = []
        chunk_num = 0
        
        for section in structure.sections:
            section_text = text[section["start"]:section["end"]].strip()
            
            if len(section_text) >= self.min_chunk_size:
                chunk_num += 1
                chunk = Chunk(
                    content=section_text[:self.max_chunk_size],
                    chunk_id=f"{document_id}_section_{chunk_num:04d}",
                    document_id=document_id,
                    section_hierarchy=[section.get("title", "Unknown")],
                    document_summary=document_summary,
                    section_summary=self._summarize_section(section_text, section.get("title", "")),
                    original_start=section["start"],
                    original_end=section["end"],
                    metadata={"is_section": True, "section_level": section.get("level", 1)},
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_semantic_chunks(
        self, text: str, document_id: str, document_summary: str
    ) -> List[Chunk]:
        """Create semantically coherent chunks (sentence-boundary aware)."""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If paragraph alone exceeds chunk size, split it
            if para_size > self.chunk_size:
                if current_chunk:
                    chunk_num += 1
                    chunks.append(self._finalize_chunk(
                        current_chunk, document_id, chunk_num, document_summary
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Split long paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if current_size + len(sent) > self.chunk_size and current_chunk:
                        chunk_num += 1
                        chunks.append(self._finalize_chunk(
                            current_chunk, document_id, chunk_num, document_summary
                        ))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sent)
                    current_size += len(sent)
            
            elif current_size + para_size > self.chunk_size and current_chunk:
                # Start new chunk
                chunk_num += 1
                chunks.append(self._finalize_chunk(
                    current_chunk, document_id, chunk_num, document_summary
                ))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_num += 1
            chunks.append(self._finalize_chunk(
                current_chunk, document_id, chunk_num, document_summary
            ))
        
        return chunks
    
    def _finalize_chunk(
        self, content_parts: List[str], document_id: str, 
        chunk_num: int, document_summary: str
    ) -> Chunk:
        """Finalize a chunk from accumulated content parts."""
        content = "\n\n".join(content_parts).strip()
        return Chunk(
            content=content,
            chunk_id=f"{document_id}_chunk_{chunk_num:04d}",
            document_id=document_id,
            document_summary=document_summary,
            original_start=0,  # Would need tracking
            original_end=len(content),
            metadata={"chunk_number": chunk_num},
        )
    
    def _add_contextual_prefixes(
        self,
        chunks: List[Chunk],
        full_document: str,
        document_summary: str,
    ) -> List[Chunk]:
        """Add contextual prefixes to chunks for improved retrieval."""
        for chunk in chunks:
            if self.context_generator and callable(self.context_generator):
                # Use LLM to generate context
                context = self.context_generator(
                    document=full_document,
                    chunk=chunk.content,
                )
                chunk.contextual_prefix = context
            else:
                # Generate simple context from metadata
                context_parts = []
                
                if chunk.document_summary:
                    # Truncate summary for prefix
                    summary = chunk.document_summary[:200]
                    context_parts.append(f"This content is from a document about: {summary}")
                
                if chunk.section_hierarchy:
                    path = " > ".join(chunk.section_hierarchy)
                    context_parts.append(f"Section: {path}")
                
                if chunk.metadata.get("section"):
                    context_parts.append(f"Within section: {chunk.metadata['section']}")
                
                chunk.contextual_prefix = "\n".join(context_parts) if context_parts else None
        
        return chunks
    
    def _generate_document_summary(
        self, text: str, title: Optional[str]
    ) -> str:
        """Generate a brief document summary."""
        if self.context_generator and callable(self.context_generator):
            # Would call LLM with summary prompt
            pass
        
        # Simple extractive summary
        # Take first few paragraphs/headings
        lines = text.split("\n")
        summary_lines = []
        
        for line in lines[:20]:  # First 20 lines
            line = line.strip()
            if line and len(line) > 10:
                if not line.isupper() or len(line) < 50:
                    summary_lines.append(line[:150])
        
        title_part = f"{title}: " if title else ""
        summary = title_part + " ".join(summary_lines[:5])
        
        return summary[:500]
    
    def _summarize_section(self, text: str, title: str) -> str:
        """Create a brief summary of a section."""
        # First sentence or two
        sentences = re.split(r'(?<=[.!?])\s+', text[:1000])
        summary = " ".join(sentences[:2])
        
        if title:
            return f"{title}: {summary}"
        return summary
    
    def extract_tables_as_chunks(
        self, 
        document_text: str, 
        table_markers: Optional[List[Tuple[int, int, str]]] = None
    ) -> List[Chunk]:
        """
        Extract tables as separate chunks for better table retrieval.
        
        Tables are often critical in financial/legal documents and
        should be indexed separately for precise retrieval.
        
        Args:
            document_text: Full document text
            table_markers: List of (start, end, table_title) tuples
                           If None, will attempt to detect tables
            
        Returns:
            List of Chunk objects for tables
        """
        chunks = []
        
        if not table_markers:
            # Attempt basic table detection
            table_markers = self._detect_tables(document_text)
        
        for i, (start, end, title) in enumerate(table_markers):
            table_text = document_text[start:end]
            
            if len(table_text) >= self.min_chunk_size:
                chunk = Chunk(
                    content=table_text,
                    chunk_id=f"table_{i:04d}",
                    document_id="",
                    section_hierarchy=["Tables", title or f"Table {i+1}"],
                    metadata={
                        "is_table": True,
                        "table_number": i + 1,
                        "table_title": title,
                    },
                )
                chunks.append(chunk)
        
        return chunks
    
    def _detect_tables(self, text: str) -> List[Tuple[int, int, str]]:
        """Basic table detection in text."""
        tables = []
        
        # Look for table-like patterns
        lines = text.split("\n")
        in_table = False
        table_start = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            # Table indicators
            has_pipes = "|" in line
            has_dashes = re.search(r'[-]{3,}', line)
            has_alignment = re.search(r'[:|-]{3,}\s*:?', line)
            
            if has_pipes or (has_dashes and has_alignment):
                if not in_table:
                    in_table = True
                    table_start = text.find(line)
                    table_lines = [line]
                else:
                    table_lines.append(line)
            elif in_table and line.strip() == "":
                # End of table
                table_text = "\n".join(table_lines)
                if len(table_text) > 50:  # Minimum table size
                    title = f"Table {len(tables) + 1}"
                    tables.append((table_start, table_start + len(table_text), title))
                in_table = False
                table_lines = []
            elif in_table:
                table_lines.append(line)
        
        # Handle table at end of document
        if in_table and table_lines:
            table_text = "\n".join(table_lines)
            if len(table_text) > 50:
                title = f"Table {len(tables) + 1}"
                tables.append((table_start, table_start + len(table_text), title))
        
        return tables


class ParentChildIndex:
    """
    Manages parent-child relationships for hierarchical retrieval.
    
    Stores parent chunks with their children for:
    1. Retrieving small semantic chunks (children)
    2. Expanding to full context (parents) when needed
    
    This enables retrieving precise information while maintaining
    the broader document context.
    """
    
    def __init__(self):
        self.parents: Dict[str, Chunk] = {}
        self.children: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.all_chunks: Dict[str, Chunk] = {}
    
    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the index."""
        self.all_chunks[chunk.chunk_id] = chunk
        
        if chunk.parent_id:
            # This is a child chunk
            if chunk.parent_id not in self.children:
                self.children[chunk.parent_id] = []
            self.children[chunk.parent_id].append(chunk.chunk_id)
        else:
            # This is a parent chunk
            self.parents[chunk.chunk_id] = chunk
    
    def get_parent(self, child_id: str) -> Optional[Chunk]:
        """Get parent chunk for a child chunk."""
        chunk = self.all_chunks.get(child_id)
        if chunk and chunk.parent_id:
            return self.all_chunks.get(chunk.parent_id)
        return None
    
    def get_children(self, parent_id: str) -> List[Chunk]:
        """Get all child chunks for a parent."""
        child_ids = self.children.get(parent_id, [])
        return [self.all_chunks[cid] for cid in child_ids if cid in self.all_chunks]
    
    def expand_to_parents(self, chunk_ids: List[str]) -> List[Chunk]:
        """Expand a list of child chunks to include their parents."""
        parent_ids = set()
        for cid in chunk_ids:
            parent = self.get_parent(cid)
            if parent:
                parent_ids.add(parent.chunk_id)
        
        return [self.all_chunks[pid] for pid in parent_ids if pid in self.all_chunks]
    
    def get_unique_parents(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get unique parent chunks for a set of child chunks."""
        seen = set()
        result = []
        
        for cid in chunk_ids:
            parent = self.get_parent(cid)
            if parent and parent.chunk_id not in seen:
                seen.add(parent.chunk_id)
                result.append(parent)
        
        return result
