"""Production RAG service for TenderSync.

Three responsibilities, kept out of the HTTP layer:

1. ``ingest_document``: contextual parent/child chunking, embedding, and
   org-scoped storage in Supabase ``documents``.
2. ``retrieve``: hybrid lexical + vector retrieval with optional reranking and
   parent-section expansion, always scoped to one organisation and optionally
   one uploaded document.
3. ``answer_query``: grounded generation with numbered citations and a
   post-generation citation/NLI-style verification pass.

The previous pipeline embedded chunks, called a non-existent RPC
(``match_documents_org`` was never defined), swallowed the error, and fell back
to the first eight rows for the whole organisation. That is why evaluation
returned zeros and why one tender could be analysed using another tender's
text. This module is the single retrieval path used by both the API and the
evaluation harness.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from services.citation_verifier import CitationVerifier
from services.contextual_chunker import Chunk, ChunkingStrategy, ContextualChunker
from services.hybrid_retriever import (
    HybridRetriever,
    HybridSearchConfig,
    RetrievedDocument,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "voyage-4-large"
DEFAULT_RERANKER_MODEL = "voyage-rerank-2"

ANSWER_SYSTEM_PROMPT = """You are TenderSync's tender analyst.

Answer the question using ONLY the numbered CONTEXT passages supplied by the user.

Rules:
1. Every factual statement must cite the passage(s) that support it, e.g. [1] or [2][3].
2. Never invent facts, figures, clause numbers, certifications or dates.
3. Preserve exact identifiers, currency amounts and dates as written.
4. If the context does not contain the answer, reply exactly:
   "Not found in the provided document."
5. Be concise. Prefer the document's wording for legal or financial terms."""


class RAGConfigurationError(RuntimeError):
    """Raised when required credentials or schema objects are missing."""


class RAGStorageError(RuntimeError):
    """Raised when chunk storage fails (usually an unapplied migration)."""


@dataclass
class IngestResult:
    document_id: str
    parent_count: int
    child_count: int
    embedded: bool
    embedding_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "embedded": self.embedded,
            "embedding_error": self.embedding_error,
        }


# ----------------------------------------------------------------------
# Provider factories
# ----------------------------------------------------------------------
def build_embedder(model: Optional[str] = None, prefer_voyage: bool = True):
    """Return a 1024-dim embedder.

    Voyage AI is used when ``VOYAGE_API_KEY`` is set; otherwise Hugging Face
    BGE-M3, which needs no paid account. Both write to ``vector(1024)``.
    """
    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if prefer_voyage and os.getenv("VOYAGE_API_KEY"):
        try:
            from services.voyage_embeddings import VoyageEmbeddingService

            voyage_model = model if model.startswith("voyage") else DEFAULT_EMBEDDING_MODEL
            service = VoyageEmbeddingService(model=voyage_model)
            if service.dimension != 1024:
                raise RAGConfigurationError(
                    f"{voyage_model} produces {service.dimension}-dim vectors; "
                    "TenderSync's schema requires 1024."
                )
            return service
        except RAGConfigurationError:
            raise
        except Exception as exc:
            logger.warning("Voyage embeddings unavailable (%s); using Hugging Face.", exc)

    from embedding_service import HuggingFaceEmbeddingService

    return HuggingFaceEmbeddingService()


def build_reranker(model: Optional[str] = None):
    """Return a Voyage reranker, or None (RRF order is kept)."""
    model = model or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    if model.strip().lower() in {"none", "off", "false", "disabled", ""}:
        return None
    if not os.getenv("VOYAGE_API_KEY"):
        logger.info("VOYAGE_API_KEY not set; skipping cross-encoder reranking.")
        return None

    try:
        from services.voyage_embeddings import VoyageReranker

        return VoyageReranker(model=model)
    except Exception as exc:
        logger.warning("Reranker unavailable (%s); keeping RRF order.", exc)
        return None


def build_groq_client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RAGConfigurationError("GROQ_API_KEY is required for answer generation.")
    return Groq(api_key=api_key)


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
def ingest_document(
    supabase_client,
    org_id: str,
    user_id: str,
    filename: str,
    document_text: str,
    embedder=None,
    chunker: Optional[ContextualChunker] = None,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.PARENT_CHILD,
) -> IngestResult:
    """Chunk, embed and persist one uploaded document.

    Parent rows are stored without embeddings; child rows carry the embedding
    of their contextualised text and a ``parent_id`` pointer. Retrieval matches
    children and expands to parents. Every row is tagged with ``org_id`` and
    the same ``document_id``, so a later analysis cannot read another tenant's
    or another tender's content.
    """
    if not supabase_client:
        raise RAGStorageError("Supabase client is required for ingestion.")
    if not org_id:
        raise RAGStorageError("org_id is required for multi-tenant isolation.")
    if not document_text or not document_text.strip():
        raise RAGStorageError(f"No text extracted from {filename}.")

    embedder = embedder if embedder is not None else build_embedder()
    chunker = chunker or ContextualChunker()
    document_id = str(uuid.uuid4())

    chunks = chunker.chunk_with_context(
        document_text=document_text,
        document_id=document_id,
        document_title=filename,
        chunking_strategy=chunking_strategy,
    )
    parents = [c for c in chunks if c.metadata.get("is_parent")]
    children = [c for c in chunks if not c.metadata.get("is_parent")]

    if not children:
        # A very short document may only produce a parent; index it as a child.
        for parent in parents:
            parent.metadata["is_parent"] = False
        children, parents = parents, []

    embeddings: List[Optional[List[float]]] = [None] * len(children)
    embedding_error: Optional[str] = None
    if embedder and children:
        try:
            vectors = embedder.embed_documents(
                [child.contextualized_content for child in children]
            )
            if len(vectors) != len(children):
                raise ValueError(
                    f"Embedding provider returned {len(vectors)} vectors for "
                    f"{len(children)} chunks."
                )
            embeddings = vectors
        except Exception as exc:
            embedding_error = str(exc)
            logger.error("Embedding failed for %s: %s", filename, exc)

    # 1) Parents first so children can point at real row ids.
    parent_id_map: Dict[str, str] = {}
    parent_rows = [
        {
            "content": parent.content,
            "embedding": None,
            "metadata": {
                "filename": filename,
                "is_parent": True,
                "section": parent.metadata.get("section"),
            },
            "user_id": user_id,
            "org_id": org_id,
            "document_id": document_id,
            "is_parent": True,
            "chunk_index": index,
            "section_path": parent.section_hierarchy,
            "contextual_prefix": parent.contextual_prefix,
            "filename": filename,
        }
        for index, parent in enumerate(parents)
    ]

    if parent_rows:
        inserted = _insert_rows(supabase_client, parent_rows, context=f"parents of {filename}")
        parent_id_map = _map_parent_ids(inserted, parents, supabase_client, document_id, org_id)

    # 2) Children with embeddings.
    child_rows = []
    for index, (child, embedding) in enumerate(zip(children, embeddings)):
        child_rows.append(
            {
                "content": child.content,
                "embedding": embedding,
                "metadata": {
                    "filename": filename,
                    "is_parent": False,
                    "section": child.metadata.get("section"),
                    "chunk_index": index,
                },
                "user_id": user_id,
                "org_id": org_id,
                "document_id": document_id,
                "parent_id": parent_id_map.get(child.parent_id or ""),
                "is_parent": False,
                "chunk_index": index,
                "section_path": child.section_hierarchy,
                "contextual_prefix": child.contextual_prefix,
                "filename": filename,
            }
        )

    if child_rows:
        _insert_rows(supabase_client, child_rows, context=f"chunks of {filename}")

    return IngestResult(
        document_id=document_id,
        parent_count=len(parents),
        child_count=len(children),
        embedded=embedding_error is None and bool(children),
        embedding_error=embedding_error,
    )


def _insert_rows(supabase_client, rows: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
    inserted: List[Dict[str, Any]] = []
    batch_size = 50
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        try:
            response = supabase_client.table("documents").insert(batch).execute()
        except Exception as exc:
            message = str(exc)
            if any(token in message.lower() for token in ("column", "schema cache", "does not exist")):
                raise RAGStorageError(
                    "Storing document chunks failed. Apply "
                    "backend/supabase/migrations/002_hybrid_search.sql so the "
                    f"documents table has the RAG columns. Original error: {message}"
                ) from exc
            raise RAGStorageError(f"Failed to store {context}: {message}") from exc
        inserted.extend(list(response.data or []))
    return inserted


def _map_parent_ids(
    inserted: Sequence[Dict[str, Any]],
    parents: Sequence[Chunk],
    supabase_client,
    document_id: str,
    org_id: str,
) -> Dict[str, int]:
    if len(inserted) == len(parents) and all(row.get("id") is not None for row in inserted):
        return {parent.chunk_id: row["id"] for parent, row in zip(parents, inserted)}

    # Some PostgREST configurations return no representation on insert.
    try:
        response = (
            supabase_client.table("documents")
            .select("id,chunk_index")
            .eq("org_id", org_id)
            .eq("document_id", document_id)
            .eq("is_parent", True)
            .order("chunk_index")
            .execute()
        )
        rows = sorted(response.data or [], key=lambda row: row.get("chunk_index") or 0)
        if len(rows) == len(parents):
            return {parent.chunk_id: row["id"] for parent, row in zip(parents, rows)}
    except Exception as exc:
        logger.warning("Could not resolve parent row ids: %s", exc)

    logger.warning("Parent chunk ids unavailable; children will not expand to parents.")
    return {}


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
def retrieve(
    supabase_client,
    org_id: str,
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
    embedder=None,
    reranker=None,
    candidate_k: int = 50,
) -> List[RetrievedDocument]:
    """Retrieve child chunks scoped to an organisation/document."""
    config = HybridSearchConfig(
        bm25_top_k=candidate_k,
        vector_top_k=candidate_k,
        fusion_top_k=candidate_k,
        final_top_k=top_k,
        reranker_top_k=top_k,
        retrieval_mode=retrieval_mode,
        expand_parents=True,
    )
    retriever = HybridRetriever(
        db_client=supabase_client,
        config=config,
        embedding_service=embedder,
        reranker=reranker,
    )
    return retriever.search(
        query=query,
        tenant_id=org_id,
        document_id=document_id,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )


def build_context(documents: Sequence[RetrievedDocument], max_chars: int = 3500) -> List[Dict[str, Any]]:
    """Deduplicate parent sections and number the passages for citation."""
    context: List[Dict[str, Any]] = []
    seen_parents = set()

    for doc in documents:
        key = doc.parent_id if doc.parent_id is not None else doc.id
        if doc.parent_content and key in seen_parents:
            continue
        if doc.parent_content:
            seen_parents.add(key)

        text = doc.context_text.strip()
        if not text:
            continue
        context.append(
            {
                "marker": len(context) + 1,
                "text": text[:max_chars],
                "chunk_id": doc.id,
                "parent_id": doc.parent_id,
                "document_id": doc.document_id,
                "section_path": doc.section_path,
                "score": doc.score,
                "source": doc.source.value,
                "is_parent_context": bool(doc.parent_content),
            }
        )
    return context


def format_citations(documents: Sequence[RetrievedDocument], limit: int = 12) -> List[Dict[str, Any]]:
    citations = []
    for index, doc in enumerate(documents[:limit], start=1):
        citations.append(
            {
                "marker": index,
                "chunk_id": doc.id,
                "document_id": doc.document_id,
                "section_path": doc.section_path,
                "score": round(float(doc.score), 6),
                "source": doc.source.value,
                "snippet": (doc.context_text or doc.content)[:400],
            }
        )
    return citations


def generate_answer(
    groq_client,
    query: str,
    context: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: int = 1200,
) -> str:
    if not context:
        return "Not found in the provided document."

    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    passages = []
    for item in context:
        path = " > ".join(item["section_path"]) if item["section_path"] else "Document"
        passages.append(f"[{item['marker']}] ({path})\n{item['text']}")
    user_prompt = (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n" + "\n\n".join(passages) +
        "\n\nAnswer with inline citations such as [1]. If the answer is absent, "
        "reply exactly: Not found in the provided document."
    )

    completion = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def answer_query(
    supabase_client,
    org_id: str,
    query: str,
    document_id: Optional[str] = None,
    groq_client=None,
    embedder=None,
    reranker=None,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
    verify: bool = True,
) -> Dict[str, Any]:
    """Full RAG answer: retrieve -> generate -> verify citations."""
    started = time.time()
    embedder = embedder if embedder is not None else build_embedder()
    documents = retrieve(
        supabase_client=supabase_client,
        org_id=org_id,
        query=query,
        document_id=document_id,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        embedder=embedder,
        reranker=reranker,
    )
    context = build_context(documents)

    if not context:
        answer = "Not found in the provided document."
    else:
        groq_client = groq_client or build_groq_client()
        answer = generate_answer(groq_client, query, context)

    verification = None
    if verify and context and answer and not answer.startswith("Not found"):
        try:
            verification = CitationVerifier().verify_response(
                answer,
                [{"id": item["chunk_id"], "content": item["text"]} for item in context],
                response_type="rag_answer",
            )
            verification.pop("outcomes", None)  # keep the API payload compact
        except Exception as exc:
            logger.warning("Citation verification failed: %s", exc)

    return {
        "query": query,
        "answer": answer,
        "document_id": document_id,
        "retrieved_chunks": [doc.to_dict() for doc in documents],
        "context": context,
        "citations": format_citations(documents),
        "verification": verification,
        "retrieval_mode": retrieval_mode,
        "latency_ms": (time.time() - started) * 1000,
    }
