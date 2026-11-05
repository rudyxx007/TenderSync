"""Hybrid retrieval for TenderSync.

Pipeline: PostgreSQL full-text ranking (``ts_rank_cd`` over a GIN tsvector)
+ pgvector HNSW cosine similarity, fused server-side with Reciprocal Rank
Fusion, then an optional cross-encoder rerank.

Everything is scoped by ``org_id``. When the database does not yet have the
hybrid function (migration 002 not applied), the retriever falls back to
org-scoped vector search and logs a warning instead of silently returning
unranked rows.

Note on naming: PostgreSQL's ``ts_rank_cd`` is a BM25-like lexical rank, not a
literal BM25 implementation. It is exposed as ``text_rank``; the legacy
``bm25_score`` field is kept for API compatibility.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    RRF = "rrf"
    WEIGHTED_SUM = "weighted_sum"
    MAX_SCORE = "max_score"


class RetrievalSource(Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrievedDocument:
    id: Any
    content: str
    score: float
    source: RetrievalSource = RetrievalSource.HYBRID
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    bm25_score: Optional[float] = None
    vector_similarity: Optional[float] = None
    parent_id: Optional[int] = None
    document_id: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    is_parent: bool = False
    contextual_prefix: Optional[str] = None
    parent_content: Optional[str] = None

    @property
    def rerank_text(self) -> str:
        """Text handed to the cross-encoder (context helps short chunks)."""
        prefix = f"{self.contextual_prefix}\n\n" if self.contextual_prefix else ""
        return f"{prefix}{self.content}"

    @property
    def context_text(self) -> str:
        """Text handed to the generator: prefer the full parent section."""
        return self.parent_content or self.rerank_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source.value,
            "rank": self.rank,
            "metadata": self.metadata or {},
            "bm25_score": self.bm25_score,
            "vector_similarity": self.vector_similarity,
            "parent_id": self.parent_id,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "is_parent": self.is_parent,
            "contextual_prefix": self.contextual_prefix,
            "parent_content": self.parent_content,
        }


@dataclass
class HybridSearchConfig:
    rrf_k: int = 50
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    bm25_top_k: int = 50
    vector_top_k: int = 50
    fusion_top_k: int = 50
    final_top_k: int = 8
    use_reranking: bool = True
    reranker_top_k: int = 8
    retrieval_mode: str = "hybrid"  # hybrid | bm25_only | vector_only
    expand_parents: bool = True


class ReciprocalRankFusion:
    """score(d) = sum(1 / (rank(d) + k)) across ranked lists."""

    def __init__(self, k: int = 50):
        if k < 1:
            raise ValueError("RRF k must be >= 1")
        self.k = k

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[RetrievedDocument]],
        top_k: int = 50,
    ) -> List[RetrievedDocument]:
        scores: Dict[str, Dict[str, Any]] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = str(doc.id)
                entry = scores.setdefault(
                    key,
                    {"doc": doc, "score": 0.0, "sources": set()},
                )
                entry["score"] += 1.0 / (rank + self.k)
                entry["sources"].add(doc.source)

        ordered = sorted(scores.values(), key=lambda e: e["score"], reverse=True)

        results: List[RetrievedDocument] = []
        for rank, entry in enumerate(ordered[:top_k], start=1):
            doc = entry["doc"]
            doc.score = float(entry["score"])
            doc.rank = rank
            doc.source = RetrievalSource.HYBRID
            results.append(doc)
        return results


class HybridRetriever:
    """Org-scoped hybrid retrieval backed by Supabase RPC functions."""

    def __init__(
        self,
        db_client=None,
        config: Optional[HybridSearchConfig] = None,
        embedding_service=None,
        reranker=None,
    ):
        self.db_client = db_client
        self.config = config or HybridSearchConfig()
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.rrf_fuser = ReciprocalRankFusion(k=self.config.rrf_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """Retrieve child chunks for a query.

        ``tenant_id`` and ``organization_id`` are aliases for the org UUID.
        Supplying neither is a programming error: an unscoped vector query
        would cross tenant boundaries.
        """
        org_id = organization_id or tenant_id
        if not org_id:
            raise ValueError("org_id is required for retrieval (multi-tenant isolation).")
        if not self.db_client:
            raise RuntimeError("HybridRetriever requires a Supabase client.")

        mode = (retrieval_mode or self.config.retrieval_mode or "hybrid").lower()
        if mode not in {"hybrid", "bm25_only", "vector_only"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        top_k = top_k or self.config.final_top_k

        query_embedding = None
        if mode != "bm25_only":
            if not self.embedding_service:
                if mode == "vector_only":
                    raise RuntimeError("vector_only retrieval requires an embedding service.")
                logger.warning("No embedding service; falling back to lexical-only retrieval.")
                mode = "bm25_only"
            else:
                query_embedding = self.embedding_service.embed_query(query)
                if not query_embedding:
                    if mode == "vector_only":
                        raise RuntimeError("Embedding provider returned an empty query vector.")
                    logger.warning("Empty query embedding; falling back to lexical-only retrieval.")
                    mode = "bm25_only"

        rows = self._hybrid_rpc(
            query=query,
            query_embedding=query_embedding,
            org_id=str(org_id),
            document_id=document_id,
            mode=mode,
        )

        results = [self._row_to_document(row, index) for index, row in enumerate(rows)]

        if self.config.use_reranking and self.reranker and results:
            results = self._rerank(query, results, top_k=self.config.reranker_top_k)

        results = results[:top_k]

        if self.config.expand_parents and results:
            self._expand_parents(results, org_id=str(org_id))

        return results

    def search_with_parents(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        retrieve_parents: bool = True,
        top_k: int = 8,
        document_id: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        results = self.search(
            query=query,
            tenant_id=tenant_id,
            organization_id=organization_id,
            top_k=top_k,
            document_id=document_id,
            retrieval_mode=retrieval_mode,
        )
        parents = [doc for doc in results if doc.parent_content] if retrieve_parents else []
        return {"children": results, "parents": parents, "all_results": results}

    # ------------------------------------------------------------------
    # RPC execution
    # ------------------------------------------------------------------
    def _hybrid_rpc(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        org_id: str,
        document_id: Optional[str],
        mode: str,
    ) -> List[Dict[str, Any]]:
        params = {
            "query_text": query,
            "query_embedding": query_embedding,
            "filter_org_id": org_id,
            "filter_document_id": document_id,
            # lexical/vector budgets are controlled by the mode
            "bm25_top_k": self.config.bm25_top_k if mode != "vector_only" else 0,
            "vector_top_k": self.config.vector_top_k if mode != "bm25_only" else 0,
            "rrf_k": self.config.rrf_k,
            "final_top_k": self.config.fusion_top_k,
        }

        try:
            response = self.db_client.rpc("hybrid_search_documents", params).execute()
            return list(response.data or [])
        except Exception as exc:
            if not self._is_missing_function(exc):
                raise

            logger.warning(
                "hybrid_search_documents is unavailable (%s). "
                "Apply supabase/migrations/002_hybrid_search.sql. "
                "Falling back to org-scoped vector search for this query.",
                exc,
            )
            if not query_embedding:
                raise RuntimeError(
                    "Hybrid retrieval function is missing and no query embedding is "
                    "available. Apply migration 002_hybrid_search.sql."
                ) from exc

            fallback = self.db_client.rpc(
                "match_documents_org",
                {
                    "query_embedding": query_embedding,
                    "match_count": self.config.vector_top_k,
                    "filter_org_id": org_id,
                    "filter_document_id": document_id,
                },
            ).execute()
            return list(fallback.data or [])

    @staticmethod
    def _is_missing_function(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "pgrst202" in text
            or "could not find the function" in text
            or "does not exist" in text and "function" in text
        )

    def _row_to_document(self, row: Dict[str, Any], index: int) -> RetrievedDocument:
        source_type = (row.get("source_type") or "hybrid").lower()
        source = {
            "bm25": RetrievalSource.BM25,
            "vector": RetrievalSource.VECTOR,
            "hybrid": RetrievalSource.HYBRID,
        }.get(source_type, RetrievalSource.HYBRID)

        score = row.get("rrf_score")
        if score is None:
            score = row.get("similarity")
        if score is None:
            score = row.get("text_rank", 0)

        return RetrievedDocument(
            id=row.get("id", f"row_{index}"),
            content=row.get("content") or "",
            score=float(score or 0.0),
            source=source,
            rank=int(row.get("rank_position") or index + 1),
            metadata=row.get("metadata") or {},
            bm25_score=row.get("text_rank", row.get("bm25_score")),
            vector_similarity=row.get("vector_similarity"),
            parent_id=row.get("parent_id"),
            document_id=str(row["document_id"]) if row.get("document_id") else None,
            section_path=list(row.get("section_path") or []),
            is_parent=bool(row.get("is_parent", False)),
            contextual_prefix=row.get("contextual_prefix"),
        )

    # ------------------------------------------------------------------
    # Rerank / parent expansion
    # ------------------------------------------------------------------
    def _rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int,
    ) -> List[RetrievedDocument]:
        try:
            reranked = self.reranker.rerank(
                query,
                [doc.rerank_text for doc in documents],
                top_k=min(top_k, len(documents)),
            )
        except Exception as exc:
            logger.warning("Reranking failed; keeping RRF order: %s", exc)
            return documents

        results: List[RetrievedDocument] = []
        seen = set()
        for item in reranked:
            index = item.get("index")
            if index is None or not (0 <= index < len(documents)):
                continue
            doc = documents[index]
            if doc.id in seen:
                continue
            seen.add(doc.id)
            doc.score = float(item.get("relevance_score", doc.score))
            doc.rank = len(results) + 1
            results.append(doc)

        # Keep any candidates the reranker did not return.
        for doc in documents:
            if doc.id not in seen:
                doc.rank = len(results) + 1
                results.append(doc)
        return results

    def _expand_parents(self, documents: List[RetrievedDocument], org_id: str) -> None:
        parent_ids = sorted({doc.parent_id for doc in documents if doc.parent_id})
        if not parent_ids:
            return

        try:
            response = (
                self.db_client.table("documents")
                .select("id,content,section_path,contextual_prefix,is_parent")
                .eq("org_id", org_id)
                .in_("id", parent_ids)
                .execute()
            )
        except Exception as exc:
            logger.warning("Parent expansion failed: %s", exc)
            return

        parents = {row["id"]: row for row in (response.data or [])}
        for doc in documents:
            parent = parents.get(doc.parent_id)
            if parent and parent.get("content"):
                doc.parent_content = parent["content"]
                if not doc.section_path and parent.get("section_path"):
                    doc.section_path = list(parent["section_path"])


def normalize_scores(scores: Sequence[float]) -> List[float]:
    """Min-max normalise scores; identical scores map to 1.0."""
    if not scores:
        return []
    values = np.asarray(scores, dtype=float)
    spread = values.max() - values.min()
    if spread == 0:
        return [1.0] * len(values)
    return ((values - values.min()) / spread).tolist()
