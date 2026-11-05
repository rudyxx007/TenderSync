"""Adapter between the evaluation harness and the TenderSync RAG pipeline.

Two modes:

* ``api_base`` set: HTTP against a running FastAPI instance.
* ``api_base`` unset: in-process call to ``rag_service.answer_query`` using
  Supabase, the configured embedder/reranker and Groq.

The previous implementation instantiated a ``HybridRetriever`` with no database
client, whose placeholder methods always returned ``[]``. Every metric was
therefore computed over zero retrieved chunks, which is exactly why the report
showed 0.00% across the board.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

import rag_service

logger = logging.getLogger(__name__)


class PipelineConfigurationError(RuntimeError):
    """Raised when the harness cannot reach a real pipeline."""


@dataclass
class RAGPipelineResult:
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    latency_ms: float
    timestamp: str


class RAGPipelineWrapper:
    def __init__(
        self,
        api_base: Optional[str] = None,
        embedding_model: str = "voyage-4-large",
        reranker_model: str = "voyage-rerank-2",
        retrieval_mode: str = "hybrid",
        top_k: int = 10,
        rerank_top_k: int = 8,
        tenant_id: Optional[str] = None,
    ):
        self.api_base = api_base.rstrip("/") if api_base else None
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.tenant_id = tenant_id or os.getenv("EVAL_TENANT_ID")

        self._supabase = None
        self._embedder = None
        self._reranker = None
        self._groq = None
        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------
    def _session_or_create(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
            api_key = os.getenv("RAG_SERVICE_API_KEY")
            if api_key:
                self._session.headers["Authorization"] = f"Bearer {api_key}"
        return self._session

    def _ensure_direct(self) -> None:
        if self._supabase is not None:
            return

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise PipelineConfigurationError(
                "Direct evaluation needs SUPABASE_URL and SUPABASE_SECRET_KEY. "
                "Set them in backend/.env, or pass --api-base for a running service."
            )

        from supabase import create_client

        self._supabase = create_client(url, key)
        try:
            self._embedder = rag_service.build_embedder(self.embedding_model)
        except Exception as exc:
            raise PipelineConfigurationError(f"Embedding provider unavailable: {exc}") from exc
        self._reranker = rag_service.build_reranker(self.reranker_model)
        try:
            self._groq = rag_service.build_groq_client()
        except Exception as exc:
            raise PipelineConfigurationError(f"Answer generation unavailable: {exc}") from exc

    @property
    def embedder(self):
        if self.api_base:
            return None
        self._ensure_direct()
        return self._embedder

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        org_id = tenant_id or organization_id or self.tenant_id
        if not org_id:
            raise PipelineConfigurationError(
                "tenant/org id is required for evaluation. Pass --tenant-id "
                "or set EVAL_TENANT_ID."
            )

        if self.api_base:
            return self._query_via_api(query, org_id, document_id)
        return self._query_direct(query, org_id, document_id)

    def _query_direct(
        self, query: str, org_id: str, document_id: Optional[str]
    ) -> Dict[str, Any]:
        self._ensure_direct()
        started = time.time()

        result = rag_service.answer_query(
            supabase_client=self._supabase,
            org_id=org_id,
            query=query,
            document_id=document_id,
            groq_client=self._groq,
            embedder=self._embedder,
            reranker=self._reranker,
            top_k=self.top_k,
            retrieval_mode=self.retrieval_mode,
        )

        # Evaluate the exact passages handed to the generator (parents included).
        retrieved = [
            {
                "id": item["chunk_id"],
                "content": item["text"],
                "score": item.get("score", 0.0),
                "metadata": {
                    "section_path": item.get("section_path", []),
                    "parent_id": item.get("parent_id"),
                    "document_id": item.get("document_id"),
                },
            }
            for item in result.get("context", [])
        ]

        return {
            "query": query,
            "answer": result.get("answer", ""),
            "retrieved_chunks": retrieved,
            "context": result.get("context", []),
            "citations": result.get("citations", []),
            "verification": result.get("verification"),
            "metadata": {
                "mode": "direct",
                "retrieval_mode": self.retrieval_mode,
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model,
                "document_id": document_id,
                "num_chunks_retrieved": len(retrieved),
            },
            "latency_ms": (time.time() - started) * 1000,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _query_via_api(
        self, query: str, org_id: str, document_id: Optional[str]
    ) -> Dict[str, Any]:
        started = time.time()
        payload = {
            "query": query,
            "tenant_id": org_id,
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
        }
        if document_id:
            payload["document_id"] = document_id

        session = self._session_or_create()
        try:
            response = session.post(
                f"{self.api_base}/api/rag/query", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            logger.warning("API query failed: %s", exc)
            result = {"answer": "", "context": [], "error": str(exc)}

        retrieved = [
            {
                "id": item.get("chunk_id"),
                "content": item.get("text", ""),
                "score": item.get("score", 0.0),
                "metadata": {"section_path": item.get("section_path", [])},
            }
            for item in result.get("context", [])
        ]

        latency_ms = (time.time() - started) * 1000
        return {
            "query": query,
            "answer": result.get("answer", ""),
            "retrieved_chunks": retrieved,
            "context": result.get("context", []),
            "verification": result.get("verification"),
            "metadata": {
                "mode": "api",
                "retrieval_mode": self.retrieval_mode,
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model,
                "error": result.get("error"),
            },
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    # ------------------------------------------------------------------
    # Health / cleanup
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        if self.api_base:
            try:
                response = self._session_or_create().get(
                    f"{self.api_base}/api/rag/health", timeout=10
                )
                return response.status_code == 200
            except Exception:
                return False
        return bool(
            os.getenv("SUPABASE_URL")
            and (os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_KEY"))
        )

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None


# Backwards-compatible alias.
class EvaluationPipeline(RAGPipelineWrapper):
    def evaluate(self, samples: List, tenant_id: str) -> List[Dict[str, Any]]:
        return [
            {"sample": sample, "result": self.query(sample.query, tenant_id=tenant_id)}
            for sample in samples
        ]
