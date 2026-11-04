"""
RAG Pipeline Wrapper - Interface to the RAG pipeline for evaluation.

Provides a clean API for the evaluation framework to interact with
the actual RAG pipeline without coupling to implementation details.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests


@dataclass
class RAGPipelineResult:
    """Result from a RAG pipeline query."""
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    latency_ms: float
    timestamp: str


class RAGPipelineWrapper:
    """
    Wrapper around the RAG pipeline for evaluation purposes.
    
    Supports both direct API calls to the running service and
    direct instantiation of pipeline components.
    """
    
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
        """
        Initialize the wrapper.
        
        Args:
            api_base: Base URL of the RAG service API (optional)
            embedding_model: Embedding model name
            reranker_model: Reranker model name
            retrieval_mode: 'hybrid', 'bm25_only', or 'vector_only'
            top_k: Number of chunks to retrieve
            rerank_top_k: Number of chunks after reranking
            tenant_id: Tenant ID for multi-tenant isolation
        """
        self.api_base = api_base
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.tenant_id = tenant_id or os.getenv("EVAL_TENANT_ID")
        
        # Lazy-initialized pipeline components
        self._pipeline = None
        self._session = None
    
    def _get_session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
            })
            # Add auth if available
            api_key = os.getenv("RAG_SERVICE_API_KEY")
            if api_key:
                self._session.headers["Authorization"] = f"Bearer {api_key}"
        return self._session
    
    def query(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a query through the RAG pipeline.
        
        Args:
            query: The user query
            tenant_id: Tenant ID (overrides instance value)
            organization_id: Organization ID for org-level access
            filters: Optional filters for retrieval
            
        Returns:
            Dict with answer, retrieved_chunks, latency, metadata
        """
        tenant_id = tenant_id or self.tenant_id
        
        if self.api_base:
            return self._query_via_api(
                query, tenant_id, organization_id, filters
            )
        else:
            return self._query_direct(query, tenant_id, organization_id, filters)
    
    def _query_via_api(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query via HTTP API."""
        start_time = time.time()
        
        session = self._get_session()
        
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
        }
        
        if organization_id:
            payload["organization_id"] = organization_id
        
        if filters:
            payload["filters"] = filters
        
        try:
            response = session.post(
                f"{self.api_base}/api/v1/rag/query",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            
        except requests.RequestException as e:
            logger.warning(f"API query failed: {e}")
            result = {
                "answer": "",
                "retrieved_chunks": [],
                "error": str(e),
            }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "answer": result.get("answer", ""),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "metadata": {
                "retrieval_mode": self.retrieval_mode,
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model,
                "api_response_time_ms": result.get("latency_ms", latency_ms),
            },
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    
    def _query_direct(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query directly using pipeline components."""
        if self._pipeline is None:
            from backend.services.voyage_embeddings import VoyageEmbeddingService
            from backend.services.contextual_chunker import ContextualChunker
            from backend.services.hybrid_retriever import HybridRetriever
            from backend.services.citation_verifier import CitationVerifier
            
            # Initialize pipeline components
            embedder = VoyageEmbeddingService(
                model=self.embedding_model,
            )
            
            retriever = HybridRetriever(
                embedding_service=embedder,
                retrieval_mode=self.retrieval_mode,
                top_k=self.top_k,
            )
            
            self._pipeline = {
                "embedder": embedder,
                "retriever": retriever,
            }
        
        start_time = time.time()
        
        # Embed query
        query_embedding = self._pipeline["embedder"].embed_query(query)
        
        # Retrieve
        retrieval_result = self._pipeline["retriever"].retrieve(
            query=query,
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            organization_id=organization_id,
            top_k=self.top_k,
        )
        
        retrieved_chunks = retrieval_result.get("chunks", [])
        
        # Rerank if applicable
        if self.reranker_model != "none" and retrieved_chunks:
            from backend.services.voyage_embeddings import VoyageReranker
            
            reranker = VoyageReranker(model=self.reranker_model)
            reranking_result = reranker.rerank(
                query=query,
                documents=[chunk.get("content", "") for chunk in retrieved_chunks],
            )
            
            # Reorder chunks based on reranking
            reranked_docs = reranking_result.get("results", [])
            original_chunks = retrieved_chunks[:len(reranked_docs)]
            
            # Sort by relevance score
            scored_chunks = list(zip(original_chunks, [r.get("relevance_score", 0) for r in reranked_docs]))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            retrieved_chunks = [chunk for chunk, _ in scored_chunks[:self.rerank_top_k]]
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate answer (simplified - in production would use LLM)
        # For evaluation, we assume the answer is pre-generated or use a mock
        answer = ""  # Would be filled by LLM in production
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "metadata": {
                "retrieval_mode": self.retrieval_mode,
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model,
                "num_chunks_retrieved": len(retrieved_chunks),
            },
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    
    def health_check(self) -> bool:
        """Check if the RAG service is healthy."""
        if not self.api_base:
            return True  # Direct mode doesn't need health check
        
        try:
            session = self._get_session()
            response = session.get(
                f"{self.api_base}/health",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close resources."""
        if self._session:
            self._session.close()
            self._session = None


# For backwards compatibility
class EvaluationPipeline:
    """Legacy compatibility class."""
    
    def __init__(self, **kwargs):
        self.wrapper = RAGPipelineWrapper(**kwargs)
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        return self.wrapper.query(query, **kwargs)
    
    def evaluate(self, samples: List, tenant_id: str) -> List[Dict]:
        """Run evaluation on samples."""
        results = []
        for sample in samples:
            result = self.wrapper.query(
                query=sample.query,
                tenant_id=tenant_id,
            )
            results.append({
                "sample": sample,
                "result": result,
            })
        return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Quick test
    wrapper = RAGPipelineWrapper(
        embedding_model="voyage-law-2",
        retrieval_mode="hybrid",
    )
    
    print(f"Pipeline wrapper initialized:")
    print(f"  API base: {wrapper.api_base or 'direct mode'}")
    print(f"  Embedding: {wrapper.embedding_model}")
    print(f"  Reranker: {wrapper.reranker_model}")
    print(f"  Retrieval: {wrapper.retrieval_mode}")
    print(f"  Top-k: {wrapper.top_k}")
