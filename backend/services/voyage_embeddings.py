"""
Voyage AI Embedding Service for TenderSync Enterprise RAG Pipeline.

Uses Voyage AI's state-of-the-art embedding models for semantic search.
Recommended models:
- voyage-4-large: Best general-purpose retrieval (1024-dim, 32K context)
- voyage-finance-2: Optimized for financial document retrieval
- voyage-law-2: Optimized for legal/contract document retrieval

References:
- https://docs.voyageai.com/docs/embeddings
- https://www.voyageai.com/
"""

import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False


@dataclass
class VoyageConfig:
    """Configuration for Voyage AI Embedding Service."""
    api_key: str
    model: str = "voyage-4-large"
    input_type: str = "document"  # 'document', 'query', or None
    truncation: bool = True
    output_dimension: Optional[int] = None  # None, 256, 512, 1024, 2048
    output_dtype: str = "float"  # 'float', 'int8', 'uint8', 'binary', 'ubinary'
    batch_size: int = 100  # Max 1000 texts per request, 320K tokens for voyage-4-large


class VoyageEmbeddingService:
    """
    Embedding service using Voyage AI's models.
    
    Provides both dense embeddings for semantic search and supports
    contextual embeddings for improved retrieval accuracy.
    
    Example:
        service = VoyageEmbeddingService(api_key="your_key")
        embeddings = service.embed_documents(["text1", "text2"])
        query_embedding = service.embed_query("search query")
    """
    
    # Model capabilities
    MODEL_DIMENSIONS = {
        "voyage-4-large": 1024,
        "voyage-4": 1024,
        "voyage-4-lite": 1024,
        "voyage-3-large": 1024,
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-multilingual-2": 1024,
    }
    
    # Token limits per model
    MODEL_TOKEN_LIMITS = {
        "voyage-4-large": 320000,
        "voyage-4": 320000,
        "voyage-4-lite": 320000,
        "voyage-3-large": 120000,
        "voyage-3.5": 120000,
        "voyage-3.5-lite": 320000,
        "voyage-3": 120000,
        "voyage-3-lite": 320000,
        "voyage-code-3": 120000,
        "voyage-finance-2": 120000,
        "voyage-law-2": 120000,
        "voyage-multilingual-2": 120000,
    }
    
    # Recommended models by domain
    RECOMMENDED_MODELS = {
        "general": "voyage-4-large",
        "finance": "voyage-finance-2",
        "legal": "voyage-law-2",
        "code": "voyage-code-3",
        "multilingual": "voyage-multilingual-2",
        "cost_optimized": "voyage-4-lite",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-4-large",
        config: Optional[VoyageConfig] = None
    ):
        """
        Initialize Voyage Embedding Service.
        
        Args:
            api_key: Voyage API key (or set VOYAGE_API_KEY env var)
            model: Embedding model name (default: voyage-4-large)
            config: Optional VoyageConfig for advanced settings
        """
        if not VOYAGE_AVAILABLE:
            raise ImportError(
                "Voyage AI package not installed. Install with: pip install voyageai"
            )
        
        self.config = config or VoyageConfig(
            api_key=api_key or os.environ.get("VOYAGE_API_KEY", ""),
            model=model,
        )
        
        if not self.config.api_key:
            raise ValueError(
                "Voyage API key not provided. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = voyageai.Client(api_key=self.config.api_key)
        self.model = self.config.model
        self._dimension = self.MODEL_DIMENSIONS.get(self.model, 1024)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension for current model."""
        return self._dimension
    
    def embed(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        model: Optional[str] = None,
        truncation: Optional[bool] = None,
        output_dimension: Optional[int] = None,
        output_dtype: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            input_type: 'document', 'query', or None
            model: Override model (uses instance model by default)
            truncation: Whether to truncate long texts
            output_dimension: Output dimension (None for model default)
            output_dtype: Output data type ('float', 'int8', etc.)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts exceed token limits
        """
        if not texts:
            return []
        
        model = model or self.model
        input_type = input_type or self.config.input_type
        truncation = truncation if truncation is not None else self.config.truncation
        output_dimension = output_dimension or self.config.output_dimension
        output_dtype = output_dtype or self.config.output_dtype
        
        # Validate batch size
        if len(texts) > 1000:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), 1000):
                batch = texts[i:i + 1000]
                batch_embeddings = self.embed(
                    batch,
                    input_type=input_type,
                    model=model,
                    truncation=truncation,
                    output_dimension=output_dimension,
                    output_dtype=output_dtype,
                )
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        
        try:
            # Only throttle when the free tier is explicitly configured. The
            # previous unconditional 21s sleep made a 100-sample evaluation
            # take 35 minutes even on paid keys.
            min_interval = float(os.environ.get("VOYAGE_MIN_INTERVAL_SECONDS", "0"))
            if min_interval > 0 and hasattr(self, '_last_embed_time'):
                elapsed = time.time() - self._last_embed_time
                if elapsed < min_interval:
                    remaining = min_interval - elapsed
                    print(f"  [Rate limit] Waiting {remaining:.1f}s before next embedding request...")
                    time.sleep(remaining)
            
            result = self.client.embed(
                texts=texts,
                model=model,
                input_type=input_type,
                truncation=truncation,
                output_dimension=output_dimension,
                output_dtype=output_dtype,
            )
            self._last_embed_time = time.time()
            return result.embeddings
        except Exception as e:
            self._last_embed_time = time.time()
            raise RuntimeError(f"Voyage AI embedding failed: {e}") from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents for storage/indexing.
        
        Uses 'document' input type for optimal document retrieval.
        """
        return self.embed(texts, input_type="document")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query for retrieval.
        
        Uses 'query' input type for optimal query-document matching.
        """
        embeddings = self.embed([query], input_type="query")
        return embeddings[0] if embeddings else []
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text (auto-detects document vs query)."""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []
    
    def get_similar_chunks(
        self,
        query: str,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to a query using cosine similarity.
        
        This is a lightweight local similarity search for small datasets.
        For production, use pgvector with HNSW indexing.
        
        Args:
            query: Search query
            chunks: List of text chunks
            chunk_embeddings: Pre-computed embeddings for chunks
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of dicts with chunk, embedding, similarity score
        """
        import numpy as np
        
        query_emb = np.array(self.embed_query(query))
        chunk_embs = np.array(chunk_embeddings)
        
        # Normalize for cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-10)
        
        # Cosine similarity
        similarities = np.dot(chunk_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= similarity_threshold:
                results.append({
                    "chunk": chunks[idx],
                    "embedding": chunk_embeddings[idx],
                    "similarity": score,
                    "index": int(idx),
                })
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health and model information."""
        return {
            "status": "healthy",
            "model": self.model,
            "dimension": self.dimension,
            "api_key_configured": bool(self.config.api_key),
            "recommended_models": self.RECOMMENDED_MODELS,
        }


class VoyageReranker:
    """
    Cross-encoder reranking using Voyage AI's reranking models.
    
    Re-ranks retrieved candidates to improve relevance.
    Uses Voyage's rerank endpoint for state-of-the-art reranking.
    
    References:
    - https://docs.voyageai.com/docs/rerank
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-rerank-2",
    ):
        """
        Initialize Voyage Reranker.
        
        Args:
            api_key: Voyage API key
            model: Rerank model name
        """
        if not VOYAGE_AVAILABLE:
            raise ImportError(
                "Voyage AI package not installed. Install with: pip install voyageai"
            )
        
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError("Voyage API key required for reranking")
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        max_chunks_per_doc: int = 4096,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            max_chunks_per_doc: Max chunks per document for processing
            
        Returns:
            List of dicts with document, relevance_score, index
        """
        try:
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k,
            )
            
            # result.results[i].index refers to the original document index;
            # zipping the original list with the reranked list mismatched texts.
            return [
                {
                    "document": documents[item.index],
                    "relevance_score": item.relevance_score,
                    "index": item.index,
                }
                for item in result.results
            ]
        except Exception as e:
            raise RuntimeError(f"Voyage AI reranking failed: {e}") from e
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        score_key: str = "content",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents with metadata.
        
        Args:
            query: Search query
            documents: List of dicts containing document content
            score_key: Key in dict containing text to rerank
            top_k: Number of top results
            
        Returns:
            List of documents with added 'relevance_score' field
        """
        docs_text = [doc[score_key] for doc in documents]
        reranked = self.rerank(query, docs_text, top_k=top_k)
        
        # Merge scores back with original documents
        result = []
        for item in reranked:
            doc = documents[item["index"]].copy()
            doc["relevance_score"] = item["relevance_score"]
            result.append(doc)
        
        return result
