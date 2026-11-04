"""
Hybrid Retriever for TenderSync Enterprise RAG Pipeline.

Combines multiple retrieval methods:
1. PostgreSQL Full-Text Search (BM25 via tsvector) - Exact keyword matching
2. pgvector HNSW Cosine Similarity - Semantic similarity
3. Reciprocal Rank Fusion (RRF) - Fusion of both methods
4. Cross-Encoder Reranking - Final refinement to top-k results

This hybrid approach is critical for enterprise documents where:
- ABN numbers, clause IDs, dollar figures need exact matching (BM25)
- Conceptual queries need semantic understanding (vectors)
- Combined approach gives best of both worlds

References:
- https://jkatz.github.io/post/postgres/hybrid-search-postgres-pgvector/
- https://www.anthropic.com/research/contextual-retrieval
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np


class FusionMethod(Enum):
    """Methods for fusing multiple retrieval results."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"
    MAX_SCORE = "max_score"
    DISCOUNTED_UNION = "discounted_union"


class RetrievalSource(Enum):
    """Sources of retrieved documents."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrievedDocument:
    """A document retrieved from the search index."""
    id: str
    content: str
    score: float
    source: RetrievalSource
    rank: int
    metadata: Dict[str, Any] = None
    bm25_score: Optional[float] = None
    vector_similarity: Optional[float] = None
    
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
        }


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    # RRF configuration
    rrf_k: int = 50  # RRF constant (higher = more weight to lower ranks)
    
    # Weights for weighted sum fusion
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    
    # Retrieval counts
    bm25_top_k: int = 50
    vector_top_k: int = 50
    fusion_top_k: int = 50  # Candidates before reranking
    final_top_k: int = 8   # Final results after reranking
    
    # Reranking
    use_reranking: bool = True
    reranker_top_k: int = 8
    
    # Score normalization
    normalize_scores: bool = True


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) for combining ranked lists.
    
    RRF formula: score(d) = sum_over_lists(1 / (rank(d) + k))
    
    Where:
    - rank(d) is the rank of document d in a particular list
    - k is a constant (typically 50-60)
    
    RRF is robust, parameter-free (mostly), and doesn't require
    score normalization across different ranking functions.
    
    Reference: https://www.blog.google/technology/search/elliptic-curve-encryption-and-quantum/
    """
    
    def __init__(self, k: int = 50):
        """
        Initialize RRF fusion.
        
        Args:
            k: RRF constant (default 50). Higher values reduce the
               impact of rank position on the final score.
        """
        self.k = k
    
    def fuse(
        self,
        ranked_lists: List[List[RetrievedDocument]],
        top_k: int = 50,
    ) -> List[RetrievedDocument]:
        """
        Fuse multiple ranked lists using RRF.
        
        Args:
            ranked_lists: List of ranked document lists from different sources
            top_k: Number of top results to return
            
        Returns:
            Fused and ranked list of documents
        """
        # Build document scores across all lists
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        for rank_list in ranked_lists:
            for rank, doc in enumerate(rank_list, start=1):
                doc_id = doc.id
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "rrf_score": 0.0,
                        "ranks": [],
                        "sources": set(),
                    }
                
                # RRF contribution: 1 / (rank + k)
                rrf_contribution = 1.0 / (rank + self.k)
                doc_scores[doc_id]["rrf_score"] += rrf_contribution
                doc_scores[doc_id]["ranks"].append(rank)
                doc_scores[doc_id]["sources"].add(doc.source)
        
        # Sort by RRF score descending
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        
        # Create result documents with fused scores
        results = []
        for rank, item in enumerate(sorted_docs[:top_k], start=1):
            doc = item["doc"]
            # Update score to be the RRF score
            doc.score = item["rrf_score"]
            doc.rank = rank
            doc.source = RetrievalSource.HYBRID
            results.append(doc)
        
        return results
    
    def fuse_with_weights(
        self,
        ranked_lists: List[Tuple[List[RetrievedDocument], float]],
        top_k: int = 50,
    ) -> List[RetrievedDocument]:
        """
        Fuse ranked lists with source weights.
        
        Args:
            ranked_lists: List of (ranked_list, weight) tuples
            top_k: Number of top results to return
            
        Returns:
            Fused and ranked list of documents
        """
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        for rank_list, weight in ranked_lists:
            for rank, doc in enumerate(rank_list, start=1):
                doc_id = doc.id
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "weighted_score": 0.0,
                        "ranks": [],
                        "sources": set(),
                    }
                
                # Weighted RRF contribution
                rrf_contribution = weight / (rank + self.k)
                doc_scores[doc_id]["weighted_score"] += rrf_contribution
                doc_scores[doc_id]["ranks"].append(rank)
                doc_scores[doc_id]["sources"].add(doc.source)
        
        # Sort by weighted score descending
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["weighted_score"],
            reverse=True,
        )
        
        results = []
        for rank, item in enumerate(sorted_docs[:top_k], start=1):
            doc = item["doc"]
            doc.score = item["weighted_score"]
            doc.rank = rank
            doc.source = RetrievalSource.HYBRID
            results.append(doc)
        
        return results


class ScoreNormalizer:
    """Normalize scores from different retrieval methods to comparable scales."""
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max scaling."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        
        if range_score == 0:
            return [1.0] * len(scores)  # All same score
        
        return [(s - min_score) / range_score for s in scores]
    
    @staticmethod
    def sigmoid_normalize(scores: List[float], temperature: float = 1.0) -> List[float]:
        """Normalize scores using sigmoid function."""
        return [1 / (1 + math.exp(-s / temperature)) for s in scores]
    
    @staticmethod
    def rank_based_normalize(ranks: List[int], total: int) -> List[float]:
        """Normalize by rank position (1 = best)."""
        return [(total - r + 1) / total for r in ranks]


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search with RRF fusion.
    
    This is the main retrieval class that orchestrates:
    1. BM25 full-text search via PostgreSQL tsvector
    2. Vector similarity search via pgvector HNSW
    3. RRF fusion of results
    4. Optional cross-encoder reranking
    
    Example:
        retriever = HybridRetriever(db_client, config)
        results = retriever.search(
            query="What are the compliance requirements?",
            tenant_id="org_123",
            top_k=8
        )
    """
    
    def __init__(
        self,
        db_client=None,  # Supabase client
        config: Optional[HybridSearchConfig] = None,
        embedding_service=None,  # VoyageEmbeddingService or similar
        reranker=None,  # VoyageReranker or similar
    ):
        """
        Initialize HybridRetriever.
        
        Args:
            db_client: Database client for queries
            config: Hybrid search configuration
            embedding_service: Service for generating query embeddings
            reranker: Optional cross-encoder reranker
        """
        self.db_client = db_client
        self.config = config or HybridSearchConfig()
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.rrf_fuser = ReciprocalRankFusion(k=self.config.rrf_k)
    
    def search(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Perform hybrid search with fusion and optional reranking.
        
        Args:
            query: Search query text
            tenant_id: Tenant ID for RLS enforcement
            organization_id: Organization ID for additional filtering
            filters: Additional metadata filters
            top_k: Final number of results (uses config.final_top_k by default)
            
        Returns:
            List of retrieved documents ranked by relevance
        """
        top_k = top_k or self.config.final_top_k
        
        # Step 1: Get results from both retrieval methods
        bm25_results = self._bm25_search(
            query, tenant_id, organization_id, filters, self.config.bm25_top_k
        )
        
        vector_results = self._vector_search(
            query, tenant_id, organization_id, filters, self.config.vector_top_k
        )
        
        # Step 2: Fuse results using RRF
        fused_results = self.rrf_fuser.fuse(
            [bm25_results, vector_results],
            top_k=self.config.fusion_top_k,
        )
        
        # Step 3: Optional cross-encoder reranking
        if self.config.use_reranking and self.reranker and fused_results:
            reranked = self._rerank(
                query, fused_results, top_k=self.config.reranker_top_k
            )
            return reranked[:top_k]
        
        return fused_results[:top_k]
    
    def _bm25_search(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str],
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """
        Perform BM25 full-text search using PostgreSQL tsvector.
        
        Requires the documents table to have a tsvector column
        and appropriate GIN index.
        
        Args:
            query: Search query
            tenant_id: Tenant ID for RLS
            organization_id: Optional org filter
            filters: Additional filters
            top_k: Number of results
            
        Returns:
            Ranked list of documents from BM25
        """
        if not self.db_client:
            # Return empty for demo/testing
            return []
        
        # Build SQL query for BM25 search
        # Note: This assumes a tsvector column 'search_text' exists
        sql = """
        SELECT 
            id,
            content,
            ts_rank_cd(search_text, plainto_tsquery('english', :query)) as bm25_score,
            metadata
        FROM documents
        WHERE 
            search_text @@ plainto_tsquery('english', :query)
            AND tenant_id = :tenant_id
            {org_filter}
            {metadata_filters}
        ORDER BY bm25_score DESC
        LIMIT :limit
        """
        
        # Build parameter dict
        params = {
            "query": query,
            "tenant_id": tenant_id,
            "limit": top_k,
        }
        
        org_filter = ""
        if organization_id:
            org_filter = "AND organization_id = :org_id"
            params["org_id"] = organization_id
        
        # Would execute query and return results
        # For now, return empty (implementation depends on DB schema)
        return self._execute_bm25_query(sql, params, top_k)
    
    def _execute_bm25_query(
        self, sql: str, params: Dict[str, Any], top_k: int
    ) -> List[RetrievedDocument]:
        """Execute BM25 query and return results."""
        # This would be implemented with actual DB execution
        # Placeholder for now
        return []
    
    def _vector_search(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str],
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """
        Perform vector similarity search using pgvector HNSW.
        
        Args:
            query: Search query
            tenant_id: Tenant ID for RLS
            organization_id: Optional org filter
            filters: Additional filters
            top_k: Number of results
            
        Returns:
            Ranked list of documents by vector similarity
        """
        if not self.db_client or not self.embedding_service:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Call pgvector similarity search
        # Using cosine distance (<=>) for similarity
        sql = """
        SELECT 
            id,
            content,
            1 - (embedding <=> :query_embedding::vector) as vector_similarity,
            metadata
        FROM documents
        WHERE 
            embedding IS NOT NULL
            AND tenant_id = :tenant_id
            {org_filter}
            {metadata_filters}
        ORDER BY embedding <=> :query_embedding::vector
        LIMIT :limit
        """
        
        params = {
            "query_embedding": query_embedding,
            "tenant_id": tenant_id,
            "limit": top_k,
        }
        
        return self._execute_vector_query(sql, params, top_k)
    
    def _execute_vector_query(
        self, sql: str, params: Dict[str, Any], top_k: int
    ) -> List[RetrievedDocument]:
        """Execute vector search query and return results."""
        # Placeholder for actual DB execution
        return []
    
    def _rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Original search query
            documents: Candidate documents to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked documents
        """
        if not self.reranker or not documents:
            return documents
        
        # Extract content for reranking
        doc_contents = [doc.content for doc in documents]
        
        try:
            reranked = self.reranker.rerank(query, doc_contents, top_k=top_k)
            
            # Update document scores
            results = []
            for item in reranked:
                # Find original document by index
                orig_doc = documents[item["index"]]
                orig_doc.score = item["relevance_score"]
                orig_doc.rank = len(results) + 1
                results.append(orig_doc)
            
            return results
        except Exception as e:
            # Reranking failed, return original order
            print(f"Reranking failed: {e}")
            return documents
    
    def search_with_parents(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str] = None,
        retrieve_parents: bool = True,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Hybrid search with parent-child expansion.
        
        First retrieves child chunks, then expands to parent sections
        for full context.
        
        Args:
            query: Search query
            tenant_id: Tenant ID
            organization_id: Optional org filter
            retrieve_parents: Whether to expand to parent chunks
            top_k: Number of final results
            
        Returns:
            Dict with 'children', 'parents', and 'all_results'
        """
        # Get hybrid search results
        results = self.search(query, tenant_id, organization_id, top_k=top_k * 2)
        
        if not retrieve_parents:
            return {
                "children": results[:top_k],
                "parents": [],
                "all_results": results[:top_k],
            }
        
        # Would expand to parents using ParentChildIndex
        # Placeholder: just return results
        return {
            "children": results[:top_k],
            "parents": [],
            "all_results": results[:top_k],
        }
    
    def hybrid_search_sql(
        self,
        query: str,
        tenant_id: str,
        organization_id: Optional[str] = None,
    ) -> str:
        """
        Generate SQL for hybrid search with RRF fusion.
        
        Returns a SQL query that performs:
        1. BM25 search via tsvector
        2. Vector search via pgvector
        3. RRF fusion in SQL
        
        This can be executed directly in PostgreSQL for efficient
        server-side fusion.
        
        Args:
            query: Search query
            tenant_id: Tenant ID
            organization_id: Optional org filter
            
        Returns:
            SQL query string
        """
        org_filter = ""
        if organization_id:
            org_filter = "AND organization_id = '" + organization_id + "'"
        
        sql = f"""
        WITH 
        -- BM25 results
        bm25_results AS (
            SELECT 
                id,
                content,
                metadata,
                ts_rank_cd(search_text, plainto_tsquery('english', :query)) as bm25_score,
                0 as vector_similarity,
                1 as source_type  -- 1 = BM25
            FROM documents
            WHERE 
                search_text @@ plainto_tsquery('english', :query)
                AND tenant_id = :tenant_id
                {org_filter}
        ),
        
        -- Vector results
        vector_results AS (
            SELECT 
                id,
                content,
                metadata,
                0 as bm25_score,
                1 - (embedding <=> :query_embedding::vector) as vector_similarity,
                2 as source_type  -- 2 = vector
            FROM documents
            WHERE 
                embedding IS NOT NULL
                AND tenant_id = :tenant_id
                {org_filter}
        ),
        
        -- Union of both
        combined AS (
            SELECT * FROM bm25_results
            UNION ALL
            SELECT * FROM vector_results
        ),
        
        -- Rank each source separately
        ranked AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY source_type 
                    ORDER BY 
                        CASE WHEN source_type = 1 THEN bm25_score END DESC NULLS LAST,
                        CASE WHEN source_type = 2 THEN vector_similarity END DESC NULLS LAST
                ) as source_rank
            FROM combined
        )
        
        -- RRF fusion
        SELECT 
            id,
            content,
            metadata,
            SUM(1.0 / (source_rank + :rrf_k)) as rrf_score,
            ARRAY_AGG(DISTINCT source_type) as sources
        FROM ranked
        GROUP BY id, content, metadata
        ORDER BY rrf_score DESC
        LIMIT :top_k
        """
        
        return sql


class BM25Postgres:
    """
    PostgreSQL BM25 search implementation using tsvector.
    
    Sets up and queries PostgreSQL's built-in full-text search
    with BM25 ranking via ts_rank_cd.
    
    Requires:
    - tsvector column with GIN index
    - OR extension like pg_search for true BM25
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
    
    def setup_tsvector_column(self, table: str = "documents", column: str = "content"):
        """
        Set up tsvector column for full-text search.
        
        Creates a generated tsvector column and GIN index.
        """
        sql = f"""
        -- Add tsvector column
        ALTER TABLE {table} 
        ADD COLUMN IF NOT EXISTS search_text tsvector;
        
        -- Populate existing rows
        UPDATE {table} 
        SET search_text = to_tsvector('english', content)
        WHERE search_text IS NULL;
        
        -- Create trigger to keep tsvector updated
        CREATE OR REPLACE FUNCTION documents_search_text_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_text := to_tsvector('english', NEW.content);
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS documents_search_text_trigger ON {table};
        CREATE TRIGGER documents_search_text_trigger
            BEFORE INSERT OR UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION documents_search_text_update();
        
        -- Create GIN index for fast search
        CREATE INDEX IF NOT EXISTS documents_search_text_idx 
        ON {table} USING GIN(search_text);
        """
        
        return sql
    
    def search(
        self,
        query: str,
        table: str = "documents",
        tenant_id: Optional[str] = None,
        top_k: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 search with ts_rank_cd ranking.
        
        Args:
            query: Search query (will be converted to tsquery)
            table: Table to search
            tenant_id: Optional tenant filter for RLS
            top_k: Number of results
            filters: Additional WHERE conditions
            
        Returns:
            List of matching documents with scores
        """
        # Build WHERE clauses
        where_clauses = ["search_text @@ plainto_tsquery('english', :query)"]
        
        params = {"query": query}
        
        if tenant_id:
            where_clauses.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value
                elif isinstance(value, list):
                    placeholders = ",".join([f":{key}_{i}" for i in range(len(value))])
                    where_clauses.append(f"{key} = ANY(ARRAY[{placeholders}]::text[])")
                    for i, v in enumerate(value):
                        params[f"{key}_{i}"] = v
        
        where_sql = " AND ".join(where_clauses)
        
        sql = f"""
        SELECT 
            id,
            content,
            ts_rank_cd(search_text, plainto_tsquery('english', :query)) as score,
            metadata
        FROM {table}
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT :top_k
        """
        params["top_k"] = top_k
        
        # Execute and return
        # Would use self.db_client.execute(sql, params)
        return []
