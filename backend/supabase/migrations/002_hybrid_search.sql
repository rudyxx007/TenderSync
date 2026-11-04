-- TenderSync Enterprise Hybrid Search Migration
-- Adds BM25 full-text search, HNSW vector index, and RRF fusion functions
-- Run after 001_initial_schema.sql

-- ============================================================
-- 1. Enable required extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;
-- tsearch2 is built-in, no extension needed

-- ============================================================
-- 2. Add full-text search columns to documents table
-- ============================================================
-- Add tsvector column for BM25 search
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS search_text tsvector;

-- Populate existing rows with tsvector from content
UPDATE documents 
SET search_text = to_tsvector('english', content)
WHERE search_text IS NULL AND content IS NOT NULL;

-- Create trigger function to keep tsvector updated
CREATE OR REPLACE FUNCTION documents_search_text_update() 
RETURNS trigger AS $$
BEGIN
    NEW.search_text := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Create trigger for automatic tsvector updates
DROP TRIGGER IF EXISTS documents_search_text_trigger ON documents;
CREATE TRIGGER documents_search_text_trigger
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW 
    EXECUTE FUNCTION documents_search_text_update();

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS documents_search_text_idx 
ON documents USING GIN(search_text);

-- ============================================================
-- 3. Upgrade vector index from IVFFlat to HNSW (better accuracy)
-- ============================================================
-- Drop old IVFFlat index if exists
DROP INDEX IF EXISTS documents_embedding_idx;

-- Create HNSW index for better recall and accuracy
-- HNSW provides better search quality than IVFFlat at cost of larger index
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================
-- 4. Add tenant_id column for proper multi-tenant isolation
-- ============================================================
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS tenant_id uuid 
REFERENCES organizations(id) ON DELETE CASCADE;

-- Also ensure we have organization_id for broader org-level filtering
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS organization_id uuid
REFERENCES organizations(id) ON DELETE CASCADE;

-- Create indexes for tenant filtering
CREATE INDEX IF NOT EXISTS documents_tenant_id_idx 
ON documents(tenant_id);

CREATE INDEX IF NOT EXISTS documents_organization_id_idx 
ON documents(organization_id);

-- ============================================================
-- 5. Create document hierarchy tables for parent-child indexing
-- ============================================================
-- Table to store document sections/subsections
CREATE TABLE IF NOT EXISTS document_sections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_section_id uuid REFERENCES document_sections(id) ON DELETE CASCADE,
    title text,
    level integer DEFAULT 1,
    start_position integer,
    end_position integer,
    content text,
    content_embedding vector(1024),
    summary text,
    created_at timestamptz DEFAULT now()
);

-- Index for section hierarchy queries
CREATE INDEX IF NOT EXISTS document_sections_document_id_idx 
ON document_sections(document_id);

CREATE INDEX IF NOT EXISTS document_sections_parent_idx 
ON document_sections(parent_section_id);

CREATE INDEX IF NOT EXISTS document_sections_embedding_idx 
ON document_sections USING hnsw (content_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Table to store parent-child relationships
CREATE TABLE IF NOT EXISTS chunk_relationships (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_chunk_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    child_chunk_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relationship_type text DEFAULT 'parent_child',
    created_at timestamptz DEFAULT now(),
    UNIQUE(parent_chunk_id, child_chunk_id)
);

-- ============================================================
-- 6. RRF (Reciprocal Rank Fusion) functions
-- ============================================================

-- RRF score calculation function
CREATE OR REPLACE FUNCTION rrf_score(rank bigint, k int DEFAULT 50)
RETURNS numeric
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $$
SELECT COALESCE(1.0 / ($1 + $2), 0.0);
$$;

-- Hybrid search function with BM25 + Vector fusion using RRF
-- This is the main hybrid retrieval function for TenderSync
CREATE OR REPLACE FUNCTION hybrid_search_documents(
    query_text text,
    query_embedding vector(1024),
    tenant_id uuid,
    organization_id uuid DEFAULT NULL,
    bm25_top_k int DEFAULT 50,
    vector_top_k int DEFAULT 50,
    rrf_k int DEFAULT 50,
    final_top_k int DEFAULT 50
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    rrf_score numeric,
    bm25_score numeric,
    vector_similarity numeric,
    source_type text,
    rank_position int
)
LANGUAGE sql
STABLE
AS $$
WITH 
-- BM25 search results
bm25_results AS (
    SELECT 
        d.id,
        d.content,
        d.metadata,
        ts_rank_cd(d.search_text, plainto_tsquery('english', query_text)) as bm25_score,
        0 as vector_similarity,
        'bm25' as source_type,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(d.search_text, plainto_tsquery('english', query_text)) DESC) as source_rank
    FROM documents d
    WHERE 
        d.search_text @@ plainto_tsquery('english', query_text)
        AND (d.tenant_id = tenant_id OR d.organization_id = organization_id)
        AND d.embedding IS NOT NULL
    ORDER BY bm25_score DESC
    LIMIT bm25_top_k
),
-- Vector search results
vector_results AS (
    SELECT 
        d.id,
        d.content,
        d.metadata,
        0 as bm25_score,
        1 - (d.embedding <=> query_embedding) as vector_similarity,
        'vector' as source_type,
        ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding) as source_rank
    FROM documents d
    WHERE 
        d.embedding IS NOT NULL
        AND (d.tenant_id = tenant_id OR d.organization_id = organization_id)
    ORDER BY d.embedding <=> query_embedding
    LIMIT vector_top_k
),
-- Combined results
combined AS (
    SELECT * FROM bm25_results
    UNION ALL
    SELECT * FROM vector_results
),
-- RRF fusion
rrf_combined AS (
    SELECT 
        id,
        content,
        metadata,
        bm25_score,
        vector_similarity,
        source_type,
        SUM(rrf_score(source_rank, rrf_k)) as rrf_score
    FROM combined
    GROUP BY id, content, metadata, bm25_score, vector_similarity, source_type
)
-- Final ranked results
SELECT 
    id,
    content,
    metadata,
    rrf_score,
    bm25_score,
    vector_similarity,
    source_type,
    ROW_NUMBER() OVER (ORDER BY rrf_score DESC) as rank_position
FROM rrf_combined
ORDER BY rrf_score DESC
LIMIT final_top_k
$$;

-- ============================================================
-- 7. Section-aware search (for parent-child retrieval)
-- ============================================================
CREATE OR REPLACE FUNCTION search_with_section_context(
    query_text text,
    query_embedding vector(1024),
    document_id bigint,
    section_levels int[] DEFAULT ARRAY[1,2],
    top_k int DEFAULT 10
)
RETURNS TABLE (
    chunk_id bigint,
    chunk_content text,
    section_path text[],
    section_summary text,
    chunk_similarity numeric,
    section_similarity numeric
)
LANGUAGE sql
STABLE
AS $$
WITH 
-- Find relevant sections
relevant_sections AS (
    SELECT 
        ds.id,
        ds.title,
        ds.level,
        ds.parent_section_id,
        ds.content,
        ds.summary,
        ds.content_embedding,
        1 - (ds.content_embedding <=> query_embedding) as section_similarity,
        ARRAY[
            (SELECT ARRAY_AGG(ds2.title ORDER BY ds2.level)
             FROM document_sections ds2
             WHERE ds2.id = ds.id 
                OR ds2.id = ANY(
                    SELECT ARRAY_AGG(ds3.id ORDER BY ds3.level DESC)
                    FROM document_sections ds3
                    WHERE ds3.parent_section_id = ds.id
                )
            )
        ] as section_path
    FROM document_sections ds
    WHERE ds.document_id = document_id
        AND ds.level = ANY(section_levels)
    ORDER BY section_similarity DESC
    LIMIT top_k * 2
),
-- Find specific chunks within sections
chunk_matches AS (
    SELECT 
        d.id as chunk_id,
        d.content as chunk_content,
        rs.section_path,
        rs.summary as section_summary,
        1 - (d.embedding <=> query_embedding) as chunk_similarity,
        rs.section_similarity
    FROM documents d
    CROSS JOIN relevant_sections rs
    WHERE d.id IN (
        SELECT child_chunk_id 
        FROM chunk_relationships 
        WHERE parent_chunk_id IN (SELECT id FROM relevant_sections)
    ) OR d.id IN (SELECT id FROM relevant_sections)
)
SELECT 
    chunk_id,
    chunk_content,
    section_path,
    section_summary,
    chunk_similarity,
    section_similarity
FROM chunk_matches
ORDER BY (chunk_similarity + COALESCE(section_similarity, 0)) DESC
LIMIT top_k
$$;

-- ============================================================
-- 8. Enhanced RLS policies for multi-tenant isolation
-- ============================================================

-- Enable RLS on new tables
ALTER TABLE document_sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_relationships ENABLE ROW LEVEL SECURITY;

-- Drop existing policies and create enhanced ones
DROP POLICY IF EXISTS "Users read own documents" ON documents;
DROP POLICY IF EXISTS "Users insert own documents" ON documents;

-- Enhanced document policies with tenant isolation
CREATE POLICY "Users read own tenant documents" ON documents
    FOR SELECT
    TO authenticated
    USING (
        (SELECT auth.uid()) = user_id 
        OR tenant_id IN (
            SELECT organization_id FROM organization_members 
            WHERE user_id = auth.uid() AND role != 'viewer'
        )
    );

CREATE POLICY "Users insert own tenant documents" ON documents
    FOR INSERT
    TO authenticated
    WITH CHECK (
        (SELECT auth.uid()) = user_id
        OR organization_id IN (
            SELECT organization_id FROM organization_members 
            WHERE user_id = auth.uid() AND role = 'owner'
        )
    );

CREATE POLICY "Users update own tenant documents" ON documents
    FOR UPDATE
    TO authenticated
    USING ((SELECT auth.uid()) = user_id)
    WITH CHECK ((SELECT auth.uid()) = user_id);

-- Policies for document_sections
CREATE POLICY "Users read own tenant sections" ON document_sections
    FOR SELECT
    TO authenticated
    USING (
        document_id IN (
            SELECT d.id FROM documents d
            WHERE (SELECT auth.uid()) = d.user_id
            OR d.tenant_id IN (
                SELECT organization_id FROM organization_members 
                WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users insert own tenant sections" ON document_sections
    FOR INSERT
    TO authenticated
    WITH CHECK (
        document_id IN (
            SELECT d.id FROM documents d
            WHERE d.user_id = auth.uid()
        )
    );

-- Policies for chunk_relationships
CREATE POLICY "Users read own tenant chunk relationships" ON chunk_relationships
    FOR SELECT
    TO authenticated
    USING (
        parent_chunk_id IN (
            SELECT id FROM documents WHERE user_id = auth.uid()
        )
        OR child_chunk_id IN (
            SELECT id FROM documents WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users insert own tenant chunk relationships" ON chunk_relationships
    FOR INSERT
    TO authenticated
    WITH CHECK (
        parent_chunk_id IN (
            SELECT id FROM documents WHERE user_id = auth.uid()
        )
        AND child_chunk_id IN (
            SELECT id FROM documents WHERE user_id = auth.uid()
        )
    );

-- ============================================================
-- 9. Update match_documents function for hybrid support
-- ============================================================
-- Keep existing function but enhance with optional hybrid mode
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    match_count int DEFAULT 8,
    filter_user_id uuid DEFAULT NULL,
    filter_tenant_id uuid DEFAULT NULL,
    use_hybrid boolean DEFAULT false,
    query_text text DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    similarity float,
    search_type text
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF use_hybrid AND query_text IS NOT NULL AND filter_tenant_id IS NOT NULL THEN
        -- Use hybrid search
        RETURN QUERY
        SELECT 
            hs.id,
            hs.content,
            hs.metadata,
            hs.rrf_score::float as similarity,
            hs.source_type as search_type
        FROM hybrid_search_documents(
            query_text,
            query_embedding,
            filter_tenant_id,
            NULL::uuid,
            50,  -- bm25_top_k
            50,  -- vector_top_k
            50,  -- rrf_k
            match_count
        ) hs;
    ELSE
        -- Use simple vector search (original behavior)
        RETURN QUERY
        SELECT
            documents.id,
            documents.content,
            documents.metadata,
            1 - (documents.embedding <=> query_embedding) AS similarity,
            'vector' as search_type
        FROM documents
        WHERE
            (filter_user_id IS NULL OR documents.user_id = filter_user_id)
            AND (filter_tenant_id IS NULL OR documents.tenant_id = filter_tenant_id
                 OR documents.organization_id = filter_tenant_id)
            AND documents.embedding IS NOT NULL
        ORDER BY documents.embedding <=> query_embedding
        LIMIT match_count;
    END IF;
END;
$$;

-- ============================================================
-- 10. Performance indexes for common query patterns
-- ============================================================

-- Composite index for tenant + vector queries
CREATE INDEX IF NOT EXISTS documents_tenant_embedding_idx 
ON documents(tenant_id, embedding) 
WHERE embedding IS NOT NULL;

-- Index for metadata queries
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING GIN(metadata);

-- Index for created_at sorting
CREATE INDEX IF NOT EXISTS documents_created_at_idx 
ON documents(created_at DESC);

-- ============================================================
-- 11. Migration tracking
-- ============================================================
INSERT INTO schema_migrations (version, applied_at)
VALUES ('002_hybrid_search', NOW())
ON CONFLICT (version) DO NOTHING;
