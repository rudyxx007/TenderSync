-- TenderSync Hybrid Retrieval Migration (canonical, idempotent)
--
-- Replaces the previous draft, which referenced a non-existent
-- `organization_members` table and a `schema_migrations` table that was never
-- created. It also created `document_sections` with a uuid FK to the
-- bigserial `documents.id`, so it could never be applied.
--
-- This migration uses `documents` as the chunk store:
--   * parent rows:   is_parent = true,  embedding = NULL
--   * child rows:    is_parent = false, embedding IS NOT NULL, parent_id -> parent row
-- Every row carries org_id (tenant), document_id (one uploaded file) and
-- section_path. Retrieval is strictly scoped to org_id.
--
-- Safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

-- ============================================================
-- 1. documents -> hierarchical chunk store
-- ============================================================
ALTER TABLE documents ADD COLUMN IF NOT EXISTS org_id uuid REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_id uuid;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parent_id uuid REFERENCES documents(id) ON DELETE CASCADE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_parent boolean NOT NULL DEFAULT false;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_index integer;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS section_path text[] NOT NULL DEFAULT '{}';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS contextual_prefix text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS filename text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_text tsvector;

-- Backfill a document_id for legacy rows, grouped by org + filename.
DO $$
DECLARE
    grp record;
    new_id uuid;
BEGIN
    FOR grp IN
        SELECT DISTINCT
            org_id,
            COALESCE(filename, metadata->>'filename', 'legacy-document') AS fname
        FROM documents
        WHERE document_id IS NULL
    LOOP
        new_id := gen_random_uuid();
        UPDATE documents
        SET document_id = new_id,
            filename = COALESCE(filename, grp.fname)
        WHERE document_id IS NULL
          AND org_id IS NOT DISTINCT FROM grp.org_id
          AND COALESCE(filename, metadata->>'filename', 'legacy-document') = grp.fname;
    END LOOP;
END $$;

-- ============================================================
-- 2. Full-text index (contextual prefix + content)
-- ============================================================
CREATE OR REPLACE FUNCTION documents_search_text_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_text := to_tsvector(
        'english',
        COALESCE(NEW.contextual_prefix, '') || ' ' || COALESCE(NEW.content, '')
    );
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_search_text_trigger ON documents;
CREATE TRIGGER documents_search_text_trigger
    BEFORE INSERT OR UPDATE OF content, contextual_prefix ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_search_text_update();

UPDATE documents
SET search_text = to_tsvector(
    'english',
    COALESCE(contextual_prefix, '') || ' ' || COALESCE(content, '')
)
WHERE search_text IS NULL;

DROP INDEX IF EXISTS documents_embedding_idx;
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS documents_search_text_idx ON documents USING GIN(search_text);
CREATE INDEX IF NOT EXISTS documents_org_document_idx ON documents(org_id, document_id);
CREATE INDEX IF NOT EXISTS documents_org_parent_idx ON documents(org_id, is_parent);
CREATE INDEX IF NOT EXISTS documents_parent_id_idx ON documents(parent_id);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING GIN(metadata);

-- ============================================================
-- 3. Query helpers
-- ============================================================

-- Postgres' plainto_tsquery ANDs every term, so natural-language questions
-- ("What ISO certifications are required?") match almost nothing. Fuse the
-- lexemes with OR instead; ts_rank_cd still rewards rarer / more matches.
CREATE OR REPLACE FUNCTION tender_tsquery(query_text text)
RETURNS tsquery
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    terms text;
BEGIN
    SELECT string_agg(DISTINCT lexeme, ' | ')
    INTO terms
    FROM unnest(to_tsvector('english', COALESCE(query_text, '')));

    IF terms IS NULL OR terms = '' THEN
        RETURN NULL;
    END IF;

    RETURN to_tsquery('english', terms);
EXCEPTION WHEN others THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION get_user_org_id()
RETURNS uuid
LANGUAGE sql
SECURITY DEFINER
STABLE
AS $$
    SELECT org_id FROM org_members WHERE user_id = auth.uid() LIMIT 1;
$$;

-- Deprecated vector-only compatibility function, org scoped.
DROP FUNCTION IF EXISTS match_documents(vector, integer, jsonb);
DROP FUNCTION IF EXISTS match_documents(vector, integer, uuid);
DROP FUNCTION IF EXISTS match_documents(vector, integer, uuid, uuid);
DROP FUNCTION IF EXISTS match_documents(vector, integer, uuid, uuid, boolean, text);
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    match_count integer DEFAULT 8,
    filter_org_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity double precision,
    parent_id uuid,
    document_id uuid,
    section_path text[],
    is_parent boolean,
    contextual_prefix text
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        d.id,
        d.content,
        d.metadata,
        (1 - (d.embedding <=> query_embedding))::double precision AS similarity,
        d.parent_id,
        d.document_id,
        d.section_path,
        d.is_parent,
        d.contextual_prefix
    FROM documents d
    WHERE filter_org_id IS NOT NULL
      AND d.org_id = filter_org_id
      AND COALESCE(d.is_parent, false) = false
      AND d.embedding IS NOT NULL
    ORDER BY d.embedding <=> query_embedding
    LIMIT GREATEST(match_count, 0);
$$;

DROP FUNCTION IF EXISTS match_documents_org(vector, integer, uuid);
DROP FUNCTION IF EXISTS match_documents_org(vector, integer, uuid, uuid);
CREATE OR REPLACE FUNCTION match_documents_org(
    query_embedding vector(1024),
    match_count integer DEFAULT 8,
    filter_org_id uuid DEFAULT NULL,
    filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity double precision,
    parent_id uuid,
    document_id uuid,
    section_path text[],
    is_parent boolean,
    contextual_prefix text
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        d.id,
        d.content,
        d.metadata,
        (1 - (d.embedding <=> query_embedding))::double precision AS similarity,
        d.parent_id,
        d.document_id,
        d.section_path,
        d.is_parent,
        d.contextual_prefix
    FROM documents d
    WHERE filter_org_id IS NOT NULL
      AND d.org_id = filter_org_id
      AND (filter_document_id IS NULL OR d.document_id = filter_document_id)
      AND COALESCE(d.is_parent, false) = false
      AND d.embedding IS NOT NULL
    ORDER BY d.embedding <=> query_embedding
    LIMIT GREATEST(match_count, 0);
$$;

-- ============================================================
-- 4. Hybrid BM25 (Postgres ts_rank_cd) + pgvector + RRF
-- ============================================================
DROP FUNCTION IF EXISTS hybrid_search_documents(text, vector, uuid, uuid, integer, integer, integer, integer);
DROP FUNCTION IF EXISTS hybrid_search_documents(text, vector, uuid, uuid, integer, integer, integer, integer, uuid);
CREATE OR REPLACE FUNCTION hybrid_search_documents(
    query_text text,
    query_embedding vector(1024),
    filter_org_id uuid,
    filter_document_id uuid DEFAULT NULL,
    bm25_top_k integer DEFAULT 50,
    vector_top_k integer DEFAULT 50,
    rrf_k integer DEFAULT 50,
    final_top_k integer DEFAULT 8
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    text_rank double precision,
    vector_similarity double precision,
    rrf_score double precision,
    source_type text,
    rank_position integer,
    parent_id uuid,
    document_id uuid,
    section_path text[],
    is_parent boolean,
    contextual_prefix text
)
LANGUAGE sql
STABLE
AS $$
WITH q AS (
    SELECT tender_tsquery(query_text) AS tsq, query_embedding AS emb
),
bm25 AS (
    SELECT
        d.id,
        d.content,
        d.metadata,
        ts_rank_cd(d.search_text, q.tsq)::double precision AS text_rank,
        0::double precision AS vector_similarity,
        'bm25'::text AS source_type,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(d.search_text, q.tsq) DESC)::integer AS source_rank,
        d.parent_id,
        d.document_id,
        d.section_path,
        d.is_parent,
        d.contextual_prefix
    FROM documents d, q
    WHERE filter_org_id IS NOT NULL
      AND d.org_id = filter_org_id
      AND COALESCE(d.is_parent, false) = false
      AND d.search_text IS NOT NULL
      AND q.tsq IS NOT NULL
      AND d.search_text @@ q.tsq
      AND (filter_document_id IS NULL OR d.document_id = filter_document_id)
    ORDER BY text_rank DESC
    LIMIT GREATEST(bm25_top_k, 0)
),
vec AS (
    SELECT
        d.id,
        d.content,
        d.metadata,
        0::double precision AS text_rank,
        (1 - (d.embedding <=> q.emb))::double precision AS vector_similarity,
        'vector'::text AS source_type,
        ROW_NUMBER() OVER (ORDER BY d.embedding <=> q.emb)::integer AS source_rank,
        d.parent_id,
        d.document_id,
        d.section_path,
        d.is_parent,
        d.contextual_prefix
    FROM documents d, q
    WHERE filter_org_id IS NOT NULL
      AND d.org_id = filter_org_id
      AND COALESCE(d.is_parent, false) = false
      AND d.embedding IS NOT NULL
      AND q.emb IS NOT NULL
      AND (filter_document_id IS NULL OR d.document_id = filter_document_id)
    ORDER BY d.embedding <=> q.emb
    LIMIT GREATEST(vector_top_k, 0)
),
combined AS (
    SELECT * FROM bm25
    UNION ALL
    SELECT * FROM vec
),
fused AS (
    SELECT
        id,
        content,
        metadata,
        MAX(text_rank) AS text_rank,
        MAX(vector_similarity) AS vector_similarity,
        SUM(1.0 / (source_rank + GREATEST(rrf_k, 1)))::double precision AS rrf_score,
        CASE
            WHEN COUNT(DISTINCT source_type) > 1 THEN 'hybrid'
            ELSE MIN(source_type)
        END AS source_type,
        parent_id,
        document_id,
        section_path,
        is_parent,
        contextual_prefix
    FROM combined
    GROUP BY id, content, metadata, parent_id, document_id, section_path, is_parent, contextual_prefix
)
SELECT
    id,
    content,
    metadata,
    text_rank,
    vector_similarity,
    rrf_score,
    source_type,
    ROW_NUMBER() OVER (ORDER BY rrf_score DESC, id)::integer AS rank_position,
    parent_id,
    document_id,
    section_path,
    is_parent,
    contextual_prefix
FROM fused
ORDER BY rrf_score DESC, id
LIMIT GREATEST(final_top_k, 0);
$$;

-- ============================================================
-- 5. Row Level Security (tenant isolation)
-- ============================================================
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users read own documents" ON documents;
DROP POLICY IF EXISTS "Users insert own documents" ON documents;
DROP POLICY IF EXISTS "Users update own documents" ON documents;
DROP POLICY IF EXISTS "Users read own tenant documents" ON documents;
DROP POLICY IF EXISTS "Users insert own tenant documents" ON documents;
DROP POLICY IF EXISTS "Users update own tenant documents" ON documents;
DROP POLICY IF EXISTS "Orgs read own documents" ON documents;
DROP POLICY IF EXISTS "Orgs insert own documents" ON documents;
DROP POLICY IF EXISTS "documents_select_own_org" ON documents;
DROP POLICY IF EXISTS "documents_insert_own_org" ON documents;
DROP POLICY IF EXISTS "documents_update_own_org" ON documents;
DROP POLICY IF EXISTS "documents_delete_own_org" ON documents;

CREATE POLICY "documents_select_own_org" ON documents
    FOR SELECT TO authenticated
    USING (org_id = get_user_org_id());

CREATE POLICY "documents_insert_own_org" ON documents
    FOR INSERT TO authenticated
    WITH CHECK (org_id = get_user_org_id());

CREATE POLICY "documents_update_own_org" ON documents
    FOR UPDATE TO authenticated
    USING (org_id = get_user_org_id())
    WITH CHECK (org_id = get_user_org_id());

CREATE POLICY "documents_delete_own_org" ON documents
    FOR DELETE TO authenticated
    USING (org_id = get_user_org_id());

-- The FastAPI backend uses the service role key, which bypasses RLS.
-- RLS protects any direct anon/authenticated client access to Supabase.

INSERT INTO schema_migrations (version) VALUES ('002_hybrid_search')
ON CONFLICT (version) DO NOTHING;
