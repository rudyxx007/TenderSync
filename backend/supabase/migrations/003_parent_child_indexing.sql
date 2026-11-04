-- TenderSync Parent-Child Indexing Migration
-- Implements hierarchical document indexing for contextual retrieval
-- Run after 002_hybrid_search.sql

-- ============================================================
-- 1. Document hierarchy tables
-- ============================================================

-- Documents table (enhanced with parent-child support)
-- Note: This extends the existing documents table

-- Section hierarchy table for storing document structure
CREATE TABLE IF NOT EXISTS document_hierarchy (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id uuid REFERENCES document_hierarchy(id) ON DELETE CASCADE,
    level integer NOT NULL DEFAULT 1,
    title text,
    heading_type text, -- 'h1', 'h2', 'h3', 'section', 'subsection', etc.
    content_start integer, -- Character offset in original document
    content_end integer,
    content_excerpt text, -- First 200 chars of section content
    full_content text, -- Complete section content
    content_embedding vector(1024),
    section_summary text, -- AI-generated section summary
    chunk_count integer DEFAULT 0, -- Number of child chunks
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Index for hierarchy traversal
CREATE INDEX IF NOT EXISTS document_hierarchy_document_idx 
ON document_hierarchy(document_id);

CREATE INDEX IF NOT EXISTS document_hierarchy_parent_idx 
ON document_hierarchy(parent_id);

CREATE INDEX IF NOT EXISTS document_hierarchy_level_idx 
ON document_hierarchy(document_id, level);

-- Vector index for section search
CREATE INDEX IF NOT EXISTS document_hierarchy_embedding_idx 
ON document_hierarchy USING hnsw (content_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Chunk table for storing individual chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    hierarchy_id uuid REFERENCES document_hierarchy(id) ON DELETE CASCADE,
    parent_chunk_id uuid REFERENCES document_chunks(id) ON DELETE CASCADE,
    chunk_order integer NOT NULL,
    content text NOT NULL,
    content_embedding vector(1024),
    contextualized_content text, -- Content with contextual prefix
    contextual_prefix text, -- The context prepended to content
    start_position integer,
    end_position integer,
    token_count integer,
    metadata jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);

-- Indexes for chunk retrieval
CREATE INDEX IF NOT EXISTS document_chunks_document_idx 
ON document_chunks(document_id, chunk_order);

CREATE INDEX IF NOT EXISTS document_chunks_hierarchy_idx 
ON document_chunks(hierarchy_id);

CREATE INDEX IF NOT EXISTS document_chunks_parent_idx 
ON document_chunks(parent_chunk_id);

-- Vector index for chunk similarity search
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
ON document_chunks USING hnsw (content_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for metadata queries
CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx 
ON document_chunks USING GIN(metadata);

-- ============================================================
-- 2. Contextual retrieval support
-- ============================================================

-- Table to store document-level summaries
CREATE TABLE IF NOT EXISTS document_summaries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    summary_type text NOT NULL DEFAULT 'auto', -- 'auto', 'manual', 'section'
    summary text NOT NULL,
    method text DEFAULT 'llm', -- 'llm', 'extractive', 'bleu'
    token_count integer,
    created_at timestamptz DEFAULT now(),
    UNIQUE(document_id, summary_type)
);

CREATE INDEX IF NOT EXISTS document_summaries_document_idx 
ON document_summaries(document_id);

-- Table to store chunk context information
CREATE TABLE IF NOT EXISTS chunk_context (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id uuid NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    context_type text NOT NULL, -- 'document_summary', 'section_summary', 'hierarchy_path', 'surrounding_text'
    context text NOT NULL,
    token_count integer,
    created_at timestamptz DEFAULT now(),
    UNIQUE(chunk_id, context_type)
);

CREATE INDEX IF NOT EXISTS chunk_context_chunk_idx 
ON chunk_context(chunk_id);

-- ============================================================
-- 3. Contextual retrieval functions
-- ============================================================

-- Function to generate contextual prefix for a chunk
CREATE OR REPLACE FUNCTION generate_chunk_context(
    chunk_id uuid,
    context_types text[] DEFAULT ARRAY['document_summary', 'section_summary', 'hierarchy_path']
)
RETURNS text
LANGUAGE plpgsql
AS $$
DECLARE
    context_parts text[];
    doc_summary text;
    section_summary text;
    hierarchy_path text;
    chunk_info record;
BEGIN
    -- Get chunk info
    SELECT * INTO chunk_info
    FROM document_chunks
    WHERE id = chunk_id;
    
    IF chunk_info IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Build context from requested types
    FOREACH context_types IN ARRAY context_types
    LOOP
        CASE context_types
            WHEN 'document_summary' THEN
                SELECT summary INTO doc_summary
                FROM document_summaries
                WHERE document_id = chunk_info.document_id
                AND summary_type = 'auto'
                LIMIT 1;
                
                IF doc_summary IS NOT NULL THEN
                    context_parts := array_append(
                        context_parts,
                        'DOCUMENT CONTEXT: ' || doc_summary
                    );
                END IF;
                
            WHEN 'section_summary' THEN
                IF chunk_info.hierarchy_id IS NOT NULL THEN
                    SELECT section_summary INTO section_summary
                    FROM document_hierarchy
                    WHERE id = chunk_info.hierarchy_id
                    LIMIT 1;
                    
                    IF section_summary IS NOT NULL THEN
                        context_parts := array_append(
                            context_parts,
                            'SECTION CONTEXT: ' || section_summary
                        );
                    END IF;
                END IF;
                
            WHEN 'hierarchy_path' THEN
                IF chunk_info.hierarchy_id IS NOT NULL THEN
                    -- Build hierarchy path from parent sections
                    WITH RECURSIVE section_path AS (
                        SELECT 
                            id,
                            title,
                            level,
                            parent_id,
                            ARRAY[title] as path
                        FROM document_hierarchy
                        WHERE id = chunk_info.hierarchy_id
                        
                        UNION ALL
                        
                        SELECT 
                            h.id,
                            h.title,
                            h.level,
                            h.parent_id,
                            sp.path || h.title
                        FROM document_hierarchy h
                        JOIN section_path sp ON h.id = sp.parent_id
                    )
                    SELECT path INTO hierarchy_path
                    FROM section_path
                    WHERE parent_id IS NULL
                    LIMIT 1;
                    
                    IF hierarchy_path IS NOT NULL THEN
                        context_parts := array_append(
                            context_parts,
                            'SECTION PATH: ' || array_to_string(hierarchy_path, ' > ')
                        );
                    END IF;
                END IF;
                
            WHEN 'surrounding_text' THEN
                -- Get surrounding chunk content
                SELECT string_agg(co.content, ' ... ' ORDER BY co.chunk_order)
                INTO context_parts[array_length(context_parts, 1) + 1]
                FROM document_chunks co
                WHERE co.document_id = chunk_info.document_id
                AND co.id != chunk_info.id
                AND ABS(co.chunk_order - chunk_info.chunk_order) <= 2
                LIMIT 3;
        END CASE;
    END LOOP;
    
    RETURN array_to_string(context_parts, E'\n\n');
END;
$$;

-- Function to get chunk with full context
CREATE OR REPLACE FUNCTION get_chunk_with_context(
    chunk_id uuid,
    include_surrounding boolean DEFAULT true
)
RETURNS TABLE (
    chunk_id uuid,
    original_content text,
    contextualized_content text,
    context_prefix text,
    section_path text[],
    document_summary text,
    section_summary text,
    surrounding_chunks text[]
)
LANGUAGE plpgsql
AS $$
DECLARE
    ctx_prefix text;
    doc_summary text;
    sec_summary text;
    sec_path text[];
    chunk_doc record;
BEGIN
    -- Get the chunk
    SELECT 
        dc.id,
        dc.content,
        dc.contextualized_content,
        dc.contextual_prefix,
        dh.title,
        dh.level,
        dh.parent_id,
        dc.document_id,
        dc.chunk_order
    INTO chunk_doc
    FROM document_chunks dc
    LEFT JOIN document_hierarchy dh ON dc.hierarchy_id = dh.id
    WHERE dc.id = chunk_id;
    
    IF chunk_doc IS NULL THEN
        RETURN;
    END IF;
    
    -- Get document summary
    SELECT summary INTO doc_summary
    FROM document_summaries
    WHERE document_id = chunk_doc.document_id
    AND summary_type = 'auto'
    LIMIT 1;
    
    -- Get section summary
    IF chunk_doc.hierarchy_id IS NOT NULL THEN
        SELECT section_summary INTO sec_summary
        FROM document_hierarchy
        WHERE id = chunk_doc.hierarchy_id
        LIMIT 1;
    END IF;
    
    -- Build section path
    IF chunk_doc.hierarchy_id IS NOT NULL THEN
        WITH RECURSIVE section_path AS (
            SELECT 
                id,
                title,
                ARRAY[title] as path
            FROM document_hierarchy
            WHERE id = chunk_doc.hierarchy_id
            
            UNION ALL
            
            SELECT 
                h.id,
                h.title,
                sp.path || h.title
            FROM document_hierarchy h
            JOIN section_path sp ON h.id = sp.parent_id
        )
        SELECT path INTO sec_path
        FROM section_path
        WHERE parent_id IS NULL
        LIMIT 1;
    END IF;
    
    -- Get surrounding chunks if requested
    RETURN QUERY
    SELECT 
        chunk_doc.id,
        chunk_doc.content,
        chunk_doc.contextualized_content,
        chunk_doc.contextual_prefix,
        sec_path,
        doc_summary,
        sec_summary,
        CASE WHEN include_surrounding THEN (
            SELECT array_agg(co.content ORDER BY co.chunk_order)
            FROM document_chunks co
            WHERE co.document_id = chunk_doc.document_id
            AND co.id != chunk_doc.id
            AND ABS(co.chunk_order - chunk_doc.chunk_order) <= 2
            LIMIT 3
        ) ELSE ARRAY[]::text[] END
    ;
END;
$$;

-- ============================================================
-- 4. Hierarchical search function
-- ============================================================

-- Search that returns both chunks and their parent contexts
CREATE OR REPLACE FUNCTION hierarchical_search(
    query_embedding vector(1024),
    query_text text DEFAULT NULL,
    tenant_id uuid,
    organization_id uuid DEFAULT NULL,
    search_level text DEFAULT 'chunk', -- 'chunk', 'section', 'document'
    top_k int DEFAULT 10,
    expand_to_parents boolean DEFAULT true,
    parent_context_window int DEFAULT 1
)
RETURNS TABLE (
    result_type text,
    id uuid,
    content text,
    context text,
    similarity numeric,
    hierarchy_level integer,
    section_title text,
    section_path text[]
)
LANGUAGE sql
STABLE
AS $$
WITH 
-- Search chunks first (most precise)
chunk_matches AS (
    SELECT 
        dc.id,
        dc.content,
        dc.contextualized_content,
        dc.hierarchy_id,
        dc.chunk_order,
        1 - (dc.content_embedding <=> query_embedding) as similarity,
        1 as result_level
    FROM document_chunks dc
    WHERE dc.document_id IN (
        SELECT d.id FROM documents d
        WHERE (d.tenant_id = tenant_id OR d.organization_id = organization_id)
    )
    ORDER BY similarity DESC
    LIMIT top_k * 3  -- Get more to allow filtering
),
-- Expand to parent sections
chunk_with_parents AS (
    SELECT 
        cm.*,
        dh.id as section_id,
        dh.title as section_title,
        dh.level as section_level,
        dh.section_summary,
        dh.content as section_content
    FROM chunk_matches cm
    LEFT JOIN document_hierarchy dh ON cm.hierarchy_id = dh.id
),
-- Get section path for each match
chunk_with_paths AS (
    SELECT 
        cwp.*,
        (SELECT ARRAY_AGG(h.title ORDER BY h.level)
         FROM document_hierarchy h
         WHERE h.id = cwp.section_id
         OR h.id = ANY(
             SELECT hierarchy_id FROM document_chunks
             WHERE hierarchy_id = cwp.section_id
         )
        ) as section_path
    FROM chunk_with_parents cwp
)
-- Final results: either chunks or sections
SELECT 
    CASE 
        WHEN search_level = 'section' AND section_id IS NOT NULL THEN 'section'
        WHEN search_level = 'document' THEN 'document'
        ELSE 'chunk'
    END as result_type,
    CASE 
        WHEN search_level = 'section' AND section_id IS NOT NULL THEN section_id::uuid
        ELSE id::uuid
    END as id,
    CASE 
        WHEN search_level = 'section' AND section_id IS NOT NULL THEN section_content
        ELSE content
    END as content,
    CASE 
        WHEN search_level = 'section' AND section_id IS NOT NULL THEN section_summary
        ELSE contextualized_content
    END as context,
    similarity,
    CASE 
        WHEN search_level = 'section' AND section_id IS NOT NULL THEN section_level
        ELSE 0
    END as hierarchy_level,
    section_title,
    section_path
FROM chunk_with_paths
GROUP BY 
    id, content, contextualized_content, section_content, section_summary,
    similarity, section_id, section_title, section_level, section_path
ORDER BY similarity DESC
LIMIT top_k
$$;

-- ============================================================
-- 5. Trigger functions for maintaining hierarchy
-- ============================================================

-- Trigger to update hierarchy chunk count
CREATE OR REPLACE FUNCTION update_hierarchy_chunk_count()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE document_hierarchy
        SET chunk_count = chunk_count + 1,
            updated_at = NOW()
        WHERE id = NEW.hierarchy_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE document_hierarchy
        SET chunk_count = GREATEST(chunk_count - 1, 0),
            updated_at = NOW()
        WHERE id = OLD.hierarchy_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger for chunk count maintenance
DROP TRIGGER IF EXISTS update_chunk_count ON document_chunks;
CREATE TRIGGER update_chunk_count
    AFTER INSERT OR DELETE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_hierarchy_chunk_count();

-- ============================================================
-- 6. Multi-tenant RLS for hierarchy tables
-- ============================================================

-- Enable RLS
ALTER TABLE document_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_context ENABLE ROW LEVEL SECURITY;

-- Document hierarchy policies
CREATE POLICY "Users read own tenant hierarchy" ON document_hierarchy
    FOR SELECT
    TO authenticated
    USING (
        document_id IN (
            SELECT d.id FROM documents d
            WHERE d.user_id = auth.uid()
            OR d.tenant_id IN (
                SELECT organization_id FROM organization_members 
                WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users insert own tenant hierarchy" ON document_hierarchy
    FOR INSERT
    TO authenticated
    WITH CHECK (
        document_id IN (
            SELECT d.id FROM documents d WHERE d.user_id = auth.uid()
        )
    );

-- Document chunks policies
CREATE POLICY "Users read own tenant chunks" ON document_chunks
    FOR SELECT
    TO authenticated
    USING (
        document_id IN (
            SELECT d.id FROM documents d
            WHERE d.user_id = auth.uid()
            OR d.tenant_id IN (
                SELECT organization_id FROM organization_members 
                WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users insert own tenant chunks" ON document_chunks
    FOR INSERT
    TO authenticated
    WITH CHECK (
        document_id IN (
            SELECT d.id FROM documents d WHERE d.user_id = auth.uid()
        )
    );

-- Document summaries policies
CREATE POLICY "Users read own tenant summaries" ON document_summaries
    FOR SELECT
    TO authenticated
    USING (
        document_id IN (
            SELECT d.id FROM documents d
            WHERE d.user_id = auth.uid()
            OR d.tenant_id IN (
                SELECT organization_id FROM organization_members 
                WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users insert own tenant summaries" ON document_summaries
    FOR INSERT
    TO authenticated
    WITH CHECK (
        document_id IN (
            SELECT d.id FROM documents d WHERE d.user_id = auth.uid()
        )
    );

-- Chunk context policies
CREATE POLICY "Users read own tenant chunk context" ON chunk_context
    FOR SELECT
    TO authenticated
    USING (
        chunk_id IN (
            SELECT dc.id FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.user_id = auth.uid()
            OR d.tenant_id IN (
                SELECT organization_id FROM organization_members 
                WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users insert own tenant chunk context" ON chunk_context
    FOR INSERT
    TO authenticated
    WITH CHECK (
        chunk_id IN (
            SELECT dc.id FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.user_id = auth.uid()
        )
    );

-- ============================================================
-- 7. Utility functions for hierarchy management
-- ============================================================

-- Function to build hierarchy from document content
CREATE OR REPLACE FUNCTION build_document_hierarchy(
    doc_id uuid,
    content text,
    title text DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    root_section_id uuid;
    section_pattern text;
    section_matches text[];
    section_bounds integer[][];
    i integer;
    current_level integer;
    current_title text;
    current_start integer;
    prev_end integer := 0;
BEGIN
    -- Insert document-level entry as root
    INSERT INTO document_hierarchy (
        document_id, level, title, heading_type,
        content_start, content_end, content_excerpt, full_content
    ) VALUES (
        doc_id, 0, title, 'document',
        0, length(content),
        left(content, 200),
        content
    ) RETURNING id INTO root_section_id;
    
    -- Simple heading detection pattern
    section_pattern := '^(#{1,6}\s+.+|^\s*\d+\.\d*\.?\s+[A-Z].+|^\s*[A-Z][A-Z\s]{5,})$';
    
    -- Find section boundaries (simplified)
    FOR i IN 0..10 LOOP
        -- Find next heading
        -- Would use regex or NLP-based section detection here
        -- Placeholder: every 2000 chars is a new section
        current_start := prev_end + 1;
        current_end := LEAST(prev_end + 2000, length(content));
        current_title := 'Section ' || (i + 1);
        current_level := 1;
        
        IF current_start > length(content) THEN
            EXIT;
        END IF;
        
        INSERT INTO document_hierarchy (
            document_id, parent_id, level, title, heading_type,
            content_start, content_end, content_excerpt, full_content
        ) VALUES (
            doc_id, root_section_id, current_level,
            current_title, 'section',
            current_start, current_end,
            left(substr(content, current_start, current_end - current_start + 1), 200),
            substr(content, current_start, current_end - current_start + 1)
        );
        
        prev_end := current_end;
    END LOOP;
    
    RETURN root_section_id;
END;
$$;

-- ============================================================
-- 8. Migration tracking
-- ============================================================
INSERT INTO schema_migrations (version, applied_at)
VALUES ('003_parent_child_indexing', NOW())
ON CONFLICT (version) DO NOTHING;
