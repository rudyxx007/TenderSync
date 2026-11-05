-- TenderSync Parent-Child Retrieval Helpers
--
-- The previous version of this migration created `document_hierarchy`,
-- `document_chunks` and friends with `document_id uuid REFERENCES documents(id)`,
-- but `documents.id` is bigserial. The migration failed on every fresh database
-- and none of those tables were ever used by the application.
--
-- Parent and child chunks now live in `documents` (see 002_hybrid_search.sql):
-- children are matched, then expanded to their parent section for context.
-- This migration only installs the helper used for that expansion.
--
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_parent_id_idx ON documents(parent_id);

-- Return the parent sections for a set of parent ids, scoped to one tenant.
CREATE OR REPLACE FUNCTION get_parent_chunks(
    parent_ids uuid[],
    filter_org_id uuid
)
RETURNS TABLE (
    parent_id uuid,
    content text,
    section_path text[],
    contextual_prefix text,
    child_count bigint
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        p.id AS parent_id,
        p.content,
        p.section_path,
        p.contextual_prefix,
        COUNT(c.id) AS child_count
    FROM documents p
    LEFT JOIN documents c
        ON c.parent_id = p.id
       AND c.org_id = p.org_id
    WHERE p.id = ANY(parent_ids)
      AND p.org_id = filter_org_id
      AND p.is_parent = true
    GROUP BY p.id, p.content, p.section_path, p.contextual_prefix;
$$;

INSERT INTO schema_migrations (version) VALUES ('003_parent_child_indexing')
ON CONFLICT (version) DO NOTHING;
