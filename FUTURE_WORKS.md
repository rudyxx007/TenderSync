# Future Works

**Last updated:** 2026-11-06  
**Branch:** `feature/enterprise-hybrid-rag-platform`

This is the tracked backlog for the enterprise RAG work. It records work that is
intentionally deferred, not work that has already been delivered. Benchmark
results and known limitations are reported separately in `RAG_EVALUATION.md`.

## 1. Move ranking and retrieval into SQL, next to the data

### Why

TenderSync should rank chunks in Postgres, not pull every candidate into Python.
The database already has the tenant boundary, the pgvector HNSW index, the
full-text GIN index, and the document/chunk rows. Doing the work in SQL:

- keeps tenant isolation in the same statement as ranking, so an application
  bug cannot return another organisation’s rows;
- avoids transferring the candidate set over the network;
- uses the indexes instead of a Python sort;
- makes the query plan, latency, and row counts observable;
- gives one round trip for lexical search, vector search, fusion, parent
  expansion and citation metadata.

### Target design

1. `hybrid_search_documents(org_id, query_text, query_embedding, ...)` runs as a
   `SECURITY INVOKER` SQL function with an explicit `org_id` predicate on every
   CTE and a matching RLS policy.
2. Compute `ts_rank_cd` over the GIN `tsvector` and cosine distance over the
   pgvector HNSW index independently.
3. Fuse with Reciprocal Rank Fusion in SQL (`1 / (rank + k)`) or a documented
   weighted sum; keep the candidate lists and component scores for audit.
4. Expand parent sections in the same query, not in a second Python pass.
5. Return only the final top-k plus the citation metadata the API needs.
6. Add `EXPLAIN (ANALYZE, BUFFERS)` regression tests at production-like row
   counts; fail the performance job if the plan loses the HNSW/GIN index or the
   p95 exceeds the agreed budget.

### Acceptance criteria

- A cross-tenant integration test returns zero rows for every retrieval path:
  vector, lexical, hybrid, parent expansion and citation lookup.
- The query is one round trip for the normal path.
- The plan uses the vector and full-text indexes; no sequential scan on the
  chunk table at the target scale.
- p95 retrieval latency is measured and recorded, not estimated.
- The SQL function is versioned in a migration and covered by a test that
  applies the migration to a clean Postgres instance.

### Deferred alternatives

- **OpenFGA/OPA for per-document grants:** useful when external counsel or
  deal teams need relationship-based access. Defer until the product has
  document-sharing requirements that a simple `org_id` cannot express.
- **A separate search service (OpenSearch/Vespa):** defer; it duplicates the
  authoritative ACL and adds an index-consistency problem before scale
  requires it.
- **Client-side ranking:** reject; it cannot enforce tenant isolation and is
  slower.

## 2. Complete the per-benchmark model comparison

Run each benchmark as a separate job; never pool scores.

- [ ] CUAD: chunking (done for the current token budget), embedding, rerank.
- [ ] FinanceBench: parse all source filings that are legally downloadable;
      record the 7 unavailable PDFs and the exact subset; do not present the
      5 usable samples as the full benchmark.
- [ ] LegalBench-RAG CUAD mini split: chunking, embedding, rerank.
- [ ] Add LegalBench-RAG’s other source groups only when their licences and
      corpus sizes are reviewed; LegalBench proper is a legal-reasoning
      benchmark, not a RAG retrieval benchmark.
- [ ] Add confidence intervals or paired bootstrap tests before declaring a
      winner on small samples.
- [ ] Record model download size, VRAM peak, tokens/second, and p50/p95 latency
      beside quality metrics.
- [ ] Re-run the comparison after any change to the tokenizer, chunk budget,
      prompt, or benchmark extraction.

## 3. Required RAGAS and TruLens audit

RAGAS and TruLens are **not optional layers**. They are required audit steps for
a release candidate, even though they are kept out of the deterministic
offline CI gate:

1. deterministic tests and schema validation;
2. push and deterministic CI;
3. live per-benchmark retrieval/generation evaluation;
4. RAGAS and TruLens audit against the same frozen report.

Future work:

- [ ] Run RAGAS on CUAD, FinanceBench and LegalBench-RAG separately.
- [ ] Run TruLens tracing on the same three report sets; retain trace IDs.
- [ ] Pin the judge model and prompt version in the report metadata.
- [ ] Store the RAGAS/TruLens reports as release artifacts, not just console
      output.
- [ ] Add a regression threshold for the audit; do not silently downgrade a
      failing audit to “optional”.
- [ ] Replace the current lexical `CitationVerifier` default with a calibrated
      entailment model for the audit’s claim-support checks.

## 4. Guardrails rollout

The full decision matrix and rejection reasons are in `GUARDRAILS.md`. The
implementation order is:

- [ ] Request-size limits and malformed-upload validation.
- [ ] Tenant-isolation integration tests against Postgres RLS.
- [ ] Append-only audit table and OpenTelemetry tracing.
- [ ] Prompt Guard 2 input and chunk scanning.
- [ ] Presidio PII detection/redaction.
- [ ] Spotlighting and a strict “retrieved text is data” prompt contract.
- [ ] Citation-span, numeric/date and claim-level verification.
- [ ] MiniCheck or HHEM entailment scoring.
- [ ] Durable human-review state machine and four-eyes approval.
- [ ] LiteLLM/Portkey gateway with per-tenant quotas and circuit breakers.
- [ ] Promptfoo/Garak/PyRIT legal-document red-team corpus in CI.

## 5. Evaluation and release engineering

- [ ] Add a benchmark manifest with source URL, retrieval date, SHA-256, licence
      and sample count for every dataset.
- [ ] Make the benchmark cache reproducible in CI without committing PDFs.
- [ ] Add a paired comparison report that shows per-question deltas, not just
      means.
- [ ] Add a cost/latency budget to the release checklist.
- [ ] Record the exact GPU, driver, PyTorch and quantization configuration for
      every local run.
- [ ] Add a “no result” state when a model or benchmark cannot run; never emit a
      zero as if it were a score.

## 6. Product and data work

- [ ] Make document-level access grants first-class for shared deal teams.
- [ ] Add a “citation not found” UX state distinct from “no answer”.
- [ ] Add reviewer annotations that feed a curated legal evaluation set.
- [ ] Add retention/deletion handling for audit traces and uploaded documents.
- [ ] Review the licence of `amal1994/distilled-voyage-modernbert` before any
      production use; it has no explicit licence tag in the model metadata.
- [ ] Revisit Fin-E5 only if a canonical public checkpoint and endpoint become
      available; it remains dropped from the current comparison.

## 7. Open questions

- Which model is the production default after the per-benchmark comparison?
  Do not choose from a pooled or smoke-test score.
- What is the acceptable false-positive rate for contract-injection detection
  on legitimate clauses such as “ignore conflicting instructions”?
- Which high-impact decisions require two reviewers versus one?
- How long must audit traces be retained, and where may they be stored?
- Can FinanceBench’s seven unavailable PDFs be obtained under a compatible
  licence, or should the benchmark be reported on the available subset only?
