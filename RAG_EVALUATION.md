# RAG Evaluation Report

**Run date:** 2026-11-06  
**Branch:** `feature/enterprise-hybrid-rag-platform`  
**Rule:** every benchmark is run and reported separately. No CUAD,
FinanceBench, or LegalBench-RAG score is pooled.

This report distinguishes measured results from runs that did not complete. A
missing cell is not zero.

## Chunking comparison

### Method and fairness controls

- All three variants use the **same Qwen3 tokenizer** and the **same token
  budget**: 1,800 tokens with 300 tokens of overlap.
- `section_aware` is the existing parent-child section-aware strategy; only its
  child rows are scored.
- `markdown_contextual` is the Snowflake-style Markdown-contextual strategy,
  with heading-path context carried in the chunk metadata.
- `docling_hybrid` is `DoclingHybridChunker` using the same tokenizer and budget.
- CUAD and LegalBench-RAG are plain text with numbered/uppercase headings, so
  both Markdown-aware paths receive the same heading-normalized input. The
  normalizer only prefixes existing heading-like lines; it does not add content.
- Evidence coverage means at least 85% of the gold-evidence tokens occur in one
  retrieved chunk. Full-span containment is stricter: the normalized gold span
  must occur inside a single chunk.

### CUAD (45 full contracts)

| Variant | Evidence coverage | Full-span containment | Mean chunks/contract | Mean tokens/chunk |
| --- | ---: | ---: | ---: | ---: |
| section_aware | 53.33% | 26.67% | 13.04 | 891.47 |
| markdown_contextual | 53.33% | 26.67% | 13.04 | 891.47 |
| **docling_hybrid** | **86.67%** | **55.56%** | 15.60 | 864.49 |

Docling wins the structural coverage comparison on CUAD at the same token
budget. The two Markdown/section variants are identical here because they share
the same splitter and differ only in whether parent rows are retained; this is
not two independent algorithms, and the result should be described that way.

### LegalBench-RAG CUAD mini split (194 questions, 29 documents)

| Variant | Evidence coverage | Full-span containment | Mean chunks/document | Mean tokens/chunk |
| --- | ---: | ---: | ---: | ---: |
| section_aware | 86.08% | 85.57% | 17.15 | 949.24 |
| markdown_contextual | 86.08% | 85.57% | 17.15 | 949.24 |
| **docling_hybrid** | **97.16%** | **96.78%** | 19.88 | 859.13 |

Docling again wins structural coverage. The mini split is the benchmark’s
published 194-test selection over its CUAD corpus; it is not the full 4,042-test
LegalBench-RAG suite.

### FinanceBench (available subset only)

Only **2 source filings** parsed successfully (`3M_2018_10K` and
`3M_2022_10K`), covering **8 evidence rows**. The other 25 source documents are
missing or were not parsed. Seven PDFs were unavailable at download time; the
Docling parse of a large filing also hit `std::bad_alloc` on some pages.

| Variant | Evidence coverage | Full-span containment | Mean chunks/document | Mean tokens/chunk |
| --- | ---: | ---: | ---: | ---: |
| section_aware | 50.00% | 37.50% | 85.25 | 427.76 |
| markdown_contextual | 50.00% | 37.50% | 85.25 | 427.76 |
| docling_hybrid | 50.00% | 12.50% | 96.12 | 528.14 |

**No FinanceBench winner is declared.** The sample is too small (8 rows), and
the full-span result favors the Markdown/section paths. Do not use this table as
a FinanceBench leaderboard result.

## Model comparison

### Embeddings

The Qwen3-0.6B smoke run on 3 CUAD questions completed. It is a smoke run, not a
model-selection result:

| Model | Dimensions | Quantization | Context recall | Context precision | NDCG |
| --- | ---: | --- | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B | 1024 | bf16 | 33.33% | 8.33% | 14.36% |

The 10-sample Qwen3-0.6B + 4B run and the 5-sample reranker run did not produce
reports: the process stayed in model download/loading and was still running when
this report was written. **No 0.6B-vs-4B or reranker winner is claimed.**

Reranker candidates defined for the run: Qwen3-Reranker-0.6B,
Qwen3-Reranker-4B, BAAI/bge-reranker-v2-m3, and
`amal1994/distilled-voyage-modernbert`. All candidates receive the same
first-stage candidate set; the baseline is the first-stage order before rerank.

### RAGAS and TruLens

The audit runner is `backend/evaluation/run_audit.py`. It uses the
Markdown-contextual chunker and a deterministic lexical retriever, then calls
Groq to generate a cited answer. This is an offline audit path; it does not use
the production Postgres/RLS hybrid query.

RAGAS is executed with Faithfulness, LLM Context Recall, and LLM Context
Precision (with reference) using the configured Groq judge. TruLens records the
same query path and a deterministic citation-support feedback. Results are
written to `backend/evaluation/results/audit/` and are not committed because the
results directory is intentionally gitignored.

### Measured audit status

The one-question CUAD audit completed on 2026-11-06:

| Framework | Metric | Result | Status |
| --- | --- | ---: | --- |
| RAGAS | Context recall | 1.0000 | measured |
| RAGAS | LLM context precision (with reference) | 0.3667 | measured |
| RAGAS | Faithfulness | not produced | `NaN` after Groq rate limiting; not a pass |
| TruLens | Citation Support feedback | recorded | 3 trace records; deterministic marker-to-context check, not an LLM judge |

The earlier three-question RAGAS attempt was also rate-limited (HTTP 429, 8,000
tokens/minute on the Groq on-demand tier) and produced only context precision;
it was not used as a final audit. The one-question result is an execution proof,
not a benchmark conclusion.

**Required follow-up:** raise the judge quota or use a judge with a larger
budget, then rerun all three metrics on the same frozen report before treating
the audit as release evidence. Do not present the partial precision score as an
overall RAGAS pass.

## What this proves and what it does not

- It proves the token-budgeted chunking harness runs and that Docling’s hybrid
  chunker has materially better structural evidence coverage on CUAD and the
  LegalBench-RAG CUAD mini split.
- It does **not** prove Docling is the best retriever for the production hybrid
  search path; retrieval quality still needs the embedding/rerank runs.
- It does **not** prove FinanceBench performance; only 8 evidence rows were
  usable.
- It does **not** claim the Markdown-contextual implementation is byte-for-byte
  identical to Snowflake’s private implementation. The available method was
  recreated from its documented Markdown-contextual behavior, not from
  Snowflake source code. The REA skill had no connected REA tools in this
  session, so no binary or shipped-artifact reverse engineering was possible.
- It does **not** treat RAGAS/TruLens as optional; a release candidate needs
  both audit artifacts.

## Reproduction

```bash
cd backend

# CUAD chunking (full)
python -m evaluation.model_comparison \
  --benchmark cuad --mode chunking \
  --max-tokens 1800 --overlap-tokens 300 \
  --output evaluation/results/model_comparison/cuad_chunking.json

# LegalBench-RAG CUAD mini split
python -m evaluation.prepare_legalbench_rag --source cuad
python -m evaluation.model_comparison \
  --benchmark legalbench_rag --mode chunking \
  --max-tokens 1800 --overlap-tokens 300 \
  --output evaluation/results/model_comparison/legalbench_rag_chunking.json

# Audit (requires GROQ_API_KEY and the benchmark venv)
python -m evaluation.run_audit --benchmark cuad --samples 3
```
