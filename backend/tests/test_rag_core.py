"""Offline regression tests for the RAG core.

Run without the app conftest:

    python -m pytest tests/test_rag_core.py -v --noconftest

No network, Supabase, Groq or embedding provider is required.
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.check_thresholds import check_thresholds
from evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGASPipelineEvaluator,
    EvaluationSample,
    create_sample_subset,
    load_golden_dataset,
)
from evaluation.ragas_optional import RagasUnavailable, run_ragas
from evaluation.validate_dataset import validate
from services.contextual_chunker import ChunkingStrategy, ContextualChunker, ParentChildIndex
from services.hybrid_retriever import (
    HybridRetriever,
    HybridSearchConfig,
    ReciprocalRankFusion,
    RetrievedDocument,
)


class FakeEmbedder:
    dimension = 1024

    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1024


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents, top_k=10):
        self.calls += 1
        ranked = list(range(len(documents)))[::-1][:top_k]
        return [
            {"index": index, "relevance_score": 1.0 / (rank + 1), "document": documents[index]}
            for rank, index in enumerate(ranked)
        ]


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def execute(self):
        return FakeResponse(self.rows)


class FakeSupabase:
    def __init__(self, rows=None, parents=None):
        self.rows = rows or []
        self.parents = parents or []
        self.calls = []

    def rpc(self, name, params):
        self.calls.append((name, params))
        return FakeQuery(self.rows)

    def table(self, name):
        self.calls.append(("table", name))
        return self

    def select(self, *args):
        return self

    def eq(self, *args):
        return self

    def in_(self, *args):
        return self

    def execute(self):
        return FakeResponse(self.parents)


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
class TestReciprocalRankFusion:
    def test_overlapping_documents_are_fused_once(self):
        a = RetrievedDocument(id=1, content="a", score=1, rank=1)
        b = RetrievedDocument(id=2, content="b", score=1, rank=2)
        c = RetrievedDocument(id=1, content="a", score=1, rank=1)

        fused = ReciprocalRankFusion(k=60).fuse([[a, b], [c]], top_k=10)

        assert [doc.id for doc in fused] == [1, 2]
        assert fused[0].score > fused[1].score

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError):
            ReciprocalRankFusion(k=0)


class TestHybridRetriever:
    def test_maps_rows_and_scopes_to_org(self):
        rows = [
            {
                "id": 10,
                "content": "ISO 27001 is mandatory",
                "rrf_score": 0.03,
                "text_rank": 0.4,
                "vector_similarity": 0.8,
                "source_type": "hybrid",
                "rank_position": 1,
                "parent_id": 3,
                "document_id": "d1",
                "section_path": ["4.2 Certifications"],
            }
        ]
        client = FakeSupabase(rows=rows)
        retriever = HybridRetriever(
            db_client=client,
            config=HybridSearchConfig(expand_parents=False, use_reranking=False, final_top_k=8),
            embedding_service=FakeEmbedder(),
        )

        results = retriever.search("ISO requirements", tenant_id="org-1", document_id="d1")

        assert len(results) == 1
        assert results[0].id == 10
        assert results[0].parent_id == 3
        name, params = client.calls[0]
        assert name == "hybrid_search_documents"
        assert params["filter_org_id"] == "org-1"
        assert params["filter_document_id"] == "d1"
        assert len(params["query_embedding"]) == 1024

    def test_missing_org_id_is_rejected(self):
        retriever = HybridRetriever(db_client=FakeSupabase(), embedding_service=FakeEmbedder())
        with pytest.raises(ValueError, match="org_id"):
            retriever.search("query")

    def test_bm25_only_does_not_embed(self):
        client = FakeSupabase(rows=[])
        retriever = HybridRetriever(
            db_client=client,
            config=HybridSearchConfig(expand_parents=False),
            embedding_service=None,
        )

        retriever.search("clause 4.2", tenant_id="org-1", retrieval_mode="bm25_only")

        _, params = client.calls[0]
        assert params["query_embedding"] is None
        assert params["vector_top_k"] == 0

    def test_reranker_order_and_parent_expansion(self):
        rows = [
            {"id": 1, "content": "first", "rrf_score": 0.3, "parent_id": 11, "source_type": "hybrid"},
            {"id": 2, "content": "second", "rrf_score": 0.2, "parent_id": 11, "source_type": "hybrid"},
        ]
        client = FakeSupabase(rows=rows, parents=[{"id": 11, "content": "full parent section"}])
        reranker = FakeReranker()
        retriever = HybridRetriever(
            db_client=client,
            config=HybridSearchConfig(expand_parents=True, use_reranking=True, final_top_k=2),
            embedding_service=FakeEmbedder(),
            reranker=reranker,
        )

        results = retriever.search("query", tenant_id="org-1")

        assert reranker.calls == 1
        assert [doc.id for doc in results] == [2, 1]
        assert all(doc.parent_content == "full parent section" for doc in results)


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
DOCUMENT = """REQUEST FOR PROPOSAL - ENTERPRISE DATA PLATFORM

1. Introduction
This RFP seeks a supplier for a governed data platform.

4. Mandatory Requirements
4.1 Insurance
The supplier must hold public liability insurance of AUD 10,000,000.

4.2 Mandatory Certifications
Bidders must hold ISO 9001:2015, ISO 27001:2013 and ISO 14001:2015.

| Certification | Requirement |
|---|---|
| ISO 27001 | Mandatory |
| ISO 9001 | Mandatory |

5. Commercial Terms
The contract value is AUD 2,500,000 exclusive of GST.
"""


class TestContextualChunker:
    def test_multi_line_document_does_not_crash(self):
        # The previous implementation raised AttributeError on the second line.
        chunks = ContextualChunker(chunk_size=300, chunk_overlap=50).chunk_with_context(
            DOCUMENT, document_id="doc-1", document_title="RFP", chunking_strategy=ChunkingStrategy.PARENT_CHILD
        )
        assert chunks

    def test_parent_child_links_and_section_path(self):
        chunks = ContextualChunker(chunk_size=250, chunk_overlap=40).chunk_with_context(
            DOCUMENT, document_id="doc-1", document_title="RFP"
        )
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent")]
        parent_ids = {c.chunk_id for c in parents}

        assert parents and children
        assert all(child.parent_id in parent_ids for child in children)
        section_text = " ".join(c.content for c in chunks)
        assert "ISO 27001:2013" in section_text
        assert any("4.2" in " ".join(chunk.section_hierarchy) for chunk in chunks)

    def test_contextual_prefix_is_prepended(self):
        chunks = ContextualChunker(chunk_size=250, chunk_overlap=40).chunk_with_context(
            DOCUMENT, document_id="doc-1", document_title="RFP"
        )
        child = next(c for c in chunks if not c.metadata.get("is_parent"))
        assert "Section:" in (child.contextual_prefix or "")
        assert child.contextualized_content.startswith(child.contextual_prefix)

    def test_unstructured_text_gets_synthetic_parents(self):
        text = " ".join(["The supplier shall maintain records for seven years."] * 40)
        chunks = ContextualChunker(chunk_size=200, chunk_overlap=20).chunk_with_context(
            text, document_id="doc-2"
        )
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent")]
        assert parents and children

    def test_table_is_not_split(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        chunks = ContextualChunker(chunk_size=120, chunk_overlap=10).chunk_with_context(
            "Intro text. " + table, document_id="doc-3"
        )
        assert any(table in chunk.content for chunk in chunks)

    def test_parent_child_index(self):
        chunks = ContextualChunker(chunk_size=250, chunk_overlap=40).chunk_with_context(
            DOCUMENT, document_id="doc-1"
        )
        index = ParentChildIndex()
        for chunk in chunks:
            index.add_chunk(chunk)
        child = next(c for c in chunks if not c.metadata.get("is_parent"))
        assert index.get_parent(child.chunk_id) is not None

    def test_markdown_contextual_strategy_is_child_only_and_contextual(self):
        chunks = ContextualChunker(chunk_size=250, chunk_overlap=40).chunk_with_context(
            DOCUMENT,
            document_id="doc-md",
            document_title="RFP",
            chunking_strategy=ChunkingStrategy.MARKDOWN_CONTEXTUAL,
        )
        assert chunks
        assert all(not chunk.metadata.get("is_parent") for chunk in chunks)
        assert all(chunk.contextual_prefix for chunk in chunks)
        assert any("4.2" in " ".join(chunk.section_hierarchy) for chunk in chunks)

    def test_docling_hybrid_chunker_keeps_heading_path(self):
        docling = pytest.importorskip("docling")
        if not isinstance(docling, types.ModuleType):
            pytest.skip("the shared test conftest mocks Docling")
        from services.docling_chunker import DoclingHybridChunker, docling_document_from_markdown

        document = docling_document_from_markdown(
            "# RFP\n\n## 4.2 Certifications\n\nISO 27001 is mandatory.\n"
        )
        chunks = DoclingHybridChunker(max_tokens=128).chunk(
            document, document_id="doc-docling", document_title="RFP"
        )
        assert chunks
        assert any("4.2 Certifications" in chunk.section_hierarchy for chunk in chunks)


class TestOptionalEvaluationLayers:
    def test_ragas_missing_package_has_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ragas", None)
        with pytest.raises(RagasUnavailable, match="requirements-ragas"):
            run_ragas([{"query": "q", "generated_answer": "a", "retrieved_contexts": ["c"], "ground_truth": "a"}])


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class TestMetrics:
    def setup_method(self):
        self.evaluator = RAGASEvaluator()

    def test_compliance_percentage_extraction_does_not_raise(self):
        # The old regex used group(2) with only one group and raised IndexError.
        assert self.evaluator._extract_compliance_score("Compliance score: 85%") == 85.0
        assert self.evaluator._extract_compliance_score("The response is 72 percent compliant") == 72.0

    def test_budget_normalisation(self):
        assert self.evaluator._normalize_budget("AUD 2,500,000") == 2_500_000
        assert self.evaluator._normalize_budget("$2.5M") == 2_500_000
        assert self.evaluator._normalize_budget("2.5 million") == 2_500_000
        assert self.evaluator._normalize_budget("Not specified") == 0.0

    def test_ndcg_perfect_and_zero(self):
        gt = ["The contract value is AUD 2,500,000."]
        perfect = self.evaluator._compute_ndcg(
            [{"content": "The contract value is AUD 2,500,000."}], gt
        )
        missed = self.evaluator._compute_ndcg(
            [{"content": "Unrelated payment terms apply."}], gt
        )
        assert perfect == pytest.approx(1.0)
        assert missed == 0.0

    def test_context_precision_rewards_top_rank(self):
        gt = ["ISO 27001 is mandatory."]
        top = self.evaluator._compute_context_precision(
            [{"content": "ISO 27001 is mandatory."}, {"content": "Unrelated."}], gt
        )
        bottom = self.evaluator._compute_context_precision(
            [{"content": "Unrelated."}, {"content": "ISO 27001 is mandatory."}], gt
        )
        assert top > bottom

    def test_faithfulness_rejects_wrong_number(self):
        chunks = [{"content": "The total contract value is AUD 2,500,000."}]
        assert self.evaluator._compute_faithfulness(
            "The total contract value is AUD 5,000,000.", chunks
        ) == 0.0

    def test_per_category_aggregation(self):
        samples = [
            EvaluationSample(id="1", query="q", ground_truth="a", relevant_chunks=["c"], category="compliance"),
            EvaluationSample(id="2", query="q", ground_truth="a", relevant_chunks=["c"], category="budget"),
        ]
        results = [
            self.evaluator.evaluate_sample(sample, "a", [{"content": "c"}]) for sample in samples
        ]
        aggregate = RAGASPipelineEvaluator(self.evaluator)._aggregate_results(results)
        assert set(aggregate.per_category) == {"compliance", "budget"}
        assert aggregate.metrics["context_recall"]["mean"] == pytest.approx(1.0)

    def test_missing_required_metric_fails_threshold_check(self):
        report = {"metrics": {"context_recall": {"mean": 0.9}}}
        config = {
            "thresholds": {
                "context_recall": {"min": 0.8},
                "faithfulness": {"min": 0.85},
            }
        }
        result = check_thresholds(report, config)
        assert result["passed"] is False
        missing = next(item for item in result["results"] if item["metric"] == "faithfulness")
        assert missing["reason"] == "metric missing from report"

    def test_sample_subset_is_deterministic(self):
        samples = [
            EvaluationSample(id=str(i), query="q", ground_truth="a", relevant_chunks=["c"])
            for i in range(100)
        ]
        first = create_sample_subset(samples, n_samples=10, seed=7)
        second = create_sample_subset(samples, n_samples=10, seed=7)
        assert [s.id for s in first] == [s.id for s in second]


# ----------------------------------------------------------------------
# Dataset validation
# ----------------------------------------------------------------------
class TestDatasetValidation:
    def _write(self, count, sources=("financebench", "cuad", "austender")):
        samples = []
        for index in range(count):
            samples.append(
                {
                    "id": f"s{index}",
                    "query": "What is the contract value?",
                    "ground_truth": "AUD 1,000,000",
                    "relevant_chunks": ["The contract value is AUD 1,000,000."],
                    "category": "budget",
                    "difficulty": "easy",
                    "metadata": {"source_dataset": sources[index % len(sources)]},
                }
            )
        payload = {
            "description": "test",
            "source_datasets": list(sources),
            "samples": samples,
        }
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        return Path(handle.name)

    def test_valid_dataset(self):
        path = self._write(100)
        assert validate(path)["valid"] is True
        path.unlink()

    def test_single_benchmark_dataset_valid(self):
        path = self._write(55, sources=("financebench",))
        assert validate(path, benchmark="financebench")["valid"] is True
        path.unlink()

    def test_single_benchmark_rejects_wrong_source(self):
        path = self._write(55, sources=("financebench",))
        result = validate(path, benchmark="cuad")
        assert result["valid"] is False
        assert any("single-source" in error or "contains source" in error for error in result["errors"])
        path.unlink()

    def test_short_dataset_rejected(self):
        path = self._write(10)
        result = validate(path)
        assert result["valid"] is False
        assert any("at least 100" in error for error in result["errors"])
        path.unlink()

    def test_duplicate_ids_rejected(self):
        path = self._write(100)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["samples"][1]["id"] = payload["samples"][0]["id"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        assert validate(path)["valid"] is False
        path.unlink()

    def test_load_golden_dataset_accepts_wrapped_file(self):
        path = self._write(2)
        samples = load_golden_dataset(str(path))
        assert len(samples) == 2
        path.unlink()
