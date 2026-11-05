"""RAG query endpoints used by the UI and the evaluation harness."""

import logging

from fastapi import APIRouter, Depends, HTTPException

import db
import rag_service
import schemas

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG Retrieval"])

_embedder = None
_reranker = None


def _components():
    """Lazily initialise providers so app startup never calls a paid API."""
    global _embedder, _reranker
    if _embedder is None:
        _embedder = rag_service.build_embedder()
    if _reranker is None:
        _reranker = rag_service.build_reranker()
    return _embedder, _reranker


@router.post("/api/rag/query")
@router.post("/api/v1/rag/query")
def rag_query(
    request: schemas.RAGQueryRequest,
    org_info: dict = Depends(db.get_current_user_org),
):
    try:
        embedder, reranker = _components()
        result = rag_service.answer_query(
            supabase_client=db.supabase_client,
            org_id=org_info["org_id"],
            query=request.query,
            document_id=request.document_id,
            groq_client=db.groq_client,
            embedder=embedder,
            reranker=reranker,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
        )
    except rag_service.RAGConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc

    return result


@router.get("/api/rag/health")
def rag_health(org_info: dict = Depends(db.get_current_user_org)):
    try:
        embedder, _ = _components()
        embedder_ready = embedder is not None
    except Exception as exc:
        embedder_ready = False
        logger.warning("Embedder unavailable: %s", exc)

    return {
        "status": "ok" if embedder_ready else "degraded",
        "org_id": org_info["org_id"],
        "embedder": type(_embedder).__name__ if _embedder else None,
        "reranker": type(_reranker).__name__ if _reranker else None,
        "retrieval": "hybrid_rrf_parent_child",
    }
