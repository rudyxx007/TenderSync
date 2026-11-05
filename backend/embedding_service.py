"""Embedding services for TenderSync.

Two interchangeable 1024-dimensional providers are supported:

* ``VoyageEmbeddingService`` (services/voyage_embeddings.py) when
  ``VOYAGE_API_KEY`` is configured.
* ``HuggingFaceEmbeddingService`` (BAAI/bge-m3) as the zero-cost default.

Both expose ``embed_documents``, ``embed_query`` and ``dimension`` so the RAG
pipeline does not care which provider is active. The pgvector column is
``vector(1024)``; ``build_embedder`` rejects a provider with another dimension
instead of silently writing unusable vectors.
"""

import os
import time
from typing import List, Optional

from huggingface_hub import InferenceClient

DEFAULT_HF_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


def get_embedding_client(token: Optional[str] = None) -> InferenceClient:
    """Instantiate a Hugging Face InferenceClient with the provided or env token."""
    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    return InferenceClient(token=resolved_token)


class HuggingFaceEmbeddingService:
    """Dense embeddings from Hugging Face Inference (BAAI/bge-m3, 1024-dim)."""

    dimension = EMBEDDING_DIMENSION

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = DEFAULT_HF_MODEL,
        max_chars: int = 24_000,
        retries: int = 2,
    ):
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.token:
            raise ValueError("HF_TOKEN is not set; cannot create Hugging Face embeddings.")
        self.model = model
        self.max_chars = max_chars
        self.retries = max(0, retries)
        self._client = get_embedding_client(self.token)

    def _embed_one(self, text: str) -> List[float]:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                vector = self._client.feature_extraction(text[: self.max_chars], model=self.model)
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                if vector and isinstance(vector[0], list):
                    vector = vector[0]
                vector = [float(x) for x in vector]
                if len(vector) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension {len(vector)} does not match the "
                        f"pgvector column ({self.dimension})."
                    )
                return vector
            except Exception as exc:  # network / rate limit / model warm-up
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"BGE-M3 embedding failed: {last_error}") from last_error

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def generate_embeddings(texts: List[str], task_type: str = "passage") -> List[List[float]]:
    """Backwards-compatible helper.

    Returns an empty list on failure, matching the old behaviour used by the
    upload pipeline. New code should use ``HuggingFaceEmbeddingService`` so it
    can distinguish "no text" from "embedding provider failed".
    """
    if not texts:
        return []

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        print("[HF BGE-M3 Embeddings] Error: HF_TOKEN is not set.")
        return []

    try:
        service = HuggingFaceEmbeddingService(token=token)
        return service.embed_documents(texts)
    except Exception as exc:
        print(f"[HF BGE-M3 Embeddings] Error: {exc}")
        return []
