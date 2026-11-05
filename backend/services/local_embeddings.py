"""Local, open-weight embedding providers for A/B evaluation.

These are intentionally separate from the production Voyage/BGE-M3 factory:
the production pgvector column is fixed at 1024 dimensions, while Qwen3-4B and
Fin-E5 have native dimensions of 2560 and 4096. The A/B runner can use Matryoshka
truncation for Qwen3, but it records the requested dimension explicitly rather
than pretending the model is identical at every size.

Fin-E5 weights were announced for release by the FinMTEB authors, but no
canonical public checkpoint was found. Set ``FIN_E5_MODEL`` to a verified local
or Hub checkpoint before running that variant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


class LocalEmbeddingUnavailable(RuntimeError):
    """Raised when an optional local embedding model cannot be loaded."""


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_id: str
    native_dimension: int
    notes: str


MODEL_SPECS = {
    "qwen3-0.6b": EmbeddingSpec(
        "qwen3-0.6b",
        "Qwen/Qwen3-Embedding-0.6B",
        1024,
        "Apache-2.0; 32K context; native 1024 dimensions",
    ),
    "qwen3-4b": EmbeddingSpec(
        "qwen3-4b",
        "Qwen/Qwen3-Embedding-4B",
        2560,
        "Apache-2.0; 32K context; native 2560, MRL can truncate to 1024",
    ),
    "fin-e5": EmbeddingSpec(
        "fin-e5",
        os.environ.get("FIN_E5_MODEL", ""),
        4096,
        "Finance-adapted e5-Mistral-7B; verified checkpoint required",
    ),
}


def available_embedders() -> dict[str, EmbeddingSpec]:
    return dict(MODEL_SPECS)


class SentenceTransformerEmbeddingService:
    """Small adapter around sentence-transformers for local A/B runs."""

    def __init__(
        self,
        model_id: str,
        dimension: Optional[int] = None,
        trust_remote_code: bool = True,
        batch_size: int = 16,
    ) -> None:
        if not model_id:
            raise LocalEmbeddingUnavailable("An embedding model id is required.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LocalEmbeddingUnavailable(
                "sentence-transformers is not installed. Install "
                "`pip install -r requirements-local-embeddings.txt`."
            ) from exc

        kwargs = {
            "trust_remote_code": trust_remote_code,
            "device": os.environ.get("EMBEDDING_DEVICE") or None,
        }
        if dimension is not None:
            kwargs["truncate_dim"] = dimension
        try:
            self._model = SentenceTransformer(model_id, **kwargs)
        except TypeError:
            # Older sentence-transformers releases do not accept truncate_dim.
            kwargs.pop("truncate_dim", None)
            self._model = SentenceTransformer(model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - model download/runtime
            raise LocalEmbeddingUnavailable(
                f"Could not load {model_id}: {exc}"
            ) from exc

        self.model_id = model_id
        self.batch_size = batch_size
        self.dimension = dimension or int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def build_local_embedder(
    name: str,
    dimension: Optional[int] = 1024,
) -> SentenceTransformerEmbeddingService:
    """Build one of the named open-weight candidates.

    ``dimension=1024`` uses Qwen3's Matryoshka representation for the 4B model.
    Fin-E5 is not Matryoshka-truncated here; its native dimension is used unless
    the caller passes an explicit dimension.
    """
    key = name.strip().lower()
    spec = MODEL_SPECS.get(key)
    if spec is None:
        raise LocalEmbeddingUnavailable(
            f"Unknown local embedder {name!r}; choose {', '.join(sorted(MODEL_SPECS))}"
        )
    if not spec.model_id:
        raise LocalEmbeddingUnavailable(
            "Fin-E5 has no canonical public checkpoint configured. Set FIN_E5_MODEL "
            "to a verified weights path or Hub id after checking the license."
        )
    effective_dimension = spec.native_dimension if key == "fin-e5" else dimension
    return SentenceTransformerEmbeddingService(
        model_id=spec.model_id,
        dimension=effective_dimension,
    )
