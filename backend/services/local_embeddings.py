"""Local open-weight embedding providers for real evaluation runs.

The production embedding path uses Voyage/BGE-M3. This module is for offline
A/B evaluation of Qwen3-Embedding models on the local GPU. It intentionally
loads the model directly with Transformers so the 4B variant can be quantized
to fit an 8 GB GPU and so the 1024-dimensional Matryoshka truncation is explicit.

Fin-E5 was dropped from the comparison because no canonical public checkpoint
was available. The old error path is retained for callers that still request it.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


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
        "Apache-2.0; 32K context; native 2560, MRL truncates to 1024",
    ),
    "fin-e5": EmbeddingSpec(
        "fin-e5",
        os.environ.get("FIN_E5_MODEL", ""),
        4096,
        "Dropped: no canonical public checkpoint",
    ),
}

EMBEDDER_VARIANTS = ("qwen3-0.6b", "qwen3-4b")


def available_embedders() -> dict[str, EmbeddingSpec]:
    return dict(MODEL_SPECS)


class Qwen3EmbeddingService:
    """Qwen3 embedding model with left padding, last-token pooling and MRL.

    Qwen3 embedding models are causal language models, not BERT encoders. The
    official pooling method is the final token's hidden state after left
    padding, followed by L2 normalization. Queries receive the official
    instruction prefix; passages do not.
    """

    INSTRUCTION = (
        "Given a legal or financial search query, retrieve relevant passages "
        "that answer the query"
    )

    def __init__(
        self,
        model_id: str,
        dimension: Optional[int] = 1024,
        quantization: str = "auto",
        max_length: int = 1024,
        batch_size: int = 2,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LocalEmbeddingUnavailable(
                "transformers and torch are required for Qwen3 embeddings. "
                "Install `pip install -r requirements-local-embeddings.txt`."
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.requested_dimension = dimension

        use_cuda = bool(torch.cuda.is_available()) and (device or "cuda").startswith("cuda")
        if not use_cuda:
            raise LocalEmbeddingUnavailable(
                "Qwen3 embedding comparison requires CUDA. The active Python is "
                "using CPU-only PyTorch; install the CUDA wheel or use CPU with "
                "an explicit, much slower fallback."
            )

        # 4B at bf16 is ~8 GB before activations and will not fit an 8 GB
        # RTX 5050. Quantize by default; 0.6B stays bf16.
        quantize = quantization == "4bit" or (
            quantization == "auto" and "4B" in model_id
        )
        model_kwargs = {"device_map": {"": 0}}
        if quantize:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise LocalEmbeddingUnavailable(
                    "bitsandbytes is required for 4-bit Qwen3-4B inference."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.quantization = "4bit"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            self.quantization = "bf16"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
                trust_remote_code=False,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=False,
                **model_kwargs,
            )
            self._model.eval()
        except Exception as exc:  # pragma: no cover - model download/runtime
            raise LocalEmbeddingUnavailable(
                f"Could not load {model_id}: {exc}"
            ) from exc

        self._device = next(self._model.parameters()).device
        native = int(self._model.config.hidden_size)
        self.native_dimension = native
        if dimension is not None and dimension > native:
            raise LocalEmbeddingUnavailable(
                f"Requested dimension {dimension} exceeds native {native} for {model_id}."
            )
        self.dimension = int(dimension or native)

    def _format_query(self, text: str) -> str:
        return f"Instruct: {self.INSTRUCTION}\nQuery: {text}"

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                output = self._model(**encoded)
                # Left padding means the final position is the real final token.
                embeddings = output.last_hidden_state[:, -1, :]
                if self.dimension != self.native_dimension:
                    embeddings = embeddings[:, : self.dimension]
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
            vectors.append(embeddings.float().cpu().numpy())
        return [list(map(float, row)) for row in np.vstack(vectors)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._encode([self._format_query(text)])[0]

    def close(self) -> None:
        del self._model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class SentenceTransformerEmbeddingService:
    """Legacy adapter retained for callers that already use sentence-transformers."""

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
            kwargs.pop("truncate_dim", None)
            self._model = SentenceTransformer(model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - model download/runtime
            raise LocalEmbeddingUnavailable(f"Could not load {model_id}: {exc}") from exc

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
    quantization: str = "auto",
    max_length: int = 1024,
    batch_size: int = 2,
) -> Qwen3EmbeddingService:
    """Build one of the named open-weight Qwen3 embedding candidates."""
    key = name.strip().lower()
    spec = MODEL_SPECS.get(key)
    if spec is None:
        raise LocalEmbeddingUnavailable(
            f"Unknown local embedder {name!r}; choose {', '.join(EMBEDDER_VARIANTS)}"
        )
    if not spec.model_id:
        raise LocalEmbeddingUnavailable(
            "Fin-E5 was dropped from the comparison: no canonical public checkpoint "
            "is available. Set FIN_E5_MODEL only if you have a verified checkpoint."
        )
    effective_dimension = dimension if dimension is not None else spec.native_dimension
    return Qwen3EmbeddingService(
        model_id=spec.model_id,
        dimension=effective_dimension,
        quantization=quantization,
        max_length=max_length,
        batch_size=batch_size,
    )
