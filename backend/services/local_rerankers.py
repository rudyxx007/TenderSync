"""Local open-weight rerankers for the TenderSync comparison.

The four candidates are kept behind one interface so the benchmark uses the
same first-stage candidate set for every reranker. Qwen3 rerankers are causal
models scored by the yes/no logits; BGE and Distilled-Voyage-ModernBERT are
sequence-classification cross-encoders.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


class LocalRerankerUnavailable(RuntimeError):
    """Raised when a reranker cannot be loaded or run."""


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    model_id: str
    kind: str
    notes: str


RERANKER_SPECS = {
    "qwen3-reranker-0.6b": RerankerSpec(
        "qwen3-reranker-0.6b",
        "Qwen/Qwen3-Reranker-0.6B",
        "qwen3",
        "Apache-2.0; 0.6B causal reranker",
    ),
    "qwen3-reranker-4b": RerankerSpec(
        "qwen3-reranker-4b",
        "Qwen/Qwen3-Reranker-4B",
        "qwen3",
        "Apache-2.0; 4B causal reranker; 4-bit on 8 GB VRAM",
    ),
    "bge-reranker-v2-m3": RerankerSpec(
        "bge-reranker-v2-m3",
        "BAAI/bge-reranker-v2-m3",
        "cross_encoder",
        "Apache-2.0; multilingual 0.6B cross-encoder",
    ),
    "distilled-voyage-modernbert": RerankerSpec(
        "distilled-voyage-modernbert",
        "amal1994/distilled-voyage-modernbert",
        "cross_encoder",
        "Open legal reranker distilled from Voyage; verify model license",
    ),
}

RERANKER_VARIANTS = tuple(RERANKER_SPECS)


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 4,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise LocalRerankerUnavailable(
                "sentence-transformers is required for cross-encoder rerankers."
            ) from exc
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise LocalRerankerUnavailable("torch is required for rerankers.") from exc

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise LocalRerankerUnavailable(
                "CUDA is unavailable; the active PyTorch build is CPU-only."
            )
        self.model_id = model_id
        self.batch_size = batch_size
        try:
            self._model = CrossEncoder(
                model_id,
                device=selected_device,
                max_length=max_length,
                trust_remote_code=False,
            )
        except Exception as exc:  # pragma: no cover
            raise LocalRerankerUnavailable(f"Could not load {model_id}: {exc}") from exc

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        pairs = [[query, document] for document in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in np.asarray(scores).reshape(-1)]

    def close(self) -> None:
        del self._model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class Qwen3Reranker:
    """Qwen3 causal reranker scored by the probability of ``yes`` vs ``no``."""

    INSTRUCTION = (
        "Given a legal or financial search query, retrieve relevant passages "
        "that answer the query"
    )
    SYSTEM = (
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    def __init__(
        self,
        model_id: str,
        quantization: str = "auto",
        max_length: int = 1024,
        batch_size: int = 1,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise LocalRerankerUnavailable(
                "transformers and torch are required for Qwen3 rerankers."
            ) from exc
        if not torch.cuda.is_available():
            raise LocalRerankerUnavailable(
                "Qwen3 reranker comparison requires CUDA. Install the CUDA PyTorch wheel."
            )

        self._torch = torch
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        quantize = quantization == "4bit" or (
            quantization == "auto" and "4B" in model_id
        )
        kwargs: dict[str, Any] = {"device_map": {"": 0}}
        if quantize:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.quantization = "4bit"
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            self.quantization = "bf16"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=False
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=False, **kwargs
            )
            self._model.eval()
        except Exception as exc:  # pragma: no cover
            raise LocalRerankerUnavailable(f"Could not load {model_id}: {exc}") from exc

        self._yes_id = self._single_token_id("yes")
        self._no_id = self._single_token_id("no")

    def _single_token_id(self, token: str) -> int:
        ids = self._tokenizer(token, add_special_tokens=False).input_ids
        if not ids:
            raise LocalRerankerUnavailable(f"Tokenizer produced no token for {token!r}")
        return int(ids[0])

    def _prompt(self, query: str, document: str) -> str:
        return (
            f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {self.INSTRUCTION}\n"
            f"<Query>: {query}\n<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        scores: List[float] = []
        # One pair at a time keeps peak VRAM controllable for the 4-bit 4B model.
        for document in documents:
            encoded = self._tokenizer(
                self._prompt(query, document),
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits[0, -1, :]
                yes = logits[self._yes_id]
                no = logits[self._no_id]
                # Stable sigmoid of the log-odds, bounded to [0, 1].
                score = self._torch.sigmoid(yes - no).float().item()
            scores.append(float(score))
        return scores

    def close(self) -> None:
        del self._model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def build_reranker(
    name: str,
    quantization: str = "auto",
    max_length: int = 1024,
    batch_size: int = 2,
) -> Any:
    key = name.strip().lower()
    spec = RERANKER_SPECS.get(key)
    if spec is None:
        raise LocalRerankerUnavailable(
            f"Unknown reranker {name!r}; choose {', '.join(RERANKER_VARIANTS)}"
        )
    if spec.kind == "qwen3":
        return Qwen3Reranker(
            spec.model_id,
            quantization=quantization,
            max_length=max_length,
            batch_size=batch_size,
        )
    return CrossEncoderReranker(
        spec.model_id,
        max_length=max_length,
        batch_size=batch_size,
    )
