"""Token counting helpers for fair chunker and embedder comparisons.

Character-based chunk sizes are not comparable across chunkers: Docling budgets
tokens, while a character budget silently gives different context sizes. The
benchmark harness therefore uses one tokenizer for every variant.

The production ingestion path can keep using character budgets; these helpers
are opt-in so the existing API remains compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


class TokenCounter(Protocol):
    model_id: str

    def count(self, text: str) -> int:
        ...


@dataclass
class TiktokenCounter:
    """Fast local token counter.

    ``cl100k_base`` is not Qwen's tokenizer, but it is stable and available
    without downloading model weights. For embedding comparisons pass a
    Hugging Face ``Qwen/Qwen3-Embedding-*`` tokenizer to ``build_token_counter``.
    """

    model_id: str = "tiktoken:cl100k_base"
    encoding_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "tiktoken is required for token-budgeted chunking. "
                "Install `tiktoken>=0.8`."
            ) from exc
        self._encoding = tiktoken.get_encoding(self.encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text or ""))


@dataclass
class HuggingFaceCounter:
    """Tokenizer-backed counter used for Qwen3 embedding/reranker runs."""

    model_id: str
    _tokenizer: Any = None

    def __post_init__(self) -> None:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "transformers is required for a Hugging Face token counter. "
                    "Install `transformers` and `sentence-transformers`."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=False,
            )

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text or ""))


def build_token_counter(
    model_id: str = "tiktoken:cl100k_base",
    tokenizer: Optional[Any] = None,
) -> TokenCounter:
    """Build a token counter from ``tiktoken:*`` or a Hugging Face model id."""
    if model_id.startswith("tiktoken:"):
        return TiktokenCounter(encoding_name=model_id.split(":", 1)[1])
    return HuggingFaceCounter(model_id=model_id, _tokenizer=tokenizer)


def count_tokens(counter: Optional[Callable[[str], int]], text: str) -> int:
    """Count tokens with a callable counter, falling back to a rough estimate."""
    if counter is None:
        return max(1, (len(text or "") + 3) // 4)
    return int(counter(text or ""))
