"""RAGAS judge integration for the required release audit.

RAGAS is not a deterministic PR gate, but it is a required audit for release
candidates. Install ``requirements-ragas.txt`` and run the audit after the
per-benchmark evaluation.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence


class RagasUnavailable(RuntimeError):
    """Raised when the required RAGAS package or judge is unavailable."""


def _build_metrics():
    try:
        from ragas.metrics import (  # type: ignore
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )

        return [
            Faithfulness(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ]
    except ImportError:
        try:
            from ragas.metrics import (  # type: ignore
                context_precision,
                context_recall,
                faithfulness,
            )

            return [faithfulness, context_recall, context_precision]
        except ImportError as exc:
            raise RagasUnavailable(
                "RAGAS is not installed. Install `pip install -r requirements-ragas.txt`."
            ) from exc


def build_groq_judge():
    """Create a Groq-backed RAGAS judge through the OpenAI-compatible API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RagasUnavailable("GROQ_API_KEY is required for the RAGAS judge.")
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RagasUnavailable(
            "RAGAS and langchain-openai are required. Install "
            "`pip install -r requirements-ragas.txt`."
        ) from exc

    model = os.getenv("RAGAS_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0,
        timeout=60,
        max_retries=1,
    )
    return LangchainLLMWrapper(client)


def _rows(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        contexts = sample.get("retrieved_contexts") or []
        answer = sample.get("generated_answer") or sample.get("response") or ""
        reference = sample.get("ground_truth") or sample.get("reference") or ""
        if not contexts or not answer or not reference:
            continue
        rows.append(
            {
                "user_input": sample.get("user_input") or sample.get("query") or "",
                "retrieved_contexts": list(contexts),
                "response": answer,
                "reference": reference,
            }
        )
    return rows


def _numeric_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    keys = {
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    summary: Dict[str, float] = {}
    for key in sorted(keys):
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and not isinstance(record.get(key), bool)
            and not (isinstance(record[key], float) and math.isnan(record[key]))
        ]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def run_ragas(
    samples: Sequence[Dict[str, Any]],
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate report samples with RAGAS and return a JSON-safe summary."""
    try:
        from ragas import EvaluationDataset, evaluate  # type: ignore
    except ImportError as exc:
        raise RagasUnavailable(
            "RAGAS is not installed. Install `pip install -r requirements-ragas.txt`."
        ) from exc

    rows = _rows(samples)
    if not rows:
        raise RagasUnavailable(
            "No report samples have retrieved_contexts, an answer and a reference."
        )
    metrics = _build_metrics()
    judge = llm or build_groq_judge()
    try:
        from ragas.run_config import RunConfig

        run_config = RunConfig(timeout=60, max_retries=1, max_workers=1)
    except ImportError:  # pragma: no cover - older RAGAS
        run_config = None
    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=metrics,
        llm=judge,
        run_config=run_config,
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    records = frame.to_dict(orient="records")
    return {
        "num_samples": len(records),
        "metrics": _numeric_summary(records),
        "samples": records,
        "judge_model": os.getenv("RAGAS_MODEL") or os.getenv("GROQ_MODEL"),
    }
