"""Optional TruLens instrumentation for the evaluation pipeline.

TruLens records the actual retrieve/generate calls made by the evaluation
runner. It is optional because it adds tracing dependencies and a local
database; the deterministic benchmark report remains the release gate.
"""

from __future__ import annotations

from typing import Any, Optional


class TruLensUnavailable(RuntimeError):
    """Raised when TruLens cannot be imported or initialized."""


class TruLensTracer:
    """Wrap a pipeline object with a TruLens app recorder."""

    def __init__(
        self,
        pipeline: Any,
        app_name: str = "TenderSync RAG",
        app_version: str = "evaluation",
    ) -> None:
        try:
            from trulens.apps.app import TruApp
        except ImportError as exc:
            raise TruLensUnavailable(
                "TruLens is not installed. Install `pip install -r requirements-trulens.txt`."
            ) from exc
        try:
            from trulens.core.otel.instrument import instrument
        except ImportError:
            try:
                from trulens.apps.app import instrument  # type: ignore
            except ImportError as exc:
                raise TruLensUnavailable(
                    "This TruLens version does not expose an instrumentation decorator."
                ) from exc

        class _InstrumentedPipeline:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            @instrument()
            def query(self, *args: Any, **kwargs: Any) -> Any:
                return self._inner.query(*args, **kwargs)

        self.pipeline = _InstrumentedPipeline(pipeline)
        try:
            self._app = TruApp(
                self.pipeline,
                app_name=app_name,
                app_version=app_version,
                feedbacks=[],
            )
        except Exception as exc:
            raise TruLensUnavailable(f"Could not initialize TruLens: {exc}") from exc

    def __enter__(self) -> "TruLensTracer":
        self._app.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return bool(self._app.__exit__(exc_type, exc, traceback))
