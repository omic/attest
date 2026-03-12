"""Optional OpenTelemetry tracing for AttestDB operations.

No-op when opentelemetry-api is not installed. Install with:
    pip install attestdb[telemetry]
"""

from __future__ import annotations

from contextlib import contextmanager

try:
    from opentelemetry import trace

    _tracer = trace.get_tracer("attestdb")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    _tracer = None


@contextmanager
def span(name: str, attributes: dict | None = None):
    """Context manager that creates an OpenTelemetry span if available."""
    if not _HAS_OTEL:
        yield None
        return
    with _tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        yield s


def record_exception(exc: Exception) -> None:
    """Record an exception on the current span if tracing is active."""
    if not _HAS_OTEL:
        return
    current = trace.get_current_span()
    if current and current.is_recording():
        current.record_exception(exc)
        current.set_status(trace.StatusCode.ERROR, str(exc))


def set_attribute(key: str, value) -> None:
    """Set an attribute on the current span if tracing is active."""
    if not _HAS_OTEL:
        return
    current = trace.get_current_span()
    if current and current.is_recording():
        current.set_attribute(key, value)
