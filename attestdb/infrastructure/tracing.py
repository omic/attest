"""Optional OpenTelemetry tracing for AttestDB operations.

No-op when opentelemetry-api is not installed. Install with:
    pip install attestdb[telemetry]

Setup OTLP export with:
    from attestdb.infrastructure.tracing import configure_otlp
    configure_otlp()  # exports to localhost:4317 by default
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

_configured = False


def configure_otlp(
    endpoint: str | None = None,
    service_name: str = "attestdb",
) -> bool:
    """Configure OTLP trace exporter. Returns True if successful.

    Requires ``pip install attestdb[telemetry]`` (opentelemetry-sdk).
    Exports spans to an OTLP-compatible collector (Jaeger, Grafana Tempo, etc.).

    Args:
        endpoint: OTLP gRPC endpoint (default: env OTEL_EXPORTER_OTLP_ENDPOINT or localhost:4317)
        service_name: Service name for trace attribution.
    """
    global _configured, _tracer
    if _configured:
        return True
    try:
        from opentelemetry import trace as _trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError:
        return False

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    kwargs = {}
    if endpoint:
        kwargs["endpoint"] = endpoint
    exporter = OTLPSpanExporter(**kwargs)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _trace.set_tracer_provider(provider)
    _tracer = _trace.get_tracer("attestdb")
    _configured = True
    return True


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
