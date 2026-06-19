"""OpenTelemetry GenAI trace ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from postmortem.ingest.base import Trace, TraceIngest, TraceMessage, TraceMetadata


class OTELTraceIngest(TraceIngest):
    """Parse OpenTelemetry GenAI semantic convention spans.

    Expects JSON export of span trees. Extracts agent traces from
    spans with gen_ai.* attributes.
    """

    def parse(self, source: str) -> list[Trace]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"OTEL trace file not found: {source}")

        data = json.loads(path.read_text())
        if isinstance(data, dict) and "resourceSpans" in data:
            return self._parse_otel_json(data)
        elif isinstance(data, list):
            return self._parse_span_list(data)
        return []

    def _parse_otel_json(self, data: dict) -> list[Trace]:
        """Parse standard OTLP JSON format."""
        traces: list[Trace] = []
        for rs in data.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                spans = ss.get("spans", [])
                traces.extend(self._group_spans(spans))
        return traces

    def _parse_span_list(self, spans: list[dict]) -> list[Trace]:
        """Parse a flat list of spans."""
        return self._group_spans(spans)

    def _group_spans(self, spans: list[dict]) -> list[Trace]:
        """Group spans by trace ID and convert to Trace objects."""
        trace_map: dict[str, list[dict]] = {}
        for span in spans:
            tid = span.get("traceId", span.get("trace_id", "unknown"))
            trace_map.setdefault(tid, []).append(span)

        traces: list[Trace] = []
        for tid, group in trace_map.items():
            traces.append(self._spans_to_trace(tid, group))
        return traces

    def _spans_to_trace(self, trace_id: str, spans: list[dict]) -> Trace:
        """Convert grouped spans into a single Trace."""
        messages: list[TraceMessage] = []
        outcome = "unknown"
        metadata = TraceMetadata()

        for span in sorted(spans, key=lambda s: s.get("startTimeUnixNano", 0)):
            attrs = self._extract_attributes(span)

            # Check for gen_ai prompt/completion events
            if "gen_ai.user.message" in attrs:
                messages.append(TraceMessage(role="user", content=attrs["gen_ai.user.message"]))
            if "gen_ai.assistant.message" in attrs:
                tool_calls = []
                if "gen_ai.tool_call.name" in attrs:
                    tool_calls.append({
                        "name": attrs["gen_ai.tool_call.name"],
                        "args": json.loads(attrs.get("gen_ai.tool_call.arguments", "{}")),
                    })
                messages.append(TraceMessage(
                    role="assistant",
                    content=attrs.get("gen_ai.assistant.message"),
                    tool_calls=tool_calls,
                ))
            if "gen_ai.tool.result" in attrs:
                messages.append(TraceMessage(role="tool", content=attrs["gen_ai.tool.result"]))

            # Extract metadata
            if "gen_ai.request.model" in attrs:
                metadata.model = attrs["gen_ai.request.model"]
            if "gen_ai.usage.total_tokens" in attrs:
                try:
                    metadata.tokens = int(attrs["gen_ai.usage.total_tokens"])
                except (ValueError, TypeError):
                    pass

            # Check status
            status = span.get("status", {})
            if status.get("code") == "ERROR" or attrs.get("error.type"):
                outcome = "error"
                metadata.error = status.get("message", attrs.get("error.type", ""))

        if outcome == "unknown":
            outcome = "success"

        return Trace(
            id=trace_id,
            messages=messages,
            metadata=metadata,
            outcome=outcome,
        )

    def _extract_attributes(self, span: dict) -> dict[str, Any]:
        """Flatten OTLP attributes into a simple dict."""
        attrs: dict[str, Any] = {}
        for attr in span.get("attributes", []):
            key = attr.get("key", "")
            val = attr.get("value", {})
            for vtype in ("stringValue", "intValue", "doubleValue", "boolValue"):
                if vtype in val:
                    attrs[key] = val[vtype]
                    break
        return attrs
