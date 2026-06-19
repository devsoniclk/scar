"""Abstract trace ingest base classes and data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceMessage:
    """A single message in a trace."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_results:
            d["tool_results"] = self.tool_results
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceMessage:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content"),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
        )


@dataclass
class TraceMetadata:
    """Metadata about a trace."""

    model: str | None = None
    tokens: int | None = None
    duration_ms: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.model:
            d["model"] = self.model
        if self.tokens is not None:
            d["tokens"] = self.tokens
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.error:
            d["error"] = self.error
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceMetadata:
        known = {"model", "tokens", "duration", "duration_ms", "error"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            model=data.get("model"),
            tokens=data.get("tokens"),
            duration_ms=data.get("duration_ms") or data.get("duration"),
            error=data.get("error"),
            extra=extra,
        )


@dataclass
class Trace:
    """A complete agent trace."""

    id: str
    messages: list[TraceMessage] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=TraceMetadata)
    outcome: str = "unknown"  # success, failure, error
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "outcome": self.outcome,
        }
        meta = self.metadata.to_dict()
        if meta:
            d["metadata"] = meta
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        return cls(
            id=data.get("id", "unknown"),
            messages=[TraceMessage.from_dict(m) for m in data.get("messages", [])],
            metadata=TraceMetadata.from_dict(data.get("metadata", {})),
            outcome=data.get("outcome", "unknown"),
            timestamp=data.get("timestamp"),
        )


class TraceIngest(ABC):
    """Abstract base class for trace ingestion."""

    @abstractmethod
    def parse(self, source: str) -> list[Trace]:
        """Parse traces from a source (file path, URL, etc.)."""
        ...
