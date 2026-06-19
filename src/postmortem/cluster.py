"""Failure clustering using text similarity."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from postmortem.ingest.base import Trace


@dataclass
class Cluster:
    """A group of similar failure traces."""

    id: str
    representative_trace: Trace
    all_traces: list[Trace] = field(default_factory=list)
    similarity_score: float = 1.0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self._auto_label()

    def _auto_label(self) -> str:
        """Generate a label from the cluster's failure patterns."""
        trace = self.representative_trace
        # Look for error patterns
        error_msgs = []
        for msg in trace.messages:
            if msg.role == "tool" and msg.content:
                content = msg.content.lower()
                if "error" in content or "exception" in content or "failed" in content:
                    error_msgs.append(msg.content[:80])

        # Check for repeated tool calls (loop detection)
        tool_names = []
        for msg in trace.messages:
            for tc in msg.tool_calls:
                tool_names.append(tc.get("name", ""))

        # Detect loops
        if self._detect_loop(tool_names):
            repeated = self._most_common(tool_names)
            return f"tool-loop:{repeated}"

        if error_msgs:
            return f"error:{error_msgs[0][:60]}"

        if trace.metadata.error:
            return f"error:{trace.metadata.error[:60]}"

        return f"failure:{trace.id[:12]}"

    @staticmethod
    def _detect_loop(names: list[str], threshold: int = 3) -> bool:
        if len(names) < threshold:
            return False
        # Check if any single tool appears 3+ times consecutively
        count = 1
        for i in range(1, len(names)):
            if names[i] == names[i - 1]:
                count += 1
                if count >= threshold:
                    return True
            else:
                count = 1
        return False

    @staticmethod
    def _most_common(items: list[str]) -> str:
        if not items:
            return "unknown"
        return Counter(items).most_common(1)[0][0]


def _tokenize(text: str) -> set[str]:
    """Simple tokenizer for similarity computation."""
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _trace_signature(trace: Trace) -> str:
    """Compute a compact signature of a trace for dedup."""
    parts = []
    for msg in trace.messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(f"tool:{tc.get('name', '')}")
        if msg.content and msg.role in ("user", "tool"):
            parts.append(msg.content[:200])
    parts.append(trace.outcome)
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


def _trace_text(trace: Trace) -> str:
    """Extract representative text from a trace for clustering."""
    parts = []
    for msg in trace.messages:
        if msg.role == "user" and msg.content:
            parts.append(msg.content[:500])
        if msg.role == "tool" and msg.content:
            parts.append(msg.content[:500])
        for tc in msg.tool_calls:
            parts.append(f"call:{tc.get('name', '')}")
    if trace.metadata.error:
        parts.append(trace.metadata.error)
    return " ".join(parts)


class FailureClusterer:
    """Cluster similar failure traces using Jaccard similarity."""

    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    def cluster(self, traces: list[Trace]) -> list[Cluster]:
        """Group similar failure traces into clusters."""
        # Filter to failures/errors only
        failures = [t for t in traces if t.outcome in ("failure", "error")]
        if not failures:
            return []

        # Deduplicate
        seen_sigs: set[str] = set()
        unique: list[Trace] = []
        for t in failures:
            sig = _trace_signature(t)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique.append(t)

        if not unique:
            return []

        # Greedy clustering
        clusters: list[Cluster] = []
        assigned: set[int] = set()

        for i, trace in enumerate(unique):
            if i in assigned:
                continue
            trace_tokens = _tokenize(_trace_text(trace))
            cluster_traces = [trace]
            cluster_scores: list[float] = []
            assigned.add(i)

            for j in range(i + 1, len(unique)):
                if j in assigned:
                    continue
                other_tokens = _tokenize(_trace_text(unique[j]))
                sim = _jaccard(trace_tokens, other_tokens)
                if sim >= self.similarity_threshold:
                    cluster_traces.append(unique[j])
                    cluster_scores.append(sim)
                    assigned.add(j)

            avg_score = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 1.0
            cid = f"cluster-{hashlib.md5(trace.id.encode()).hexdigest()[:8]}"
            clusters.append(Cluster(
                id=cid,
                representative_trace=trace,
                all_traces=cluster_traces,
                similarity_score=avg_score,
            ))

        return clusters
