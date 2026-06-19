"""Tests for failure clustering."""

import pytest

from postmortem.ingest.base import Trace, TraceMessage, TraceMetadata
from postmortem.cluster import FailureClusterer, Cluster, _jaccard, _tokenize


class TestJaccard:
    def test_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self):
        sim = _jaccard({"a", "b"}, {"b", "c"})
        assert sim == pytest.approx(1 / 3)

    def test_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_one_empty(self):
        assert _jaccard({"a"}, set()) == 0.0


def _make_trace(tid: str, error_msg: str) -> Trace:
    return Trace(
        id=tid,
        messages=[
            TraceMessage(role="user", content="do something"),
            TraceMessage(role="tool", content=f"Error: {error_msg}"),
        ],
        metadata=TraceMetadata(error=error_msg),
        outcome="error",
    )


class TestFailureClusterer:
    def test_cluster_similar_failures(self):
        t1 = _make_trace("t1", "permission denied on /tmp")
        t2 = _make_trace("t2", "permission denied on /var")
        t3 = _make_trace("t3", "connection timeout to database")

        clusterer = FailureClusterer(similarity_threshold=0.3)
        clusters = clusterer.cluster([t1, t2, t3])
        assert len(clusters) >= 1

    def test_cluster_with_exact_duplicates(self):
        t1 = _make_trace("t1", "same error")
        t2 = _make_trace("t2", "same error")
        t3 = Trace(
            id="t3",
            messages=[TraceMessage(role="user", content="ok")],
            outcome="success",
        )

        clusterer = FailureClusterer()
        clusters = clusterer.cluster([t1, t2, t3])
        # t3 is success so should be filtered
        assert len(clusters) >= 1
        # Deduplication should group t1 and t2
        total_traces = sum(len(c.all_traces) for c in clusters)
        assert total_traces == 1  # deduplicated

    def test_no_failures(self):
        t1 = Trace(id="t1", outcome="success")
        clusterer = FailureClusterer()
        assert clusterer.cluster([t1]) == []

    def test_cluster_label_generation(self):
        t = Trace(
            id="t1",
            messages=[
                TraceMessage(role="user", content="do it"),
                TraceMessage(role="assistant", tool_calls=[{"name": "search"}]),
                TraceMessage(role="tool", content="Error: timeout"),
            ],
            outcome="error",
        )
        clusterer = FailureClusterer()
        clusters = clusterer.cluster([t])
        assert len(clusters) == 1
        assert clusters[0].label  # has an auto-generated label
