"""Tests for trace ingestion."""

import json
import tempfile
from pathlib import Path

import pytest

from postmortem.ingest.base import Trace, TraceMessage, TraceMetadata
from postmortem.ingest.jsonl import JSONLTraceIngest


class TestTraceMessage:
    def test_to_dict(self):
        msg = TraceMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_to_dict_with_tool_calls(self):
        msg = TraceMessage(role="assistant", content=None, tool_calls=[{"name": "foo", "args": {}}])
        d = msg.to_dict()
        assert d["tool_calls"] == [{"name": "foo", "args": {}}]
        assert "content" not in d  # None content excluded

    def test_from_dict(self):
        data = {"role": "user", "content": "test"}
        msg = TraceMessage.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "test"


class TestTrace:
    def test_from_dict(self):
        data = {
            "id": "t1",
            "messages": [{"role": "user", "content": "hi"}],
            "outcome": "success",
            "metadata": {"model": "gpt-4"},
        }
        trace = Trace.from_dict(data)
        assert trace.id == "t1"
        assert len(trace.messages) == 1
        assert trace.metadata.model == "gpt-4"
        assert trace.outcome == "success"

    def test_roundtrip(self):
        trace = Trace(
            id="t1",
            messages=[TraceMessage(role="user", content="test")],
            outcome="failure",
        )
        d = trace.to_dict()
        restored = Trace.from_dict(d)
        assert restored.id == trace.id
        assert restored.outcome == trace.outcome


class TestJSONLIngest:
    def test_parse_single_trace(self, tmp_path):
        data = {"id": "t1", "messages": [{"role": "user", "content": "hi"}], "outcome": "success"}
        f = tmp_path / "trace.jsonl"
        f.write_text(json.dumps(data))
        ingest = JSONLTraceIngest()
        traces = ingest.parse(str(f))
        assert len(traces) == 1
        assert traces[0].id == "t1"

    def test_parse_jsonl_multiple(self, tmp_path):
        lines = [
            json.dumps({"id": "t1", "messages": [{"role": "user", "content": "a"}], "outcome": "success"}),
            json.dumps({"id": "t2", "messages": [{"role": "user", "content": "b"}], "outcome": "error"}),
        ]
        f = tmp_path / "traces.jsonl"
        f.write_text("\n".join(lines))
        ingest = JSONLTraceIngest()
        traces = ingest.parse(str(f))
        assert len(traces) == 2
        assert traces[1].outcome == "error"

    def test_parse_json_array(self, tmp_path):
        data = [
            {"id": "t1", "messages": [], "outcome": "success"},
            {"id": "t2", "messages": [], "outcome": "failure"},
        ]
        f = tmp_path / "traces.json"
        f.write_text(json.dumps(data))
        ingest = JSONLTraceIngest()
        traces = ingest.parse(str(f))
        assert len(traces) == 2

    def test_parse_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        ingest = JSONLTraceIngest()
        assert ingest.parse(str(f)) == []

    def test_file_not_found(self):
        ingest = JSONLTraceIngest()
        with pytest.raises(FileNotFoundError):
            ingest.parse("/nonexistent/path.jsonl")

    def test_parse_example_failing_trace(self):
        path = Path(__file__).parent.parent / "examples" / "traces" / "failing_trace.jsonl"
        if path.exists():
            ingest = JSONLTraceIngest()
            traces = ingest.parse(str(path))
            assert len(traces) == 1
            assert traces[0].id == "trace-001"
            assert traces[0].outcome == "error"

    def test_parse_example_loop_trace(self):
        path = Path(__file__).parent.parent / "examples" / "traces" / "loop_trace.jsonl"
        if path.exists():
            ingest = JSONLTraceIngest()
            traces = ingest.parse(str(path))
            assert len(traces) == 1
            assert traces[0].id == "trace-002"
            assert traces[0].outcome == "failure"
