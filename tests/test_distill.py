"""Tests for trace distillation."""

import pytest

from postmortem.ingest.base import Trace, TraceMessage, TraceMetadata
from postmortem.distill import Distiller
from postmortem.case import Assertion


class TestDistiller:
    def test_distill_error_trace(self):
        trace = Trace(
            id="t1",
            messages=[
                TraceMessage(role="user", content="List files"),
                TraceMessage(role="assistant", tool_calls=[{"name": "list_files", "args": {"path": "/tmp"}}]),
                TraceMessage(role="tool", content="Error: permission denied"),
            ],
            outcome="error",
            metadata=TraceMetadata(error="permission denied"),
        )
        distiller = Distiller()
        case = distiller.distill(trace)

        assert case.id == "case-t1"
        assert case.input["messages"]
        assert len(case.assertions) >= 1

        # Should have a not_contains assertion for the error
        error_assertions = [a for a in case.assertions if a.type == "not_contains"]
        assert len(error_assertions) >= 1

    def test_distill_loop_trace(self):
        trace = Trace(
            id="t2",
            messages=[
                TraceMessage(role="user", content="Search commits"),
                TraceMessage(role="assistant", tool_calls=[{"name": "git_log"}]),
                TraceMessage(role="tool", content="No repository found"),
                TraceMessage(role="assistant", tool_calls=[{"name": "git_log"}]),
                TraceMessage(role="tool", content="No repository found"),
                TraceMessage(role="assistant", tool_calls=[{"name": "git_log"}]),
                TraceMessage(role="tool", content="No repository found"),
            ],
            outcome="failure",
        )
        distiller = Distiller()
        case = distiller.distill(trace)

        # Should detect loop
        loop_assertions = [a for a in case.assertions if a.type == "not_repeated"]
        assert len(loop_assertions) >= 1
        assert loop_assertions[0].expected == "git_log"

    def test_distill_preserves_user_input(self):
        trace = Trace(
            id="t3",
            messages=[
                TraceMessage(role="system", content="You are helpful"),
                TraceMessage(role="user", content="Hello"),
                TraceMessage(role="assistant", content="Hi there"),
            ],
            outcome="error",
        )
        distiller = Distiller()
        case = distiller.distill(trace)

        msgs = case.input["messages"]
        # Should have system + user messages
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert "user" in roles

    def test_distill_infers_label(self):
        trace = Trace(
            id="t4",
            messages=[
                TraceMessage(role="user", content="Do something"),
                TraceMessage(role="tool", content="Error: permission denied"),
            ],
            outcome="error",
        )
        distiller = Distiller()
        case = distiller.distill(trace)
        assert "permission" in case.label.lower()
