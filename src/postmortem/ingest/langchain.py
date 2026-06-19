"""LangChain/LangSmith trace ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from postmortem.ingest.base import Trace, TraceIngest, TraceMessage, TraceMetadata


class LangChainTraceIngest(TraceIngest):
    """Parse LangChain/LangSmith trace format.

    Supports both LangSmith run format and LangChain callback dumps.
    """

    def parse(self, source: str) -> list[Trace]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"LangChain trace file not found: {source}")

        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [self._parse_run(r) for r in data]
        elif isinstance(data, dict):
            if "runs" in data:
                return [self._parse_run(r) for r in data["runs"]]
            return [self._parse_run(data)]
        return []

    def _parse_run(self, run: dict[str, Any]) -> Trace:
        """Convert a LangChain/LangSmith run to a Trace."""
        trace_id = run.get("id", run.get("run_id", "unknown"))
        outcome = "unknown"
        messages: list[TraceMessage] = []
        metadata = TraceMetadata()

        # Extract model info
        if "extra" in run and "metadata" in run["extra"]:
            meta = run["extra"]["metadata"]
            metadata.model = meta.get("ls_model_name", meta.get("model"))

        # Parse inputs
        inputs = run.get("inputs", {})
        if "messages" in inputs:
            for msg_group in inputs["messages"]:
                if isinstance(msg_group, list):
                    for msg in msg_group:
                        messages.append(self._lc_message_to_trace(msg))
                elif isinstance(msg_group, dict):
                    messages.append(self._lc_message_to_trace(msg_group))
        elif "input" in inputs:
            messages.append(TraceMessage(role="user", content=str(inputs["input"])))
        elif "query" in inputs:
            messages.append(TraceMessage(role="user", content=inputs["query"]))

        # Parse outputs
        outputs = run.get("outputs", {})
        if "generations" in outputs:
            for gen_list in outputs["generations"]:
                for gen in gen_list:
                    text = gen.get("text", gen.get("message", {}).get("content", ""))
                    if text:
                        messages.append(TraceMessage(role="assistant", content=text))
        elif "output" in outputs:
            messages.append(TraceMessage(role="assistant", content=str(outputs["output"])))
        elif "text" in outputs:
            messages.append(TraceMessage(role="assistant", content=outputs["text"]))

        # Parse tool calls from intermediate steps
        for step in run.get("intermediate_steps", []):
            if isinstance(step, list) and len(step) >= 2:
                action, result = step[0], step[1]
                if isinstance(action, dict):
                    tool_name = action.get("tool", action.get("name", "unknown"))
                    tool_input = action.get("tool_input", action.get("input", {}))
                    messages.append(TraceMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[{"name": tool_name, "args": tool_input}],
                    ))
                messages.append(TraceMessage(role="tool", content=str(result)))

        # Determine outcome
        error = run.get("error")
        if error:
            outcome = "error"
            metadata.error = str(error)
        elif run.get("status") in ("error", "FAILURE"):
            outcome = "error"
        elif outputs:
            outcome = "success"

        # Token usage
        if "usage_metadata" in outputs:
            usage = outputs["usage_metadata"]
            metadata.tokens = usage.get("total_tokens")

        return Trace(
            id=str(trace_id),
            messages=messages,
            metadata=metadata,
            outcome=outcome,
            timestamp=run.get("start_time"),
        )

    def _lc_message_to_trace(self, msg: dict[str, Any]) -> TraceMessage:
        """Convert a LangChain message dict to TraceMessage."""
        role = msg.get("type", msg.get("role", "unknown"))
        # Map LangChain types to standard roles
        role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        role = role_map.get(role, role)

        content = msg.get("content", "")
        tool_calls = []
        if "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
        elif msg.get("additional_kwargs", {}).get("tool_calls"):
            tool_calls = msg["additional_kwargs"]["tool_calls"]

        return TraceMessage(role=role, content=content, tool_calls=tool_calls)
