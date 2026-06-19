"""Portable eval-case format with YAML/JSON serialization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Assertion:
    """A single assertion to check against agent output."""

    type: str  # contains, not_contains, tool_called, status_code, regex_match, json_path, not_repeated
    expected: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "expected": self.expected}
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assertion:
        return cls(
            type=data["type"],
            expected=data.get("expected"),
            description=data.get("description", ""),
        )

    def check(self, output: str, tool_calls: list[str] | None = None) -> tuple[bool, str]:
        """Check this assertion against actual output.

        Returns (passed, detail_message).
        """
        tool_calls = tool_calls or []
        output = output or ""

        if self.type == "contains":
            passed = str(self.expected) in output
            return passed, f"Expected '{self.expected}' in output" + ("" if passed else " — not found")

        elif self.type == "not_contains":
            passed = str(self.expected) not in output
            return passed, f"Expected '{self.expected}' not in output" + ("" if passed else " — found it")

        elif self.type == "tool_called":
            passed = str(self.expected) in tool_calls
            return passed, f"Expected tool '{self.expected}' called" + ("" if passed else " — not called")

        elif self.type == "not_repeated":
            count = tool_calls.count(str(self.expected))
            passed = count < 3
            return passed, f"Tool '{self.expected}' called {count} times" + ("" if passed else " — loop detected")

        elif self.type == "status_code":
            if self.expected == "success":
                passed = "error" not in output.lower() and "exception" not in output.lower()
                return passed, "Expected success" + ("" if passed else " — error found in output")
            return output == str(self.expected), f"Expected status '{self.expected}'"

        elif self.type == "regex_match":
            try:
                passed = bool(re.search(str(self.expected), output))
                return passed, f"Expected regex '{self.expected}' match" + ("" if passed else " — no match")
            except re.error as e:
                return False, f"Invalid regex: {e}"

        elif self.type == "json_path":
            try:
                data = json.loads(output)
                parts = str(self.expected).split(".")
                current = data
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return False, f"JSON path '{self.expected}' not found"
                return True, f"JSON path '{self.expected}' found: {current}"
            except (json.JSONDecodeError, TypeError):
                return False, "Output is not valid JSON"

        return False, f"Unknown assertion type: {self.type}"


@dataclass
class EvalCase:
    """A portable eval case that can be serialized to YAML or JSON."""

    id: str
    label: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    assertions: list[Assertion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id}
        if self.label:
            d["label"] = self.label
        if self.input:
            d["input"] = self.input
        if self.context:
            d["context"] = self.context
        d["assertions"] = [a.to_dict() for a in self.assertions]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        return cls(
            id=data["id"],
            label=data.get("label", ""),
            input=data.get("input", {}),
            context=data.get("context", {}),
            assertions=[Assertion.from_dict(a) for a in data.get("assertions", [])],
        )

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize to YAML. If path given, write to file."""
        text = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> EvalCase:
        """Load from a YAML string or file path."""
        p = Path(source)
        if p.exists():
            text = p.read_text()
        else:
            text = str(source)
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON. If path given, write to file."""
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> EvalCase:
        """Load from a JSON string or file path."""
        p = Path(source)
        if p.exists():
            text = p.read_text()
        else:
            text = str(source)
        data = json.loads(text)
        return cls.from_dict(data)
