"""Tests for eval case format and assertions."""

import pytest
import tempfile
from pathlib import Path

from postmortem.case import EvalCase, Assertion


class TestAssertion:
    def test_contains_pass(self):
        a = Assertion(type="contains", expected="hello")
        passed, _ = a.check("hello world")
        assert passed

    def test_contains_fail(self):
        a = Assertion(type="contains", expected="goodbye")
        passed, _ = a.check("hello world")
        assert not passed

    def test_not_contains_pass(self):
        a = Assertion(type="not_contains", expected="error")
        passed, _ = a.check("success")
        assert passed

    def test_not_contains_fail(self):
        a = Assertion(type="not_contains", expected="Error")
        passed, _ = a.check("Error: something broke")
        assert not passed

    def test_tool_called_pass(self):
        a = Assertion(type="tool_called", expected="search")
        passed, _ = a.check("ok", tool_calls=["search", "lookup"])
        assert passed

    def test_tool_called_fail(self):
        a = Assertion(type="tool_called", expected="search")
        passed, _ = a.check("ok", tool_calls=["lookup"])
        assert not passed

    def test_not_repeated_pass(self):
        a = Assertion(type="not_repeated", expected="search")
        passed, _ = a.check("ok", tool_calls=["search", "lookup"])
        assert passed

    def test_not_repeated_fail(self):
        a = Assertion(type="not_repeated", expected="search")
        passed, _ = a.check("ok", tool_calls=["search", "search", "search"])
        assert not passed

    def test_regex_match(self):
        a = Assertion(type="regex_match", expected=r"\d+ items")
        passed, _ = a.check("Found 5 items")
        assert passed

    def test_regex_no_match(self):
        a = Assertion(type="regex_match", expected=r"\d+ items")
        passed, _ = a.check("Found items")
        assert not passed

    def test_status_code_success(self):
        a = Assertion(type="status_code", expected="success")
        passed, _ = a.check("Here is your result")
        assert passed

    def test_status_code_success_with_error(self):
        a = Assertion(type="status_code", expected="success")
        passed, _ = a.check("Error: something failed")
        assert not passed

    def test_json_path(self):
        a = Assertion(type="json_path", expected="count")
        passed, _ = a.check('{"count": 42}')
        assert passed

    def test_to_dict_from_dict(self):
        a = Assertion(type="contains", expected="hello", description="greeting check")
        d = a.to_dict()
        restored = Assertion.from_dict(d)
        assert restored.type == a.type
        assert restored.expected == a.expected
        assert restored.description == a.description


class TestEvalCase:
    def test_to_dict_from_dict(self):
        case = EvalCase(
            id="c1",
            label="test",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="hello")],
        )
        d = case.to_dict()
        restored = EvalCase.from_dict(d)
        assert restored.id == "c1"
        assert len(restored.assertions) == 1

    def test_yaml_roundtrip(self, tmp_path):
        case = EvalCase(
            id="c1",
            label="test",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="hello", description="greeting")],
        )
        path = tmp_path / "case.yaml"
        case.to_yaml(path)

        loaded = EvalCase.from_yaml(path)
        assert loaded.id == "c1"
        assert loaded.assertions[0].type == "contains"

    def test_json_roundtrip(self, tmp_path):
        case = EvalCase(
            id="c1",
            assertions=[Assertion(type="not_contains", expected="error")],
        )
        path = tmp_path / "case.json"
        case.to_json(path)

        loaded = EvalCase.from_json(path)
        assert loaded.id == "c1"

    def test_yaml_string(self):
        case = EvalCase(
            id="c1",
            assertions=[Assertion(type="contains", expected="ok")],
        )
        text = case.to_yaml()
        assert "id: c1" in text
        assert "contains" in text
