"""Tests for suite runner."""

import pytest

from postmortem.case import EvalCase, Assertion
from postmortem.runner import SuiteRunner, RunReport


def _write_case(path, case: EvalCase):
    """Write an eval case to a YAML file."""
    case.to_yaml(path / f"{case.id}.yaml")


def _mock_agent(messages, **kwargs):
    """Mock agent that returns a fixed response."""
    return {"output": "Hello, I can help you with that.", "tool_calls": []}


def _mock_agent_with_error(messages, **kwargs):
    """Mock agent that returns an error."""
    return {"output": "Error: something went wrong", "tool_calls": []}


def _mock_agent_with_tools(messages, **kwargs):
    """Mock agent that calls tools."""
    return {"output": "Result", "tool_calls": ["search", "lookup"]}


def _mock_agent_looping(messages, **kwargs):
    """Mock agent that loops."""
    return {"output": "Looping", "tool_calls": ["search", "search", "search"]}


class TestSuiteRunner:
    def test_run_passing_suite(self, tmp_path):
        case = EvalCase(
            id="c1",
            label="test",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="Hello")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent)

        assert report.total == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.no_regressions

    def test_run_failing_suite(self, tmp_path):
        case = EvalCase(
            id="c1",
            label="test",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="not_contains", expected="Error")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent_with_error)

        assert report.total == 1
        assert report.failed == 1
        assert report.passed == 0

    def test_run_multiple_cases(self, tmp_path):
        case1 = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="Hello")],
        )
        case2 = EvalCase(
            id="c2",
            input={"messages": [{"role": "user", "content": "err"}]},
            assertions=[Assertion(type="not_contains", expected="Error")],
        )
        _write_case(tmp_path, case1)
        _write_case(tmp_path, case2)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent)

        assert report.total == 2

    def test_regression_detection(self, tmp_path):
        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="Hello")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()

        # First run - passes
        report1 = runner.run(str(tmp_path), _mock_agent, save=True)
        assert report1.passed == 1

        # Second run with failing agent - regression
        report2 = runner.run(str(tmp_path), _mock_agent_with_error, save=True)
        assert "c1" in report2.regressions

    def test_improvement_detection(self, tmp_path):
        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="Hello")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()

        # First run - fails
        report1 = runner.run(str(tmp_path), _mock_agent_with_error, save=True)
        assert report1.failed == 1

        # Second run - passes (improvement)
        report2 = runner.run(str(tmp_path), _mock_agent, save=True)
        assert "c1" in report2.improvements

    def test_run_case_tool_check(self, tmp_path):
        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "search"}]},
            assertions=[
                Assertion(type="tool_called", expected="search"),
                Assertion(type="not_repeated", expected="search"),
            ],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent_with_tools)
        assert report.passed == 1

    def test_run_case_loop_detection(self, tmp_path):
        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "search"}]},
            assertions=[
                Assertion(type="not_repeated", expected="search"),
            ],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent_looping)
        assert report.failed == 1

    def test_agent_exception(self, tmp_path):
        def bad_agent(messages, **kwargs):
            raise RuntimeError("Agent crashed")

        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="ok")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), bad_agent)
        assert report.failed == 1
        assert report.results[0].error == "Agent crashed"

    def test_pass_rate(self, tmp_path):
        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="Hello")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent)
        assert report.pass_rate == 1.0

    def test_summary(self, tmp_path):
        case = EvalCase(
            id="c1",
            input={"messages": [{"role": "user", "content": "hi"}]},
            assertions=[Assertion(type="contains", expected="Hello")],
        )
        _write_case(tmp_path, case)

        runner = SuiteRunner()
        report = runner.run(str(tmp_path), _mock_agent)
        summary = report.summary()
        assert "Total: 1" in summary
        assert "Passed: 1" in summary
