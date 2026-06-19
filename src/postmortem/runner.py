"""Suite runner — replay eval cases against an agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from postmortem.case import EvalCase, Assertion


@dataclass
class AssertionResult:
    """Result of checking a single assertion."""

    assertion: Assertion
    passed: bool
    detail: str


@dataclass
class CaseResult:
    """Result of running a single eval case."""

    case_id: str
    label: str
    passed: bool
    output: str
    tool_calls: list[str] = field(default_factory=list)
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunReport:
    """Summary report of a suite run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    results: list[CaseResult] = field(default_factory=list)

    @property
    def no_regressions(self) -> bool:
        return len(self.regressions) == 0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def summary(self) -> str:
        lines = [
            f"Total: {self.total}  Passed: {self.passed}  Failed: {self.failed}",
            f"Pass rate: {self.pass_rate:.0%}",
        ]
        if self.regressions:
            lines.append(f"Regressions ({len(self.regressions)}): {', '.join(self.regressions)}")
        if self.improvements:
            lines.append(f"Improvements ({len(self.improvements)}): {', '.join(self.improvements)}")
        return "\n".join(lines)


def _load_suite(suite_dir: str) -> list[EvalCase]:
    """Load all eval cases from a directory (YAML or JSON files)."""
    p = Path(suite_dir)
    if not p.exists():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    cases: list[EvalCase] = []
    for ext in ("*.yaml", "*.yml", "*.json"):
        for f in sorted(p.glob(ext)):
            try:
                if f.suffix == ".json":
                    cases.append(EvalCase.from_json(f))
                else:
                    cases.append(EvalCase.from_yaml(f))
            except Exception:
                continue
    return cases


def _load_previous_results(suite_dir: str) -> dict[str, bool]:
    """Load previous run results for regression tracking."""
    results_path = Path(suite_dir) / ".results.json"
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text())
        return {r["case_id"]: r["passed"] for r in data.get("results", [])}
    except Exception:
        return {}


def _save_results(suite_dir: str, report: RunReport):
    """Save current results for future regression comparison."""
    results_path = Path(suite_dir) / ".results.json"
    data = {
        "results": [
            {"case_id": r.case_id, "passed": r.passed}
            for r in report.results
        ]
    }
    results_path.write_text(json.dumps(data, indent=2))


class SuiteRunner:
    """Run an eval suite against an agent function."""

    def run(self, suite_dir: str, agent_fn: Callable, *, save: bool = True) -> RunReport:
        """Run all eval cases in suite_dir against agent_fn.

        agent_fn should accept (messages: list[dict], **context) -> dict
        where the return dict has at least 'output' (str) and optionally
        'tool_calls' (list[str]).
        """
        cases = _load_suite(suite_dir)
        previous = _load_previous_results(suite_dir)

        report = RunReport()
        for case in cases:
            result = self.run_case(case, agent_fn)
            report.results.append(result)
            report.total += 1
            if result.passed:
                report.passed += 1
            else:
                report.failed += 1

            # Check for regressions / improvements
            if case.id in previous:
                was_passed = previous[case.id]
                if was_passed and not result.passed:
                    report.regressions.append(case.id)
                elif not was_passed and result.passed:
                    report.improvements.append(case.id)

        if save:
            _save_results(suite_dir, report)

        return report

    def run_case(self, case: EvalCase, agent_fn: Callable) -> CaseResult:
        """Run a single eval case against the agent."""
        try:
            # Call the agent
            messages = case.input.get("messages", [])
            response = agent_fn(messages, **case.context)

            # Normalize response
            if isinstance(response, str):
                output = response
                tool_calls: list[str] = []
            elif isinstance(response, dict):
                output = response.get("output", "")
                tool_calls = response.get("tool_calls", [])
            else:
                output = str(response)
                tool_calls = []

            # Check assertions
            all_passed = True
            assertion_results: list[AssertionResult] = []
            for assertion in case.assertions:
                passed, detail = assertion.check(output, tool_calls)
                assertion_results.append(AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    detail=detail,
                ))
                if not passed:
                    all_passed = False

            return CaseResult(
                case_id=case.id,
                label=case.label,
                passed=all_passed,
                output=output,
                tool_calls=tool_calls,
                assertion_results=assertion_results,
            )

        except Exception as e:
            return CaseResult(
                case_id=case.id,
                label=case.label,
                passed=False,
                output="",
                error=str(e),
            )
