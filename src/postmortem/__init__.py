"""Postmortem — Turn production agent failures into regression test cases."""

from postmortem.ingest.base import Trace, TraceMessage, TraceMetadata, TraceIngest
from postmortem.ingest.jsonl import JSONLTraceIngest
from postmortem.case import EvalCase, Assertion
from postmortem.distill import Distiller
from postmortem.cluster import FailureClusterer
from postmortem.runner import SuiteRunner, RunReport


def from_trace(path: str, **kwargs) -> EvalCase:
    """Convert a trace file to a minimal eval case with assertions."""
    ingest = JSONLTraceIngest()
    traces = ingest.parse(path)
    if not traces:
        raise ValueError(f"No traces found in {path}")
    distiller = Distiller()
    return distiller.distill(traces[0])


def run_suite(suite_dir: str, agent, **kwargs) -> RunReport:
    """Run an eval suite against an agent function."""
    runner = SuiteRunner()
    return runner.run(suite_dir, agent)


__all__ = [
    "from_trace",
    "run_suite",
    "Trace",
    "TraceMessage",
    "TraceMetadata",
    "TraceIngest",
    "JSONLTraceIngest",
    "EvalCase",
    "Assertion",
    "Distiller",
    "FailureClusterer",
    "SuiteRunner",
    "RunReport",
]
