"""CLI for postmortem."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

# Default directory for storing labeled traces
TRACES_DIR = Path.home() / ".postmortem" / "traces"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_agent(script_path: str):
    """Dynamically load an agent function from a Python script."""
    path = Path(script_path)
    if not path.exists():
        raise click.ClickException(f"Agent script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("agent_module", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Cannot load module: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    # Look for common agent function names
    for name in ("agent", "run", "predict", "invoke", "call"):
        if hasattr(module, name) and callable(getattr(module, name)):
            return getattr(module, name)

    raise click.ClickException(
        f"No agent function found in {script_path}. "
        "Define a function named 'agent', 'run', 'predict', 'invoke', or 'call'."
    )


@click.group()
def main():
    """Postmortem — Turn production failures into regression tests."""
    pass


@main.command()
@click.argument("trace_file")
@click.option("--label", "-l", default="", help="Label for the trace")
def add(trace_file: str, label: str):
    """Ingest and optionally label a trace file."""
    from postmortem.ingest.jsonl import JSONLTraceIngest
    from postmortem.distill import Distiller

    ingest = JSONLTraceIngest()
    traces = ingest.parse(trace_file)
    if not traces:
        raise click.ClickException(f"No traces found in {trace_file}")

    console.print(f"[green]✓[/green] Parsed {len(traces)} trace(s)")

    _ensure_dir(TRACES_DIR)
    distiller = Distiller()

    for trace in traces:
        case = distiller.distill(trace)
        if label:
            case.label = label

        out_path = TRACES_DIR / f"{case.id}.yaml"
        case.to_yaml(out_path)
        console.print(f"  [cyan]{case.id}[/cyan] → {out_path}")
        if case.assertions:
            for a in case.assertions:
                console.print(f"    • {a.type}: {a.expected} ({a.description})")


@main.command()
@click.option("--out", "-o", default="suite", help="Output directory for eval suite")
@click.option("--traces-dir", "-t", default=None, help="Directory with labeled traces")
def build(out: str, traces_dir: str | None):
    """Build an eval suite from labeled traces."""
    import json
    import shutil

    src = Path(traces_dir) if traces_dir else TRACES_DIR
    if not src.exists():
        raise click.ClickException(f"No traces found at {src}")

    out_path = Path(out)
    _ensure_dir(out_path)

    count = 0
    for f in sorted(src.glob("*.yaml")):
        shutil.copy2(f, out_path / f.name)
        count += 1
    for f in sorted(src.glob("*.json")):
        shutil.copy2(f, out_path / f.name)
        count += 1

    console.print(f"[green]✓[/green] Built suite with {count} case(s) → {out}")


@main.command()
@click.argument("suite_dir")
@click.option("--agent", "-a", required=True, help="Path to agent Python script")
def run(suite_dir: str, agent: str):
    """Run eval suite against an agent."""
    from postmortem.runner import SuiteRunner

    agent_fn = _load_agent(agent)
    runner = SuiteRunner()
    report = runner.run(suite_dir, agent_fn)

    table = Table(title="Postmortem Run Report")
    table.add_column("Case", style="cyan")
    table.add_column("Label")
    table.add_column("Result", style="bold")
    table.add_column("Details")

    for r in report.results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        details = ""
        if r.error:
            details = f"Error: {r.error}"
        elif r.assertion_results:
            failed = [a for a in r.assertion_results if not a.passed]
            if failed:
                details = failed[0].detail
        table.add_row(r.case_id, r.label, status, details[:60])

    console.print(table)
    console.print()
    console.print(report.summary())

    if report.regressions:
        console.print(f"\n[red bold]⚠ {len(report.regressions)} regression(s) detected![/red bold]")
        sys.exit(1)


@main.command("list")
def list_cmd():
    """List available traces and cases."""
    table = Table(title="Postmortem Inventory")
    table.add_column("Type", style="cyan")
    table.add_column("Location")
    table.add_column("Count", justify="right")

    # Traces
    if TRACES_DIR.exists():
        traces = list(TRACES_DIR.glob("*.yaml")) + list(TRACES_DIR.glob("*.json"))
        table.add_row("Labeled traces", str(TRACES_DIR), str(len(traces)))

    # Look for common suite dirs
    for suite_name in ("suite", "evals", "tests/evals"):
        suite_path = Path(suite_name)
        if suite_path.exists():
            cases = list(suite_path.glob("*.yaml")) + list(suite_path.glob("*.json"))
            if cases:
                table.add_row("Eval suite", str(suite_path), str(len(cases)))

    console.print(table)


@main.command()
@click.argument("trace_dir")
@click.option("--threshold", "-t", default=0.5, help="Similarity threshold (0-1)")
def cluster(trace_dir: str, threshold: float):
    """Cluster similar failure traces."""
    from postmortem.ingest.jsonl import JSONLTraceIngest
    from postmortem.cluster import FailureClusterer

    path = Path(trace_dir)
    ingest = JSONLTraceIngest()

    all_traces = []
    for ext in ("*.jsonl", "*.json"):
        for f in sorted(path.glob(ext)):
            all_traces.extend(ingest.parse(f))

    if not all_traces:
        raise click.ClickException(f"No traces found in {trace_dir}")

    console.print(f"[green]✓[/green] Loaded {len(all_traces)} trace(s)")

    clusterer = FailureClusterer(similarity_threshold=threshold)
    clusters = clusterer.cluster(all_traces)

    if not clusters:
        console.print("[yellow]No failure clusters found[/yellow]")
        return

    table = Table(title="Failure Clusters")
    table.add_column("Cluster", style="cyan")
    table.add_column("Label")
    table.add_column("Size", justify="right")
    table.add_column("Similarity", justify="right")

    for c in clusters:
        table.add_row(c.id, c.label, str(len(c.all_traces)), f"{c.similarity_score:.2f}")

    console.print(table)


if __name__ == "__main__":
    main()
