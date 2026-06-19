# crucible

Your evals are green. Production is still unreliable. The gap is the failures you watched happen and never turned into tests.

crucible takes real agent traces from production failures and turns them into eval cases — minimal reproducing inputs with assertions. The same failure can't ship twice because it's now in your regression suite.

## The idea

```
prod failure trace
      │
      ├─ cluster with similar failures
      │
      ├─ distill to minimal repro input
      │
      └─ generate eval case + assertion
              │
              └─ regression gate on next deploy
```

The clustering step is what makes this practical. You typically have hundreds of failures but only a handful of distinct root causes. Jaccard similarity over tool call sequences groups them, then you distill one representative case per cluster instead of one per failure.

## Usage

```bash
pip install -e .

# ingest a failure trace
crucible add examples/traces/failing_trace.jsonl --label "tool-loop"

# build eval suite from labeled traces
crucible build --out examples/suite

# run suite against your agent
crucible run examples/suite --agent my_agent.py

# find patterns in unlabeled traces
crucible cluster examples/traces
```

## Trace format

```json
{"id": "trace-1", "messages": [
  {"role": "user", "content": "List files in /tmp"},
  {"role": "assistant", "content": null, "tool_calls": [{"name": "list_files", "args": {"path": "/tmp"}}]},
  {"role": "tool", "content": "Error: permission denied"}
], "outcome": "error", "metadata": {"model": "gpt-4o", "tokens": 150}}
```

Accepts JSONL, OpenTelemetry spans, and LangChain trace format.

## Eval case output

```yaml
id: case-trace-1
label: tool-loop-on-empty-result
input:
  messages:
    - role: user
      content: 'List files in /tmp'
assertions:
  - type: not_contains
    expected: 'Error: permission denied'
  - type: tool_called
    expected: 'list_files'
```

## Python API

```python
from crucible import from_trace, run_suite

case = from_trace("trace.jsonl")

report = run_suite("suite/", agent=my_agent)
assert report.no_regressions
```

## License

MIT
