"""JSONL trace ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from postmortem.ingest.base import Trace, TraceIngest


class JSONLTraceIngest(TraceIngest):
    """Parse JSONL trace files. Each line is one trace object."""

    def parse(self, source: str) -> list[Trace]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {source}")

        traces: list[Trace] = []
        text = path.read_text().strip()
        if not text:
            return traces

        # Try parsing as a single JSON object first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    traces.append(Trace.from_dict(item))
            elif isinstance(data, dict):
                traces.append(Trace.from_dict(data))
            return traces
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL (one JSON object per line)
        for line_num, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                traces.append(Trace.from_dict(data))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

        return traces
