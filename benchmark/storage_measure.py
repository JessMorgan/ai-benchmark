"""Measure local run-storage size and persistence latency.

This tool intentionally uses synthetic records rather than model calls. It is
useful for comparing JSON and SQLite profiles on the same machine and for
recording a reproducible baseline in CI or a benchmark report.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .runtime_records import BenchmarkAttemptRecord, PluginRecord, TargetRecord
from .persistence.sqlite_schema import connect_database
from .persistence.storage import RunIdentity, SQLiteRunStore


def measure_storage(*, targets: int = 8, plugins: int = 4, payload_chars: int = 1024,
                    attempts: int = 2, flush_interval: float = 0.01) -> dict[str, Any]:
    """Return size and latency measurements for synthetic JSON and SQLite runs."""
    if min(targets, plugins, payload_chars, attempts) <= 0:
        raise ValueError("targets, plugins, payload_chars, and attempts must be positive")
    with tempfile.TemporaryDirectory(prefix="ai-benchmark-storage-") as tmp:
        root = Path(tmp)
        json_path = root / "benchmark_state.json"
        json_rows = []
        text = "x" * payload_chars
        for target_index in range(targets):
            row: dict[str, Any] = {
                "model": f"model-{target_index}",
                "state_key": f"model-{target_index}",
                "runner": "http",
                "source": "synthetic",
                "status": "ok",
            }
            for plugin_index in range(plugins):
                pid = f"plugin-{plugin_index}"
                row[f"{pid}_score"] = 10 + plugin_index
                row[f"{pid}_prompt"] = text
                row[f"{pid}_content"] = text
                row[f"{pid}_thinking"] = text
                row[f"{pid}_attempts"] = [
                    {"attempt": attempt, "content": text, "thinking": text}
                    for attempt in range(1, attempts + 1)
                ]
            json_rows.append(row)
        json_state = {
            "active_plugins": [f"plugin-{i}" for i in range(plugins)],
            "model_info": {},
            "results": json_rows,
        }
        json_path.write_text(json.dumps(json_state, ensure_ascii=False), encoding="utf-8")

        sqlite_path = root / "run.sqlite3"
        store = SQLiteRunStore(str(sqlite_path), flush_interval=flush_interval)
        store.start_run(RunIdentity("synthetic", 1), runner_mode="http")
        plugin_records = [
            PluginRecord(
                plugin_id=f"plugin-{i}", plugin_version="1.0.0",
                name=f"Plugin {i}", max_score=20.0, supports_streaming=True,
            )
            for i in range(plugins)
        ]
        target_records = [
            TargetRecord(
                logical_name=f"model-{i}", runner="http", source="synthetic",
                api_model=f"model-{i}", target_signature=f"synthetic/model-{i}",
            )
            for i in range(targets)
        ]
        store.prepare_run(target_records, plugin_records)
        latencies: list[float] = []
        for target in target_records:
            for plugin in plugin_records:
                cell_id = store.get_cell_id(target.logical_name, "http", plugin.plugin_id)
                if cell_id is None:
                    raise RuntimeError("synthetic cell was not registered")
                for attempt in range(1, attempts + 1):
                    started = time.perf_counter()
                    store.record_benchmark_attempt(
                        cell_id,
                        BenchmarkAttemptRecord(
                            attempt_number=attempt, prompt=text, content=text,
                            thinking=text, output_tokens=payload_chars // 4,
                            thinking_tokens=payload_chars // 4,
                            total_tokens=payload_chars // 2,
                            score=10 + plugins, status="completed",
                        ),
                        selected=attempt == attempts,
                    )
                    latencies.append(time.perf_counter() - started)
        flush_started = time.perf_counter()
        store.flush(timeout=30)
        flush_latency = time.perf_counter() - flush_started
        if not store.close(timeout=30):
            raise RuntimeError("synthetic SQLite writer did not close")
        sqlite_size = sum(
            path.stat().st_size
            for path in root.glob("run.sqlite3*")
            if path.is_file()
        )
        connection = connect_database(str(sqlite_path))
        try:
            payload_count = int(connection.execute("SELECT count(*) FROM payloads").fetchone()[0])
            attempt_count = int(connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0])
        finally:
            connection.close()
        json_size = json_path.stat().st_size
        return {
            "parameters": {
                "targets": targets, "plugins": plugins,
                "payload_chars": payload_chars, "attempts": attempts,
            },
            "json_bytes": json_size,
            "sqlite_bytes": sqlite_size,
            "sqlite_to_json_ratio": round(sqlite_size / json_size, 4),
            "payload_rows": payload_count,
            "attempt_rows": attempt_count,
            "record_latency_ms": {
                "count": len(latencies),
                "median": round(statistics.median(latencies) * 1000, 4),
                "p95": round(_percentile(latencies, 0.95) * 1000, 4),
                "max": round(max(latencies) * 1000, 4),
            },
            "flush_latency_ms": round(flush_latency * 1000, 4),
        }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--plugins", type=int, default=4)
    parser.add_argument("--payload-chars", type=int, default=1024)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--flush-interval", type=float, default=0.01)
    args = parser.parse_args(argv)
    print(json.dumps(measure_storage(
        targets=args.targets, plugins=args.plugins,
        payload_chars=args.payload_chars, attempts=args.attempts,
        flush_interval=args.flush_interval,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
