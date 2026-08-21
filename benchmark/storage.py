"""Backend-neutral persistence façade for benchmark runs."""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .sqlite_benchmarks import SQLiteBenchmarkStore
from .sqlite_judges import SQLiteJudgeStore
from .sqlite_writer import SQLiteWriteQueue


@dataclass(frozen=True)
class RunIdentity:
    """Logical run and continuation revision identity."""

    run_id: str
    revision_id: int | None


@runtime_checkable
class RunStore(Protocol):
    """Persistence façade used by benchmark and judge runtime code."""

    backend_name: str

    def start_run(self, identity: RunIdentity, **metadata: Any) -> None:
        """Initialize or attach the current run/revision."""
        ...

    def record_result(self, result: dict[str, Any]) -> None:
        """Record one benchmark result read-model update."""
        ...

    def update_model(self, model_name: str, **fields: Any) -> None:
        """Update live model state or its backend projection."""
        ...

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        """Record a judge projection update."""
        ...

    def save_snapshot(self, path: str | None = None,
                      plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        """Flush/persist a backend snapshot."""
        ...

    def latest_results(self) -> list[dict[str, Any]]:
        """Return the current result read model."""
        ...

    def flush(self, timeout: float | None = None) -> None:
        """Wait for queued persistence work."""
        ...

    def close(self, timeout: float | None = None) -> bool:
        """Close the backend and return whether shutdown completed."""
        ...


@runtime_checkable
class PayloadStore(Protocol):
    """Canonical storage for large prompt and response payloads."""

    def put(self, kind: str, data: bytes) -> int:
        ...

    def get(self, payload_id: int) -> bytes:
        ...


@runtime_checkable
class DebugLogStore(Protocol):
    """Append-only diagnostic log contract."""

    def append(self, path: str, data: str | bytes) -> None:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class ReportSource(Protocol):
    """Read-only source used to construct derived reports."""

    def load_results(self, path: str) -> tuple[list[dict[str, Any]], list[str], int | None]:
        ...


class JsonRunStore:
    """Facade adapter preserving the existing JSON ``BenchmarkState`` behavior."""

    backend_name = "json"

    def __init__(self, state: Any):
        self._state = state
        self.identity: RunIdentity | None = None

    def start_run(self, identity: RunIdentity, **metadata: Any) -> None:
        del metadata
        self.identity = identity

    def record_result(self, result: dict[str, Any]) -> None:
        self._state.add_result(result)

    def update_model(self, model_name: str, **fields: Any) -> None:
        # BenchmarkState owns the live JSON mutation; this method is only a
        # façade hook for non-JSON backends.
        del model_name, fields

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        self._state.update_judge_result(state_key, runner, plugin_id, **fields)

    def save_snapshot(self, path: str | None = None,
                      plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        if path is None:
            raise ValueError("JSON snapshots require a state path")
        return self._state.compact_journal(
            path, plugin_versions=plugin_versions,
            raise_on_error=raise_on_error,
        )

    def latest_results(self) -> list[dict[str, Any]]:
        return self._state.latest_results()

    def flush(self, timeout: float | None = None) -> None:
        del timeout

    def close(self, timeout: float | None = None) -> bool:
        del timeout
        return True


class SQLiteRunStore:
    """Facade over the SQLite schema stores and one background writer thread.

    The façade deliberately accepts backend operations as callables so callers
    do not share SQLite connections. Each operation runs on the writer thread
    and commits with the configured batch transaction. The normalized benchmark
    and judge stores remain available through ``benchmark`` and ``judges`` for
    migration code and advanced callers.
    """

    backend_name = "sqlite"

    def __init__(
        self,
        path: str,
        *,
        batch_size: int = 64,
        flush_interval: float = 0.25,
        synchronous: str = "NORMAL",
        failure_callback: Any = None,
    ):
        self.path = path
        self.writer = SQLiteWriteQueue(
            path, batch_size=batch_size, flush_interval=flush_interval,
            synchronous=synchronous, failure_callback=failure_callback,
        )
        self.identity: RunIdentity | None = None
        self._metadata: dict[str, Any] = {}
        self._results: dict[tuple[Any, Any], dict[str, Any]] = {}
        self._connection: sqlite3.Connection | None = None
        self.benchmark: SQLiteBenchmarkStore | None = None
        self.judges: SQLiteJudgeStore | None = None

    def start_run(self, identity: RunIdentity, **metadata: Any) -> None:
        self.identity = identity
        self._metadata = dict(metadata)
        self.writer.start()
        self.writer.flush(timeout=10)

    def submit(self, operation: Any) -> Future[Any]:
        """Submit a normalized SQLite operation to the background writer."""
        return self.writer.submit(operation)

    def record_result(self, result: dict[str, Any]) -> None:
        key = (result.get("state_key", result.get("model")), result.get("runner", "http"))
        self._results[key] = dict(result)

    def update_model(self, model_name: str, **fields: Any) -> None:
        del model_name, fields
        # Live model state remains owned by BenchmarkState. Normalized model
        # projections are written by the benchmark store in the next runtime
        # integration layer; this no-op keeps the façade safe during migration.

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        key = (state_key, runner)
        row = self._results.setdefault(key, {
            "model": state_key, "state_key": state_key, "runner": runner,
        })
        row.update({f"{plugin_id}_{name}": value for name, value in fields.items()})

    def save_snapshot(self, path: str | None = None,
                      plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        del path, plugin_versions, raise_on_error
        try:
            self.writer.flush(timeout=10)
        except (OSError, RuntimeError, TimeoutError):
            return False
        return not self.writer.failures

    def latest_results(self) -> list[dict[str, Any]]:
        return list(self._results.values())

    def flush(self, timeout: float | None = None) -> None:
        self.writer.flush(timeout=timeout)

    def close(self, timeout: float | None = None) -> bool:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        return self.writer.close(timeout=timeout)


class JsonReportSource:
    """Read the legacy JSON state format for report-only generation."""

    def load_results(self, path: str) -> tuple[list[dict[str, Any]], list[str], int | None]:
        state_path = path
        if os.path.isdir(path):
            state_path = os.path.join(path, "benchmark_state.json")
        if state_path.endswith((".sqlite3", ".db")):
            raise RuntimeError("Use SQLiteReportSource for SQLite report loading")
        with open(state_path, encoding="utf-8") as handle:
            data = json.load(handle)
        results = data.get("results")
        if not isinstance(results, list):
            raise TypeError(f"{state_path} does not contain a results list")
        active_plugins = data.get("active_plugins") or []
        if not isinstance(active_plugins, list) or not all(
            isinstance(plugin_id, str) for plugin_id in active_plugins
        ):
            raise TypeError(f"{state_path} contains invalid active_plugins metadata")
        return results, active_plugins, data.get("session_seed")


class JsonPayloadStore:
    """Compatibility placeholder for JSON's embedded payload representation."""

    def put(self, kind: str, data: bytes) -> int:
        del kind, data
        raise NotImplementedError("JSON storage has no separate payload table")

    def get(self, payload_id: int) -> bytes:
        del payload_id
        raise NotImplementedError("JSON storage has no separate payload table")


class JsonDebugLogStore:
    """Compatibility adapter for existing plaintext JSON-run diagnostics."""

    def append(self, path: str, data: str | bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = data.encode("utf-8") if isinstance(data, str) else data
        with open(path, "ab") as handle:
            handle.write(payload)

    def close(self) -> None:
        return None


def latest_result_rows(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the last row for each state-key/runner identity."""
    latest: dict[tuple[Any, Any], dict[str, Any]] = {}
    for result in results:
        key = (
            result.get("state_key", result.get("model")),
            result.get("runner", "http"),
        )
        latest[key] = result
    return list(latest.values())
