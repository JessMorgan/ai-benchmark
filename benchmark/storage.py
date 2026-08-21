"""Storage backend contracts for benchmark runs.

Stage 1 keeps JSON as the active backend while giving benchmark and judge
callers a stable seam for the SQLite implementation.  The JSON adapter
intentionally delegates to ``BenchmarkState`` so this stage does not alter
resume, journaling, or result semantics.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunStore(Protocol):
    """Persistence operations used by benchmark and judge execution."""

    backend_name: str

    def record_result(self, result: dict[str, Any]) -> None:
        """Record one benchmark result row."""
        ...

    def update_model(self, model_name: str, **fields: Any) -> None:
        """Update live model state."""
        ...

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        """Record a judge projection update."""
        ...

    def save_snapshot(self, path: str, plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        """Persist a backend snapshot."""
        ...

    def latest_results(self) -> list[dict[str, Any]]:
        """Return the current result read model."""
        ...


@runtime_checkable
class PayloadStore(Protocol):
    """Canonical storage for large prompt and response payloads."""

    def put(self, kind: str, data: bytes) -> int:
        """Store or deduplicate a payload and return its ID."""
        ...

    def get(self, payload_id: int) -> bytes:
        """Return a payload by ID."""
        ...


@runtime_checkable
class DebugLogStore(Protocol):
    """Append-only diagnostic log contract."""

    def append(self, path: str, data: str | bytes) -> None:
        """Append diagnostic data to a log."""
        ...

    def close(self) -> None:
        """Flush and close owned log resources."""
        ...


@runtime_checkable
class ReportSource(Protocol):
    """Read-only source used to construct derived reports."""

    def load_results(self, path: str) -> tuple[list[dict[str, Any]], list[str], int | None]:
        """Load result rows, active plugin IDs, and the session seed."""
        ...


class JsonRunStore:
    """RunStore adapter that preserves the existing JSON state implementation."""

    backend_name = "json"

    def __init__(self, state: Any):
        self._state = state

    def record_result(self, result: dict[str, Any]) -> None:
        self._state.add_result(result)

    def update_model(self, model_name: str, **fields: Any) -> None:
        self._state.update(model_name, **fields)

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        self._state.update_judge_result(
            state_key, runner, plugin_id, **fields,
        )

    def save_snapshot(self, path: str, plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        return self._state.compact_journal(
            path, plugin_versions=plugin_versions,
            raise_on_error=raise_on_error,
        )

    def latest_results(self) -> list[dict[str, Any]]:
        return self._state.latest_results()


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
    """Explicit placeholder for JSON payload compatibility.

    JSON state currently embeds payload-bearing fields in result dictionaries.
    The canonical payload implementation arrives with the SQLite stage; this
    class makes that limitation explicit instead of silently inventing a
    second JSON representation.
    """

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
