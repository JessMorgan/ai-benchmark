"""Handlers for CLI commands that do not start a benchmark run."""
from __future__ import annotations

import os
import sqlite3
from typing import Any

from benchmark.outputs import save_outputs
from benchmark.persistence.sqlite_integrity import check_integrity
from benchmark.persistence.sqlite_reports import SQLiteReportSource, sqlite_path_from_report_path
from benchmark.persistence.storage import JsonReportSource, latest_result_rows
from plugins import discover_plugins


def generate_reports(path: str, output_formats: list[str], revision: int | None = None) -> list[str]:
    """Generate selected reports from either JSON state or SQLite storage."""
    sqlite_path = sqlite_path_from_report_path(path)
    source = None
    try:
        if sqlite_path is not None:
            source = SQLiteReportSource.open(sqlite_path)
            results, active_ids, seed, _ = source.load_results(
                revision=revision, include_reused=True,
            )
            output_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
        else:
            output_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
            results, active_ids, seed = JsonReportSource().load_results(path)
            results = latest_result_rows(results)
        plugins = [p for p in discover_plugins() if p.id in active_ids]
        missing = set(active_ids) - {p.id for p in plugins}
        if missing:
            raise ValueError(f"plugins are unavailable: {', '.join(sorted(missing))}")
        return save_outputs(results, output_dir, plugins,
                            output_formats=output_formats, session_seed=seed)
    finally:
        if source is not None:
            source.close()


def check_sqlite(path: str) -> dict[str, Any]:
    """Return an integrity report for a SQLite run."""
    sqlite_path = sqlite_path_from_report_path(path)
    if sqlite_path is None:
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"file:{os.path.abspath(sqlite_path)}?mode=ro", uri=True,
    )
    try:
        connection.row_factory = sqlite3.Row
        return check_integrity(connection).as_dict()
    finally:
        connection.close()


def list_plugins() -> str:
    """Return the formatted discovered plugin list."""
    from plugins import format_plugin_list
    return format_plugin_list(discover_plugins())
