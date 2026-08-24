"""Developer-only offline evaluation helpers for inspecting saved responses.

This module intentionally exposes native evaluator scores and raw rubric
breakdowns for debugging. It is not part of the benchmark's public result or
persistence schema; normal runs use ``benchmark.core`` to produce normalized
percentage-v1 data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plugins import discover_plugins


def evaluate_saved_response(plugin_id: str, response_path: str | Path) -> dict[str, Any]:
    """Return native developer diagnostics, not public benchmark scores."""
    plugins = {plugin.id: plugin for plugin in discover_plugins()}
    try:
        plugin = plugins[plugin_id]
    except KeyError as exc:
        available = ", ".join(sorted(plugins))
        raise ValueError(f"Unknown plugin {plugin_id!r}; available plugins: {available}") from exc

    path = Path(response_path)
    response_text = path.read_text(encoding="utf-8")
    evaluation = plugin.evaluate(response_text)
    result = evaluation.diagnostic_data()
    result.update({
        "diagnostic_schema": "native-evaluator-v1",
        "plugin": plugin.id,
        "plugin_version": plugin.version,
        "response_path": str(path),
        "response_length": len(response_text),
    })
    return result


def write_evaluation_diagnostics(plugin_id: str, response_path: str | Path) -> None:
    """Print one saved-response evaluation as formatted JSON."""
    print(json.dumps(evaluate_saved_response(plugin_id, response_path), indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate one saved benchmark response offline.")
    parser.add_argument("plugin", help="challenge plugin ID")
    parser.add_argument("response", help="path to the saved response text")
    arguments = parser.parse_args()
    write_evaluation_diagnostics(arguments.plugin, arguments.response)
