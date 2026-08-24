"""Application entrypoint for the benchmark process.

Argument construction remains in :mod:`benchmark.cli`; this module owns the
process-level decision about where the orchestrator and live TUI run.
"""
from __future__ import annotations

import sys
import threading

from benchmark import cli


def main() -> None:
    """Start the benchmark, keeping Textual on the interpreter main thread."""
    if not cli._textual_tui_enabled():
        cli._run_benchmark()
        return

    handoff = {"ready": threading.Event(), "interrupt": threading.Event()}
    outcome: dict[str, object] = {}

    def run_orchestrator() -> None:
        try:
            cli._run_benchmark(handoff)
        except SystemExit as exc:
            outcome["exit_code"] = exc.code
        except BaseException as exc:  # noqa: BLE001 - re-raise fatal worker errors
            outcome["error"] = exc

    orchestrator = threading.Thread(target=run_orchestrator, daemon=True)
    orchestrator.start()
    while not handoff["ready"].is_set() and orchestrator.is_alive():
        handoff["ready"].wait(0.05)

    if "args" in handoff:
        try:
            cli.tui_main(*handoff["args"])
        except KeyboardInterrupt:
            handoff["interrupt"].set()
            handoff["stop_event"].set()

    orchestrator.join()
    if "error" in outcome:
        error = outcome["error"]
        if isinstance(error, BaseException):
            raise error
    if "exit_code" in outcome:
        sys.exit(outcome["exit_code"])


__all__ = ["main"]
