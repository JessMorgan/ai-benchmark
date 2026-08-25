"""ShutdownCoordinator – drain persistence and close storage backends.

Encapsulates the three shutdown phases previously defined as closures in
``_run_benchmark``: stop the background flusher, save a final snapshot,
and close the storage backend.  Failures are recorded via the same
``report_persistence_failure`` callback used elsewhere in the orchestrator.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# Function signature of ``report_persistence_failure(stage: str, exc: Exception)``.
FailureReporter = Callable[[str, Exception], None]


class ShutdownCoordinator:
    """Drain all persistence channels during benchmark shutdown.

    Parameters
    ----------
    flusher:
        Background flusher instance with a ``stop(timeout: float) -> bool`` method.
    shutdown_timeout:
        Seconds to wait for each shutdown phase.
    report_persistence_failure:
        Callable that records and prints a persistence failure.
    state:
        ``BenchmarkState`` instance whose ``run_store`` and ``close_run_store``
        are driven at shutdown.
    state_file:
        Path to the state snapshot file (JSON).
    plugin_versions:
        Dict of plugin ids → version string.
    persistence_lock:
        Lock held while saving the final snapshot.
    """

    def __init__(
        self,
        *,
        flusher: Any,
        shutdown_timeout: float,
        report_persistence_failure: FailureReporter,
        state: Any,
        state_file: str,
        plugin_versions: dict[str, str],
        persistence_lock: threading.Lock,
    ) -> None:
        self._flusher = flusher
        self._timeout = shutdown_timeout
        self._report = report_persistence_failure
        self._state = state
        self._state_file = state_file
        self._plugin_versions = plugin_versions
        self._lock = persistence_lock

    # ── Phase implementations ────────────────────────────────────────────

    def stop_flusher(self) -> bool:
        if self._flusher.stop(timeout=self._timeout):
            return True
        self._report(
            "background flush shutdown timeout",
            TimeoutError(f"state flusher did not stop within {self._timeout:g}s"),
        )
        return False

    def save_final_state(self) -> bool:
        with self._lock:
            self._state.run_store.save_snapshot(
                self._state_file,
                plugin_versions=self._plugin_versions,
                raise_on_error=True,
            )
            journal_failures = self._state.consume_journal_failures()
            if journal_failures:
                raise RuntimeError("; ".join(journal_failures))
        return True

    def close_backend(self) -> bool:
        if self._state.close_run_store(timeout=self._timeout):
            return True
        self._report(
            "storage backend close timeout",
            TimeoutError(
                f"storage backend did not close within {self._timeout:g}s"
            ),
        )
        return False

    def check_sqlite_writer_failures(self) -> None:
        """Report any deferred write failures from the SQLite backend."""
        backend = getattr(self._state, "_run_store", None)
        if backend is not None and getattr(backend, "backend_name", "json") == "sqlite":
            for failure in backend.writer.failures:
                self._report("sqlite writer", failure)
