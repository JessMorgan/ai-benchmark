"""Lifecycle invariants and bounded shutdown supervision."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class LifecycleInvariantError(ValueError):
    """Raised when a runtime model record violates its lifecycle contract."""


_ALLOWED_STATUSES = {
    "pending", "queued", "running", "completed", "failed", "error",
}


def validate_model_info(model_name: str, info: dict[str, Any]) -> None:
    """Validate fields that coordinate scheduling, display, and shutdown."""
    if not isinstance(model_name, str) or not model_name:
        raise LifecycleInvariantError("model name must be a non-empty string")
    if not isinstance(info, dict):
        raise LifecycleInvariantError(f"{model_name}: model info must be an object")
    status = info.get("status")
    if status not in _ALLOWED_STATUSES and not (
        isinstance(status, str) and status.startswith("running_")
    ):
        raise LifecycleInvariantError(f"{model_name}: unsupported status {status!r}")
    running = info.get("running_pids", [])
    if not isinstance(running, list) or not all(isinstance(pid, str) and pid for pid in running):
        raise LifecycleInvariantError(f"{model_name}: running_pids must contain strings")
    if len(running) != len(set(running)):
        raise LifecycleInvariantError(f"{model_name}: running_pids contains duplicates")
    for key in ("attempt", "elapsed"):
        value = info.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            raise LifecycleInvariantError(f"{model_name}: {key} must be non-negative")
    if not isinstance(info.get("preloading", False), bool):
        raise LifecycleInvariantError(f"{model_name}: preloading must be boolean")


def validate_snapshot(snapshot: dict[str, dict[str, Any]]) -> None:
    """Validate every model record in a copied runtime snapshot."""
    if not isinstance(snapshot, dict):
        raise LifecycleInvariantError("snapshot must be an object")
    for model_name, info in snapshot.items():
        validate_model_info(model_name, info)


@dataclass(frozen=True)
class ShutdownPhaseResult:
    """Outcome of one named shutdown phase."""

    name: str
    elapsed: float
    completed: bool
    error: str | None = None


@dataclass
class ShutdownSupervisor:
    """Run named cleanup phases with a shared deadline and diagnostics."""

    timeout: float
    results: list[ShutdownPhaseResult] = field(default_factory=list)
    _deadline: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ValueError("shutdown timeout must be numeric")
        if self.timeout <= 0:
            raise ValueError("shutdown timeout must be positive")
        self._deadline = time.monotonic() + float(self.timeout)

    def run(self, name: str, action: Callable[[], Any]) -> bool:
        """Run one phase without blocking past the remaining deadline.

        Python cannot safely terminate a running thread. On timeout the phase
        is therefore marked failed and the daemon helper is left to finish in
        the background; callers receive a bounded shutdown result instead of
        hanging indefinitely.
        """
        started = time.monotonic()
        remaining = self._remaining()
        if remaining <= 0:
            self.results.append(ShutdownPhaseResult(name, 0.0, False, "deadline exceeded"))
            return False
        outcome: dict[str, Any] = {}
        finished = threading.Event()

        def invoke() -> None:
            try:
                outcome["value"] = action()
            except Exception as exc:  # noqa: BLE001 - supervisor records phase failures
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                finished.set()

        worker = threading.Thread(target=invoke, name=f"shutdown-{name}", daemon=True)
        worker.start()
        finished.wait(remaining)
        elapsed = time.monotonic() - started
        completed: bool
        error: str | None
        if not finished.is_set():
            completed = False
            error = f"phase exceeded remaining deadline ({remaining:.3f}s)"
        elif "error" in outcome:
            completed = False
            error = outcome["error"]
        else:
            completed = outcome.get("value") is not False
            error = None if completed else "phase reported timeout"
        self.results.append(ShutdownPhaseResult(name, elapsed, completed, error))
        return completed

    def _remaining(self) -> float:
        """Return time left on the supervisor's monotonic total deadline."""
        return self._deadline - time.monotonic()

    @property
    def successful(self) -> bool:
        """Whether all recorded phases completed successfully."""
        return all(result.completed for result in self.results)

    def as_dict(self) -> list[dict[str, Any]]:
        """Serialize phase diagnostics for run-info.json."""
        return [
            {
                "name": result.name,
                "elapsed": round(result.elapsed, 6),
                "completed": result.completed,
                "error": result.error,
            }
            for result in self.results
        ]
