"""Observer callbacks shared by streaming benchmark transports.

Observers keep live-progress concerns out of the HTTP/OpenCode request APIs.
Callbacks are best-effort: a display or telemetry failure must never abort a
model request.
"""
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass


@dataclass
class TaskObserver:
    """Receive live content, thinking, and transport-retry notifications."""

    model_name: str = ""
    pid: str = ""
    on_chunk: Callable[[str], None] | None = None
    on_think_chunk: Callable[[str], None] | None = None
    on_retry: Callable[[], None] | None = None

    def chunk(self, delta: str) -> None:
        """Report a content delta without allowing observer errors to escape."""
        if self.on_chunk is not None:
            with suppress(Exception):
                self.on_chunk(delta)

    def think_chunk(self, delta: str) -> None:
        """Report a thinking delta without affecting request correctness."""
        if self.on_think_chunk is not None:
            with suppress(Exception):
                self.on_think_chunk(delta)

    def retry(self) -> None:
        """Report a transport-level retry without affecting request correctness."""
        if self.on_retry is not None:
            with suppress(Exception):
                self.on_retry()

    @classmethod
    def noop(cls) -> "TaskObserver":
        """Return an observer that intentionally does nothing."""
        return cls()
