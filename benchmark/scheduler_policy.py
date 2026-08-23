"""Pure scheduling policy for benchmark and judge capacity."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSchedulingPolicy:
    """Capacity policy for one source's benchmark and judge pipelines."""

    source: str
    model_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_limit, int) or isinstance(self.model_limit, bool):
            raise ValueError(f"{self.source}: model_limit must be a positive integer")
        if self.model_limit <= 0:
            raise ValueError(f"{self.source}: model_limit must be a positive integer")

    def capacity(self, *, benchmark_active: bool) -> tuple[int, int]:
        """Return ``(benchmark_slots, judge_slots)`` for current source work."""
        if benchmark_active:
            return self.model_limit, 0
        return self.model_limit, self.model_limit

    def benchmark_has_priority(self, *, benchmark_active: bool) -> bool:
        """Whether benchmark work currently owns all source model slots."""
        return bool(benchmark_active)

    def can_start_judges(self, *, benchmark_active: bool) -> bool:
        """Whether judge workers may be activated for this source."""
        return not benchmark_active
