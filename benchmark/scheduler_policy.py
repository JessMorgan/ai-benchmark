"""Pure scheduling policy for benchmark and judge capacity."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSchedulingPolicy:
    """Capacity policy for one source's benchmark and judge pipelines.

    Benchmarks always keep priority and claim the full source model limit.
    Judge workers may use the model slots benchmarks cannot fill, and a
    benchmarking model that is also a judge may share its plugin capacity.
    """

    source: str
    model_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_limit, int) or isinstance(self.model_limit, bool):
            raise ValueError(f"{self.source}: model_limit must be a positive integer")
        if self.model_limit <= 0:
            raise ValueError(f"{self.source}: model_limit must be a positive integer")

    def benchmark_slots(self, queued: int = 0) -> int:
        """Benchmarks always claim the full source model limit (priority)."""
        return self.model_limit

    def judge_model_slots(self, queued: int = 0) -> int:
        """Model slots benchmarks cannot use become judge slots.

        ``queued`` is the number of benchmark models pending or running for
        the source; at most ``model_limit`` run at once, so the judge budget
        is the leftover capacity. A benchmarking model that is also a judge
        rides its model's benchmark slot instead (see the pool's dual-role
        handling), so it does not consume a free slot here.
        """
        try:
            queued = max(0, int(queued))
        except (TypeError, ValueError):
            queued = 0
        return max(0, self.model_limit - min(self.model_limit, queued))

    def can_start_judges(self, queued: int = 0) -> bool:
        """Whether free model slots exist for regular (non-dual-role) judges.

        Dual-role judges can also start while the source is benchmarking; that
        exemption is enforced at the pool level, not by this policy.
        """
        return self.judge_model_slots(queued) > 0

    def benchmark_has_priority(self) -> bool:
        """Benchmark work always owns the source's model slots."""
        return True
