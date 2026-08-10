"""Abstract base classes and typed contracts for AI benchmark plugins."""
import abc
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from numbers import Real
from typing import Any

SCORE_SCHEMA = "percentage-v1"


def _decimal_value(value, name):
    """Convert a numeric value to Decimal without accepting non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(  # noqa: TRY004 - public numeric contract uses ValueError
            f"{name} must be numeric, got {value!r}"
        )
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not decimal.is_finite():
        raise ValueError(f"{name} must be finite, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return decimal


def normalize_score(raw_score: float, max_score: float) -> int:
    """Clamp a native plugin score and return an integer percentage.

    Native rubric scores remain in each plugin's task-specific scale. This is
    the single public-schema conversion point: Decimal half-up rounding makes
    ties deterministic instead of using Python's ties-to-even ``round``.
    """
    raw = _decimal_value(raw_score, "raw_score")
    maximum = _decimal_value(max_score, "max_score")
    if maximum <= 0:
        raise ValueError(f"max_score must be positive, got {max_score!r}")
    raw = max(Decimal(0), min(raw, maximum))
    percentage = (Decimal(100) * raw / maximum).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    return int(max(Decimal(0), min(percentage, Decimal(100))))


def serialize_rubric(rubric: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert native rubric entries to the persisted ``points/total`` shape.

    Plugins continue to return their original native rubric (``earned``,
    ``max``, and ``missed``). The benchmark boundary is responsible for the
    persisted representation: criterion credit is called ``points`` and the
    criterion denominator is called ``total``. Nested evidence and negative
    findings retain their native point values as well; no percentage rubric
    values are fabricated or stored.
    """
    serialized = []
    for criterion in rubric:
        if not isinstance(criterion, dict):
            raise TypeError(f"rubric criterion must be an object, got {criterion!r}")
        if "earned" not in criterion or "max" not in criterion:
            raise ValueError("rubric criterion must contain earned and max")
        item = {
            "name": criterion.get("name", ""),
            "points": criterion["earned"],
            "total": criterion["max"],
        }
        for key in ("matched", "evidence", "errors", "negative_findings"):
            if key in criterion:
                item[key] = criterion[key]
        serialized.append(item)
    return serialized


def sanitize_diagnostics(value: Any) -> Any:
    """Return diagnostics unchanged for compatibility with developer tooling.

    The persisted public rubric is converted separately by
    :func:`serialize_rubric`; evaluator diagnostics may legitimately contain
    native validator evidence and should not be silently stripped.
    """
    return value


@dataclass(frozen=True)
class EvaluationResult:
    """Native score, rubric breakdown, and diagnostics from a task plugin.

    ``score`` and ``rubric`` are intentionally native evaluator values. The
    benchmark core normalizes the evaluated score exactly once before exposing
    or persisting public benchmark results.
    """

    score: float
    rubric: list[dict[str, Any]]
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self):
        """Give every evaluation, including direct early returns, diagnostics."""
        if self.diagnostics is None:
            object.__setattr__(self, "diagnostics", {
                "source": "plugin.evaluate",
                "criterion_count": len(self.rubric),
                "matched_criterion_count": sum(
                    1 for criterion in self.rubric if criterion.get("matched")
                ),
                "errors": [],
            })

    def diagnostic_data(self) -> dict[str, Any]:
        """Return native evaluator diagnostics for developer-facing tooling."""
        return {
            "score": self.score,
            "rubric": self.rubric,
            "diagnostics": self.diagnostics or {},
        }


@dataclass(frozen=True)
class PluginTaskResult:
    """Outcome of running one benchmark task."""

    result: dict[str, Any] | None
    error: str | None


class BenchmarkOutputPlugin(abc.ABC):
    """A report-output generator that persists benchmark results to disk.

    Each output plugin writes a single report file (e.g. results.md,
    results.csv) into the output directory.  The main benchmark runner
    discovers output plugins from the ``plugins/`` directory and invokes
    them after all task plugins have completed.
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Stable machine-readable identifier, e.g. 'output-markdown'."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Markdown Report'."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def extension(self) -> str:
        """File extension (without dot), e.g. 'md', 'csv', 'html'."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, results, active_plugins, output_dir=None, session_seed=None):
        """Write the report file into *output_dir* and return the file path."""
        raise NotImplementedError


class BenchmarkTaskPlugin(abc.ABC):
    """A benchmark task with a native rubric and task-specific score scale."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Stable machine-readable identifier, e.g. 'code-review'."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Semantic version for result correlation, e.g. '1.0.0'."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable task name, e.g. 'Rate Limiter'."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_score(self) -> float:
        """Internal native maximum for this task's rubric."""
        raise NotImplementedError

    @property
    def supports_streaming(self) -> bool:
        """Whether the task should use the streaming API path."""
        return True

    @abc.abstractmethod
    def get_prompt(self) -> str:
        """Return the prompt text sent to the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_temperature(self, global_config: dict) -> float | None:
        """Return the temperature to use for this task, or None to omit it."""
        raise NotImplementedError

    @abc.abstractmethod
    def score(self, response_text: str) -> float:
        """Return the native score in this plugin's task-specific scale."""
        raise NotImplementedError

    def evaluate(self, response_text: str) -> EvaluationResult:
        """Return the native score and detailed rubric for internal evaluation."""
        return EvaluationResult(
            self.score(response_text), [],
            {"source": "plugin.score", "criterion_count": 0, "errors": []},
        )
