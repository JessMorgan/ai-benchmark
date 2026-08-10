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


def _normalize_rubric_value(value, criterion_max):
    """Convert a criterion-relative numeric diagnostic to a percentage."""
    return normalize_score(value, criterion_max)


def normalize_rubric(rubric: list[dict[str, Any]], max_score: float) -> list[dict[str, Any]]:
    """Convert native rubric diagnostics to the public percentage-only shape."""
    normalized = []
    for criterion in rubric:
        criterion_max = criterion.get("max", 0)
        if _decimal_value(criterion_max, "criterion max") <= 0:
            raise ValueError(f"criterion max must be positive, got {criterion_max!r}")
        item = {
            "name": criterion.get("name", ""),
            "score_percent": _normalize_rubric_value(
                criterion.get("earned", 0), criterion_max
            ),
            "weight_percent": normalize_score(criterion_max, max_score),
        }
        for key in ("matched", "evidence", "errors"):
            if key in criterion:
                item[key] = criterion[key]
        if "negative_findings" in criterion:
            findings = []
            for finding in criterion["negative_findings"]:
                if isinstance(finding, dict):
                    public_finding = {
                        key: value for key, value in finding.items() if key != "points"
                    }
                    if "points" in finding:
                        public_finding["points_percent"] = normalize_score(
                            finding["points"], criterion_max
                        )
                    findings.append(public_finding)
                else:
                    findings.append(finding)
            item["negative_findings"] = findings
        evidence = item.get("evidence")
        if isinstance(evidence, list):
            public_evidence = []
            for entry in evidence:
                if isinstance(entry, dict) and "points" in entry:
                    public_entry = {
                        key: value for key, value in entry.items() if key != "points"
                    }
                    public_entry["points_percent"] = normalize_score(
                        entry["points"], criterion_max
                    )
                    public_evidence.append(public_entry)
                else:
                    public_evidence.append(entry)
            item["evidence"] = public_evidence
        normalized.append(item)
    return normalized


def sanitize_diagnostics(value: Any) -> Any:
    """Remove native point-scale fields from persisted diagnostics.

    Diagnostics remain useful in benchmark results, but arbitrary plugin or
    validator evidence must not reintroduce the native rubric scale through
    nested ``max``, ``earned``, ``missed``, or ``points`` keys. Percentage
    variants (for example ``points_percent``) are retained.
    """
    raw_keys = {"max", "max_score", "earned", "missed", "points", "raw_score"}
    if isinstance(value, dict):
        return {
            key: sanitize_diagnostics(item)
            for key, item in value.items()
            if key not in raw_keys
        }
    if isinstance(value, list):
        return [sanitize_diagnostics(item) for item in value]
    return value


@dataclass(frozen=True)
class EvaluationResult:
    """Native score, rubric breakdown, and diagnostics from a task plugin.

    ``score`` and ``rubric`` are intentionally native evaluator values. The
    benchmark core and the public :meth:`BenchmarkTaskPlugin.score` method
    normalize them exactly once before exposing or persisting public results.
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
    """A benchmark task with a native rubric and a normalized public score."""

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

    def __init_subclass__(cls, **kwargs):
        """Adapt legacy native ``score`` implementations to the public API."""
        super().__init_subclass__(**kwargs)
        native_score = cls.__dict__.get("score")
        if native_score is not None:
            cls._native_score = native_score
            cls.score = BenchmarkTaskPlugin.score

    def score(self, response_text: str) -> int:
        """Return the normalized public score as an integer percentage."""
        evaluation = self.evaluate(response_text)
        return normalize_score(evaluation.score, self.max_score)

    def evaluate(self, response_text: str) -> EvaluationResult:
        """Return the native score and detailed rubric for internal evaluation.

        Legacy third-party plugins that still implement ``score`` are treated
        as native evaluators here; built-in plugins override this method with
        their detailed rubric implementation.
        """
        legacy_score = getattr(type(self), "_native_score", None)
        if legacy_score is None:
            raise NotImplementedError
        return EvaluationResult(
            legacy_score(self, response_text), [],
            {"source": "plugin.score", "criterion_count": 0, "errors": []},
        )
