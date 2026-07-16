"""Abstract base class for AI benchmark task plugins."""
import abc


class BenchmarkTaskPlugin(abc.ABC):
    """A single benchmark task that can be run against a model.

    Each plugin defines a prompt, a scoring function, and metadata such as
    a stable ID, version, and maximum score. The main benchmark runner
    discovers plugins from the ``plugins/`` directory and runs the active set
    against every model.
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Stable machine-readable identifier, e.g. 'rate-limiter'."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Semantic version for result correlation, e.g. '1.0.0'."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable task name, e.g. 'Rate Limiter'."""
        ...

    @property
    @abc.abstractmethod
    def max_score(self) -> float:
        """Maximum possible score for this task."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether the task should use the streaming API path.

        Defaults to True. Set to False for tasks where only the final
        response is needed.
        """
        return True

    @abc.abstractmethod
    def get_prompt(self) -> str:
        """Return the prompt text sent to the model."""
        ...

    @abc.abstractmethod
    def get_temperature(self, global_config: dict) -> float | None:
        """Return the temperature to use for this task, or None to omit it."""
        ...

    @abc.abstractmethod
    def score(self, response_text: str) -> float:
        """Score the model's response and return a float."""
        ...

    def evaluate(self, response_text: str) -> tuple[float, list[dict]]:
        """Score the model's response and return a detailed rubric.

        Returns a tuple of ``(score, rubric)`` where ``rubric`` is a list of
        dictionaries describing each graded criterion. Plugins may override
        this method to provide a breakdown; the default implementation
        returns ``(self.score(response_text), [])`` for backward compatibility.
        """
        return self.score(response_text), []
