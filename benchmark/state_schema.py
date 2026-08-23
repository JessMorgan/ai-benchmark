"""Validation for the persisted JSON benchmark state schema."""
from __future__ import annotations

from typing import Any


class StateSchemaError(ValueError):
    """Raised when a persisted state file cannot safely be resumed."""


def validate_state_data(data: Any) -> None:
    """Validate the durable top-level state contract before it is consumed."""
    if not isinstance(data, dict):
        raise StateSchemaError("state must be a JSON object")
    required = ("model_info", "results", "active_plugins", "score_schema")
    missing = [key for key in required if key not in data]
    if missing:
        raise StateSchemaError(f"state is missing required fields: {', '.join(missing)}")
    if not isinstance(data["model_info"], dict):
        raise StateSchemaError("state.model_info must be an object")
    if not isinstance(data["results"], list):
        raise StateSchemaError("state.results must be an array")
    if not isinstance(data["active_plugins"], list) or not all(
        isinstance(plugin_id, str) and plugin_id for plugin_id in data["active_plugins"]
    ):
        raise StateSchemaError("state.active_plugins must contain non-empty strings")
    if not isinstance(data["score_schema"], str) or not data["score_schema"]:
        raise StateSchemaError("state.score_schema must be a non-empty string")
    plugin_versions = data.get("plugin_versions", {})
    if not isinstance(plugin_versions, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in plugin_versions.items()
    ):
        raise StateSchemaError("state.plugin_versions must be a string-to-string object")
    sequence = data.get("journal_sequence", 0)
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise StateSchemaError("state.journal_sequence must be a non-negative integer")
    runner = data.get("runner", "http")
    if not isinstance(runner, str) or not runner:
        raise StateSchemaError("state.runner must be a non-empty string")

    for index, result in enumerate(data["results"]):
        if not isinstance(result, dict):
            raise StateSchemaError(f"state.results[{index}] must be an object")
        identity = result.get("state_key", result.get("model"))
        if not isinstance(identity, str) or not identity:
            raise StateSchemaError(
                f"state.results[{index}] must contain a non-empty model or state_key"
            )

    for model_name, info in data["model_info"].items():
        if not isinstance(model_name, str) or not model_name:
            raise StateSchemaError("state.model_info keys must be non-empty strings")
        if not isinstance(info, dict):
            raise StateSchemaError(f"state.model_info[{model_name!r}] must be an object")
        status = info.get("status")
        if status is not None and status not in {
            "pending", "queued", "running", "completed", "failed", "error",
        } and not (
            isinstance(status, str) and status.startswith("running_")
        ):
            raise StateSchemaError(
                f"state.model_info[{model_name!r}].status has unsupported value {status!r}"
            )


def validate_journal_event(event: Any) -> None:
    """Validate one structured journal event before replay."""
    if not isinstance(event, dict):
        raise StateSchemaError("journal event must be an object")
    event_type = event.get("type")
    if event_type not in {"result", "judge"}:
        raise StateSchemaError(f"unsupported journal event type: {event_type!r}")
    sequence = event.get("seq")
    if sequence is not None and (
        isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1
    ):
        raise StateSchemaError("journal event seq must be a positive integer")
    if not isinstance(event.get("data"), dict):
        raise StateSchemaError("journal event data must be an object")
