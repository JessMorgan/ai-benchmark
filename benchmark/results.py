"""Result construction and artifact persistence helpers.

Execution and scoring stay in ``core``/``transport``. This module owns the
stable result dictionaries and the response/sidecar files that represent them.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .outputs import sanitize_filename
from .plugin import SCORE_SCHEMA
from .types import JSONObject


def _write_text(path: str, content: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError:
        pass


def _write_json(path: str, value: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, default=str)
    except OSError:
        pass


def save_task_result(
    execution: Any,
    *,
    state: Any,
    model_name: str,
    pid: str,
    plugin: Any,
    output_dir: str | None = None,
    save_responses: bool = False,
    judge_input_dir: str | None = None,
    judge_enqueue: Any = None,
    artifact_target: str | None = None,
    runner: str = "http",
    request_applied: bool = True,
    score: Any = None,
    rubric: list[dict[str, Any]] | None = None,
    diagnostics: JSONObject | None = None,
    score_error: str | None = None,
    score_traceback: str | None = None,
    selected_prompt: str = "",
    selected_text: str = "",
    selected_think: str = "",
    judge_prompt: str | None = None,
    response_time: float | None = None,
    output_tokens: int = 0,
    thinking_tokens: int = 0,
    total_tokens: int = 0,
    tps: float | None = None,
    seed: int = 0,
    max_tokens: int = 0,
    attempts: list[dict[str, Any]] | None = None,
    selected_attempt: int = 1,
    retry_reason: str | None = None,
    prompt_altered: str = "none",
    response_nature: str | None = None,
    finish_reason: str | None = None,
    truncated: bool = False,
    truncated_due_to_time: bool = False,
    failure_cause: str | None = None,
    repeating: bool = False,
    stream_ok: bool = False,
    empty_reason: str | None = None,
    disposition: str | None = None,
    schema_metadata: dict[str, Any] | None = None,
    selected_error: str | None = None,
    api_model: str | None = None,
    opencode_model: str | None = None,
    runner_metadata: dict[str, Any] | None = None,
    is_agent: bool = False,
    system_prompt: str | None = None,
    prepare_judge_sidecar_fn: Any = None,
    judge_sidecar_path_fn: Any = None,
) -> dict[str, Any]:
    """Build and persist one benchmark plugin result.

    ``execution`` is retained in the API so callers can pass the shared
    ``TaskExecution`` object; the already-scored attempt records are supplied
    explicitly because evaluation is intentionally caller-owned. ``state`` and
    ``judge_enqueue`` are accepted as extension points for future storage
    backends and preserve the phase-plan API.
    """
    del execution, state, judge_enqueue, request_applied
    rubric = rubric or []
    diagnostics = diagnostics or {}
    schema_metadata = schema_metadata or {}
    runner_metadata = runner_metadata or {}
    attempts = attempts or []
    target = artifact_target or model_name
    attempt_history = [
        {key: value for key, value in attempt.items() if key != "score_traceback"}
        for attempt in attempts
    ]
    retry_reasons = [
        attempt["retry_reason"] for attempt in attempts
        if attempt.get("retry_reason") is not None
    ]

    if judge_input_dir and prepare_judge_sidecar_fn and judge_sidecar_path_fn:
        sidecar = judge_sidecar_path_fn(judge_input_dir, target, runner, pid)
        try:
            prepare_judge_sidecar_fn(
                sidecar, plugin, (judge_prompt or selected_prompt), selected_text,
                target=target, state_key=model_name, runner=runner,
            )
        except OSError:
            pass

    meta = {
        "plugin": pid,
        "plugin_version": plugin.version,
        "target": target,
        "model": api_model,
        "runner": runner,
        "opencode_model": opencode_model,
        "runner_metadata": runner_metadata,
        "is_agent": is_agent,
        "system_prompt": system_prompt,
        "score": score,
        "score_schema": SCORE_SCHEMA,
        "rubric": rubric,
        "diagnostics": diagnostics,
        **schema_metadata,
        "response_time": response_time,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "tps": tps,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_tokens": max_tokens,
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "retried": len(attempts) > 1,
        "retry_reasons": retry_reasons,
        "selected_attempt": selected_attempt,
        "retry_reason": retry_reason,
        "prompt_altered": prompt_altered,
        "response_nature": response_nature,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "truncated_due_to_time": truncated_due_to_time,
        "failure_cause": failure_cause,
        "attempts": attempt_history,
        "think_text": selected_think,
    }
    if score_error:
        meta["error"] = score_error
        meta["traceback"] = score_traceback
    if selected_error:
        meta["error"] = selected_error
        meta["stream_error"] = selected_error
    if empty_reason is not None:
        meta["empty_reason"] = empty_reason
    if disposition is not None:
        meta["disposition"] = disposition

    if save_responses and output_dir:
        responses_dir = os.path.join(output_dir, "responses", sanitize_filename(target))
        os.makedirs(responses_dir, exist_ok=True)
        files = {
            f"{pid}.prompt.txt": selected_prompt,
            f"{pid}.content.txt": selected_text,
            f"{pid}.txt": (
                f"<thinking>\n{selected_think}\n</thinking>\n\n{selected_text}"
                if selected_think else selected_text
            ),
        }
        if selected_think:
            files[f"{pid}.think.txt"] = selected_think
        for filename, content in files.items():
            _write_text(os.path.join(responses_dir, filename), content)
        _write_json(os.path.join(responses_dir, f"{pid}.meta.json"), meta)

    return {
        **{f"{pid}_{key}": value for key, value in schema_metadata.items()},
        f"{pid}_score": score,
        f"{pid}_rubric": rubric,
        f"{pid}_diagnostics": diagnostics,
        f"{pid}_response_time": response_time,
        f"{pid}_output_tokens": output_tokens,
        f"{pid}_thinking_tokens": thinking_tokens,
        f"{pid}_total_tokens": total_tokens,
        f"{pid}_tps": tps,
        f"{pid}_truncated": truncated,
        f"{pid}_repeating": repeating,
        f"{pid}_stream_ok": stream_ok,
        f"{pid}_empty_reason": empty_reason,
        f"{pid}_disposition": disposition,
        f"{pid}_max_tokens": max_tokens,
        f"{pid}_attempt_count": len(attempts),
        f"{pid}_retry_count": max(0, len(attempts) - 1),
        f"{pid}_retried": len(attempts) > 1,
        f"{pid}_retry_reasons": retry_reasons,
        f"{pid}_selected_attempt": selected_attempt,
        f"{pid}_retry_reason": retry_reason,
        f"{pid}_prompt_altered": prompt_altered,
        f"{pid}_response_nature": response_nature,
        f"{pid}_finish_reason": finish_reason,
        f"{pid}_truncated_due_to_time": truncated_due_to_time,
        f"{pid}_failure_cause": failure_cause,
        f"{pid}_attempts": attempt_history,
        f"{pid}_runner_metadata": runner_metadata,
    }


def save_judge_result(
    result: Any,
    *,
    model_name: str,
    judge_prompt_version: str,
    judge_contract_id: str | None,
    parsed_judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable, versioned vote dictionary for one judge attempt."""
    parsed_judge = parsed_judge or {}

    def value(name: str, default: Any = None) -> Any:
        if name in parsed_judge:
            return parsed_judge[name]
        return getattr(result, name, default) if result is not None else default

    return {
        "model": model_name,
        "score": value("score"),
        "confidence": value("confidence"),
        "rationale": value("rationale"),
        "criteria": value("criteria", []) or [],
        "judge_prompt_version": judge_prompt_version,
        "judge_contract_id": judge_contract_id,
        "error": value("error"),
    }
