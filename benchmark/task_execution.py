"""Execution of one benchmark plugin cell."""
from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .http import nonstream_request, stream_request
from .judging import judge_sidecar_path, prepare_judge_sidecar
from .observer import TaskObserver
from .opencode import resolve_opencode_timeout, run_process
from .plugin import PluginTaskResult, normalize_score, serialize_rubric
from .request_models import (
    GenerationFields,
    HTTPRequest,
    OpenCodeRequest,
    PiRequest,
    TransportRequest,
)
from .response_classification import count_tokens
from .results import save_task_result
from .runtime_records import BenchmarkAttemptRecord
from .transport import BENCHMARK_RETRY_POLICY, _retry_prompt_alteration, execute_task
from .transport_options import HTTPTransportOptions, OpenCodeTransportOptions, PiTransportOptions

DEFAULT_MAX_THINKING_TOKENS = 32768
DEFAULT_MAX_CONTENT_TOKENS = 16384

@dataclass(frozen=True)
class TaskExecutionDependencies:
    stream_request_fn: Callable[..., Any] = stream_request
    nonstream_request_fn: Callable[..., Any] = nonstream_request
    run_process_fn: Callable[..., Any] = run_process
    resolve_stream_guards_fn: Callable[[dict[str, Any], str], tuple[int, int, bool]] | None = None


def resolve_stream_guards(source_config: dict[str, Any], source: str) -> tuple[int, int, bool]:
    cfg = source_config.get(source)
    if not isinstance(cfg, dict):
        return DEFAULT_MAX_CONTENT_TOKENS, DEFAULT_MAX_THINKING_TOKENS, True
    def tokens(key: str, default: int) -> int:
        try:
            value = int(cfg.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default
    return tokens("max_content_tokens", DEFAULT_MAX_CONTENT_TOKENS), tokens("max_thinking_tokens", DEFAULT_MAX_THINKING_TOKENS), bool(cfg.get("repetition_guard", True))


def _schema_request_metadata(plugin: Any, request_params: dict[str, Any] | None = None, *, response_schema_valid: bool | None = None, error: str | None = None, request_applied: bool = True, schema_fallback_used: bool = False, schema_fallback_error: str | None = None) -> dict[str, Any]:
    get_schema = getattr(plugin, "get_response_schema", None)
    declared_schema = get_schema() if callable(get_schema) else None
    response_format = request_params.get("response_format") if isinstance(request_params, dict) else None
    requested = bool(declared_schema or (isinstance(response_format, dict) and response_format.get("type") in {"json", "json_schema"}))
    if not requested:
        return {"schema_requested": False, "schema_request_status": "schema_not_requested", "response_schema_valid": None, "schema_enforcement_verified": None, "schema_fallback_used": False, "schema_fallback_error": None}
    status = "schema_not_applied_by_runner" if not request_applied else ("schema_fallback_json_object_failed" if schema_fallback_used and error else "schema_fallback_json_object_valid" if schema_fallback_used and response_schema_valid is True else "schema_fallback_json_object_invalid" if schema_fallback_used and response_schema_valid is False else "schema_accepted_valid" if response_schema_valid is True else "schema_accepted_invalid" if response_schema_valid is False else "schema_transport_error" if error else "schema_accepted_unknown")
    return {"schema_requested": True, "schema_request_status": status, "response_schema_valid": response_schema_valid, "schema_enforcement_verified": False, "schema_fallback_used": schema_fallback_used, "schema_fallback_error": schema_fallback_error}


def _run_plugin_task(target_name: str, api_model: str, source: str, plugin: Any, source_config: dict[str, Any], timeout: float,
                     max_tokens: int, session_seed: int, log_file: str | None, global_cfg: dict[str, Any], state: Any,
                     stop_event: threading.Event | None = None, save_responses: bool = False, output_dir: str | None = None,
                     judge_input_dir: str | None = None, judge_enqueue: Callable[[str, str, str, str], Any] | None = None,
                     system_prompt: str | None = None, is_agent: bool = False, runner: str = "http",
                     opencode_config_path: str | None = None, opencode_model: str | None = None,
                     opencode_agent: str | None = None, opencode_binary: str | None = None,
                     pi_node: str | None = None, pi_worker: str | None = None, pi_config: dict[str, Any] | None = None,
                     artifact_target_name: str | None = None,
                     config_target_name: str | None = None, debug_logs: bool = False,
                     dependencies: TaskExecutionDependencies | None = None) -> PluginTaskResult:
    """Run one benchmark cell with one scalar budget and at most one policy retry.

    Transport retries preserve the prompt. Token-limit and repetition retries
    append a machine-readable, purpose-specific instruction. Timeouts and
    cancellation are terminal and retain any partial response.
    """
    pid = plugin.id
    cfg = source_config.get(source)
    if runner == "http" and cfg is None:
        return PluginTaskResult(None, f"Unknown source '{source}' — not in SOURCE_CONFIG")
    if runner not in {"http", "opencode", "pi"}:
        return PluginTaskResult(None, f"Unknown runner {runner!r}")
    if stop_event and stop_event.is_set():
        return PluginTaskResult(None, "Cancelled")
    try:
        budget = int(max_tokens)
    except (TypeError, ValueError):
        return PluginTaskResult(None, f"Invalid max_tokens value: {max_tokens!r}")
    if budget <= 0:
        return PluginTaskResult(None, f"Invalid max_tokens value: {max_tokens!r}")

    guard_values = (dependencies.resolve_stream_guards_fn if dependencies and dependencies.resolve_stream_guards_fn else resolve_stream_guards)(source_config, source)
    max_content_tokens, max_thinking_tokens, repetition_guard = guard_values
    base_prompt = plugin.get_prompt()
    temperature = plugin.get_temperature(global_cfg or {})
    config_target_name = config_target_name or target_name
    artifact_target = artifact_target_name or config_target_name
    raw_model_cfg = ((global_cfg or {}).get("models", {}).get(config_target_name)
                     or (global_cfg or {}).get("agents", {}).get(config_target_name))
    drop_params = raw_model_cfg.get("drop_params", []) if isinstance(raw_model_cfg, dict) else []
    get_request_params = getattr(plugin, "get_request_params", None)
    request_params = get_request_params(global_cfg or {}) if callable(get_request_params) else {}
    if not isinstance(request_params, dict):
        request_params = {}
    schema_fallback_used = False
    schema_fallback_error = None
    schema_request_applied = runner == "http" and (
        not isinstance(cfg, dict) or cfg.get("api_protocol") not in {"1min", "chatplayground"}
    )

    def on_retry() -> None:
        state.start_plugin_run(target_name, pid)

    def on_chunk(delta: str) -> None:
        state.mark_first_chunk_seen(target_name, pid, ts=time.time())
        state.add_bytes_received(target_name, pid, len(delta))

    def on_think_chunk(delta: str) -> None:
        state.mark_first_chunk_seen(target_name, pid, ts=time.time())
        state.add_thinking_bytes_received(target_name, pid, len(delta))

    task_observer = TaskObserver(
        model_name=target_name,
        pid=pid,
        on_chunk=on_chunk,
        on_think_chunk=on_think_chunk,
        on_retry=on_retry,
    )

    common = GenerationFields(
        prompt=base_prompt,
        max_tokens=budget,
        source_config=source_config,
        api_model=api_model,
        source=source,
        timeout=timeout,
        temperature=temperature,
        reasoning=bool((pi_config or {}).get("reasoning", False)),
        system_prompt=system_prompt,
        drop_params=drop_params,
        session_seed=session_seed,
        log_path=log_file,
        log_label=f"{plugin.name} (attempt {{attempt}})",
        pid=pid,
        stop_event=stop_event,
        observer=task_observer,
        debug_logs=debug_logs,
    )
    http_options = HTTPTransportOptions(
        supports_streaming=plugin.supports_streaming,
        max_content_tokens=max_content_tokens,
        max_thinking_tokens=max_thinking_tokens,
        repetition_guard=repetition_guard,
        request_params=request_params,
    )
    if runner == "opencode":
        request: TransportRequest = OpenCodeRequest(
            common,
            OpenCodeTransportOptions(
                config_path=opencode_config_path,
                model=opencode_model,
                agent=opencode_agent,
                binary=opencode_binary,
                output_dir=output_dir,
                no_output_grace=resolve_opencode_timeout(source_config, source),
                target_key=artifact_target,
                plugin_id=pid,
            ),
        )
    elif runner == "pi":
        request = PiRequest(
            common,
            PiTransportOptions(
                node=pi_node,
                worker=pi_worker,
                config=pi_config,
                target_key=artifact_target,
                plugin_id=pid,
            ),
        )
    else:
        request = HTTPRequest(common, http_options)

    def on_attempt(attempt_number: int) -> None:
        state.set_plugin_attempt(target_name, pid, attempt_number)

    def evaluate(text: str) -> tuple[int | str, list[dict[str, Any]], Any, str | None, str | None]:
        try:
            value = plugin.evaluate(text)
            return normalize_score(value.score, plugin.max_score), serialize_rubric(value.rubric), value.diagnostics or {}, None, None
        except Exception as exc:  # noqa: BLE001 - retain evaluator failures as metadata
            return "fail", [], {}, f"plugin.evaluate raised {type(exc).__name__}: {exc}", traceback.format_exc()

    deps = dependencies or TaskExecutionDependencies()
    execution = execute_task(
        request,
        retry_policy=BENCHMARK_RETRY_POLICY,
        base_prompt=base_prompt,
        prompt_alterer=_retry_prompt_alteration,
        attempt_callback=on_attempt,
        stream_request_fn=deps.stream_request_fn,
        nonstream_request_fn=deps.nonstream_request_fn,
        run_process_fn=deps.run_process_fn,
    )
    if not execution.attempts:
        return PluginTaskResult(None, "No benchmark attempt was executed")

    attempts: list[dict[str, Any]] = []
    bodies = {}
    for index, task_attempt in enumerate(execution.attempts):
        raw = task_attempt.result
        text = raw.text
        think_text = raw.think_text
        schema_fallback_used = schema_fallback_used or raw.schema_fallback_used
        if raw.schema_fallback_error:
            schema_fallback_error = raw.schema_fallback_error
        nature = raw.response_nature
        thinking_tokens = raw.thinking_tokens
        if raw.error and not text.strip():
            score_result: int | str = "fail"
            rubric: list[dict[str, Any]] = []
            diagnostics: Any = {}
            score_error_result: str | None = None
            score_traceback: str | None = None
        else:
            eval_result = evaluate(text)
            score_result = eval_result[0]
            rubric = eval_result[1]
            diagnostics = eval_result[2]
            score_error_result = eval_result[3]
            score_traceback = eval_result[4]
        if score_error_result is not None:
            nature = "plugin_error"
        usable = (
            bool(text.strip())
            and nature not in {"timeout", "cancelled", "transport_error"}
            and score_error_result is None
        )
        attempt_failure_cause = raw.error or score_error_result
        if attempt_failure_cause is None and nature == "token_limit":
            attempt_failure_cause = (
                f"finish_reason:{raw.finish_reason}"
                if raw.finish_reason else "token_limit"
            )
        elif attempt_failure_cause is None and nature not in {"completed", "empty"}:
            attempt_failure_cause = nature
        attempt_record = {
            "attempt": task_attempt.attempt_number,
            "max_tokens": budget,
            "prompt_altered": task_attempt.prompt_altered,
            "retry_reason": task_attempt.retry_reason,
            "response_nature": nature,
            "finish_reason": raw.finish_reason,
            "error": raw.error,
            "failure_cause": attempt_failure_cause,
            "response_time": raw.response_time,
            "gen_time": raw.gen_time,
            "output_tokens": int(count_tokens(text)),
            "thinking_tokens": thinking_tokens,
            "total_tokens": int(count_tokens(text)) + thinking_tokens,
            "stream_ok": raw.stream_ok,
            "truncated": raw.finish_reason == "length",
            "truncated_due_to_time": nature == "timeout",
            "repeating": raw.repeating,
            "empty_reason": raw.empty_reason,
            "score": score_result,
            "rubric": rubric,
            "diagnostics": diagnostics,
            "score_error": score_error_result,
            "score_traceback": score_traceback,
            "usable": usable,
            "selected": False,
            "prompt_sha256": raw.prompt_sha256,
            "response_sha256": raw.response_sha256,
            "runner_metadata": raw.runner_metadata,
        }
        if index + 1 < len(execution.attempts):
            attempt_record["retry_scheduled"] = True
        attempts.append(attempt_record)
        bodies[task_attempt.attempt_number] = (task_attempt.request_prompt, text, think_text)

    usable_attempts = [attempt for attempt in attempts if attempt["usable"]]
    pool = usable_attempts or attempts
    selected = max(
        pool,
        key=lambda attempt: (
            attempt["score"]
            if isinstance(attempt["score"], (int, float))
            and not isinstance(attempt["score"], bool) else -1,
            -attempt["attempt"],
        ),
    )
    selected["selected"] = True
    selected_task_attempt = next(
        attempt for attempt in execution.attempts
        if attempt.attempt_number == selected["attempt"]
    )
    execution.select(selected_task_attempt)

    # Persist each immutable attempt through the storage facade. JSON adapts
    # this to a no-op; SQLite records one benchmark_attempts row per transport
    # attempt plus the revision-local selection. Recording is fire-and-forget
    # so the transport worker is never blocked on the background writer.
    cell_id = state.run_store.get_cell_id(target_name, runner, pid)
    if cell_id is not None:
        for attempt in attempts:
            attempt_number = attempt["attempt"]
            prompt, content, thinking = bodies[attempt_number]
            attempt_gen_time = attempt.get("gen_time") or 0
            attempt_tps = (
                round(attempt["output_tokens"] / attempt_gen_time, 2)
                if attempt_gen_time > 0 else None
            )
            status = (
                "completed"
                if attempt.get("usable") and not attempt.get("error") else "failed"
            )
            state.run_store.record_benchmark_attempt(
                cell_id,
                BenchmarkAttemptRecord(
                    attempt_number=attempt_number,
                    prompt=prompt,
                    content=content,
                    thinking=thinking,
                    max_tokens=attempt.get("max_tokens"),
                    output_tokens=attempt.get("output_tokens"),
                    thinking_tokens=attempt.get("thinking_tokens"),
                    total_tokens=attempt.get("total_tokens"),
                    tps=attempt_tps,
                    response_time=attempt.get("response_time"),
                    gen_time=attempt.get("gen_time"),
                    finish_reason=attempt.get("finish_reason"),
                    response_nature=attempt.get("response_nature"),
                    retry_reason=attempt.get("retry_reason"),
                    prompt_altered=attempt.get("prompt_altered"),
                    truncated=attempt.get("truncated"),
                    truncated_due_to_time=attempt.get("truncated_due_to_time"),
                    failure_cause=attempt.get("failure_cause"),
                    stream_ok=attempt.get("stream_ok"),
                    repeating=attempt.get("repeating"),
                    empty_reason=attempt.get("empty_reason"),
                    error=attempt.get("error"),
                    score=attempt.get("score"),
                    rubric=attempt.get("rubric", []),
                    diagnostics=attempt.get("diagnostics", {}),
                    status=status,
                ),
                selected=bool(attempt.get("selected")),
            )

    selected_prompt, selected_text, selected_think = bodies[selected["attempt"]]
    selected_nature = selected["response_nature"]
    selected_error = selected["error"]
    score = selected["score"]
    rubric = selected["rubric"]
    diagnostics = selected["diagnostics"]
    score_error = selected["score_error"]
    response_time = selected["response_time"]
    gen_time = selected["gen_time"]
    output_tokens = selected["output_tokens"]
    thinking_tokens = selected["thinking_tokens"]
    total_tokens = selected["total_tokens"]
    tps = round(output_tokens / gen_time, 2) if gen_time > 0 else None
    empty_reason = selected["empty_reason"]
    stream_ok = selected["stream_ok"]
    truncated = selected["truncated"]
    repeating = selected["repeating"]
    selected_alteration = selected["prompt_altered"]
    selected_retry_reason = selected["retry_reason"]
    runner_metadata = selected.get("runner_metadata", {})
    schema_metadata = _schema_request_metadata(
        plugin, request_params,
        response_schema_valid=diagnostics.get("response_schema_valid")
        if isinstance(diagnostics, dict) else None,
        request_applied=schema_request_applied,
        schema_fallback_used=schema_fallback_used,
        schema_fallback_error=schema_fallback_error,
        error=selected_error or score_error,
    )

    # Build the terminal disposition: why the selected attempt ended.
    # Token-limit and timeout are legitimate endpoint categories — the cell
    # is complete, graded with whatever content was available, and never
    # re-run.  Chunk deficit (no content at all) is noted for easy
    # post-run purge but still treated as complete.
    disposition = selected_nature
    if selected_nature == "token_limit" and not selected_text.strip():
        disposition = "token_exhausted_empty"
    elif selected_nature == "timeout" and not selected_text.strip():
        disposition = "timeout_empty"

    task_error: str | None = score_error
    if selected_nature in {"token_limit", "timeout"}:
        # Grade whatever content arrived and treat the cell as complete.
        # Convert unresolved "fail" scores to a numeric 0 so the resume
        # gate sees a reusable result instead of re-running every time.
        task_error = None
        if score == "fail":
            score = 0
            rubric = []
            diagnostics = {}
    elif selected_nature in {"transport_error", "cancelled"} and not selected_text.strip():
        task_error = selected_error or score_error or selected_nature
    result = save_task_result(
        execution,
        state=state,
        model_name=target_name,
        pid=pid,
        plugin=plugin,
        output_dir=output_dir,
        save_responses=save_responses,
        judge_input_dir=judge_input_dir,
        judge_enqueue=judge_enqueue,
        artifact_target=artifact_target,
        runner=runner,
        request_applied=schema_request_applied,
        score=score,
        rubric=rubric,
        diagnostics=diagnostics,
        score_error=score_error,
        score_traceback=selected.get("score_traceback"),
        selected_prompt=selected_prompt,
        selected_text=selected_text,
        selected_think=selected_think,
        response_time=response_time,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
        tps=tps,
        seed=session_seed,
        max_tokens=budget,
        attempts=attempts,
        selected_attempt=selected["attempt"],
        retry_reason=selected_retry_reason,
        prompt_altered=selected_alteration,
        response_nature=selected_nature,
        finish_reason=selected["finish_reason"],
        truncated=truncated,
        truncated_due_to_time=selected["truncated_due_to_time"],
        failure_cause=selected["failure_cause"],
        repeating=repeating,
        stream_ok=stream_ok,
        empty_reason=empty_reason,
        disposition=disposition,
        schema_metadata=schema_metadata,
        selected_error=selected_error,
        api_model=api_model,
        opencode_model=opencode_model,
        runner_metadata=runner_metadata,
        is_agent=is_agent,
        system_prompt=system_prompt,
        prepare_judge_sidecar_fn=prepare_judge_sidecar,
        judge_sidecar_path_fn=judge_sidecar_path,
    )
    return PluginTaskResult(result, task_error)

