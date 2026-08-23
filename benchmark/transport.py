"""Reusable transport normalization and logical-attempt execution.

This module owns one-attempt HTTP/OpenCode normalization and the bounded retry
engine. Scoring, judge parsing, and state persistence remain caller-owned.
"""
from __future__ import annotations

import contextlib
import hashlib
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .http import NonStreamResult, StreamResult, nonstream_request, stream_request
from .observer import TaskObserver
from .opencode import OPENCODE_BINARY, run_process
from .pi import PI_DEFAULT_NODE, PiProcessResult
from .pi import run_process as run_pi_process

# Default generation budget for benchmark tasks when no explicit max_tokens
# can be parsed. Deliberately separate from the judge default so benchmark
# tasks never fall back to the (smaller) judging budget.
BENCHMARK_DEFAULT_MAX_TOKENS = 16384


def _split_token_budget(total_budget, fallback: int) -> tuple[int, int, int]:
    """Return the reported, thinking, and answer budgets for a generation.

    Report 75% of the real budget so the model self-regulates and leaves
    headroom under the actual generation limit; reserve half of that for
    internal thinking and the rest for the final answer. Falls back to
    ``fallback`` when the budget cannot be parsed so each caller (judges vs
    benchmark tasks) gets its own appropriate default.
    """
    try:
        real_budget = max(1, int(total_budget))
    except (TypeError, ValueError):
        real_budget = fallback
    reported = max(1, (real_budget * 3) // 4)
    thinking_budget = max(1, reported // 2)
    return reported, thinking_budget, reported - thinking_budget


@dataclass(frozen=True)
class RequestIdentity:
    """Stable identity shared by transport logs, attempts, and diagnostics."""

    run_id: str = "unknown-run"
    revision_id: str | int = "unknown-revision"
    target: str = "unknown-target"
    plugin: str = "unknown-plugin"
    runner: str = "http"
    attempt: int = 1

    @property
    def request_id(self) -> str:
        """Return a compact, log-safe request identifier."""
        return ":".join(
            str(value).replace(":", "_")
            for value in (
                self.run_id, self.revision_id, self.target,
                self.plugin, self.runner, self.attempt,
            )
        )


@dataclass(frozen=True)
class TransportRequest:
    """All inputs needed to execute one transport attempt."""

    prompt: str
    max_tokens: int
    source_config: dict
    api_model: str
    source: str
    timeout: float
    temperature: float | None = 0.0
    reasoning: bool = False
    prompt_altered: str = "none"
    system_prompt: str | None = None
    drop_params: list | None = None
    request_params: dict | None = None
    session_seed: int = 0
    log_path: str | None = None
    log_label: str = ""
    attempt: int = 1
    pid: str = ""
    stop_event: Any = None
    observer: TaskObserver | None = None
    max_content_tokens: int | None = None
    max_thinking_tokens: int | None = None
    repetition_guard: int | bool | None = None
    transport: Literal["http", "opencode", "pi"] = "http"
    supports_streaming: bool = True
    opencode_config_path: str | None = None
    opencode_model: str | None = None
    opencode_agent: str | None = None
    opencode_binary: str | None = None
    opencode_output_dir: str | None = None
    opencode_no_output_grace: float | None = None
    opencode_target_key: str | None = None
    opencode_plugin_id: str | None = None
    pi_node: str | None = None
    pi_worker: str | None = None
    pi_config: dict | None = None
    pi_target_key: str | None = None
    pi_plugin_id: str | None = None
    identity: RequestIdentity | None = None


@dataclass(frozen=True)
class TransportResult:
    """Normalized response from one transport attempt."""

    text: str
    think_text: str
    error: str | None
    finish_reason: str | None
    response_time: float
    gen_time: float
    stream_ok: bool
    repeating: bool
    usage: dict[str, Any]
    response_nature: str
    empty_reason: str | None
    thinking_tokens: int
    prompt_sha256: str
    response_sha256: str
    schema_fallback_used: bool = False
    schema_fallback_error: str | None = None
    stream_fallback_used: bool = False
    stream_fallback_error: str | None = None
    runner_metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = "unknown-request"
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class RetryPolicy:
    """Controls which classifications may start a second logical attempt."""

    max_attempts: int = 2
    retry_on_transport_error: bool = True
    retry_on_token_limit: bool = True
    retry_on_repetition: bool = True
    retry_on_json_error: bool = False
    retry_on_timeout: bool = False


BENCHMARK_RETRY_POLICY = RetryPolicy(
    retry_on_transport_error=True,
    retry_on_token_limit=True,
    retry_on_repetition=True,
)

JUDGE_RETRY_POLICY = RetryPolicy(
    retry_on_transport_error=False,
    retry_on_token_limit=False,
    retry_on_repetition=False,
    retry_on_json_error=True,
    retry_on_timeout=False,
)


@dataclass
class TaskAttempt:
    """One logical attempt, including the prompt that was actually sent."""

    result: TransportResult
    attempt_number: int
    prompt_altered: str
    retry_reason: str | None
    request_prompt: str


@dataclass
class TaskExecution:
    """Complete bounded execution and its default selected attempt."""

    attempts: list[TaskAttempt]
    selected: TaskAttempt | None
    retry_reasons: list[str]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def retry_reason(self) -> str | None:
        return self.selected.retry_reason if self.selected is not None else None

    def select(self, attempt: TaskAttempt) -> None:
        """Allow a caller's evaluator to replace the transport default."""
        if attempt not in self.attempts:
            raise ValueError("cannot select an attempt outside this execution")
        self.selected = attempt


class StreamingTaskExecution:
    """Live content plus metadata for one logical attempt.

    ``next_attempt`` is populated before ``metadata_future`` resolves when the
    shared retry policy schedules another logical attempt. Transport-level
    retries remain inside the current future and do not create another node.
    """

    def __init__(self, stream: Iterator[str], metadata_future: Future[TaskAttempt],
                 stop_event: threading.Event) -> None:
        self.stream = stream
        self.metadata_future = metadata_future
        self.next_attempt: StreamingTaskExecution | None = None
        self._stop_event = stop_event

    def cancel(self) -> None:
        """Request cancellation of the active transport and any retry."""
        self._stop_event.set()
        if self.next_attempt is not None:
            self.next_attempt.cancel()


def classify_empty_reason(text: str, think_text: str = "", finish_reason: str | None = None,
                          error: str | None = None) -> str | None:
    """Classify an empty response without conflating it with transport failure."""
    if text and text.strip():
        return None
    if error:
        return "error"
    if think_text and finish_reason == "length":
        return "thinking-truncation"
    if think_text:
        return "thinking-only"
    if finish_reason == "length":
        return "max-tokens"
    return "empty"


def is_repeating(text: str, min_seq: int = 80, repeats: int = 3) -> bool:
    """Detect a repeated tail in generated content."""
    if len(text) < min_seq * repeats:
        return False
    tail = text[-min_seq:]
    return text.count(tail) >= repeats


def response_nature(*, text: str, error: str | None, finish_reason: str | None,
                    repeating: bool = False, cancelled: bool = False) -> str:
    """Classify the machine-observable end of one transport attempt."""
    lowered = str(error or "").lower()
    if cancelled or "cancelled" in lowered or "canceled" in lowered:
        return "cancelled"
    if any(marker in lowered for marker in ("timeout", "timed out", "readtimeout")):
        return "timeout"
    if repeating or "repetition" in lowered or "repeated" in lowered:
        return "repetition_abort"
    if finish_reason == "length" or "token limit" in lowered or "budget exceeded" in lowered:
        return "token_limit"
    if error:
        return "transport_error"
    if not text or not text.strip():
        return "empty"
    return "completed"


def _response_reasoning_tokens(response: Any) -> int:
    """Prefer provider reasoning usage, falling back to the char/4 estimate."""
    usage = getattr(response, "usage", {})
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("completion_tokens_details") or usage.get("completion_token_details")
    containers = (usage, details if isinstance(details, dict) else {})
    for container in containers:
        for key in ("reasoning_tokens", "thinking_tokens"):
            value = container.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    thinking = getattr(response, "think_text", "") or ""
    if not isinstance(thinking, str):
        return 0
    return int(len(thinking) / 4) if thinking else 0


def _is_schema_grammar_error(error: str | None) -> bool:
    lowered = str(error or "").lower()
    return any(marker in lowered for marker in (
        "failed to initialize samplers",
        "grammar sampler",
        "failed to parse grammar",
        "error initializing grammar",
    ))


def _json_object_fallback_params(request_params: dict | None) -> dict | None:
    if not isinstance(request_params, dict):
        return None
    response_format = request_params.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return None
    fallback = dict(request_params)
    fallback["response_format"] = {"type": "json_object"}
    return fallback


def _retry_prompt_alteration(nature: str, thinking_tokens: int | None,
                             max_tokens: int) -> tuple[str, str]:
    """Return the benchmark retry label and prompt guidance for a response."""
    if nature == "repetition_abort":
        return (
            "avoid_repetition",
            (
                "\n\nRETRY GUIDANCE: Do not repeat phrases, paragraphs, code blocks, "
                "or reasoning loops from the previous attempt. Produce new, "
                "task-relevant content and finish the requested answer."
            ),
        )
    if nature != "token_limit":
        return "none", ""
    reported, thinking_budget, answer_budget = _split_token_budget(
        max_tokens, BENCHMARK_DEFAULT_MAX_TOKENS,
    )
    if thinking_tokens is not None and thinking_tokens >= 0.8 * max_tokens:
        return (
            "thinking_50_percent",
            (
                "\n\nRETRY GUIDANCE: On this retry you MUST keep internal thinking or "
                f"reasoning below {thinking_budget} tokens and the entire response "
                f"below {reported} total tokens ({answer_budget} tokens are reserved "
                "for the final answer). Exceeding either limit is considered a failure."
            ),
        )
    if thinking_tokens is not None and thinking_tokens > 0.5 * max_tokens:
        return (
            "thinking_30_percent",
            (
                "\n\nRETRY GUIDANCE: On this retry you MUST keep internal thinking or "
                f"reasoning below {max(1, int(reported * 0.3))} tokens and the entire "
                f"response below {reported} total tokens. Exceeding either limit is "
                "considered a failure."
            ),
        )
    return (
        "response_under_budget",
        (
            "\n\nRETRY GUIDANCE: Complete the required answer while keeping the total "
            f"response below {reported} tokens. Exceeding the limit is considered "
            "a failure."
        ),
    )


def _thinking_consumed_budget(diagnostics: dict[str, Any] | None) -> bool:
    """Return whether reasoning consumed at least 80% of a length-limited budget."""
    if not isinstance(diagnostics, dict):
        return False
    max_tokens = diagnostics.get("request_max_tokens")
    reasoning_tokens = diagnostics.get("response_reasoning_tokens")
    return (
        diagnostics.get("response_finish_reason") == "length"
        and isinstance(max_tokens, (int, float))
        and not isinstance(max_tokens, bool)
        and isinstance(reasoning_tokens, (int, float))
        and not isinstance(reasoning_tokens, bool)
        and reasoning_tokens >= 0.8 * max_tokens
    )


def _is_streaming_rejection(error: str | None) -> bool:
    """Recognize explicit provider refusal of streaming, not generic failures."""
    lowered = str(error or "").lower()
    return any(marker in lowered for marker in (
        "streaming is not supported",
        "streaming not supported",
        "does not support streaming",
        "unsupported streaming",
        "stream is unsupported",
    ))


def _observer(request: TransportRequest) -> TaskObserver:
    return request.observer or TaskObserver(pid=request.pid)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(request: TransportRequest, *, text: str, think_text: str,
               error: str | None, finish_reason: str | None,
               response_time: float, gen_time: float, stream_ok: bool,
               repeating: bool, usage: dict[str, Any] | None = None,
               schema_fallback_used: bool = False,
               schema_fallback_error: str | None = None,
               stream_fallback_used: bool = False,
               stream_fallback_error: str | None = None,
               runner_metadata: dict[str, Any] | None = None) -> TransportResult:
    text = text or ""
    think_text = think_text or ""
    usage = usage if isinstance(usage, dict) else {}
    cancelled = bool(request.stop_event and request.stop_event.is_set())
    nature = response_nature(
        text=text,
        error=error,
        finish_reason=finish_reason,
        repeating=repeating,
        cancelled=cancelled,
    )
    return TransportResult(
        text=text,
        think_text=think_text,
        error=error,
        finish_reason=finish_reason,
        response_time=max(0.0, float(response_time)),
        gen_time=max(0.0, float(gen_time)),
        stream_ok=stream_ok,
        repeating=repeating,
        usage=usage,
        response_nature=nature,
        empty_reason=classify_empty_reason(text, think_text, finish_reason, error),
        thinking_tokens=_response_reasoning_tokens(
            type("Response", (), {"usage": usage, "think_text": think_text})()
        ),
        prompt_sha256=_hash_text(request.prompt),
        response_sha256=_hash_text(text),
        schema_fallback_used=schema_fallback_used,
        schema_fallback_error=schema_fallback_error,
        stream_fallback_used=stream_fallback_used,
        stream_fallback_error=stream_fallback_error,
        runner_metadata=runner_metadata or {},
        request_id=(request.identity or RequestIdentity(
            target=request.api_model,
            plugin=request.pid or "unknown-plugin",
            runner=request.transport,
            attempt=request.attempt,
        )).request_id,
        timeout_seconds=request.timeout,
    )


def _normalize_stream(request: TransportRequest, response: StreamResult, started: float,
                      *, stream_fallback_used: bool = False,
                      stream_fallback_error: str | None = None) -> TransportResult:
    first = getattr(response, "first_tok", None)
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        first = None
    ended = getattr(response, "stream_end", None)
    if isinstance(ended, bool) or not isinstance(ended, (int, float)) or not ended:
        ended = time.time()
    generation = ended - first if first is not None else ended - started
    text = getattr(response, "text", "") or ""
    if not isinstance(text, str):
        text = ""
    think_text = getattr(response, "think_text", "") or ""
    if not isinstance(think_text, str):
        think_text = ""
    error = getattr(response, "error", None)
    if not isinstance(error, str):
        error = None
    finish_reason = getattr(response, "finish_reason", None)
    if not isinstance(finish_reason, str):
        finish_reason = None
    usage = getattr(response, "usage", {}) or {}
    repeating = is_repeating(text) or "repetition" in str(error or "").lower()
    return _normalize(
        request,
        text=text,
        think_text=think_text,
        error=error,
        finish_reason=finish_reason,
        response_time=ended - started,
        gen_time=generation,
        stream_ok=error is None and first is not None,
        repeating=repeating,
        usage=usage,
        stream_fallback_used=stream_fallback_used,
        stream_fallback_error=stream_fallback_error,
    )


def _normalize_nonstream(request: TransportRequest, response: NonStreamResult,
                         started: float, *, schema_fallback_used: bool = False,
                         schema_fallback_error: str | None = None,
                         stream_fallback_used: bool = False,
                         stream_fallback_error: str | None = None) -> TransportResult:
    text = response.text or ""
    return _normalize(
        request,
        text=text,
        think_text=response.think_text or "",
        error=response.error,
        finish_reason=response.finish_reason,
        response_time=response.gen_time,
        gen_time=response.gen_time,
        stream_ok=False,
        repeating=is_repeating(text),
        usage=getattr(response, "usage", {}) or {},
        schema_fallback_used=schema_fallback_used,
        schema_fallback_error=schema_fallback_error,
        stream_fallback_used=stream_fallback_used,
        stream_fallback_error=stream_fallback_error,
    )


def _execute_http_nonstream(request: TransportRequest, started: float, *,
                            nonstream_request_fn=None,
                            stream_fallback_used: bool = False,
                            stream_fallback_error: str | None = None) -> TransportResult:
    observer = _observer(request)
    nonstream_request_fn = nonstream_request_fn or nonstream_request
    params = request.request_params
    response = nonstream_request_fn(
        request.source_config, request.timeout, request.api_model, request.source,
        request.prompt, request.max_tokens,
        log_path=request.log_path,
        log_label=request.log_label,
        session_seed=request.session_seed,
        temperature=request.temperature,
        drop_params=request.drop_params,
        stop_event=request.stop_event,
        system_prompt=request.system_prompt,
        request_params=params,
        observer=observer,
        pid=request.pid,
        max_content_tokens=request.max_content_tokens,
        max_thinking_tokens=request.max_thinking_tokens,
        repetition_guard=request.repetition_guard,
    )
    schema_fallback_used = False
    schema_fallback_error = None
    fallback_params = _json_object_fallback_params(params) if _is_schema_grammar_error(response.error) else None
    if fallback_params is not None:
        schema_fallback_used = True
        schema_fallback_error = response.error
        if isinstance(params, dict):
            params.clear()
            params.update(fallback_params)
        response = nonstream_request_fn(
            request.source_config, request.timeout, request.api_model, request.source,
            request.prompt, request.max_tokens,
            log_path=request.log_path,
            log_label=f"{request.log_label} (JSON-object schema fallback)",
            session_seed=request.session_seed,
            temperature=request.temperature,
            drop_params=request.drop_params,
            stop_event=request.stop_event,
            system_prompt=request.system_prompt,
            request_params=request.request_params,
            observer=observer,
            pid=request.pid,
            max_content_tokens=request.max_content_tokens,
            max_thinking_tokens=request.max_thinking_tokens,
            repetition_guard=request.repetition_guard,
        )
    return _normalize_nonstream(
        request, response, started,
        schema_fallback_used=schema_fallback_used,
        schema_fallback_error=schema_fallback_error,
        stream_fallback_used=stream_fallback_used,
        stream_fallback_error=stream_fallback_error,
    )


def _execute_http(request: TransportRequest, *, stream_request_fn=None,
                  nonstream_request_fn=None) -> TransportResult:
    started = time.time()
    stream_request_fn = stream_request_fn or stream_request
    nonstream_request_fn = nonstream_request_fn or nonstream_request
    observer = _observer(request)
    if not request.supports_streaming:
        return _execute_http_nonstream(
            request, started, nonstream_request_fn=nonstream_request_fn,
        )
    response = stream_request_fn(
        request.source_config, request.timeout, request.api_model, request.source,
        request.prompt, request.max_tokens,
        log_path=request.log_path,
        log_label=request.log_label,
        session_seed=request.session_seed,
        temperature=request.temperature,
        drop_params=request.drop_params,
        stop_event=request.stop_event,
        system_prompt=request.system_prompt,
        request_params=request.request_params,
        observer=observer,
        pid=request.pid,
        max_content_tokens=request.max_content_tokens,
        max_thinking_tokens=request.max_thinking_tokens,
        repetition_guard=request.repetition_guard,
    )
    result = _normalize_stream(request, response, started)
    if (
        result.error
        and not result.text
        and not result.think_text
        and _is_streaming_rejection(result.error)
    ):
        return _execute_http_nonstream(
            request,
            started,
            nonstream_request_fn=nonstream_request_fn,
            stream_fallback_used=True,
            stream_fallback_error=result.error,
        )
    return result


def _execute_opencode(request: TransportRequest, *, run_process_fn=None) -> TransportResult:
    run_process_fn = run_process_fn or run_process
    if not request.opencode_config_path or not request.opencode_model:
        return _normalize(
            request,
            text="",
            think_text="",
            error="OpenCode runner is missing generated config or model mapping",
            finish_reason=None,
            response_time=0.0,
            gen_time=0.0,
            stream_ok=False,
            repeating=False,
        )
    response = run_process_fn(
        request.prompt,
        config_path=request.opencode_config_path,
        model=request.opencode_model,
        timeout=request.timeout,
        binary=request.opencode_binary or OPENCODE_BINARY,
        agent=request.opencode_agent,
        output_dir=request.opencode_output_dir,
        target_key=request.opencode_target_key or request.source,
        plugin_id=request.opencode_plugin_id or request.pid or "plugin",
        stop_event=request.stop_event,
        no_output_grace=request.opencode_no_output_grace or 0,
    )
    return _normalize(
        request,
        text=response.text or "",
        think_text=response.think_text or "",
        error=response.error,
        finish_reason=None,
        response_time=response.elapsed,
        gen_time=response.elapsed,
        stream_ok=False,
        repeating=is_repeating(response.text or ""),
        usage={},
    )


def _retry_plan(
    result: TransportResult,
    *,
    retry_policy: RetryPolicy,
    max_tokens: int,
    prompt_alterer: Callable[[str, int | None, int], tuple[str, str]] | None,
    json_error_prompt_alterer: Callable[[TransportResult], str | None] | None,
) -> tuple[str, str, str | None]:
    """Return ``(prompt label, instruction, reason)`` for the next attempt."""
    nature = result.response_nature
    next_alteration = "none"
    instruction = ""
    reason: str | None = None
    if (
        (nature == "transport_error" and retry_policy.retry_on_transport_error)
        or (nature == "timeout" and retry_policy.retry_on_timeout)
    ):
        reason = nature
    elif (
        (nature == "token_limit" and retry_policy.retry_on_token_limit)
        or (nature == "repetition_abort" and retry_policy.retry_on_repetition)
    ):
        alter = prompt_alterer or _retry_prompt_alteration
        next_alteration, instruction = alter(
            nature, result.thinking_tokens, max_tokens,
        )
        reason = nature if next_alteration != "none" else None
    elif (
        retry_policy.retry_on_json_error
        and json_error_prompt_alterer is not None
        and not result.error
    ):
        json_instruction = json_error_prompt_alterer(result)
        if json_instruction:
            next_alteration = "json_error"
            instruction = json_instruction
            reason = "json_error"
    return next_alteration, instruction, reason


def _execute_pi(request: TransportRequest) -> TransportResult:
    """Execute one isolated Pi SDK worker attempt."""
    response: PiProcessResult = run_pi_process(
        request.prompt,
        source_config=request.source_config,
        source=request.source,
        api_model=request.api_model,
        max_tokens=request.max_tokens,
        timeout=request.timeout,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        reasoning=request.reasoning,
        prompt_altered=request.prompt_altered,
        attempt=request.attempt,
        pi_config=request.pi_config,
        node=request.pi_node or PI_DEFAULT_NODE,
        worker=request.pi_worker,
        output_dir=request.opencode_output_dir,
        target_key=request.pi_target_key or request.source,
        plugin_id=request.pi_plugin_id or request.pid or "plugin",
        stop_event=request.stop_event,
        observer=request.observer,
    )
    return _normalize(
        request,
        text=response.text,
        think_text=response.think_text,
        error=response.error,
        finish_reason=response.finish_reason,
        response_time=response.elapsed,
        gen_time=response.elapsed,
        stream_ok=response.error is None,
        repeating=is_repeating(response.text),
        usage=response.usage,
        runner_metadata={
            "runner": "pi",
            "adapter_version": response.adapter_version,
            "worker_version": response.worker_version,
            "sdk_version": response.sdk_version,
            "provider": response.provider,
            "requested_tools": list(response.requested_tools),
            "tools": list(response.tools),
            "permissions": response.permissions,
            "tool_called": response.tool_called,
            "truncated": response.truncated,
        },
    )


def execute_task(

    request: TransportRequest,
    *,
    retry_policy: RetryPolicy,
    base_prompt: str,
    prompt_alterer: Callable[[str, int | None, int], tuple[str, str]] | None = None,
    json_error_prompt_alterer: Callable[[TransportResult], str | None] | None = None,
    attempt_callback: Callable[[int], None] | None = None,
    stream_request_fn=None,
    nonstream_request_fn=None,
    run_process_fn=None,
) -> TaskExecution:
    """Execute a bounded sequence of logical attempts.

    Transport-level retries (429 backoff, for example) remain inside the
    request functions and do not create another ``TaskAttempt``. This engine
    only decides whether a completed transport leg warrants the caller's one
    policy retry. Scoring and JSON parsing stay with the caller.
    """
    max_attempts = max(1, int(retry_policy.max_attempts))
    attempts: list[TaskAttempt] = []
    retry_reasons: list[str] = []
    request_prompt = base_prompt
    prompt_altered = "none"
    retry_reason: str | None = None

    for attempt_number in range(1, max_attempts + 1):
        if request.stop_event is not None and request.stop_event.is_set():
            break
        if attempt_callback is not None:
            # Live observers are advisory and must not affect execution.
            with contextlib.suppress(Exception):
                attempt_callback(attempt_number)
        log_label = request.log_label
        if "{attempt}" in log_label:
            log_label = log_label.replace("{attempt}", str(attempt_number))
        else:
            log_label = f"{log_label} (attempt {attempt_number})"
        attempt_request = replace(
            request,
            prompt=request_prompt,
            log_label=log_label,
            attempt=attempt_number,
            prompt_altered=prompt_altered,
        )
        result = execute_transport(
            attempt_request,
            stream_request_fn=stream_request_fn,
            nonstream_request_fn=nonstream_request_fn,
            run_process_fn=run_process_fn,
        )
        attempt = TaskAttempt(
            result=result,
            attempt_number=attempt_number,
            prompt_altered=prompt_altered,
            retry_reason=retry_reason,
            request_prompt=request_prompt,
        )
        attempts.append(attempt)
        if attempt_number >= max_attempts:
            break

        next_alteration, instruction, retry_reason = _retry_plan(
            result,
            retry_policy=retry_policy,
            max_tokens=request.max_tokens,
            prompt_alterer=prompt_alterer,
            json_error_prompt_alterer=json_error_prompt_alterer,
        )

        if retry_reason is None:
            break
        retry_reasons.append(retry_reason)
        prompt_altered = next_alteration
        request_prompt = base_prompt + instruction

    selected = attempts[-1] if attempts else None
    return TaskExecution(
        attempts=attempts,
        selected=selected,
        retry_reasons=retry_reasons,
    )


def execute_task_streaming(
    request: TransportRequest,
    *,
    retry_policy: RetryPolicy,
    base_prompt: str,
    prompt_alterer: Callable[[str, int | None, int], tuple[str, str]] | None = None,
    json_error_prompt_alterer: Callable[[TransportResult], str | None] | None = None,
    attempt_callback: Callable[[int], None] | None = None,
    stream_request_fn=None,
    nonstream_request_fn=None,
    run_process_fn=None,
    _attempt_number: int = 1,
    _prompt_altered: str = "none",
    _retry_reason: str | None = None,
) -> StreamingTaskExecution:
    """Start one live attempt and return before transport completion.

    Content deltas are yielded from ``stream`` as the transport observer sees
    them. The future resolves to the completed ``TaskAttempt``. If the policy
    schedules another logical attempt, ``next_attempt`` points to its live
    execution after the first future resolves.
    """
    stop_event = request.stop_event or threading.Event()
    request = replace(request, stop_event=stop_event)
    items: queue.Queue[str | object] = queue.Queue()
    sentinel = object()
    metadata: Future[TaskAttempt] = Future()

    completed = False

    def stream() -> Iterator[str]:
        nonlocal completed
        try:
            while True:
                item = items.get()
                if item is sentinel:
                    completed = True
                    return
                yield item  # type: ignore[misc]
        finally:
            if not completed:
                stop_event.set()

    execution = StreamingTaskExecution(stream(), metadata, stop_event)

    def run() -> None:
        try:
            if attempt_callback is not None:
                with contextlib.suppress(Exception):
                    attempt_callback(_attempt_number)
            log_label = request.log_label
            if "{attempt}" in log_label:
                log_label = log_label.replace("{attempt}", str(_attempt_number))
            else:
                log_label = f"{log_label} (attempt {_attempt_number})"
            base_observer = request.observer or TaskObserver.noop()

            def on_content(delta: str) -> None:
                base_observer.chunk(delta)
                items.put(delta)

            observer = TaskObserver(
                model_name=base_observer.model_name,
                pid=base_observer.pid,
                on_chunk=on_content,
                on_think_chunk=base_observer.think_chunk,
                on_retry=base_observer.retry,
            )
            result = execute_transport(
                replace(
                    request,
                    log_label=log_label,
                    observer=observer,
                    attempt=_attempt_number,
                    prompt_altered=_prompt_altered,
                ),
                stream_request_fn=stream_request_fn,
                nonstream_request_fn=nonstream_request_fn,
                run_process_fn=run_process_fn,
            )
            attempt = TaskAttempt(
                result=result,
                attempt_number=_attempt_number,
                prompt_altered=_prompt_altered,
                retry_reason=_retry_reason,
                request_prompt=request.prompt,
            )
            next_alteration, instruction, reason = _retry_plan(
                result,
                retry_policy=retry_policy,
                max_tokens=request.max_tokens,
                prompt_alterer=prompt_alterer,
                json_error_prompt_alterer=json_error_prompt_alterer,
            )
            if (
                reason is not None
                and _attempt_number < max(1, int(retry_policy.max_attempts))
                and not stop_event.is_set()
            ):
                execution.next_attempt = execute_task_streaming(
                    replace(request, prompt=base_prompt + instruction),
                    retry_policy=retry_policy,
                    base_prompt=base_prompt,
                    prompt_alterer=prompt_alterer,
                    json_error_prompt_alterer=json_error_prompt_alterer,
                    attempt_callback=attempt_callback,
                    stream_request_fn=stream_request_fn,
                    nonstream_request_fn=nonstream_request_fn,
                    run_process_fn=run_process_fn,
                    _attempt_number=_attempt_number + 1,
                    _prompt_altered=next_alteration,
                    _retry_reason=reason,
                )
            metadata.set_result(attempt)
        except Exception as exc:  # noqa: BLE001 - surface worker failures via the future
            metadata.set_exception(exc)
        finally:
            items.put(sentinel)

    threading.Thread(target=run, name="transport-stream", daemon=True).start()
    return execution


def execute_transport(request: TransportRequest, *, stream_request_fn=None,
                      nonstream_request_fn=None,
                      run_process_fn=None) -> TransportResult:
    """Execute exactly one logical attempt through HTTP or OpenCode.

    The injectable callables are intentionally keyword-only: production uses
    the module defaults, while orchestration callers and tests can preserve a
    local transport seam without mutating this module globally.
    """
    if request.transport == "http":
        return _execute_http(
            request,
            stream_request_fn=stream_request_fn,
            nonstream_request_fn=nonstream_request_fn,
        )
    if request.transport == "opencode":
        return _execute_opencode(request, run_process_fn=run_process_fn)
    if request.transport == "pi":
        return _execute_pi(request)
    return _normalize(
        request,
        text="",
        think_text="",
        error=f"Unknown transport {request.transport!r}",
        finish_reason=None,
        response_time=0.0,
        gen_time=0.0,
        stream_ok=False,
        repeating=False,
    )
