"""Reusable transport normalization and logical-attempt execution.

This module owns one-attempt HTTP/OpenCode normalization and the bounded retry
engine. Scoring, judge parsing, and state persistence remain caller-owned.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from .http import NonStreamResult, StreamResult, nonstream_request, stream_request
from .observer import TaskObserver
from .opencode import OPENCODE_BINARY, run_process


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
    system_prompt: str | None = None
    drop_params: list | None = None
    request_params: dict | None = None
    session_seed: int = 0
    log_path: str | None = None
    log_label: str = ""
    pid: str = ""
    stop_event: Any = None
    observer: TaskObserver | None = None
    max_content_tokens: int | None = None
    max_thinking_tokens: int | None = None
    repetition_guard: int | bool | None = None
    transport: Literal["http", "opencode"] = "http"
    supports_streaming: bool = True
    opencode_config_path: str | None = None
    opencode_model: str | None = None
    opencode_agent: str | None = None
    opencode_binary: str | None = None
    opencode_output_dir: str | None = None
    opencode_no_output_grace: float | None = None
    opencode_target_key: str | None = None
    opencode_plugin_id: str | None = None


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
    budget = max(1, int(max_tokens))
    if thinking_tokens is not None and thinking_tokens >= 0.8 * budget:
        return (
            "thinking_50_percent",
            (
                "\n\nRETRY GUIDANCE: Limit internal thinking or reasoning to approximately "
                f"{max(1, budget // 2)} tokens (about half of the {budget}-token "
                "generation budget). Reserve the remaining budget for the required "
                "final answer. Do not spend the whole retry budget thinking."
            ),
        )
    if thinking_tokens is not None and thinking_tokens > 0.5 * budget:
        return (
            "thinking_30_percent",
            (
                "\n\nRETRY GUIDANCE: Limit internal thinking or reasoning to approximately "
                f"{max(1, int(budget * 0.3))} tokens (about 30% of the {budget}-token "
                "generation budget). Reserve most of the budget for the required "
                "final answer."
            ),
        )
    return (
        "response_under_budget",
        (
            "\n\nRETRY GUIDANCE: Complete the required answer while keeping the total "
            f"response just below the {budget}-token generation limit. Be concise "
            "enough to finish before the limit."
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
               stream_fallback_error: str | None = None) -> TransportResult:
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
        attempt_request = replace(request, prompt=request_prompt, log_label=log_label)
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

        next_alteration = "none"
        instruction = ""
        nature = result.response_nature
        if (
            (nature == "transport_error" and retry_policy.retry_on_transport_error)
            or (nature == "timeout" and retry_policy.retry_on_timeout)
        ):
            retry_reason = nature
        elif (
            (nature == "token_limit" and retry_policy.retry_on_token_limit)
            or (nature == "repetition_abort" and retry_policy.retry_on_repetition)
        ):
            alter = prompt_alterer or _retry_prompt_alteration
            next_alteration, instruction = alter(
                nature, result.thinking_tokens, request.max_tokens,
            )
            retry_reason = nature if next_alteration != "none" else None
        elif (
            retry_policy.retry_on_json_error
            and json_error_prompt_alterer is not None
            and not result.error
        ):
            json_instruction = json_error_prompt_alterer(result)
            if json_instruction:
                next_alteration = "json_error"
                instruction = json_instruction
                retry_reason = "json_error"

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
