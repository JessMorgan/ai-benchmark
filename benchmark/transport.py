"""Normalized one-attempt execution for benchmark transports.

This module deliberately does not own semantic retries, scoring, or state
persistence. It executes one prompt through HTTP or OpenCode and returns the
same result shape for both transports.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
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
    first = response.first_tok
    ended = response.stream_end or time.time()
    generation = ended - first if first is not None else ended - started
    text = response.text or ""
    repeating = is_repeating(text) or "repetition" in str(response.error or "").lower()
    return _normalize(
        request,
        text=text,
        think_text=response.think_text or "",
        error=response.error,
        finish_reason=response.finish_reason,
        response_time=ended - started,
        gen_time=generation,
        stream_ok=response.error is None and first is not None,
        repeating=repeating,
        usage=getattr(response, "usage", {}) or {},
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
