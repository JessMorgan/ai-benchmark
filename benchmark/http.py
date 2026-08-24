"""HTTP request helpers for the AI benchmark.

This module contains the low-level request logic (streaming and non-streaming)
used by ``benchmark_core.py``. Keeping it separate makes ``benchmark_core.py"
smaller and makes the request helpers easier to test and reason about.
"""
import contextlib
import copy
import email.utils
import json
import os
import random
import shlex
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from .logs import AppendOnlyGzipLog, recover_log
from .observer import TaskObserver


@dataclass(frozen=True)
class PostRequestResult:
    """Outcome yielded by the HTTP request context manager."""

    response: requests.Response | None
    error: str | None
    curl_cmd: str | None


@dataclass(frozen=True)
class SSEParseResult:
    """Updated state after parsing one SSE line."""

    first_tok: float | None
    text: str
    think_text: str
    finish_reason: str | None
    usage: dict[str, Any]
    done: bool
    # Accumulated native ``tool_calls`` deltas (merged by index, with
    # ``function.arguments`` fragments concatenated). Empty when the model
    # emitted no tool calls. See ``_merge_tool_calls``.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Server-side error surfaced inside the SSE stream (e.g. litellm/Ollama
    # emitting ``{"error": {...}}`` as a final data line before closing the
    # connection mid-reasoning). None for healthy streams. Detecting this is
    # what turns an aborted stream from a silent empty completion into a
    # diagnosed ``stream_error``.
    error: str | None = None


@dataclass(frozen=True)
class StreamResult:
    """Structured result returned by :func:`stream_request`."""

    text: str
    think_text: str
    first_tok: float | None
    stream_end: float
    error: str | None
    finish_reason: str | None
    usage: dict[str, Any]
    # Native tool calls accumulated from the stream (merged by index),
    # also rendered into ``text`` as ``<tool_call>{...}</tool_call>`` blocks
    # so the tool-calling plugin can score them. Empty when the model
    # emitted no tool calls.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class NonStreamResult:
    """Structured result returned by :func:`nonstream_request`."""

    text: str
    think_text: str
    usage: dict[str, Any]
    gen_time: float
    error: str | None
    finish_reason: str | None
    # Native tool calls from the response message (also rendered into
    # ``text`` as ``<tool_call>{...}</tool_call>`` blocks). Empty when the
    # model emitted no tool calls.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ResponseBodyResult:
    """Result of reading a non-streaming response body."""

    text: str | None
    error: str | None


@dataclass(frozen=True)
class _StreamLineError:
    """Internal sentinel for an exception raised while reading SSE lines."""

    error: str


def _safe_iter_lines(resp: requests.Response) -> Any:
    """Yield SSE lines while converting iterator failures to a sentinel."""
    try:
        yield from resp.iter_lines(decode_unicode=True)
    except Exception as exc:  # noqa: BLE001 - any iterator failure becomes a stream error sentinel
        yield _StreamLineError(f"{type(exc).__name__}: {exc}")


def _api_protocol(cfg: dict[str, Any]) -> str:
    """Return the request/response protocol for a source config.

    ``"openai"`` is the default OpenAI-compatible format used by every
    pre-existing source. ``"1min"`` selects the 1min.ai native
    ``/api/chat-with-ai`` endpoint, which uses a different request body
    (``type``/``model``/``promptObject``) and a different response/SSE shape.
    ``"chatplayground"`` selects the Playwright-driven interactive-web
    transport in :mod:`benchmark.chatplayground`. The choice is driven by the
    optional ``api_protocol`` source key; unknown or missing values fall back
    to the OpenAI format.
    """
    if isinstance(cfg, dict) and cfg.get("api_protocol") in ("1min", "chatplayground"):
        return str(cfg["api_protocol"])
    return "openai"


def _build_1min_request_body(model: str, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
    """Build the 1min.ai ``/api/chat-with-ai`` request body.

    The 1min.ai chat endpoint accepts ``type``, ``model``, and a
    ``promptObject`` with a single ``prompt`` string. It has no system-message
    field and no ``max_tokens``/``temperature``/``seed`` parameters, so a
    supplied system prompt is folded into the user prompt to preserve the
    persona and all benchmark generation knobs are ignored. Streaming is
    selected via the ``?isStreaming=true`` query parameter (handled in
    ``_post_request_context``), not via a body field.
    """
    text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    return {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": model,
        "promptObject": {"prompt": text},
    }


def _parse_1min_result(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract ``(text, error)`` from a 1min.ai non-streaming response body.

    A success body carries the generated text under
    ``aiRecord.aiRecordDetail.resultObject`` (a list of strings). Error bodies
    carry ``{"success": false, "error": {"message": ...}}``; a 200 response
    whose ``aiRecord.status`` is not ``SUCCESS`` is also surfaced as an error
    so a silent empty result is never mis-scored.
    """
    if isinstance(data, dict) and data.get("success") is False:
        err = data.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return "", f"1min error: {msg}"
    record = data.get("aiRecord") if isinstance(data, dict) else None
    if not isinstance(record, dict):
        return "", "Invalid 1min response: missing aiRecord"
    status = record.get("status")
    if status not in (None, "SUCCESS"):
        return "", f"1min aiRecord status: {status}"
    result_object = (record.get("aiRecordDetail") or {}).get("resultObject")
    if isinstance(result_object, list):
        text = "\n".join(str(item) for item in result_object)
    elif isinstance(result_object, str):
        text = result_object
    elif result_object is not None:
        text = json.dumps(result_object, ensure_ascii=False)
    else:
        text = ""
    return text, None


def _iter_1min_sse_events(resp: requests.Response) -> Iterator[tuple[str, Any] | _StreamLineError]:
    """Yield ``(event, data)`` pairs from a 1min.ai named-event SSE stream.

    1min.ai streams use named SSE events (``event: content``, ``event: done``,
    ``event: error``, ``event: result``) instead of the OpenAI ``data:``-only
    delta format. Each ``data:`` payload is paired with the most recent
    ``event:`` name; transport failures are surfaced as ``_StreamLineError``.
    """
    event_name: str | None = None
    for line in _safe_iter_lines(resp):
        if isinstance(line, _StreamLineError):
            yield line
            return
        if not isinstance(line, str):
            continue
        line = line.strip()
        if not line:
            event_name = None
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            yield (event_name or "message", data)


_log_lock = threading.Lock()
_GZIP_LOGS_RECOVERED: set[str] = set()


class _StopAwareRequestWatchdog:
    """Close a streaming response after a deadline, or soon after cancellation.

    ``close_active_requests()`` closes every response registered at the moment
    a quit lands, but a request whose ``requests.post()`` was still in flight
    then is added to ``_active_requests`` only after that pass, leaving the
    worker blocked on a socket read for up to the full request timeout
    (minutes) while the shutdown path's joins wait on it. This watchdog
    observes the same ``stop_event``: once it fires, the response is closed
    within a fraction of a second, so a quit can never strand a worker on a
    stale connection.
    """

    def __init__(self, resp: Any, timeout: float, stop_event: threading.Event | None) -> None:
        self._resp = resp
        self._timeout = max(0.0, float(timeout))
        self._stop_event = stop_event
        self._cancel = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="request-watchdog", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """Disarm the watchdog (request completed normally)."""
        self._cancel.set()

    def _run(self) -> None:
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._cancel.wait(min(remaining, 0.25)):
                return
        with contextlib.suppress(Exception):
            close_fn = getattr(self._resp, "close", None)
            if callable(close_fn):
                close_fn()

# Active HTTP responses so Ctrl+C can close them and unblock plugin threads.
_active_requests_lock = threading.Lock()
_active_requests: set[Any] = set()


def close_active_requests() -> None:
    """Close all in-flight HTTP responses to unblock worker threads."""
    with _active_requests_lock:
        for resp in list(_active_requests):
            with contextlib.suppress(Exception):
                resp.close()


# HTTP 429 activity tracked for the TUI live status section. Keyed by
# ``(source, model, pid)`` so the dashboard can tell the operator not only
# *that* a source/model is in backoff but also *which* plugin is blocked.
# All mutations and reads happen under ``_429_lock``.
_429_lock = threading.Lock()
_429_stats: dict[str, Any] = {
    # Total number of times any request entered a 429 backoff sleep this run.
    "total_retries": 0,
    # Map ``(source, model, pid) -> {wake_ts, attempts, max_attempts}``. Entries
    # are inserted when a sleep begins and removed when it ends (cleanly or
    # via Ctrl+C). ``wake_ts`` is an absolute ``time.time()`` value so the
    # TUI can render a countdown without holding the lock while sleeping.
    "sleeping": {},
    # Aggregate per-plugin retry statistics for post-run analysis.
    # ``pid -> {"retries": int, "total_sleep_time": float}``.
    "plugin_stats": {},
}


def get_active_request_count() -> int:
    """Return the number of in-flight HTTP responses across the run.

    Used by the TUI to show how many concurrent requests are outstanding so
    operators can correlate a slow model row with the actual request count.
    """
    with _active_requests_lock:
        return len(_active_requests)


def get_429_stats() -> dict[str, Any]:
    """Return a thread-safe snapshot of HTTP 429 retry/backoff state.

    The snapshot is decoupled from the internal ``_429_stats`` dict so a
    caller iterating over ``sleeping`` cannot observe a half-removed entry
    torn by a concurrent sleep completion. ``sleeping`` keys are strings
    of the form ``"source|model|pid"`` for JSON-friendly consumption.
    """
    with _429_lock:
        sleeping = {}
        for (src, model, pid), info in _429_stats["sleeping"].items():
            sleeping[f"{src}|{model}|{pid}"] = dict(info)
        return {
            "total_retries": _429_stats["total_retries"],
            "sleeping": sleeping,
            "plugin_stats": copy.deepcopy(_429_stats["plugin_stats"]),
        }


def reset_429_stats() -> None:
    """Reset 429 activity tracking. Intended for unit tests only."""
    global _429_stats
    with _429_lock:
        _429_stats = {"total_retries": 0, "sleeping": {}, "plugin_stats": {}}


def _set_429_sleep(source: str, model: str, pid: str, wake_ts: float, attempts: int, max_attempts: int, delay: float) -> None:
    """Record that ``(source, model, pid)`` is entering a 429 backoff sleep."""
    with _429_lock:
        _429_stats["total_retries"] += 1
        _429_stats["sleeping"][(source, model, pid)] = {
            "wake_ts": wake_ts,
            "attempts": attempts,
            "max_attempts": max_attempts,
        }
        plugin_id = pid or "?"
        p_stats = _429_stats["plugin_stats"].setdefault(plugin_id, {
            "retries": 0,
            "total_sleep_time": 0.0,
        })
        p_stats["retries"] += 1
        p_stats["total_sleep_time"] += float(delay)


def _clear_429_sleep(source: str, model: str, pid: str) -> None:
    """Remove a ``(source, model, pid)`` entry from the 429 sleeping set, if any."""
    with _429_lock:
        _429_stats["sleeping"].pop((source, model, pid), None)


def fetch_models_v1(base_url: str, api_key: str | None = None) -> list[str]:
    """Call GET {base_url}/v1/models and return a list of model IDs."""
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    return [m["id"] for m in models if "id" in m]


def build_curl_cmd(model: str, prompt: str, max_tokens: int, stream: bool, api_url: str, headers: dict[str, str], system_prompt: str | None = None,
                   request_body: dict[str, Any] | None = None) -> str:
    """Build a curl command string for the given API request.

    ``request_body`` is the already-merged body passed to ``requests.post``.
    Supplying it keeps the diagnostic curl command faithful to the actual
    request, including provider-specific parameters such as
    ``chat_template_kwargs`` and ``response_format``. The individual
    arguments remain as a backward-compatible fallback for callers that only
    need to construct a basic request.
    """
    if request_body is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
    data = json.dumps(request_body, ensure_ascii=False)
    # Render every configured header (not just Authorization/Content-Type) so
    # providers that authenticate via other headers — e.g. 1min.ai's
    # ``API-KEY`` — produce a fully replayable curl command.
    header_lines = "".join(
        f"  -H {shlex.quote(f'{key}: {value}')} \\\n"
        for key, value in headers.items()
    )
    return (
        f"curl -s -X POST {shlex.quote(api_url)} \\\n"
        f"{header_lines}"
        f"  -d {shlex.quote(data)}"
    )


def log_request_entry(log_path: str, curl_cmd: str, response_body: str, request_label: str | None = None) -> None:
    """Append a request/response block as plaintext or a gzip member."""
    block = ""
    if request_label:
        block += f"\n# === {request_label} ===\n"
    block += f"{curl_cmd}\n\n"
    block += f"{response_body}\n"
    block += "\n" + "-" * 60 + "\n"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with _log_lock:
        if log_path.endswith(".gz"):
            absolute_path = os.path.abspath(log_path)
            recover_tail = absolute_path not in _GZIP_LOGS_RECOVERED
            if recover_tail:
                recover_log(log_path, repair=True)
                _GZIP_LOGS_RECOVERED.add(absolute_path)
            writer = AppendOnlyGzipLog(
                log_path, recover_tail=False, sync_policy="batch",
            )
            writer.append_record([block])
            writer.close(sync=False)
            return
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(block)


def _check_total_timeout(start_time: float, timeout: float, error: str | None, finish_reason: str | None = None) -> str | None:
    """Return a timeout error if the overall request duration was exceeded."""
    if not error and not finish_reason and time.time() - start_time > timeout:
        return f"Total timeout ({timeout}s) exceeded"
    return error


def _log_response(log_path: str | None, curl_cmd: str | None, response_body: str | None, log_label: str | None) -> None:
    """Write the response body to the request log if logging is enabled."""
    if log_path and curl_cmd:
        log_request_entry(log_path, curl_cmd, response_body or "(empty response)", log_label)


def _build_request_body(model: str, prompt: str, max_tokens: int, session_seed: int, temperature: float | None, drop_params: list[str] | None, stream: bool,
                        system_prompt: str | None = None, request_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the JSON body for an API request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if session_seed:
        body["seed"] = session_seed
    if isinstance(request_params, dict):
        body.update(copy.deepcopy(request_params))
    for p in drop_params or []:
        body.pop(p, None)
    return body


@contextmanager
def _post_request_context(source_config: dict[str, Any], source: str, body: dict[str, Any], timeout: float, stream: bool, log_path: str | None, log_label: str | None,
                          stop_event: threading.Event | None = None, pid: str | None = None, on_retry: Callable[[], None] | None = None) -> Iterator[PostRequestResult]:
    """Make a POST request and yield the response, handling cleanup.

    Yields a :class:`PostRequestResult`. ``response`` is the
    requests Response object on success, or ``None`` if an error occurred
    before or during the request. Cleanup (watchdog cancellation, active
    request tracking removal, response close) is performed automatically.

    ``pid`` identifies the plugin making the request so that 429 backoff
    state can be tracked per-plugin rather than per-model. ``on_retry`` is
    an optional callable invoked at the start of each retry attempt (after
    the first) so the caller can reset per-request timing/bookkeeping.

    HTTP 429 is retried with exponential backoff + jitter up to
    ``max_429_retries`` times per source. Defaults (``backoff_seconds=30``,
    ``backoff_factor=2.0``, ``max_backoff_seconds=300``, ``max_429_retries=2``)
    are opt-out: setting ``max_429_retries: 0`` restores the previous
    fail-fast behaviour. The ``Retry-After`` header (numeric form) is honoured
    as a *floor* on the delay. Sleeps go through ``stop_event.wait`` so
    Ctrl+C aborts immediately.
    """
    cfg = source_config.get(source, {})
    protocol = _api_protocol(cfg)
    default_url = (
        "https://api.1min.ai/api/chat-with-ai"
        if protocol == "1min" else "http://localhost:11434/chat/completions"
    )
    api_url = cfg.get("api_url", default_url)
    # 1min.ai selects streaming via a URL query parameter, not a body field.
    if protocol == "1min" and stream:
        api_url = api_url + ("&" if "?" in api_url else "?") + "isStreaming=true"
    headers = cfg.get("headers", {"Content-Type": "application/json"})
    model = body.get("model", "")
    system_prompt = None
    if body.get("messages") and body["messages"][0].get("role") == "system":
        system_prompt = body["messages"][0]["content"]
    prompt = ""
    for msg in body.get("messages", []):
        if msg.get("role") == "user":
            prompt = msg["content"]
            break
    max_tokens = body.get("max_tokens", 2048)
    curl_cmd = (
        build_curl_cmd(
            model, prompt, max_tokens, stream, api_url, headers,
            system_prompt=system_prompt, request_body=body,
        )
        if log_path else None
    )
    resp = None
    watchdog = None
    # Use a short connection timeout so a stuck connect() returns quickly,
    # but keep the user's full timeout for reading so slow models are not
    # aborted prematurely.
    connect_timeout = min(float(timeout), 5.0)
    request_timeout = (connect_timeout, timeout)

    # ── HTTP 429 retry config (opt-out via ``max_429_retries: 0``) ──
    try:
        max_retries = max(0, int(cfg.get("max_429_retries", 2)))
    except (TypeError, ValueError):
        max_retries = 2
    try:
        base_delay = float(cfg.get("backoff_seconds", 30))
    except (TypeError, ValueError):
        base_delay = 30.0
    try:
        backoff_factor = float(cfg.get("backoff_factor", 2.0))
    except (TypeError, ValueError):
        backoff_factor = 2.0
    try:
        max_backoff = float(cfg.get("max_backoff_seconds", 300))
    except (TypeError, ValueError):
        max_backoff = 300.0

    plugin_id = pid or "?"
    try:
        for attempt in range(max_retries + 1):
            # Cancellation check before each request attempt.
            if stop_event is not None and stop_event.is_set():
                yield PostRequestResult(None, "Cancelled", curl_cmd)
                return

            # Give the caller a chance to reset per-request bookkeeping
            # (e.g. the plugin start timestamp) at the beginning of each
            # retry. The first attempt already started at dispatch time.
            if attempt > 0 and on_retry is not None:
                try:
                    on_retry()
                except Exception as exc:  # noqa: BLE001 - a buggy observer must not abort the retry loop
                    # swallowing it silently makes state bugs hard to find.
                    with contextlib.suppress(Exception):
                        sys.stderr.write(
                            f"benchmark_http: on_retry observer failed: {exc}\n"
                        )
                        traceback.print_exc(file=sys.stderr)

            try:
                resp = requests.post(
                    api_url, headers=headers, json=body, stream=True,
                    timeout=request_timeout)
            except Exception as e:  # noqa: BLE001 - a transport failure becomes a failed request result
                error = f"{type(e).__name__}: {e}"
                if log_path and curl_cmd:
                    log_request_entry(log_path, curl_cmd, f"ERROR: {error}", log_label)
                yield PostRequestResult(None, error, curl_cmd)
                return

            with _active_requests_lock:
                _active_requests.add(resp)
            # Guard ``resp.close`` against ``None``/non-Response objects
            # (e.g. mocks whose side_effect yields a function rather than
            # a Response). Without this guard the ``Timer`` constructor
            # itself fails — not 30s later when the timer fires — so the
            # thread crashes during ``__enter__`` and the test fixture
            # never sees the response. The ``finally`` below already
            # tolerates ``watchdog is None`` (early return path skips it
            # entirely).
            close_fn = getattr(resp, "close", None)
            if callable(close_fn):
                watchdog = _StopAwareRequestWatchdog(resp, timeout, stop_event)
                watchdog.start()
            else:
                watchdog = None

            if resp.status_code != 429:
                # Success or non-429 error: yield to caller; outer ``finally``
                # handles watchdog/active-requests/close cleanup.
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:150]}"
                    if log_path and curl_cmd:
                        log_request_entry(
                            log_path, curl_cmd,
                            f"HTTP {resp.status_code}: {resp.text[:500]}", log_label)
                    yield PostRequestResult(None, error, curl_cmd)
                else:
                    yield PostRequestResult(resp, None, curl_cmd)
                return

            # HTTP 429 — decide whether to surface or retry.
            if attempt >= max_retries:
                # Out of retries; surface as a failure.
                error = f"HTTP {resp.status_code}: {resp.text[:150]}"
                if log_path and curl_cmd:
                    log_request_entry(
                        log_path, curl_cmd,
                        f"HTTP {resp.status_code}: {resp.text[:500]}", log_label)
                yield PostRequestResult(None, error, curl_cmd)
                return

            # Compute retry-after delay: exponential backoff with Retry-After
            # as a floor and bounded by ``max_backoff_seconds``.
            # Retry-After can be either a numeric number of seconds (RFC 7231
            # §7.1.3 form 1) or an HTTP-date (form 2). Both are honoured.
            # Anything unparseable silently falls back to the computed delay.
            retry_after_raw = resp.headers.get("Retry-After", "").strip()
            retry_after = 0.0
            if retry_after_raw:
                try:
                    retry_after = float(retry_after_raw)
                except ValueError:
                    try:
                        parsed_dt = email.utils.parsedate_to_datetime(retry_after_raw)
                    except (TypeError, ValueError):
                        parsed_dt = None
                    if parsed_dt is not None:
                        if parsed_dt.tzinfo is None:
                            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                        delta = (parsed_dt - datetime.now(timezone.utc)).total_seconds()
                        retry_after = max(0.0, delta)
            delay = base_delay * (backoff_factor ** attempt)
            delay = max(retry_after, delay)
            delay = min(delay, max_backoff)
            # Apply ±20 % jitter ONLY when Retry-After is absent. We do not
            # extend a precise server-suggested wait — providers expect us
            # to honour it as-is (±jitter would risk re-throttling).
            if not retry_after:
                delay *= random.uniform(0.8, 1.2)
                delay = min(delay, max_backoff)

            # Log the retry decision into the per-request log file so the
            # operator can see why the wall-clock grew. Counter is over the
            # total attempts including the initial request — operators
            # reading the log can map "attempt 2/3" to "initial + 1 retry
            # so far" without ref-counting.
            if log_path and curl_cmd:
                log_request_entry(
                    log_path, curl_cmd,
                    f"# HTTP 429; sleeping {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries + 1})", log_label)

            # Publish 429 state for the TUI live status section. We do this
            # AFTER the log entry so the operator can correlate the two,
            # and we always clear it in a finally block below so Ctrl+C
            # cannot leave a stale "sleeping" marker on screen. ``source``
            # and the model name come from the function args / request body
            # so the dashboard can attribute the sleep to a specific source.
            attempts_done = attempt + 1
            max_attempts = max_retries + 1
            model = body.get("model", "?")
            wake_ts = time.time() + delay
            _set_429_sleep(source, model, plugin_id, wake_ts, attempts_done, max_attempts, delay)

            # Tear down the current response before sleeping so we don't
            # leak the connection or hold a watchdog reference.
            if watchdog is not None:
                with contextlib.suppress(Exception):
                    watchdog.cancel()
            with _active_requests_lock:
                _active_requests.discard(resp)
            with contextlib.suppress(Exception):
                resp.close()
            resp = None
            watchdog = None

            # Interruptible sleep. With ``stop_event`` we sleep through
            # ``Event.wait``, which returns immediately if the event is set.
            # Without it we ``time.sleep`` (the previous version accidentally
            # short-circuited the wait and never slept, which made Retry-After
            # / backoff tests flake on 0.0s runs).
            try:
                if stop_event is not None:
                    if stop_event.wait(delay):
                        yield PostRequestResult(None, "Cancelled", curl_cmd)
                        return
                else:
                    time.sleep(delay)
            finally:
                _clear_429_sleep(source, model, plugin_id)
    finally:
        if watchdog is not None:
            with contextlib.suppress(Exception):
                watchdog.cancel()
        if resp is not None:
            with contextlib.suppress(Exception), _active_requests_lock:
                _active_requests.discard(resp)
            with contextlib.suppress(Exception):
                resp.close()


def _merge_tool_calls(acc: list[dict[str, Any]], fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge SSE ``tool_calls`` delta fragments into the accumulated list.

    Streaming APIs send tool calls as partial objects keyed by ``index``:
    the first fragment carries ``id``/``type``/``function.name`` and
    subsequent fragments append ``function.arguments`` chunks. This helper
    merges them into a list of complete tool-call objects, preserving the
    server's ``index`` ordering.
    """
    result = list(acc)
    for frag in fragments or []:
        if not isinstance(frag, dict):
            continue
        idx = frag.get("index", 0)
        while len(result) <= idx:
            result.append({})
        slot = result[idx]
        if frag.get("id"):
            slot["id"] = frag["id"]
        if frag.get("type"):
            slot["type"] = frag["type"]
        fn_frag = frag.get("function") or {}
        fn = slot.setdefault("function", {})
        if fn_frag.get("name"):
            fn["name"] = fn_frag["name"]
        if fn_frag.get("arguments"):
            fn["arguments"] = fn.get("arguments", "") + fn_frag["arguments"]
    return result


def _render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Render accumulated native tool calls as ``<tool_call>`` blocks.

    Converts the OpenAI-style ``{id, type, function: {name, arguments}}``
    form (with ``arguments`` as a JSON string) into the benchmark's textual
    ``<tool_call>{"name": ..., "args": {...}}</tool_call>`` format so the
    tool-calling plugin can score native tool-call emissions.
    """
    blocks = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, ValueError):
            args = raw_args
        blocks.append(f"<tool_call>{json.dumps({'name': name, 'args': args}, ensure_ascii=False)}</tool_call>")
    return "\n".join(blocks)


# Live-stream repetition guard. An abort is a destructive act (the model
# slot frees and the partial text is scored), so the live rule is deliberately
# more selective than the post-hoc ``is_repeating`` flag in ``core.py``:
#
# * the repeated block must be found again *adjacent* to the stream tail
#   (within ``REPETITION_GUARD_ADJACENCY`` chars) - a true echo loop re-emits
#   its last phrase immediately, while legitimate repetition (three generated
#   classes sharing an ``__init__(self, limit, window_seconds)`` scaffold, an
#   ASCII diagram's repeated ``+---+`` borders) is interleaved with distinct
#   content;
# * ``_decorative_block`` skips repeats that are mostly typographic
#   decoration (box-drawing borders, dashes, pipes) rather than words/code.
#
# Longer-period loops that pass both filters are still bounded by the
# content/thinking token budgets, so the repetition guard only needs to
# catch the dense self-echo cases.
REPETITION_GUARD_MIN_SEQ = 80
REPETITION_GUARD_REPEATS = 3
REPETITION_GUARD_WINDOW = 4096  # chars of history searched per check
REPETITION_GUARD_ADJACENCY = 256  # previous repeat must end within this many chars of the tail
REPETITION_GUARD_DECORATION_RATIO = 0.35  # block with fewer alnum chars than this is typographic


class _StreamGuards:
    """Abort a live stream that exceeds its budgets or falls into a loop.

    Two budgets split by stream: ``reasoning_content`` is capped at
    ``max_thinking_tokens`` and final ``content`` at ``max_content_tokens``
    (both estimated as ``len(text) / 4``, matching ``count_tokens``). A
    per-stream repetition detector aborts content or thinking that repeats
    its tail block at least 3 times inside the recent history with the
    previous repeat *adjacent* to the stream tail (see the module constants
    above). The post-hoc ``is_repeating`` flag in ``core.py`` uses the
    simpler 80-char x3 rule because it only marks a completed response;
    aborting a live stream demands more evidence.

    ``check`` is called after every parsed SSE delta; it returns an error
    string once a budget or the repetition guard fires, and ``None``
    otherwise. Checking is skipped when a stream has not grown since the
    last call, so an idle/heartbeat stream costs nothing.
    """

    def __init__(self, max_content_tokens: int | None = None, max_thinking_tokens: int | None = None,
                 repetition_guard: int | bool = False) -> None:
        self.max_content_tokens = max_content_tokens or 0
        self.max_thinking_tokens = max_thinking_tokens or 0
        self.repetition_guard = repetition_guard
        self._checked_content_len = 0
        self._checked_think_len = 0

    def check(self, text: str, think_text: str) -> str | None:
        """Return an abort error message, or ``None`` when the stream is fine."""
        if len(text) > self._checked_content_len:
            self._checked_content_len = len(text)
            if self.max_content_tokens and len(text) / 4 > self.max_content_tokens:
                return f"Content budget exceeded ({int(len(text) / 4)} tokens)"
            if self.repetition_guard and _repeats_detected(text):
                return "Repetition detected in content — stream aborted"
        if len(think_text) > self._checked_think_len:
            self._checked_think_len = len(think_text)
            if self.max_thinking_tokens and len(think_text) / 4 > self.max_thinking_tokens:
                return f"Thinking budget exceeded ({int(len(think_text) / 4)} tokens)"
            if self.repetition_guard and _repeats_detected(think_text):
                return "Repetition detected in thinking — stream aborted"
        return None


def _decorative_block(block: str) -> bool:
    """Return whether ``block`` is mostly typographic decoration.

    ASCII architecture/wireframe diagrams legitimately repeat box-drawing
    borders, dashes and pipes; a repeated block that is mostly those
    characters is a structural pattern, not a generation loop.
    """
    if not block:
        return True
    meaningful = sum(1 for ch in block if ch.isalnum())
    return meaningful / len(block) < REPETITION_GUARD_DECORATION_RATIO


def _repeats_detected(text: str) -> bool:
    """Return whether ``text`` is stuck in a dense echo loop.

    The newest ``REPETITION_GUARD_MIN_SEQ``-char tail must appear
    ``REPETITION_GUARD_REPEATS`` times total, the previous occurrence must
    end within ``REPETITION_GUARD_ADJACENCY`` chars of the stream tail (a
    genuine loop re-emits its last phrase immediately), and the block must
    not be mostly typographic decoration. The search is bounded to the most
    recent ``REPETITION_GUARD_WINDOW`` characters so an unbounded stream
    cannot grow the per-check cost.
    """
    if len(text) < REPETITION_GUARD_MIN_SEQ * REPETITION_GUARD_REPEATS:
        return False
    tail = text[-REPETITION_GUARD_MIN_SEQ:]
    if _decorative_block(tail):
        return False
    history = text[-REPETITION_GUARD_MIN_SEQ - REPETITION_GUARD_WINDOW:-REPETITION_GUARD_MIN_SEQ]
    if history.count(tail) < REPETITION_GUARD_REPEATS - 1:
        return False
    prev = history.rfind(tail)
    return prev != -1 and (len(history) - prev) <= REPETITION_GUARD_ADJACENCY


def _parse_sse_line(line: str, first_tok: float | None, text: str,
                    think_text: str, finish_reason: str | None,
                    usage: dict[str, Any],
                    tool_calls: list[dict[str, Any]] | None = None) -> SSEParseResult:
    """Parse a single Server-Sent Events line and update streaming state.

    Returns an :class:`SSEParseResult`. ``done`` is True when the ``[DONE]``
    sentinel is encountered.

    ``think_text`` accumulates ``reasoning_content`` from SSE deltas so
    thinking-capable models' chain-of-thought is preserved separately from
    the final content. Models that don't emit ``reasoning_content`` leave
    ``think_text`` unchanged (empty string).

    ``tool_calls`` accumulates native ``tool_calls`` delta fragments so
    agent-style responses that emit tool calls instead of text are not
    silently dropped (previously ``delta["tool_calls"]`` was ignored and
    those legs scored 0 with empty content).

    A data payload with an ``error`` key (how litellm/Ollama and several
    OpenAI-compatible proxies signal a mid-stream failure) is surfaced via
    the returned ``error`` field instead of being swallowed -- the stream
    that produced it is aborted, not treated as a clean empty completion.
    """
    tool_calls = tool_calls or []
    error = None
    if not line.startswith("data: "):
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False, tool_calls, error)
    payload = line[6:]
    if payload.strip() == "[DONE]":
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, True, tool_calls, error)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False, tool_calls, error)
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        if isinstance(err, dict):
            error = err.get("message") or json.dumps(err)
        else:
            error = str(err)
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False, tool_calls, error)
    if first_tok is None:
        first_tok = time.time()
    for ch in data.get("choices", []):
        delta = ch.get("delta", {})
        text += delta.get("content", "")
        think_text += delta.get("reasoning_content", "")
        tc = delta.get("tool_calls")
        if tc:
            tool_calls = _merge_tool_calls(tool_calls, tc)
        fr = ch.get("finish_reason")
        if fr:
            finish_reason = fr
    if "usage" in data:
        usage = data["usage"]
    return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False, tool_calls, error)


def stream_request(source_config: dict[str, Any], timeout: float, model: str, source: str, prompt: str, max_tokens: int = 2048,
                   log_path: str | None = None, log_label: str | None = None, session_seed: int = 0, temperature: float | None = None,
                   drop_params: list[str] | None = None, stop_event: threading.Event | None = None, system_prompt: str | None = None,
                   request_params: dict[str, Any] | None = None,
                   observer: TaskObserver | None = None,
                   on_chunk: Callable[[str], None] | None = None,
                   on_think_chunk: Callable[[str], None] | None = None,
                   pid: str | None = None,
                   on_retry: Callable[[], None] | None = None,
                   max_content_tokens: int | None = None, max_thinking_tokens: int | None = None,
                   repetition_guard: int | bool = False) -> StreamResult:
    """Make a streaming chat-completion request and return parsed results.

    Returns a :class:`StreamResult` with parsed fields for the assembled
    text, timing, finish reason, usage, and any transport error.
    ``think_text`` contains any reasoning/thinking content emitted by
    thinking-capable models (conversational ``reasoning_content`` field from
    SSE deltas). For standard models it is an empty string.

    ``on_chunk`` (optional) is called once per parsed SSE delta with the new
    text accumulated in that iteration, so callers (notably the live TUI)
    can update per-plugin byte/tok counts as tokens arrive instead of
    waiting for the full response. Exceptions raised by ``on_chunk`` are
    swallowed so a buggy observer cannot abort the stream read -- the TUI
    is a display concern, not a correctness concern.

    ``max_content_tokens`` / ``max_thinking_tokens`` / ``repetition_guard``
    enable the live ``_StreamGuards`` watchdog: the stream is aborted
    (returning an error with whatever text accumulated) the moment final
    content or reasoning exceeds its budget, or the content/thinking repeats
    itself. Defaults are off so preload/judge/utility callers are
    unaffected; the benchmark task path opts in with per-source config.
    """
    if observer is None:
        observer = TaskObserver(
            pid=pid or "",
            on_chunk=on_chunk,
            on_think_chunk=on_think_chunk,
            on_retry=on_retry,
        )
    start = time.time()
    first_tok = None
    text = ""
    think_text = ""
    error = None
    finish_reason = None
    usage: dict[str, Any] = {}
    tool_calls: list[dict[str, Any]] = []
    guards = _StreamGuards(max_content_tokens, max_thinking_tokens, repetition_guard)
    cfg = source_config.get(source) or {}
    protocol = _api_protocol(cfg)
    if protocol == "chatplayground":
        from . import chatplayground

        text, error, _gen_time = chatplayground.request(
            cfg, model, prompt, timeout=timeout, stop_event=stop_event,
            system_prompt=system_prompt,
        )
        if log_path:
            log_request_entry(
                log_path,
                "# ChatPlayground (browser) request",
                text or "(empty response)",
                log_label,
            )
        return StreamResult(text, "", None, time.time(), error, None, {}, [])
    if protocol == "1min":
        body = _build_1min_request_body(model, prompt, system_prompt=system_prompt)
    else:
        body = _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params,
                                   stream=True, system_prompt=system_prompt,
                                   request_params=request_params)
    with _post_request_context(source_config, source, body, timeout, True, log_path, log_label,
                               stop_event=stop_event, pid=pid, on_retry=observer.retry) as request:
        if request.error:
            return StreamResult(text, think_text, first_tok, time.time(),
                                request.error, finish_reason, usage)
        resp = request.response
        if resp is None:
            return StreamResult(text, think_text, first_tok, time.time(),
                                "HTTP request returned no response", finish_reason, usage)
        if protocol == "1min":
            for item in _iter_1min_sse_events(resp):
                if isinstance(item, _StreamLineError):
                    error = item.error
                    break
                event, data = item
                if stop_event and stop_event.is_set():
                    error = "Cancelled"
                    break
                if event == "error":
                    msg = (data.get("message") or data.get("error")
                           or "Unknown 1min stream error") if isinstance(data, dict) else data
                    error = f"1min stream error: {msg}"
                    break
                if event == "done":
                    break
                if event == "content":
                    chunk = data.get("content", "") if isinstance(data, dict) else str(data)
                    if chunk:
                        if first_tok is None:
                            first_tok = time.time()
                        text += chunk
                        observer.chunk(chunk)
                        guard_error = guards.check(text, "")
                        if guard_error:
                            error = guard_error
                            break
            error = _check_total_timeout(start, timeout, error, finish_reason)
            _log_response(log_path, request.curl_cmd, text, log_label)
            return StreamResult(text, think_text, first_tok, time.time(),
                                error, finish_reason, usage, tool_calls)
        prev_text_len = 0
        prev_think_len = 0
        for line in _safe_iter_lines(resp):
            if isinstance(line, _StreamLineError):
                error = line.error
                break
            if stop_event and stop_event.is_set():
                error = "Cancelled"
                break
            if not line:
                continue
            try:
                parsed = _parse_sse_line(
                    line, first_tok, text, think_text, finish_reason, usage, tool_calls)
            except Exception as exc:  # noqa: BLE001 - malformed server lines become a stream error
                error = f"SSE parse error: {type(exc).__name__}: {exc}"
                break
            first_tok = parsed.first_tok
            text = parsed.text
            think_text = parsed.think_text
            finish_reason = parsed.finish_reason
            usage = parsed.usage
            tool_calls = parsed.tool_calls
            if parsed.error:
                # The server sent ``{"error": ...}`` inside the stream
                # (litellm/Ollama abort mid-reasoning with EOF). Abort and
                # surface it: without this the aborted stream was treated
                # as a clean empty completion (`stream_ok=True`, score 0).
                error = parsed.error
                break
            if parsed.done:
                break
            # Notify the caller of the content delta accumulated in this
            # iteration. We compute the delta from ``text`` length so a
            # single SSE data event that pumps multiple ``choices`` deltas
            # still produces one observer call with the joined delta,
            # which keeps the callback rate proportional to wall-clock
            # arrivals rather than choice array sizes.
            if len(text) > prev_text_len:
                delta = text[prev_text_len:]
                prev_text_len = len(text)
                observer.chunk(delta)
            # Parallel reasoning / thinking callback. The thinking
            # counter increments independently of the content counter so
            # a deepseek-r1 / Qwen3 / o1-style stream that emits 2 000
            # chars of ``reasoning_content`` BEFORE ``content`` shows a
            # real ticking ``[streaming - N think-tok]`` rather than
            # the seconds-only ``[streaming - Ns]`` placeholder
            # during the entire thinking phase. The ``prev_think_len``
            # tracking parallels ``prev_text_len``; a non-thinking model
            # never produces a non-empty ``think_delta`` so the
            # branch is a no-op for them. Same exception-swallowing
            # contract as ``on_chunk`` so a buggy observer cannot
            # abort the stream read.
            if len(think_text) > prev_think_len:
                think_delta = think_text[prev_think_len:]
                prev_think_len = len(think_text)
                observer.think_chunk(think_delta)
            guard_error = guards.check(text, think_text)
            if guard_error:
                error = guard_error
                break
        error = _check_total_timeout(start, timeout, error, finish_reason)
        # Render any captured native tool calls into the final text so the
        # tool-calling plugin can score them (they arrive in ``tool_calls``
        # deltas, not ``content``, and were previously dropped -> empty
        # response with score 0).
        if tool_calls:
            rendered = _render_tool_calls(tool_calls)
            if rendered:
                text = (text.rstrip() + "\n" + rendered) if text else rendered
        _log_response(log_path, request.curl_cmd, text, log_label)
    return StreamResult(text, think_text, first_tok, time.time(), error,
                        finish_reason, usage, tool_calls)


def _read_response_body(resp: requests.Response, stop_event: threading.Event | None) -> ResponseBodyResult:
    """Read a non-streaming response body in chunks, honouring cancellation."""
    chunks = []
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            if stop_event and stop_event.is_set():
                return ResponseBodyResult(None, "Cancelled")
            if chunk:
                chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001 - body read failures become a failed result
        return ResponseBodyResult(None, f"{type(exc).__name__}: {exc}")
    return ResponseBodyResult(
        b"".join(chunks).decode("utf-8", errors="replace"), None
    )


def nonstream_request(source_config: dict[str, Any], timeout: float, model: str, source: str, prompt: str, max_tokens: int = 2048,
                      log_path: str | None = None, log_label: str | None = None, session_seed: int = 0, temperature: float | None = None,
                      drop_params: list[str] | None = None, stop_event: threading.Event | None = None, system_prompt: str | None = None,
                      request_params: dict[str, Any] | None = None,
                      observer: TaskObserver | None = None,
                      pid: str | None = None,
                      on_retry: Callable[[], None] | None = None,
                      max_content_tokens: int | None = None, max_thinking_tokens: int | None = None,
                      repetition_guard: int | bool = False) -> NonStreamResult:
    """Make a non-streaming chat-completion request and return parsed results.

    Returns a :class:`NonStreamResult` with named fields for response text,
    timing, finish reason, usage, and any transport error.
    ``think_text`` contains any reasoning/thinking content from the API
    response (``message.reasoning_content`` field). For standard models it
    is an empty string.

    ``max_content_tokens`` / ``max_thinking_tokens`` / ``repetition_guard``
    enable the ``_StreamGuards`` watchdog on the completed response: a
    response that already exceeded its content/thinking budget or repeats
    itself is returned as an error instead of a completed result. Defaults
    are off so preload/judge/utility callers are unaffected; the benchmark
    task path opts in with per-source config.
    """
    if observer is None:
        observer = TaskObserver(pid=pid or "", on_retry=on_retry)
    start = time.time()
    error = None
    text = ""
    think_text = ""
    usage: dict[str, Any] = {}
    finish_reason = None
    tool_calls: list[dict[str, Any]] = []
    guards = _StreamGuards(max_content_tokens, max_thinking_tokens, repetition_guard)
    cfg = source_config.get(source) or {}
    protocol = _api_protocol(cfg)
    if protocol == "chatplayground":
        from . import chatplayground

        text, error, gen_time = chatplayground.request(
            cfg, model, prompt, timeout=timeout, stop_event=stop_event,
            system_prompt=system_prompt,
        )
        if log_path:
            log_request_entry(
                log_path,
                "# ChatPlayground (browser) request",
                text or "(empty response)",
                log_label,
            )
        return NonStreamResult(text, "", {}, gen_time, error, None)
    if protocol == "1min":
        body = _build_1min_request_body(model, prompt, system_prompt=system_prompt)
    else:
        body = _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params,
                                   stream=False, system_prompt=system_prompt,
                                   request_params=request_params)
    raw_resp_text = None
    with _post_request_context(source_config, source, body, timeout, False, log_path, log_label,
                               stop_event=stop_event, pid=pid, on_retry=observer.retry) as request:
        if request.error:
            return NonStreamResult(text, think_text, usage, time.time() - start,
                                    request.error, finish_reason)
        if request.response is None:
            return NonStreamResult(text, think_text, usage, time.time() - start,
                                   "HTTP request returned no response", finish_reason)
        response = _read_response_body(request.response, stop_event)
        if response.error:
            return NonStreamResult(text, think_text, usage, time.time() - start,
                                   response.error, finish_reason)
        raw_resp_text = response.text
        if raw_resp_text is None:
            return NonStreamResult(text, think_text, usage, time.time() - start,
                                   "Empty response body", finish_reason)
        if protocol == "1min":
            try:
                data = json.loads(raw_resp_text)
            except json.JSONDecodeError as exc:
                error = f"Invalid 1min response: {type(exc).__name__}: {exc}"
            else:
                parsed_text, one_min_error = _parse_1min_result(data)
                text = parsed_text or ""
                if one_min_error:
                    error = one_min_error
        else:
            try:
                data = json.loads(raw_resp_text)
                message = data["choices"][0]["message"]
                text = message["content"] or ""
                think_text = message.get("reasoning_content", "")
                usage = data.get("usage", {})
                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                tool_calls = message.get("tool_calls") or []
                # Native tool calls (OpenAI-style ``message.tool_calls``) are
                # rendered into the final text so the tool-calling plugin can
                # score them, mirroring the streaming path.
                if tool_calls:
                    rendered = _render_tool_calls(tool_calls)
                    if rendered:
                        text = (text.rstrip() + "\n" + rendered) if text else rendered
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                error = f"Invalid completion response: {type(exc).__name__}: {exc}"
                tool_calls = []
        guard_error = guards.check(text, think_text)
        if not error and guard_error:
            error = guard_error
        _log_response(log_path, request.curl_cmd, raw_resp_text, log_label)
    error = _check_total_timeout(start, timeout, error, finish_reason)
    return NonStreamResult(text, think_text, usage, time.time() - start,
                           error, finish_reason, tool_calls)
