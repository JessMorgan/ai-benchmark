"""HTTP request helpers for the AI benchmark.

This module contains the low-level request logic (streaming and non-streaming)
used by ``benchmark_core.py``. Keeping it separate makes ``benchmark_core.py"
smaller and makes the request helpers easier to test and reason about.
"""
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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

import requests


@dataclass(frozen=True)
class PostRequestResult:
    """Outcome yielded by the HTTP request context manager."""

    response: Optional[requests.Response]
    error: Optional[str]
    curl_cmd: Optional[str]


@dataclass(frozen=True)
class SSEParseResult:
    """Updated state after parsing one SSE line."""

    first_tok: Optional[float]
    text: str
    think_text: str
    finish_reason: Optional[str]
    usage: dict[str, Any]
    done: bool


@dataclass(frozen=True)
class StreamResult:
    """Structured result returned by :func:`stream_request`."""

    text: str
    think_text: str
    first_tok: Optional[float]
    stream_end: float
    error: Optional[str]
    finish_reason: Optional[str]
    usage: dict[str, Any]


@dataclass(frozen=True)
class NonStreamResult:
    """Structured result returned by :func:`nonstream_request`."""

    text: str
    think_text: str
    usage: dict[str, Any]
    gen_time: float
    error: Optional[str]
    finish_reason: Optional[str]


@dataclass(frozen=True)
class ResponseBodyResult:
    """Result of reading a non-streaming response body."""

    text: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class _StreamLineError:
    """Internal sentinel for an exception raised while reading SSE lines."""

    error: str


def _safe_iter_lines(resp: requests.Response):
    """Yield SSE lines while converting iterator failures to a sentinel."""
    try:
        yield from resp.iter_lines(decode_unicode=True)
    except Exception as exc:
        yield _StreamLineError(f"{type(exc).__name__}: {exc}")


_log_lock = threading.Lock()

# Active HTTP responses so Ctrl+C can close them and unblock plugin threads.
_active_requests_lock = threading.Lock()
_active_requests: set = set()


def close_active_requests():
    """Close all in-flight HTTP responses to unblock worker threads."""
    with _active_requests_lock:
        for resp in list(_active_requests):
            try:
                resp.close()
            except Exception:
                pass


# HTTP 429 activity tracked for the TUI live status section. Keyed by
# ``(source, model, pid)`` so the dashboard can tell the operator not only
# *that* a source/model is in backoff but also *which* plugin is blocked.
# All mutations and reads happen under ``_429_lock``.
_429_lock = threading.Lock()
_429_stats: dict = {
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


def get_429_stats() -> dict:
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


def reset_429_stats():
    """Reset 429 activity tracking. Intended for unit tests only."""
    global _429_stats
    with _429_lock:
        _429_stats = {"total_retries": 0, "sleeping": {}, "plugin_stats": {}}


def _set_429_sleep(source, model, pid, wake_ts, attempts, max_attempts, delay):
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


def _clear_429_sleep(source, model, pid):
    """Remove a ``(source, model, pid)`` entry from the 429 sleeping set, if any."""
    with _429_lock:
        _429_stats["sleeping"].pop((source, model, pid), None)


def fetch_models_v1(base_url, api_key=None):
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


def build_curl_cmd(model, prompt, max_tokens, stream, api_url, headers, system_prompt=None):
    """Build a curl command string for the given API request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream
    }, ensure_ascii=False)
    auth_value = headers.get("Authorization", "")
    content_type = headers.get("Content-Type", "application/json")
    auth_header = (
        f"  -H {shlex.quote('Authorization: ' + auth_value)} \\\n"
        if auth_value else ""
    )
    return (
        f"curl -s -X POST {shlex.quote(api_url)} \\\n"
        f"{auth_header}"
        f"  -H {shlex.quote('Content-Type: ' + content_type)} \\\n"
        f"  -d {shlex.quote(data)}"
    )


def log_request_entry(log_path, curl_cmd, response_body, request_label=None):
    """Append a curl command and response body to the log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with _log_lock:
        with open(log_path, 'a') as f:
            if request_label:
                f.write(f"\n# === {request_label} ===\n")
            f.write(f"{curl_cmd}\n\n")
            f.write(f"{response_body}\n")
            f.write("\n" + "-" * 60 + "\n")


def _check_total_timeout(start_time, timeout, error, finish_reason=None):
    """Return a timeout error if the overall request duration was exceeded."""
    if not error and not finish_reason and time.time() - start_time > timeout:
        return f"Total timeout ({timeout}s) exceeded"
    return error


def _log_response(log_path, curl_cmd, response_body, log_label):
    """Write the response body to the request log if logging is enabled."""
    if log_path and curl_cmd:
        log_request_entry(log_path, curl_cmd, response_body or "(empty response)", log_label)


def _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params, stream,
                        system_prompt=None):
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
    for p in drop_params or []:
        body.pop(p, None)
    return body


@contextmanager
def _post_request_context(source_config, source, body, timeout, stream, log_path, log_label,
                          stop_event=None, pid=None, on_retry=None) -> Iterator[PostRequestResult]:
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
    api_url = cfg.get("api_url", "http://localhost:11434/chat/completions")
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
    curl_cmd = build_curl_cmd(model, prompt, max_tokens, stream, api_url, headers, system_prompt=system_prompt) if log_path else None
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
                except Exception as exc:
                    # A buggy observer must not abort the retry loop, but
                    # swallowing it silently makes state bugs hard to find.
                    try:
                        sys.stderr.write(
                            f"benchmark_http: on_retry observer failed: {exc}\n"
                        )
                        traceback.print_exc(file=sys.stderr)
                    except Exception:
                        pass

            try:
                resp = requests.post(
                    api_url, headers=headers, json=body, stream=True,
                    timeout=request_timeout)
            except Exception as e:
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
                watchdog = threading.Timer(timeout, close_fn)
                watchdog.daemon = True
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
            try:
                watchdog.cancel()
            except Exception:
                pass
            with _active_requests_lock:
                _active_requests.discard(resp)
            try:
                resp.close()
            except Exception:
                pass
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
            try:
                watchdog.cancel()
            except Exception:
                pass
        if resp is not None:
            try:
                with _active_requests_lock:
                    _active_requests.discard(resp)
            except Exception:
                pass
            try:
                resp.close()
            except Exception:
                pass


def _parse_sse_line(line: str, first_tok: Optional[float], text: str,
                    think_text: str, finish_reason: Optional[str],
                    usage: dict[str, Any]) -> SSEParseResult:
    """Parse a single Server-Sent Events line and update streaming state.

    Returns an :class:`SSEParseResult`. ``done`` is True when the ``[DONE]``
    sentinel is encountered.

    ``think_text`` accumulates ``reasoning_content`` from SSE deltas so
    thinking-capable models' chain-of-thought is preserved separately from
    the final content. Models that don't emit ``reasoning_content`` leave
    ``think_text`` unchanged (empty string).
    """
    if not line.startswith("data: "):
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False)
    payload = line[6:]
    if payload.strip() == "[DONE]":
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, True)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False)
    if first_tok is None:
        first_tok = time.time()
    for ch in data.get("choices", []):
        delta = ch.get("delta", {})
        text += delta.get("content", "")
        think_text += delta.get("reasoning_content", "")
        fr = ch.get("finish_reason")
        if fr:
            finish_reason = fr
    if "usage" in data:
        usage = data["usage"]
    return SSEParseResult(first_tok, text, think_text, finish_reason, usage, False)


def stream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                   log_path=None, log_label=None, session_seed=0, temperature=None,
                   drop_params=None, stop_event=None, system_prompt=None,
                   on_chunk: Optional[Callable[[str], None]] = None,
                   on_think_chunk: Optional[Callable[[str], None]] = None,
                   pid: Optional[str] = None,
                   on_retry: Optional[Callable[[], None]] = None) -> StreamResult:
    """Make a streaming chat-completion request and return parsed results.

    Returns a :class:`StreamResult` with named fields for the assembled text,
    timing, finish reason, usage, and any transport error.
    ``think_text`` contains any reasoning/thinking content emitted by
    thinking-capable models (conversational ``reasoning_content`` field from
    SSE deltas). For standard models it is an empty string.

    ``on_chunk`` (optional) is called once per parsed SSE delta with the new
    text accumulated in that iteration, so callers (notably the live TUI)
    can update per-plugin byte/tok counts as tokens arrive instead of
    waiting for the full response. Exceptions raised by ``on_chunk`` are
    swallowed so a buggy observer cannot abort the stream read -- the TUI
    is a display concern, not a correctness concern.
    """
    start = time.time()
    first_tok = None
    text = ""
    think_text = ""
    error = None
    finish_reason = None
    usage = {}
    body = _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params,
                               stream=True, system_prompt=system_prompt)
    with _post_request_context(source_config, source, body, timeout, True, log_path, log_label,
                               stop_event=stop_event, pid=pid, on_retry=on_retry) as request:
        if request.error:
            return StreamResult(text, think_text, first_tok, time.time(),
                                request.error, finish_reason, usage)
        resp = request.response
        if resp is None:
            return StreamResult(text, think_text, first_tok, time.time(),
                                "HTTP request returned no response", finish_reason, usage)
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
                    line, first_tok, text, think_text, finish_reason, usage)
            except Exception as exc:
                error = f"SSE parse error: {type(exc).__name__}: {exc}"
                break
            first_tok = parsed.first_tok
            text = parsed.text
            think_text = parsed.think_text
            finish_reason = parsed.finish_reason
            usage = parsed.usage
            if parsed.done:
                break
            # Notify the caller of the content delta accumulated in this
            # iteration. We compute the delta from ``text`` length so a
            # single SSE data event that pumps multiple ``choices`` deltas
            # still produces one observer call with the joined delta,
            # which keeps the callback rate proportional to wall-clock
            # arrivals rather than choice array sizes.
            if on_chunk is not None and len(text) > prev_text_len:
                delta = text[prev_text_len:]
                prev_text_len = len(text)
                try:
                    on_chunk(delta)
                except Exception:
                    # A buggy observer must not abort the stream read.
                    pass
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
            if on_think_chunk is not None and len(think_text) > prev_think_len:
                think_delta = think_text[prev_think_len:]
                prev_think_len = len(think_text)
                try:
                    on_think_chunk(think_delta)
                except Exception:
                    # A buggy observer must not abort the stream read.
                    pass
        error = _check_total_timeout(start, timeout, error, finish_reason)
        _log_response(log_path, request.curl_cmd, text, log_label)
    return StreamResult(text, think_text, first_tok, time.time(), error,
                        finish_reason, usage)


def _read_response_body(resp: requests.Response, stop_event) -> ResponseBodyResult:
    """Read a non-streaming response body in chunks, honouring cancellation."""
    chunks = []
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            if stop_event and stop_event.is_set():
                return ResponseBodyResult(None, "Cancelled")
            if chunk:
                chunks.append(chunk)
    except Exception as exc:
        return ResponseBodyResult(None, f"{type(exc).__name__}: {exc}")
    return ResponseBodyResult(
        b"".join(chunks).decode("utf-8", errors="replace"), None
    )


def nonstream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                      log_path=None, log_label=None, session_seed=0, temperature=None,
                      drop_params=None, stop_event=None, system_prompt=None,
                      pid: Optional[str] = None,
                      on_retry: Optional[Callable[[], None]] = None) -> NonStreamResult:
    """Make a non-streaming chat-completion request and return parsed results.

    Returns a :class:`NonStreamResult` with named fields for response text,
    timing, finish reason, usage, and any transport error.
    ``think_text`` contains any reasoning/thinking content from the API
    response (``message.reasoning_content`` field). For standard models it
    is an empty string.
    """
    start = time.time()
    error = None
    text = ""
    think_text = ""
    usage = {}
    finish_reason = None
    body = _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params,
                               stream=False, system_prompt=system_prompt)
    raw_resp_text = None
    with _post_request_context(source_config, source, body, timeout, False, log_path, log_label,
                               stop_event=stop_event, pid=pid, on_retry=on_retry) as request:
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
        try:
            data = json.loads(raw_resp_text)
            text = data["choices"][0]["message"]["content"]
            think_text = data["choices"][0]["message"].get("reasoning_content", "")
            usage = data.get("usage", {})
            finish_reason = data.get("choices", [{}])[0].get("finish_reason")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            error = f"Invalid completion response: {type(exc).__name__}: {exc}"
        _log_response(log_path, request.curl_cmd, raw_resp_text, log_label)
    error = _check_total_timeout(start, timeout, error, finish_reason)
    return NonStreamResult(text, think_text, usage, time.time() - start,
                           error, finish_reason)
