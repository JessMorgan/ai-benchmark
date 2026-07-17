"""HTTP request helpers for the AI benchmark.

This module contains the low-level request logic (streaming and non-streaming)
used by ``benchmark_core.py``. Keeping it separate makes ``benchmark_core.py"
smaller and makes the request helpers easier to test and reason about.
"""
import json
import os
import threading
import time
from contextlib import contextmanager

import requests


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


def build_curl_cmd(model, prompt, max_tokens, stream, api_url, headers):
    """Build a curl command string for the given API request."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream
    }, ensure_ascii=False)
    return (
        f"curl -s -X POST '{api_url}' \\\n"
        f"  -H 'Authorization: Bearer {headers['Authorization']}' \\\n"
        f"  -H 'Content-Type: {headers['Content-Type']}' \\\n"
        f"  -d '{data}'"
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


def _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params, stream):
    """Build the JSON body for an API request."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
def _post_request_context(source_config, source, body, timeout, stream, log_path, log_label):
    """Make a POST request and yield the response, handling cleanup.

    Yields a tuple of ``(response, error, curl_cmd)``. ``response`` is the
    requests Response object on success, or ``None`` if an error occurred
    before or during the request. Cleanup (watchdog cancellation, active
    request tracking removal, response close) is performed automatically.
    """
    cfg = source_config.get(source, {})
    api_url = cfg.get("api_url", "http://localhost:11434/chat/completions")
    headers = cfg.get("headers", {"Content-Type": "application/json"})
    model = body.get("model", "")
    prompt = body["messages"][0]["content"] if body.get("messages") else ""
    max_tokens = body.get("max_tokens", 2048)
    curl_cmd = build_curl_cmd(model, prompt, max_tokens, stream, api_url, headers) if log_path else None
    resp = None
    # Use a short connection timeout so a stuck connect() returns quickly,
    # but keep the user's full timeout for reading so slow models are not
    # aborted prematurely.
    connect_timeout = min(float(timeout), 5.0)
    request_timeout = (connect_timeout, timeout)
    try:
        resp = requests.post(
            api_url, headers=headers, json=body, stream=True,
            timeout=request_timeout)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if log_path and curl_cmd:
            log_request_entry(log_path, curl_cmd, f"ERROR: {error}", log_label)
        yield None, error, curl_cmd
        return

    with _active_requests_lock:
        _active_requests.add(resp)
    watchdog = threading.Timer(timeout, resp.close)
    watchdog.daemon = True
    watchdog.start()
    try:
        if resp.status_code != 200:
            error = f"HTTP {resp.status_code}: {resp.text[:150]}"
            if log_path and curl_cmd:
                log_request_entry(log_path, curl_cmd,
                                  f"HTTP {resp.status_code}: {resp.text[:500]}", log_label)
            yield None, error, curl_cmd
        else:
            yield resp, None, curl_cmd
    finally:
        watchdog.cancel()
        with _active_requests_lock:
            _active_requests.discard(resp)
        resp.close()


def _parse_sse_line(line, first_tok, text, finish_reason, usage):
    """Parse a single Server-Sent Events line and update streaming state.

    Returns a tuple of ``(first_tok, text, finish_reason, usage, done)``.
    ``done`` is True when the ``[DONE]`` sentinel is encountered.
    """
    if not line.startswith("data: "):
        return first_tok, text, finish_reason, usage, False
    payload = line[6:]
    if payload.strip() == "[DONE]":
        return first_tok, text, finish_reason, usage, True
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return first_tok, text, finish_reason, usage, False
    if first_tok is None:
        first_tok = time.time()
    for ch in data.get("choices", []):
        text += ch.get("delta", {}).get("content", "")
        fr = ch.get("finish_reason")
        if fr:
            finish_reason = fr
    if "usage" in data:
        usage = data["usage"]
    return first_tok, text, finish_reason, usage, False


def stream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                   log_path=None, log_label=None, session_seed=0, temperature=None,
                   drop_params=None, stop_event=None):
    """Make a streaming chat-completion request and return parsed results."""
    start = time.time()
    first_tok = None
    text = ""
    error = None
    finish_reason = None
    usage = {}
    body = _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params, stream=True)
    with _post_request_context(source_config, source, body, timeout, True, log_path, log_label) as (resp, err, curl_cmd):
        if err:
            return text, first_tok, time.time(), err, finish_reason, usage
        for line in resp.iter_lines(decode_unicode=True):
            if stop_event and stop_event.is_set():
                error = "Cancelled"
                break
            if not line:
                continue
            first_tok, text, finish_reason, usage, done = _parse_sse_line(
                line, first_tok, text, finish_reason, usage)
            if done:
                break
        error = _check_total_timeout(start, timeout, error, finish_reason)
        _log_response(log_path, curl_cmd, text, log_label)
    return text, first_tok, time.time(), error, finish_reason, usage


def _read_response_body(resp, stop_event):
    """Read a non-streaming response body in chunks, honouring cancellation."""
    chunks = []
    for chunk in resp.iter_content(chunk_size=8192):
        if stop_event and stop_event.is_set():
            return None, "Cancelled"
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace"), None


def nonstream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                      log_path=None, log_label=None, session_seed=0, temperature=None,
                      drop_params=None, stop_event=None):
    """Make a non-streaming chat-completion request and return parsed results."""
    start = time.time()
    error = None
    text = ""
    usage = {}
    finish_reason = None
    body = _build_request_body(model, prompt, max_tokens, session_seed, temperature, drop_params, stream=False)
    raw_resp_text = None
    with _post_request_context(source_config, source, body, timeout, False, log_path, log_label) as (resp, err, curl_cmd):
        if err:
            return text, usage, time.time() - start, err, finish_reason
        raw_resp_text, read_error = _read_response_body(resp, stop_event)
        if read_error:
            return text, usage, time.time() - start, read_error, finish_reason
        data = json.loads(raw_resp_text)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        finish_reason = data.get("choices", [{}])[0].get("finish_reason")
        _log_response(log_path, curl_cmd, raw_resp_text, log_label)
    error = _check_total_timeout(start, timeout, error, finish_reason)
    return text, usage, time.time() - start, error, finish_reason
