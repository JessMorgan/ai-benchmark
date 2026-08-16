"""ChatPlayground.ai interactive-web source adapter (Playwright-driven).

ChatPlayground.ai (https://web.chatplayground.ai) is a closed, JavaScript-
rendered web app with no public API: it authenticates with a username/password
and renders every chat interaction client-side. This module drives that UI with
Playwright — log in with the credentials configured on the source, select a
model, submit the prompt, and read back the completed (buffered) answer.

Playwright is imported lazily inside ``_get_page`` so the rest of the benchmark
(and the test suite, which mocks the driver) never needs a browser at import
time. All browser operations are serialized under a module lock: the sync
Playwright API is not thread-safe, and a single logged-in session is reused
across plugin tasks instead of re-authenticating for every request.
"""

import contextlib
import threading
import time

DEFAULT_BASE_URL = "https://web.chatplayground.ai"

# Best-effort CSS selectors for the single-page app. Every key can be overridden
# through the source's ``selectors`` mapping. The site is JavaScript-rendered,
# so these cannot be read from a static page; run ``probe(cfg)`` (or
# ``python -m benchmark.chatplayground --probe``) against the live site to
# capture the current selectors and adjust this mapping as needed.
DEFAULT_SELECTORS = {
    # Route the login form lives at (joined to ``base_url``).
    "login_url": "/login",
    "email_input": "input[type=email], input[name=email], input[autocomplete=username]",
    "password_input": "input[type=password]",
    "login_submit": "button[type=submit], button:has-text('Log in'), button:has-text('Sign in')",
    # Route the chat view lives at (joined to ``base_url``).
    "chat_url": "/",
    # Opens the model picker.
    "model_trigger": "button:has-text('Model'), [data-testid=model-select]",
    # A container selector matching the individual options in the picker.
    "model_option": "[role=option], [data-testid=model-option], li",
    # The prompt composer and its submit affordance.
    "prompt_input": "textarea, div[role=textbox], [contenteditable=true]",
    "send_button": "button[type=submit], button:has-text('Send')",
    # Shown only while a response is still generating; when it disappears the
    # answer is complete. Optional — when unset, completion is detected via the
    # response container plus a settle delay.
    "stop_generation": "button:has-text('Stop'), [data-testid=stop-generating]",
    # Container(s) holding completed assistant answers; the last one is read.
    "response": "[data-testid=assistant-message], .assistant-message, [data-message-author=assistant]",
    # Extra settle time (ms) after the stop affordance disappears.
    "settle_ms": 1500,
    # Overall selector wait timeout (ms).
    "wait_timeout_ms": 15000,
}

# RLock: ``probe`` holds the lock while calling ``list_models``, which also
# acquires it.
_lock = threading.RLock()
_state = None


def is_chatplayground(cfg) -> bool:
    """Return whether a source config selects the ChatPlayground protocol."""
    return isinstance(cfg, dict) and cfg.get("api_protocol") == "chatplayground"


def credentials(cfg) -> tuple[str, str]:
    """Return ``(email, password)`` from a ChatPlayground source config."""
    if not isinstance(cfg, dict):
        return "", ""
    email = cfg.get("email") or cfg.get("username") or ""
    password = cfg.get("password") or ""
    return str(email), str(password)


def selectors(cfg) -> dict:
    """Merge the default selectors with any per-source ``selectors`` overrides."""
    merged = dict(DEFAULT_SELECTORS)
    if isinstance(cfg, dict) and isinstance(cfg.get("selectors"), dict):
        merged.update(cfg["selectors"])
    return merged


def _close_session():
    """Tear down the cached browser session, if any."""
    global _state
    if _state is None:
        return
    for closer in ("context", "playwright"):
        obj = _state.get(closer)
        if obj is not None:
            with contextlib.suppress(Exception):
                obj.stop() if closer == "playwright" else obj.close()
    _state = None


def _login(page, email, password, base_url, sel):
    """Navigate to the login form and authenticate with email/password."""
    page.goto(base_url.rstrip("/") + (sel.get("login_url") or "/login"))
    page.fill(sel["email_input"], email)
    page.fill(sel["password_input"], password)
    page.click(sel["login_submit"])
    # Wait until the SPA leaves the login form. The chat view is the success
    # signal; we don't assert a specific URL because the app may use hashes.
    page.wait_for_selector(sel["prompt_input"], timeout=sel.get("wait_timeout_ms", 15000))


def _get_page(cfg):
    """Return a logged-in page, reusing (or re-establishing) the cached session."""
    global _state
    email, password = credentials(cfg)
    base_url = cfg.get("base_url", DEFAULT_BASE_URL)
    key = (base_url, email)
    if _state is not None and _state.get("key") != key:
        _close_session()
    if _state is None:
        from playwright.sync_api import sync_playwright  # lazy import

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=bool(cfg.get("headless", True)))
        context = browser.new_context()
        page = context.new_page()
        # Register the session before logging in so a login failure tears down
        # the freshly-launched browser via ``_close_session`` instead of leaking it.
        _state = {
            "playwright": pw,
            "browser": browser,
            "context": context,
            "page": page,
            "key": key,
        }
        try:
            _login(page, email, password, base_url, selectors(cfg))
        except Exception:
            _close_session()
            raise
    return _state["page"]


def _select_model(page, model, sel):
    """Open the model picker and choose ``model`` (best-effort by visible text)."""
    page.click(sel["model_trigger"])
    page.wait_for_selector(sel["model_option"], timeout=sel.get("wait_timeout_ms", 15000))
    page.locator(sel["model_option"]).filter(has_text=model).first.click()


def _submit_prompt(page, prompt, sel):
    """Type the prompt and send it."""
    page.fill(sel["prompt_input"], prompt)
    page.click(sel["send_button"])


def _wait_for_completion(page, timeout, stop_event, sel):
    """Wait for the in-flight answer to finish, honouring ``stop_event``."""
    deadline = time.monotonic() + timeout
    stop = sel.get("stop_generation")
    if stop:
        # Generation starts when the stop affordance appears...
        with contextlib.suppress(Exception):
            page.wait_for_selector(stop, timeout=sel.get("wait_timeout_ms", 15000))
        # ...and finishes when it detaches/hides. Poll so a cancelled run can
        # interrupt the wait.
        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                return False
            if page.locator(stop).count() == 0:
                break
            time.sleep(0.25)
    else:
        page.wait_for_selector(sel["response"], timeout=sel.get("wait_timeout_ms", 15000))
    settle = int(sel.get("settle_ms", 1500))
    if settle > 0:
        end = min(time.monotonic() + settle / 1000.0, deadline)
        while time.monotonic() < end:
            if stop_event is not None and stop_event.is_set():
                return False
            time.sleep(0.1)
    return True


def _read_response(page, sel):
    """Return the text of the last completed assistant message."""
    locator = page.locator(sel["response"]).last
    return locator.inner_text()


def _send_prompt(page, model, prompt, timeout, stop_event, cfg):
    """Run one chat turn and return the buffered answer text."""
    sel = selectors(cfg)
    base_url = cfg.get("base_url", DEFAULT_BASE_URL)
    page.goto(base_url.rstrip("/") + (sel.get("chat_url") or "/"))
    page.wait_for_selector(sel["prompt_input"], timeout=sel.get("wait_timeout_ms", 15000))
    if model:
        _select_model(page, model, sel)
    _submit_prompt(page, prompt, sel)
    if not _wait_for_completion(page, timeout, stop_event, sel):
        raise TimeoutError("ChatPlayground request cancelled")
    return _read_response(page, sel)


def request(cfg, model, prompt, *, timeout, stop_event=None, system_prompt=None):
    """Send ``prompt`` through a logged-in ChatPlayground session.

    Returns ``(text, error, elapsed_seconds)``. The whole browser turn is
    serialized under the module lock; the answer is buffered (no per-token
    streaming), and an optional ``system_prompt`` is folded into the prompt.
    """
    started = time.time()
    text = ""
    error = None
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    try:
        with _lock:
            page = _get_page(cfg)
            text = _send_prompt(page, model, prompt, timeout, stop_event, cfg)
    except Exception as exc:  # noqa: BLE001 - any browser failure becomes an error result
        error = f"{type(exc).__name__}: {exc}"
    return text, error, round(time.time() - started, 1)


def list_models(cfg) -> list[str]:
    """Enumerate the model names exposed by the UI's model picker."""
    with _lock:
        page = _get_page(cfg)
        sel = selectors(cfg)
        page.click(sel["model_trigger"])
        page.wait_for_selector(sel["model_option"], timeout=sel.get("wait_timeout_ms", 15000))
        names = page.locator(sel["model_option"]).all_inner_texts()
    return [name.strip() for name in names if name and name.strip()]


def probe(cfg) -> dict:
    """Capture diagnostic DOM information for selector finalization."""
    with _lock:
        page = _get_page(cfg)
        info = {
            "url": page.url,
            "title": page.title(),
            "textarea_count": page.locator("textarea").count(),
            "input_count": page.locator("input").count(),
            "button_count": page.locator("button").count(),
            "buttons": page.locator("button").all_inner_texts(),
        }
        try:
            info["models"] = list_models(cfg)
        except Exception as exc:  # noqa: BLE001 - model enumeration is best-effort
            info["models_error"] = f"{type(exc).__name__}: {exc}"
        return info


def _cli_probe() -> dict:
    """Run a probe from a minimal source config built out of environment vars."""
    import os

    cfg = {
        "api_protocol": "chatplayground",
        "base_url": os.environ.get("CHATPLAYGROUND_BASE_URL", DEFAULT_BASE_URL),
        "email": os.environ.get("CHATPLAYGROUND_EMAIL", ""),
        "password": os.environ.get("CHATPLAYGROUND_PASSWORD", ""),
        "headless": os.environ.get("CHATPLAYGROUND_HEADLESS", "1") == "1",
    }
    return probe(cfg)


if __name__ == "__main__":  # pragma: no cover - diagnostic entry point
    import json

    result = _cli_probe()
    print(json.dumps(result, indent=2, default=str))
