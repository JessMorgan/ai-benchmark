"""ChatPlayground.ai browser worker subprocess.

The parent :mod:`benchmark.chatplayground` module proxies browser operations to
this module, which runs in its own interpreter process. Keeping Playwright
(and its greenlet/Chromium machinery) in a separate process means a native
crash in the browser stack can no longer take down the benchmark runner: the
parent detects the worker's death and surfaces it as a per-request error
instead of segfaulting the whole run.

The worker owns the Playwright sync session and serves a small JSON-lines
protocol over stdin/stdout:

* request (parent -> worker): ``{"id": N, "op": "send"|"list_models"|"probe",
  "config": {...}, ...}``
* response (worker -> parent): ``{"id": N, "ok": true, ...}`` or
  ``{"id": N, "ok": false, "error": "..."}``

The worker exits when its stdin reaches EOF (the parent closed the pipe),
which also tears down the browser session. Pure helpers (``credentials``,
``selectors``, ``DEFAULT_SELECTORS``) are imported from the parent module,
which never imports Playwright itself.
"""

import contextlib
import faulthandler
import json
import sys
import threading
import time

from . import chatplayground as _cp

DEFAULT_BASE_URL = _cp.DEFAULT_BASE_URL
DEFAULT_SELECTORS = _cp.DEFAULT_SELECTORS
credentials = _cp.credentials
selectors = _cp.selectors

# RLock: ``probe`` holds the lock while calling the model enumeration helper,
# and every browser operation is serialized under it so the single logged-in
# session is never touched by two in-flight requests at once.
_lock = threading.RLock()
_state = None

# JS that returns the text of the last assistant answer. The answer is a sibling
# of the ``ASSISTANT`` label inside its content wrapper; reading the wrapper's
# ``innerText`` (minus the label itself) preserves code/newline formatting that
# the plugin evaluators rely on.
_READ_RESPONSE_JS = """
(label) => {
  const labels = [...document.querySelectorAll('p')]
    .filter(p => (p.innerText || '').trim() === label);
  if (!labels.length) return '';
  const wrapper = labels[labels.length - 1].parentElement;
  if (!wrapper) return '';
  return wrapper.innerText.replace(label, '').trim();
}
"""

# JS that returns the model slugs from the sidebar's "AI MODELS" section. The
# heading is a leaf node whose parent holds the model links; scoping there keeps
# conversation-history links (also ``/chat/...``) out of the result.
_LIST_MODELS_JS = """
() => {
  const headings = [...document.querySelectorAll('*')]
    .filter(e => e.children.length === 0 && (e.innerText || '').trim() === 'AI MODELS');
  if (!headings.length) return [];
  const container = headings[0].parentElement;
  if (!container) return [];
  return [...container.querySelectorAll('a[href*="/chat/"]')]
    .map(a => a.getAttribute('href'))
    .filter(href => href && href.includes('/chat/'))
    .map(href => href.split('/chat/')[1].split('?')[0])
    .filter(slug => slug && slug !== 'new');
}
"""


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
    """Navigate to the Clerk login form and authenticate with email/password."""
    page.goto(base_url.rstrip("/") + (sel.get("login_url") or "/login"))
    page.fill(sel["email_input"], email)
    page.fill(sel["password_input"], password)
    page.get_by_role("button", name=sel["login_submit"], exact=True).click()
    # Wait until the SPA leaves the login form and renders the chat composer.
    page.wait_for_selector(sel["prompt_input"], timeout=sel.get("wait_timeout_ms", 30000))


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


def _submit_prompt(page, prompt, sel):
    """Type the prompt and send it."""
    page.fill(sel["prompt_input"], prompt)
    page.get_by_role("button", name=sel["send_button"], exact=True).click()


def _has_response(page, sel):
    """Return whether an assistant answer has appeared yet."""
    return page.get_by_text(sel["assistant_label"], exact=True).count() > 0


def _wait_for_completion(page, timeout, stop_event, sel):
    """Wait for the in-flight answer to finish, honouring ``stop_event``.

    The "Stop" affordance is present while a response is generating. A turn is
    complete once that affordance has appeared and then disappeared; a very fast
    response may never show it, so an assistant answer already being present is
    also treated as completion.
    """
    deadline = time.monotonic() + timeout
    stop = sel["stop_generation"]
    saw_stop = False
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return False
        with contextlib.suppress(Exception):
            if page.locator(stop).count() > 0:
                saw_stop = True
                time.sleep(0.25)
                continue
        if saw_stop or _has_response(page, sel):
            break
        time.sleep(0.25)
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
    return str(page.evaluate(_READ_RESPONSE_JS, sel["assistant_label"]) or "")


def _send_prompt(page, model, prompt, timeout, stop_event, cfg):
    """Run one chat turn on ``model`` and return the buffered answer text."""
    sel = selectors(cfg)
    base_url = cfg.get("base_url", DEFAULT_BASE_URL)
    chat_path = (sel.get("chat_path") or "/chat").rstrip("/")
    if model:
        page.goto(f"{base_url.rstrip('/')}{chat_path}/{model}")
    else:
        page.goto(base_url.rstrip("/"))
    page.wait_for_selector(sel["prompt_input"], timeout=sel.get("wait_timeout_ms", 30000))
    _submit_prompt(page, prompt, sel)
    if not _wait_for_completion(page, timeout, stop_event, sel):
        raise TimeoutError("ChatPlayground request timed out")
    return _read_response(page, sel)


def _browser_send(cfg, model, prompt, *, timeout, system_prompt=None):
    """Run one chat turn and return ``(text, error, elapsed_seconds)``."""
    started = time.time()
    text = ""
    error = None
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    try:
        with _lock:
            page = _get_page(cfg)
            text = _send_prompt(page, model, prompt, timeout, None, cfg)
    except Exception as exc:  # noqa: BLE001 - any browser failure becomes an error result
        error = f"{type(exc).__name__}: {exc}"
    return text, error, round(time.time() - started, 1)


def _browser_list_models_locked(page) -> list[str]:
    """Enumerate model slugs from an already-obtained page (lock held)."""
    slugs = page.evaluate(_LIST_MODELS_JS) or []
    return [str(slug) for slug in slugs if slug]


def _browser_list_models(cfg) -> list[str]:
    """Enumerate the model slugs exposed by the sidebar's "AI MODELS" list."""
    with _lock:
        page = _get_page(cfg)
        return _browser_list_models_locked(page)


def _browser_probe(cfg) -> dict:
    """Capture diagnostic DOM information for selector finalization."""
    with _lock:
        page = _get_page(cfg)
        info = {
            "url": page.url,
            "title": page.title(),
            "textarea_count": page.locator("textarea").count(),
            "input_count": page.locator("input").count(),
            "button_count": page.locator("button").count(),
        }
        try:
            info["models"] = _browser_list_models_locked(page)
        except Exception as exc:  # noqa: BLE001 - model enumeration is best-effort
            info["models_error"] = f"{type(exc).__name__}: {exc}"
        return info


def handle(msg) -> dict:
    """Dispatch one request message and return the response dict."""
    req_id = msg.get("id")
    op = msg.get("op")
    cfg = msg.get("config") or {}
    try:
        if op == "send":
            text, error, _elapsed = _browser_send(
                cfg,
                msg.get("model"),
                msg.get("prompt", ""),
                timeout=float(msg.get("timeout") or 0),
                system_prompt=msg.get("system_prompt"),
            )
            if error is not None:
                return {"id": req_id, "ok": False, "error": error}
            return {"id": req_id, "ok": True, "text": text}
        if op == "list_models":
            return {"id": req_id, "ok": True, "models": _browser_list_models(cfg)}
        if op == "probe":
            return {"id": req_id, "ok": True, "probe": _browser_probe(cfg)}
        return {"id": req_id, "ok": False, "error": f"unknown op: {op!r}"}
    except Exception as exc:  # noqa: BLE001 - any failure becomes an error response
        return {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    """Serve the JSON-lines request protocol until stdin closes."""
    # Dump the Python stack on a native crash (Playwright/greenlet/Chromium
    # segfault). The parent captures this worker's stderr and surfaces it in
    # the per-request error, turning an opaque crash into a diagnosable one.
    faulthandler.enable()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            resp = handle(msg)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    finally:
        _close_session()


if __name__ == "__main__":  # pragma: no cover - spawned by the parent proxy
    main()
