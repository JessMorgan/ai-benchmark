"""ChatPlayground.ai interactive-web source adapter (Playwright-driven).

ChatPlayground.ai (https://web.chatplayground.ai) is a closed, JavaScript-
rendered web app with no public API: it authenticates with a username/password
(Clerk) and renders every chat interaction client-side. This module drives that
UI with Playwright — log in with the credentials configured on the source,
navigate to a single-model chat route, submit the prompt, and read back the
completed (buffered) answer.

Each model is addressed by the slug used in the site's ``/chat/<slug>`` route
(e.g. ``deepseek-v4-pro``, ``gpt-5.6-terra``, ``gemini-3-flash``). Use
``list_models(cfg)`` (or ``python -m benchmark.chatplayground``) to enumerate
the slugs exposed by the sidebar's "AI MODELS" list.

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

# Selectors for the single-page app, captured against the live site. Every key
# can be overridden through the source's ``selectors`` mapping. ``login_submit``
# and ``send_button`` are accessible names (matched exactly via
# ``get_by_role``), not CSS selectors, because the buttons share a text
# substring with other controls ("Continue with Google", "Send" vs nothing).
DEFAULT_SELECTORS = {
    # Route the Clerk login form lives at (joined to ``base_url``).
    "login_url": "/login",
    "email_input": "input[name=identifier]",
    "password_input": "input[name=password]",
    # Accessible name of the Clerk submit button ("Continue", not "Continue with Google").
    "login_submit": "Continue",
    # Prefix for single-model chat routes; the model slug is appended.
    "chat_path": "/chat",
    # The prompt composer and its submit affordance.
    "prompt_input": "textarea[name=input]",
    "send_button": "Send",
    # Shown only while a response is still generating; its disappearance marks
    # completion.
    "stop_generation": "button:has-text('Stop')",
    # Exact label preceding each assistant answer in the transcript.
    "assistant_label": "ASSISTANT",
    # Extra settle time (ms) after the stop affordance disappears.
    "settle_ms": 1500,
    # Overall selector wait timeout (ms).
    "wait_timeout_ms": 30000,
}

# RLock: ``probe`` holds the lock while calling ``list_models``, which also
# acquires it.
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
    """Enumerate the model slugs exposed by the sidebar's "AI MODELS" list."""
    with _lock:
        page = _get_page(cfg)
        slugs = page.evaluate(_LIST_MODELS_JS) or []
    return [str(slug) for slug in slugs if slug]


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
        }
        try:
            info["models"] = list_models(cfg)
        except Exception as exc:  # noqa: BLE001 - model enumeration is best-effort
            info["models_error"] = f"{type(exc).__name__}: {exc}"
        return info


def config_from_env() -> dict:
    """Build a ChatPlayground source config from environment variables.

    ``CHATPLAYGROUND_EMAIL``/``CHATPLAYGROUND_PASSWORD`` supply the Clerk
    credentials; ``CHATPLAYGROUND_BASE_URL`` and ``CHATPLAYGROUND_HEADLESS``
    (default ``"1"``) override the site and browser mode.
    """
    import os

    return {
        "api_protocol": "chatplayground",
        "base_url": os.environ.get("CHATPLAYGROUND_BASE_URL", DEFAULT_BASE_URL),
        "email": os.environ.get("CHATPLAYGROUND_EMAIL", ""),
        "password": os.environ.get("CHATPLAYGROUND_PASSWORD", ""),
        "headless": os.environ.get("CHATPLAYGROUND_HEADLESS", "1") == "1",
    }


def _complete_source_config(source_cfg: dict) -> dict:
    """Return ``source_cfg`` with the browser-safe defaults benchmark needs.

    Browser work is serialized under a module lock regardless of scheduler
    settings, but pinning both thread limits to 1 (and skipping the HTTP-only
    preload probe) keeps the scheduler from queueing parallel browser turns.
    """
    cfg = dict(source_cfg)
    cfg.setdefault("api_protocol", "chatplayground")
    cfg.setdefault("model_thread_limit", 1)
    cfg.setdefault("plugin_thread_limit", 1)
    cfg.setdefault("preload", False)
    return cfg


def generate_config(source_cfg: dict | None = None, models: list[str] | None = None) -> dict:
    """Return a ready-to-run benchmark config dict for a ChatPlayground source.

    When ``source_cfg`` is omitted it is built from the environment (see
    :func:`config_from_env`). When ``models`` is omitted the model slugs are
    enumerated from the live sidebar's "AI MODELS" list. Each discovered slug
    becomes a ``models`` entry pointing at the ``ChatPlayground`` source.
    """
    if source_cfg is None:
        source_cfg = config_from_env()
    if not credentials(source_cfg)[0]:
        raise ValueError(
            "ChatPlayground credentials are missing; set CHATPLAYGROUND_EMAIL "
            "and CHATPLAYGROUND_PASSWORD"
        )
    if models is None:
        models = list_models(source_cfg)
    if not models:
        raise RuntimeError("No ChatPlayground models discovered from the UI")
    return {
        "output_dir": "benchmark-results",
        "timeout": 600,
        "token_levels": [16384],
        "plugins_whitelist": [],
        "plugins_blacklist": [],
        "sources": {"ChatPlayground": _complete_source_config(source_cfg)},
        "models": {slug: "ChatPlayground" for slug in models},
    }


def _cli_probe() -> dict:
    """Run a probe from a minimal source config built out of environment vars."""
    return probe(config_from_env())


if __name__ == "__main__":  # pragma: no cover - diagnostic entry point
    import json

    result = _cli_probe()
    print(json.dumps(result, indent=2, default=str))
