"""Tests for the ChatPlayground.ai Playwright-driven source adapter.

No real browser is launched here: the sync-Playwright driver is mocked, so the
suite stays offline and fast. Selector-based helpers are exercised with a fake
page that records the calls the adapter makes.
"""

import threading
import unittest
from unittest import mock

import benchmark.chatplayground as cp
from benchmark.http import nonstream_request, stream_request

DEFAULT_BASE_URL = cp.DEFAULT_BASE_URL


def _cfg(**overrides):
    base = {
        "api_protocol": "chatplayground",
        "base_url": DEFAULT_BASE_URL,
        "email": "a@b.com",
        "password": "hunter2",
        "headless": True,
        "selectors": {"settle_ms": 0},
    }
    base.update(overrides)
    return base


class _FakeLocator:
    def __init__(self, texts=(), inner="", count=0):
        self.texts = list(texts)
        self.inner = inner
        self._count = count
        self.first = self
        self.last = self

    def filter(self, has_text=None):
        return self

    def click(self):
        pass

    def count(self):
        return self._count

    def inner_text(self):
        return self.inner

    def all_inner_texts(self):
        return self.texts


class _FakePage:
    def __init__(self, models=("gpt-4o", "claude-sonnet"), response_text="the answer",
                 stop_counts=()):
        self.models = list(models)
        self.response_text = response_text
        self.stop_counts = list(stop_counts)
        self.url = DEFAULT_BASE_URL + "/"
        self.fills = {}
        self.clicks = []
        self.gotos = []

    def goto(self, url):
        self.gotos.append(url)

    def fill(self, selector, value):
        self.fills[selector] = value

    def click(self, selector):
        self.clicks.append(selector)

    def wait_for_selector(self, selector, timeout=None):
        return None

    def locator(self, selector):
        if selector == cp.DEFAULT_SELECTORS["stop_generation"] and self.stop_counts:
            return _FakeLocator(
                texts=self.models, inner=self.response_text,
                count=self.stop_counts.pop(0),
            )
        return _FakeLocator(texts=self.models, inner=self.response_text)

    def title(self):
        return "ChatPlayground AI"


class TestConfigHelpers(unittest.TestCase):
    def test_is_chatplayground(self):
        self.assertTrue(cp.is_chatplayground(_cfg()))
        self.assertFalse(cp.is_chatplayground({}))
        self.assertFalse(cp.is_chatplayground(None))
        self.assertFalse(cp.is_chatplayground({"api_protocol": "1min"}))

    def test_credentials_email_and_username_fallback(self):
        self.assertEqual(cp.credentials(_cfg()), ("a@b.com", "hunter2"))
        self.assertEqual(
            cp.credentials({"email": "x@y.z", "password": "pw"}), ("x@y.z", "pw")
        )
        self.assertEqual(
            cp.credentials({"username": "user", "password": "pw"}), ("user", "pw")
        )
        self.assertEqual(cp.credentials(None), ("", ""))

    def test_selectors_merge_overrides(self):
        merged = cp.selectors(_cfg(selectors={"settle_ms": 0, "prompt_input": "textarea#x"}))
        self.assertEqual(merged["settle_ms"], 0)
        self.assertEqual(merged["prompt_input"], "textarea#x")
        # Untouched keys keep their defaults.
        self.assertEqual(merged["email_input"], cp.DEFAULT_SELECTORS["email_input"])


class TestRequest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cp._close_session)

    def test_request_returns_buffered_text_and_sends_prompt(self):
        page = _FakePage()
        with mock.patch.object(cp, "_get_page", return_value=page):
            text, error, elapsed = cp.request(_cfg(), "gpt-4o", "hi", timeout=10)

        self.assertEqual(text, "the answer")
        self.assertIsNone(error)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(page.fills[cp.DEFAULT_SELECTORS["prompt_input"]], "hi")
        # Model trigger and send button were clicked.
        self.assertIn(cp.DEFAULT_SELECTORS["model_trigger"], page.clicks)
        self.assertIn(cp.DEFAULT_SELECTORS["send_button"], page.clicks)

    def test_request_folds_system_prompt(self):
        page = _FakePage()
        with mock.patch.object(cp, "_get_page", return_value=page):
            cp.request(_cfg(), "gpt-4o", "hi", timeout=10, system_prompt="You are a coder.")

        self.assertEqual(
            page.fills[cp.DEFAULT_SELECTORS["prompt_input"]],
            "You are a coder.\n\nhi",
        )

    def test_request_surfaces_login_errors(self):
        with mock.patch.object(cp, "_get_page", side_effect=RuntimeError("login failed")):
            text, error, _elapsed = cp.request(_cfg(), "gpt-4o", "hi", timeout=10)

        self.assertEqual(text, "")
        self.assertIn("RuntimeError: login failed", error)

    def test_request_cancel_becomes_error(self):
        page = _FakePage()
        stop = threading.Event()
        stop.set()
        with mock.patch.object(cp, "_get_page", return_value=page):
            text, error, _elapsed = cp.request(
                _cfg(), "gpt-4o", "hi", timeout=10, stop_event=stop
            )

        self.assertEqual(text, "")
        self.assertIn("cancelled", error)

    def test_list_models(self):
        page = _FakePage(models=("gpt-4o", "claude-sonnet", "gemini"))
        with mock.patch.object(cp, "_get_page", return_value=page):
            self.assertEqual(
                cp.list_models(_cfg()), ["gpt-4o", "claude-sonnet", "gemini"]
            )

    def test_probe_reports_dom_and_models(self):
        page = _FakePage(models=("gpt-4o",))
        with mock.patch.object(cp, "_get_page", return_value=page):
            info = cp.probe(_cfg())

        self.assertEqual(info["url"], DEFAULT_BASE_URL + "/")
        self.assertEqual(info["title"], "ChatPlayground AI")
        self.assertEqual(info["models"], ["gpt-4o"])


class TestSessionReuse(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cp._close_session)

    def test_session_is_reused_across_requests(self):
        page = _FakePage()
        fake_pw = mock.MagicMock()
        fake_browser = mock.MagicMock()
        fake_context = mock.MagicMock()
        fake_pw.start.return_value = fake_pw
        fake_pw.chromium.launch.return_value = fake_browser
        fake_browser.new_context.return_value = fake_context
        fake_context.new_page.return_value = page

        with mock.patch("playwright.sync_api.sync_playwright", return_value=fake_pw):
            for _ in range(2):
                text, error, _elapsed = cp.request(
                    _cfg(), "gpt-4o", "hi", timeout=10
                )
                self.assertEqual(text, "the answer")
                self.assertIsNone(error)

        # The browser is launched exactly once; the second request reuses it.
        self.assertEqual(fake_pw.chromium.launch.call_count, 1)
        self.assertEqual(fake_context.new_page.call_count, 1)


class TestHttpDelegation(unittest.TestCase):
    def test_nonstream_request_routes_chatplayground(self):
        cfg = {"cp": {"api_protocol": "chatplayground", "email": "a@b.com", "password": "pw"}}
        with mock.patch.object(cp, "request", return_value=("hello", None, 1.5)) as req:
            result = nonstream_request(cfg, timeout=5, model="gpt-4o", source="cp",
                                       prompt="hi", max_tokens=10)
        self.assertEqual(result.text, "hello")
        self.assertIsNone(result.error)
        self.assertEqual(result.gen_time, 1.5)
        req.assert_called_once()

    def test_stream_request_routes_chatplayground(self):
        cfg = {"cp": {"api_protocol": "chatplayground", "email": "a@b.com", "password": "pw"}}
        with mock.patch.object(cp, "request", return_value=("hello", None, 1.5)) as req:
            result = stream_request(cfg, timeout=5, model="gpt-4o", source="cp",
                                    prompt="hi", max_tokens=10)
        self.assertEqual(result.text, "hello")
        self.assertIsNone(result.error)
        # Buffered: no first-token time for browser sources.
        self.assertIsNone(result.first_tok)
        req.assert_called_once()


class TestSessionLifecycle(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cp._close_session)

    def test_close_session_tears_down_context_and_playwright(self):
        ctx = mock.MagicMock()
        pw = mock.MagicMock()
        cp._state = {"context": ctx, "playwright": pw, "page": mock.MagicMock(), "key": ("u", "e")}
        cp._close_session()
        ctx.close.assert_called_once()
        pw.stop.assert_called_once()
        self.assertIsNone(cp._state)

    def test_close_session_noop_when_empty(self):
        cp._state = None
        cp._close_session()
        self.assertIsNone(cp._state)

    def test_get_page_relogs_in_when_credentials_change(self):
        page = _FakePage()
        fake_pw = mock.MagicMock()
        fake_browser = mock.MagicMock()
        fake_context = mock.MagicMock()
        fake_pw.start.return_value = fake_pw
        fake_pw.chromium.launch.return_value = fake_browser
        fake_browser.new_context.return_value = fake_context
        fake_context.new_page.return_value = page

        with mock.patch("playwright.sync_api.sync_playwright", return_value=fake_pw):
            cp._get_page(_cfg(email="a@b.com"))
            cp._get_page(_cfg(email="other@b.com"))
        self.assertEqual(fake_pw.chromium.launch.call_count, 2)

    def test_get_page_login_failure_closes_and_raises(self):
        fake_pw = mock.MagicMock()
        fake_browser = mock.MagicMock()
        fake_context = mock.MagicMock()
        fake_pw.start.return_value = fake_pw
        fake_pw.chromium.launch.return_value = fake_browser
        fake_browser.new_context.return_value = fake_context
        fake_context.new_page.return_value = _FakePage()

        with (
            mock.patch("playwright.sync_api.sync_playwright", return_value=fake_pw),
            mock.patch.object(cp, "_login", side_effect=ValueError("bad creds")),
            self.assertRaises(ValueError),
        ):
            cp._get_page(_cfg())
        self.assertIsNone(cp._state)
        fake_pw.stop.assert_called_once()


class TestCompletionEdgeCases(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cp._close_session)

    def test_completion_without_stop_selector_uses_response(self):
        cfg = _cfg(selectors={"stop_generation": "", "settle_ms": 0})
        page = _FakePage()
        with mock.patch.object(cp, "_get_page", return_value=page):
            text, error, _elapsed = cp.request(cfg, "gpt-4o", "hi", timeout=10)
        self.assertEqual(text, "the answer")
        self.assertIsNone(error)

    def test_completion_polls_until_stop_disappears(self):
        page = _FakePage(stop_counts=(1, 0))
        cfg = _cfg(selectors={"settle_ms": 0})
        with mock.patch.object(cp, "_get_page", return_value=page):
            text, error, _elapsed = cp.request(cfg, "gpt-4o", "hi", timeout=10)
        self.assertEqual(text, "the answer")
        self.assertIsNone(error)

    def test_send_prompt_skips_model_when_empty(self):
        page = _FakePage()
        with mock.patch.object(cp, "_get_page", return_value=page):
            cp.request(_cfg(), "", "hi", timeout=10)
        self.assertNotIn(cp.DEFAULT_SELECTORS["model_trigger"], page.clicks)

    def test_probe_surfaces_model_enumeration_errors(self):
        page = _FakePage()
        with mock.patch.object(cp, "_get_page", return_value=page), \
             mock.patch.object(cp, "list_models", side_effect=RuntimeError("no picker")):
            info = cp.probe(_cfg())
        self.assertIn("models_error", info)
        self.assertIn("no picker", info["models_error"])

    def test_cli_probe_reads_env(self):
        env = {"CHATPLAYGROUND_EMAIL": "e", "CHATPLAYGROUND_PASSWORD": "p",
               "CHATPLAYGROUND_HEADLESS": "0"}
        with mock.patch.dict("os.environ", env, clear=True), \
             mock.patch.object(cp, "probe", return_value={"ok": True}) as pr:
            out = cp._cli_probe()
        self.assertEqual(out, {"ok": True})
        pr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
