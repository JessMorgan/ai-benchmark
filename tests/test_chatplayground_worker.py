"""Tests for the ChatPlayground.ai browser worker subprocess logic.

No real browser is launched here: the sync-Playwright driver is mocked, so the
suite stays offline and fast. Selector-based helpers are exercised with a fake
page that records the calls the worker makes. The worker's JSON-lines protocol
dispatch (``handle``) is tested with mocked browser operations.
"""

import io
import json
import unittest
from unittest import mock

import benchmark.chatplayground_worker as cpw

DEFAULT_BASE_URL = cpw.DEFAULT_BASE_URL


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


class _FakeRole:
    def __init__(self, page, name):
        self._page = page
        self._name = name

    def click(self):
        self._page.clicks.append(f"role:{self._name}")


class _FakeText:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


class _FakeLocator:
    def __init__(self, count=0):
        self._count = count

    def count(self):
        return self._count


class _FakePage:
    def __init__(self, models=("gpt-5.6-terra", "deepseek-v4-pro"), response_text="the answer",
                 stop_counts=(0,), assistant_count=1):
        self.url = DEFAULT_BASE_URL + "/"
        self.models = list(models)
        self.response_text = response_text
        self._stop_counts = list(stop_counts)
        self._assistant_count = assistant_count
        self.fills = {}
        self.clicks = []
        self.gotos = []

    def goto(self, url):
        self.gotos.append(url)

    def fill(self, selector, value):
        self.fills[selector] = value

    def get_by_role(self, role, name=None, exact=False):
        return _FakeRole(self, name)

    def get_by_text(self, text, exact=False):
        return _FakeText(count=self._assistant_count)

    def wait_for_selector(self, selector, timeout=None):
        return None

    def locator(self, selector):
        if selector == cpw.DEFAULT_SELECTORS["stop_generation"] and self._stop_counts:
            return _FakeLocator(count=self._stop_counts.pop(0))
        return _FakeLocator(count=0)

    def evaluate(self, js, arg=None):
        if "AI MODELS" in js:
            return self.models
        return self.response_text

    def title(self):
        return "ChatPlayground AI"


class TestBrowserSend(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cpw._close_session)

    def test_send_returns_buffered_text_and_navigates_to_model(self):
        page = _FakePage()
        with mock.patch.object(cpw, "_get_page", return_value=page):
            text, error, elapsed = cpw._browser_send(_cfg(), "gpt-5.6-terra", "hi", timeout=10)

        self.assertEqual(text, "the answer")
        self.assertIsNone(error)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(page.fills[cpw.DEFAULT_SELECTORS["prompt_input"]], "hi")
        self.assertIn("role:Send", page.clicks)
        # Single-model route is used.
        self.assertEqual(page.gotos[-1], DEFAULT_BASE_URL + "/chat/gpt-5.6-terra")

    def test_send_folds_system_prompt(self):
        page = _FakePage()
        with mock.patch.object(cpw, "_get_page", return_value=page):
            cpw._browser_send(_cfg(), "gpt-5.6-terra", "hi", timeout=10,
                              system_prompt="You are a coder.")

        self.assertEqual(
            page.fills[cpw.DEFAULT_SELECTORS["prompt_input"]],
            "You are a coder.\n\nhi",
        )

    def test_send_surfaces_login_errors(self):
        with mock.patch.object(cpw, "_get_page", side_effect=RuntimeError("login failed")):
            text, error, _elapsed = cpw._browser_send(_cfg(), "gpt-5.6-terra", "hi", timeout=10)

        self.assertEqual(text, "")
        self.assertIn("RuntimeError: login failed", error)

    def test_completion_polls_until_stop_disappears(self):
        page = _FakePage(stop_counts=(1, 0), assistant_count=0)
        with mock.patch.object(cpw, "_get_page", return_value=page):
            text, error, _elapsed = cpw._browser_send(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
        self.assertEqual(text, "the answer")
        self.assertIsNone(error)


class TestBrowserSession(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cpw._close_session)

    def test_close_session_tears_down_context_and_playwright(self):
        ctx = mock.MagicMock()
        pw = mock.MagicMock()
        cpw._state = {"context": ctx, "playwright": pw, "page": mock.MagicMock(), "key": ("u", "e")}
        cpw._close_session()
        ctx.close.assert_called_once()
        pw.stop.assert_called_once()
        self.assertIsNone(cpw._state)

    def test_close_session_noop_when_empty(self):
        cpw._state = None
        cpw._close_session()
        self.assertIsNone(cpw._state)

    def test_session_is_reused_across_sends(self):
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
                text, error, _elapsed = cpw._browser_send(_cfg(), "gpt-5.6-terra", "hi", timeout=10)
                self.assertEqual(text, "the answer")
                self.assertIsNone(error)

        # The browser is launched exactly once; the second request reuses it.
        self.assertEqual(fake_pw.chromium.launch.call_count, 1)
        self.assertEqual(fake_context.new_page.call_count, 1)

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
            cpw._get_page(_cfg(email="a@b.com"))
            cpw._get_page(_cfg(email="other@b.com"))
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
            mock.patch.object(cpw, "_login", side_effect=ValueError("bad creds")),
            self.assertRaises(ValueError),
        ):
            cpw._get_page(_cfg())
        self.assertIsNone(cpw._state)
        fake_pw.stop.assert_called_once()


class TestBrowserProbe(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cpw._close_session)

    def test_probe_reports_dom_and_models(self):
        page = _FakePage(models=("gpt-5.6-terra",))
        with mock.patch.object(cpw, "_get_page", return_value=page):
            info = cpw._browser_probe(_cfg())

        self.assertEqual(info["url"], DEFAULT_BASE_URL + "/")
        self.assertEqual(info["title"], "ChatPlayground AI")
        self.assertEqual(info["models"], ["gpt-5.6-terra"])

    def test_probe_surfaces_model_enumeration_errors(self):
        page = _FakePage()
        with mock.patch.object(cpw, "_get_page", return_value=page), \
             mock.patch.object(cpw, "_browser_list_models_locked",
                               side_effect=RuntimeError("no picker")):
            info = cpw._browser_probe(_cfg())
        self.assertIn("models_error", info)
        self.assertIn("no picker", info["models_error"])


class TestHandleDispatch(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cpw._close_session)

    def test_handle_send_ok(self):
        with mock.patch.object(cpw, "_browser_send", return_value=("hello", None, 1.5)):
            resp = cpw.handle({"id": 7, "op": "send", "config": _cfg(),
                               "model": "gpt-5.6-terra", "prompt": "hi", "timeout": 10})
        self.assertEqual(resp, {"id": 7, "ok": True, "text": "hello"})

    def test_handle_send_error(self):
        with mock.patch.object(cpw, "_browser_send",
                               return_value=("", "TimeoutError: boom", 1.5)):
            resp = cpw.handle({"id": 7, "op": "send", "config": _cfg(),
                               "model": "gpt-5.6-terra", "prompt": "hi", "timeout": 10})
        self.assertEqual(resp["ok"], False)
        self.assertIn("TimeoutError: boom", resp["error"])

    def test_handle_send_exception_becomes_error(self):
        with mock.patch.object(cpw, "_browser_send", side_effect=ValueError("bad")):
            resp = cpw.handle({"id": 1, "op": "send", "config": _cfg(), "prompt": "hi"})
        self.assertFalse(resp["ok"])
        self.assertIn("ValueError: bad", resp["error"])

    def test_handle_list_models(self):
        with mock.patch.object(cpw, "_browser_list_models", return_value=["a", "b"]):
            resp = cpw.handle({"id": 2, "op": "list_models", "config": _cfg()})
        self.assertEqual(resp, {"id": 2, "ok": True, "models": ["a", "b"]})

    def test_handle_probe(self):
        with mock.patch.object(cpw, "_browser_probe", return_value={"url": "u"}):
            resp = cpw.handle({"id": 3, "op": "probe", "config": _cfg()})
        self.assertEqual(resp, {"id": 3, "ok": True, "probe": {"url": "u"}})

    def test_handle_unknown_op(self):
        resp = cpw.handle({"id": 4, "op": "nope", "config": _cfg()})
        self.assertFalse(resp["ok"])
        self.assertIn("unknown op", resp["error"])

    def test_handle_send_without_timeout_defaults_zero(self):
        with mock.patch.object(cpw, "_browser_send", return_value=("x", None, 0.1)) as bs:
            resp = cpw.handle({"id": 5, "op": "send", "config": _cfg(),
                               "model": "m", "prompt": "p"})
        self.assertEqual(resp["ok"], True)
        self.assertEqual(bs.call_args.kwargs["timeout"], 0.0)


class TestMainLoop(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cpw._close_session)

    def test_main_serves_protocol_until_eof(self):
        stdin = io.StringIO(
            '{"id": 1, "op": "send", "config": {}, "model": "m", "prompt": "p", "timeout": 5}\n'
            "not-json\n"
            '{"id": 2, "op": "nope", "config": {}}\n'
            '"just-a-string"\n'
            "\n"
        )
        stdout = io.StringIO()
        with (
            mock.patch.object(cpw, "_browser_send", return_value=("hello", None, 0.1)),
            mock.patch("sys.stdin", stdin),
            mock.patch("sys.stdout", stdout),
            mock.patch.object(cpw, "_close_session") as close,
        ):
            cpw.main()
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0]["id"], 1)
        self.assertTrue(lines[0]["ok"])
        self.assertEqual(lines[0]["text"], "hello")
        self.assertEqual(lines[1]["id"], 2)
        self.assertFalse(lines[1]["ok"])
        self.assertIn("unknown op", lines[1]["error"])
        close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
