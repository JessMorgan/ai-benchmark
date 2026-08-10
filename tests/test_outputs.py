"""Tests for the benchmark.outputs wrapper module.

These cover the thin delegation layer (``gen_markdown``/``gen_csv``/
``gen_html``/``gen_pdf``/``_save_outputs``) that forwards to the output
plugins discovered from ``plugins/outputs/``, plus the ``_get_output_plugin``
lookup and the helper functions.
"""
import tempfile
import unittest
from unittest import mock

from benchmark.outputs import (
    _get_output_plugin,
    _numeric_score,
    _plugin_token_counts,
    _plugin_total_score,
    _save_outputs,
    gen_csv,
    gen_html,
    gen_markdown,
    gen_pdf,
    sanitize_filename,
)
from plugins import discover_plugins


class TestSanitizeFilename(unittest.TestCase):
    def test_strips_illegal_chars_and_collapses_whitespace(self):
        self.assertEqual(sanitize_filename("My Model (v2)"), "My_Model_(v2)")
        self.assertEqual(sanitize_filename("a/b\\c:d"), "a_b_c_d")
        self.assertEqual(sanitize_filename("  spaced   out  "), "spaced_out")
        self.assertEqual(sanitize_filename("plain"), "plain")


class _FakePlugin:
    """Minimal task-plugin stand-in carrying id/version/name/max_score."""

    def __init__(self, pid="rate-limiter", version="1.0.0"):
        self.id = pid
        self.version = version
        self.name = pid.replace("-", " ").title()
        self.max_score = 20.0


class TestOutputHelpers(unittest.TestCase):
    def test_plugin_total_score_returns_normalized_mean(self):
        plugins = [_FakePlugin("a"), _FakePlugin("b"), _FakePlugin("c")]
        result = {"a_score": 10, "b_score": "fail", "c_score": 8}
        self.assertEqual(_plugin_total_score(result, plugins), 9)

    def test_plugin_total_score_missing_scores_is_blank(self):
        result = {}
        self.assertIsNone(_plugin_total_score(result, [_FakePlugin("a")]))

    def test_plugin_token_counts_modern_split(self):
        result = {
            "p_output_tokens": 100,
            "p_thinking_tokens": 23,
            "p_total_tokens": 123,
        }
        self.assertEqual(_plugin_token_counts(result, "p"), (23, 100, 123))

    def test_plugin_token_counts_legacy_derives_split(self):
        result = {"p_output_tokens": 100}
        self.assertEqual(_plugin_token_counts(result, "p"), (0, 100, 100))

    def test_plugin_token_counts_legacy_fail_mirrors_content(self):
        result = {"p_output_tokens": "fail"}
        self.assertEqual(_plugin_token_counts(result, "p"), ("fail", "fail", "fail"))

    def test_plugin_token_counts_missing_returns_dashes(self):
        self.assertEqual(_plugin_token_counts({}, "p"), ("-", "-", "-"))

    def test_numeric_score_falls_back_to_default(self):
        self.assertEqual(_numeric_score({"p_score": 5}, "p"), 5)
        self.assertEqual(_numeric_score({"p_score": "fail"}, "p"), 0)
        self.assertEqual(_numeric_score({}, "p", default=-1), -1)


class TestGetOutputPlugin(unittest.TestCase):
    def test_returns_matching_plugin(self):
        plugin = _get_output_plugin("output-markdown")
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.id, "output-markdown")

    def test_unknown_id_returns_none(self):
        self.assertIsNone(_get_output_plugin("no-such-output-plugin"))


class TestGenWrappers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.active_plugins = discover_plugins()
        cls.results = [
            {
                "model": "model-a",
                "state_key": "model-a",
                "api_model": "model-a",
                "source": "Local",
                "runner": "http",
                "status": "ok",
                "stream_ok": True,
                "ttft": 1.1,
                "total_time": 8.0,
                "rate-limiter_score": 60,
                "rate-limiter_response_time": 3.2,
                "rate-limiter_output_tokens": 120,
                "rate-limiter_thinking_tokens": 10,
                "rate-limiter_total_tokens": 130,
                "rate-limiter_tps": 37.5,
                "rate-limiter_stream_ok": True,
            }
        ]

    def test_gen_markdown_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen_markdown(self.results, self.active_plugins, output_dir=tmpdir)
            self.assertTrue(path)
            with open(path, encoding="utf-8") as handle:
                self.assertIn("model-a", handle.read())

    def test_gen_csv_returns_rows(self):
        # Without an output_dir the CSV plugin returns its content inline
        # rather than writing a file.
        content = gen_csv(self.results, self.active_plugins)
        self.assertIsInstance(content, str)
        self.assertIn("model-a", content)
        self.assertIn("rate-limiter_Score", content)

    def test_gen_html_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen_html(self.results, self.active_plugins, output_dir=tmpdir)
            self.assertTrue(path)
            with open(path, encoding="utf-8") as handle:
                self.assertIn("model-a", handle.read())

    def test_gen_pdf_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen_pdf(self.results, self.active_plugins, tmpdir)
            self.assertTrue(path)
            with open(path, "rb") as handle:
                self.assertTrue(handle.read().startswith(b"%PDF"))

    def test_gen_markdown_delegates_without_output_dir(self):
        # The wrappers must tolerate output_dir=None (the plugin picks its own
        # default location or returns None) without raising.
        with tempfile.TemporaryDirectory(), \
                mock.patch("benchmark.outputs._get_output_plugin") as getter:
            fake = mock.MagicMock()
            fake.id = "output-markdown"
            fake.generate.return_value = None
            getter.return_value = fake
            self.assertIsNone(gen_markdown(self.results, self.active_plugins))
            fake.generate.assert_called_once_with(
                self.results, self.active_plugins,
                output_dir=None, session_seed=None)

    def test_gen_wrappers_return_none_when_plugin_missing(self):
        with mock.patch("benchmark.outputs._get_output_plugin", return_value=None):
            self.assertIsNone(gen_markdown([], []))
            self.assertIsNone(gen_csv([], []))
            self.assertIsNone(gen_html([], []))
            self.assertIsNone(gen_pdf([], [], "/tmp"))


class _FakeState:
    def __init__(self, results, session_seed=None):
        self._results = results
        self.session_seed = session_seed

    def latest_results(self):
        return self._results


class TestSaveOutputs(unittest.TestCase):
    def test_save_outputs_calls_every_discovered_plugin(self):
        fake_plugin = mock.MagicMock()
        fake_plugin.generate.return_value = "some/path"

        state = _FakeState([{"model": "m1"}], session_seed=42)
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("plugins.discover_output_plugins",
                           return_value=[fake_plugin]):
            _save_outputs(state, tmpdir, [])
        fake_plugin.generate.assert_called_once_with(
            [{"model": "m1"}], [], output_dir=tmpdir, session_seed=42)

    def test_save_outputs_swallows_plugin_exceptions(self):
        bad_plugin = mock.MagicMock()
        bad_plugin.generate.side_effect = RuntimeError("boom")
        good_plugin = mock.MagicMock()
        good_plugin.generate.return_value = None

        state = _FakeState([{"model": "m1"}], session_seed=None)
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("plugins.discover_output_plugins",
                           return_value=[bad_plugin, good_plugin]):
            # Must not raise even though the first plugin crashed.
            _save_outputs(state, tmpdir, [])
        good_plugin.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
