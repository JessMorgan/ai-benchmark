"""Tests for the per-plugin merged-cell visualization in ``ai-benchmark.py``.

Python identifiers disallow hyphens so the file ``ai-benchmark.py`` cannot
be imported by ``import ai_benchmark`` directly. We load the module via
``importlib.util.spec_from_file_location`` so unittest can still exercise
the helper functions.
"""
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


_THIS_DIR = pathlib.Path(__file__).resolve().parent
_AI_BENCHMARK_PATH = _THIS_DIR.parent / "ai-benchmark.py"
_spec = importlib.util.spec_from_file_location("_ai_benchmark_module", _AI_BENCHMARK_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load spec for {_AI_BENCHMARK_PATH}")
ai_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["_ai_benchmark_module"] = ai_benchmark
_spec.loader.exec_module(ai_benchmark)


class TestPluginCellBlock(unittest.TestCase):
    """The ``_plugin_cell_block`` helper produces a single 32-char cell per
    plugin, collapsing the existing 5-cell results layout to a bracket-
    delimited status message when the plugin is in flight or 429-sleeping.
    """

    def setUp(self):
        self.p_streaming = mock.MagicMock()
        self.p_streaming.id = "rate-limiter"
        self.p_streaming.supports_streaming = True
        self.p_nonstream = mock.MagicMock()
        self.p_nonstream.id = "counter"
        self.p_nonstream.supports_streaming = False

    def test_block_is_always_exactly_32_chars(self):
        """All branches produce a fixed-width cell so vertical alignment
        holds against the existing ``plugin_cols`` table."""
        cases = [
            ({"running_pids": []}, None),                              # queued
            ({"running_pids": ["rate-limiter"]}, None),                # in-flight
            ({"running_pids": ["rate-limiter"]}, 24),                  # 429 sleep
            ({
                "running_pids": [],
                "rate-limiter_score": 95.0,
                "rate-limiter_output_tokens": 123,
                "rate-limiter_response_time": 45.6,
                "rate-limiter_tps": 2.5,
            }, None),                                                   # completed
        ]
        for s, sleeping_remaining in cases:
            with self.subTest(s=s, sleeping_remaining=sleeping_remaining):
                block = ai_benchmark._plugin_cell_block(
                    "rate-limiter", s, self.p_streaming, sleeping_remaining)
                self.assertEqual(len(block), ai_benchmark.PLUGIN_BLOCK_WIDTH)

    def test_queued_no_results_shows_dash_placeholders(self):
        """A model not yet running shows the standard 5-cell layout
        with ``-`` placeholders for missing values."""
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", {"running_pids": []}, self.p_streaming, None)
        self.assertNotIn("[", block)
        self.assertTrue(block.endswith("-"),
                        "st column is '-' when no streaming event observed")

    def test_in_flight_streaming_plugin_shows_streaming_after_first_token(self):
        """If a streaming-capable plugin has received its first token,
        the merged cell says ``[streaming]``."""
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        self.assertNotIn("[waiting]", block)

    def test_in_flight_streaming_plugin_shows_waiting_without_first_token(self):
        """If a streaming-capable plugin is in flight but has NOT yet
        received its first token, the merged cell says ``[waiting]``."""
        s = {"running_pids": ["rate-limiter"]}
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[waiting]", block)
        self.assertNotIn("[streaming]", block)

    def test_in_flight_non_streaming_plugin_shows_in_flight_label(self):
        """A non-streaming plugin in flight has no first-token concept;
        use ``[in flight]`` (transport-only label) instead of ``[running]``
        (which would collide with status="running")."""
        s = {"running_pids": ["counter"]}
        block = ai_benchmark._plugin_cell_block(
            "counter", s, self.p_nonstream, None)
        self.assertIn("[in flight]", block)
        self.assertNotIn("[running]", block,
                         "label must not collide with status=running glyph")
        self.assertNotIn("[streaming]", block)

    def test_429_sleep_overrides_in_flight_status(self):
        """If the model is in a 429 backoff, the bracket text shows the
        countdown regardless of per-plugin transport state -- the
        operator cares more about wall-clock backoff than the per-
        plugin status."""
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,  # would otherwise be [streaming]
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, sleeping_remaining=24)
        self.assertIn("[429 sleeping 24s]", block)
        self.assertNotIn("[streaming]", block,
                         "429 must override the per-plugin streaming label")
        self.assertNotIn("[waiting]", block,
                         "429 must override the per-plugin waiting label")

    def test_429_sleep_when_not_in_flight_still_shows_bracket(self):
        """Even when running_pids is empty the 429 indicator still
        renders -- if the inner ``_post_request_context`` has cleared
        the pid at the moment the snapshot is taken, the model-level
        sleep counter is still the operator's primary signal."""
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", {"running_pids": []}, self.p_streaming, 7)
        self.assertIn("[429 sleeping 7s]", block)

    def test_429_sleep_clamps_at_zero_seconds(self):
        """A wake_ts in the past clamps to ``0`` so the bracket stays
        well-formed rather than rendering a negative duration."""
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", {"running_pids": ["rate-limiter"]},
            self.p_streaming, sleeping_remaining=0)
        self.assertIn("[429 sleeping 0s]", block)

    def test_completed_plugin_shows_numeric_results(self):
        """A plugin whose task has finished shows the standard 5-cell
        layout with the recorded score / tokens / time / tps and a
        post-flight ``-`` streaming glyph."""
        s = {
            "running_pids": [],
            "rate-limiter_score": 95.0,
            "rate-limiter_output_tokens": 123,
            "rate-limiter_response_time": 45.6,
            "rate-limiter_tps": 2.5,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        # No bracket text in completed state.
        self.assertNotIn("[", block)
        self.assertNotIn("]", block)
        # Numeric results present in their respective columns.
        self.assertIn("95.0", block)
        self.assertIn("123", block)
        self.assertIn("45.6", block)
        self.assertIn("2.5", block)
        # Last 5 chars (the st column) should be a single "-".
        self.assertEqual(block[-5:], "    -",
                         "st column is '-' post-flight regardless of stream state")


class TestBuildLiveIndicators(unittest.TestCase):
    """Tests for ``_build_live_indicators`` in ai-benchmark.py.

    The helper feeds the per-model row of the live TUI's ``Live:``
    section. It must show ALL streaming-capable plugins in
    ``running_pids`` (not just the first), in ``running_pids``
    insertion order, comma-separated; non-streaming plugins are
    omitted so we never lie about a transport detail we cannot
    observe. With parallel plugin threads (``max_workers > 1``) a
    model can be running several streaming plugins simultaneously;
    the operator needs to see every active thread's state.

    Example expected output for three in-flight streaming plugins:
        ``"rate-limiter (stream), moe-dense (wait), wireframes (stream)"``
    """

    @staticmethod
    def _plugin(pid, *, streaming=True):
        return type("P", (), {"id": pid, "supports_streaming": streaming})()

    def test_multiple_streaming_plugins_all_shown_in_running_pids_order(self):
        """With three streaming-capable plugins in flight, each one's
        ``(stream)`` or ``(wait)`` indicator appears in the output, in
        the order the pids appear in ``running_pids``. The format is
        ``"<pid1> (stream), <pid2> (wait), <pid3> (stream)"``.
        """
        s = {
            # running_pids is the source-of-truth insertion order;
            # rate-limiter started first, wireframes second, moe-dense
            # last (the order matches the user's example output).
            "running_pids": ["rate-limiter", "moe-dense", "wireframes"],
            "rate-limiter_first_tok_ts": 1.5,
            # moe-dense deliberately has no first_tok_ts -> wait branch
            "wireframes_first_tok_ts": 2.0,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("moe-dense", streaming=True),
            self._plugin("wireframes", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(
            out,
            "rate-limiter (stream), moe-dense (wait), wireframes (stream)",
        )

    def test_non_streaming_plugin_in_flight_is_omitted(self):
        """A non-streaming-capable plugin (e.g. ``structured-output``)
        that is in flight does NOT get a ``(stream)/(wait)`` glyph --
        we cannot observe its transport state so we don't add a
        misleading indicator to the row.
        """
        s = {
            "running_pids": ["rate-limiter", "structured-output"],
            "rate-limiter_first_tok_ts": 1.5,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("structured-output", streaming=False),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(out, "rate-limiter (stream)")

    def test_plugin_not_in_running_pids_is_excluded(self):
        """A plugin that completed (no longer in ``running_pids``) and a
        plugin that never ran (also not in ``running_pids``) are both
        excluded from the indicator string. Only ids in
        ``running_pids`` are iterated.
        """
        s = {
            # completed_plugin ran but finished -- not in running_pids
            # wireframes never started -- not in running_pids
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 0,  # still waiting, no first token
            # These would shape up as (stream)/(stream) if shown:
            "completed_plugin_first_tok_ts": 5.0,
            "wireframes_first_tok_ts": 3.0,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("wireframes", streaming=True),
            self._plugin("completed_plugin", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(out, "rate-limiter (wait)")

    def test_empty_running_pids_returns_empty_string(self):
        """No in-flight plugins -> empty string. The caller
        (``_render_live_activity``) skips the ``"  <indicators>"``
        prefix when the result is empty.
        """
        s = {"running_pids": []}
        plugins = [self._plugin("any"), self._plugin("other")]
        self.assertEqual(ai_benchmark._build_live_indicators(s, plugins), "")

    def test_all_non_streaming_returns_empty_string(self):
        """All in-flight plugins are non-streaming -> empty string.
        We don't surface any glyph on rows whose transport state we
        cannot observe, even when running_pids has entries.
        """
        s = {"running_pids": ["structured-output", "tool-calling"]}
        plugins = [
            self._plugin("structured-output", streaming=False),
            self._plugin("tool-calling", streaming=False),
        ]
        self.assertEqual(ai_benchmark._build_live_indicators(s, plugins), "")


if __name__ == "__main__":
    unittest.main()
