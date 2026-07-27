"""Tests for the per-plugin merged-cell visualization in ``ai-benchmark.py``.

Python identifiers disallow hyphens so the file ``ai-benchmark.py`` cannot
be imported by ``import ai_benchmark`` directly. We load the module via
``importlib.util.spec_from_file_location`` so unittest can still exercise
the helper functions.
"""
import importlib.util
import pathlib
import sys
import time
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

    def test_streaming_plugin_shows_tok_count_after_bytes_accumulate(self):
        """Once the streaming callback has accumulated chars (mocked
        here as a stored counter), the cell shows
        ``[streaming - N tok]`` where N is chars // 4 (matching the
        ``count_tokens`` estimator + ``benchmark_core.add_bytes_received``
        which counts chars, not bytes, so live and completion numbers
        align for ASCII=CJK=emoji alike). The indicator gives the
        operator a wall-clock ticker on the in-flight progress.
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,
            "rate-limiter_bytes_received": 64,  # 64 // 4 = 16 tok
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming - 16 tok]", block)
        self.assertNotIn("[waiting]", block)
        # Bare "[streaming]" must NOT appear after substituting out
        # the new indicator -- the cell shows the enriched status,
        # not the legacy no-counter form.
        leftover = block.replace("[streaming - 16 tok]", "")
        self.assertNotIn("[streaming]", leftover,
                         "cell should enrich to [streaming - N tok] when bytes accumulate")

    def test_streaming_plugin_keeps_bare_streaming_before_first_byte(self):
        """If first_tok_ts is set but bytes_received is 0 (just
        landed first-tok event but no delta has accumulated yet),
        keep the legacy ``[streaming]`` form rather than showing
        ``[streaming - 0 tok]`` (which would be visually noisy and
        duplicate the rare ``ft > 0 but no bytes`` edge case).
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 1234.5,
            "rate-limiter_bytes_received": 0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s, self.p_streaming, None)
        self.assertIn("[streaming]", block)
        self.assertNotIn("tok", block,
                         "no bytes yet -> don't show 0-tok indicator")

    def test_in_flight_waiting_plugin_shows_elapsed_suffix_above_threshold(self):
        """Streaming-capable plugin in flight with no first token yet:
        once the wall-clock wait exceeds 2s, the bracket becomes
        ``[waiting - Ns]`` so the operator can tell a slow / stuck
        wait from a normal <2s one. Below the threshold the bare
        ``[waiting]`` form is kept (no visual noise on quick plugins).
        A missing ``attempt_start`` also keeps the bare form (we
        don't fabricate a meaningless elapsed value from epoch 0).
        """
        now = time.time()
        # Above threshold (5s ago): expect elapsed suffix.
        s_above = {
            "running_pids": ["rate-limiter"],
            "attempt_start": now - 5.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s_above, self.p_streaming, None)
        self.assertIn("[waiting - 5s]", block)
        # Below threshold (1s ago): bare bracket, no suffix.
        s_below = {
            "running_pids": ["rate-limiter"],
            "attempt_start": now - 1.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s_below, self.p_streaming, None)
        self.assertIn("[waiting]", block)
        self.assertNotIn("[waiting -", block,
                         "below-threshold wait should not carry an elapsed suffix")
        # Missing attempt_start: bare bracket (no fabricated elapsed).
        s_none = {
            "running_pids": ["rate-limiter"],
            # no attempt_start key -- simulates a model that hasn't been dispatched yet
        }
        block = ai_benchmark._plugin_cell_block(
            "rate-limiter", s_none, self.p_streaming, None)
        self.assertIn("[waiting]", block)
        self.assertNotIn("[waiting -", block,
                         "missing attempt_start should keep bare bracket (no fabricated elapsed)")

    def test_in_flight_non_streaming_plugin_shows_elapsed_suffix_above_threshold(self):
        """Non-streaming-capable plugin in flight: once elapsed
        exceeds 2s, the bracket becomes ``[in flight - Ns]`` so the
        operator can spot hung non-streaming requests. Below the
        threshold the bare ``[in flight]`` form is kept.
        """
        now = time.time()
        # Above threshold (5s ago): expect elapsed suffix.
        s_above = {
            "running_pids": ["counter"],
            "attempt_start": now - 5.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "counter", s_above, self.p_nonstream, None)
        self.assertIn("[in flight - 5s]", block)
        # Below threshold (1s ago): bare bracket.
        s_below = {
            "running_pids": ["counter"],
            "attempt_start": now - 1.0,
        }
        block = ai_benchmark._plugin_cell_block(
            "counter", s_below, self.p_nonstream, None)
        self.assertIn("[in flight]", block)
        self.assertNotIn("[in flight -", block,
                         "below-threshold non-streaming should not carry an elapsed suffix")


class TestBuildLiveIndicators(unittest.TestCase):
    """Tests for ``_build_live_indicators`` in ai-benchmark.py.

    The helper feeds the per-model row of the live TUI's ``Live:``
    section. With parallel plugin threads (``max_workers > 1``), a
    model can be running several streaming plugins simultaneously;
    the operator needs to see every per-thread streaming state AND
    a single aggregate count of plugins still waiting for their
    first token.

    Format requirements:
    * Space-separated ``"[<pid>: <N> tok]"`` entries in
      ``running_pids`` insertion order for streaming plugins with
      bytes accumulated.
    * Trailing ``"[waiting: K]"`` aggregate (count of streaming-
      capable in-flight plugins with no first_tok_ts yet).
    * Non-streaming in-flight plugins are omitted entirely (the
      table cell already shows ``[in flight]`` per-cell, and we have
      no transport state to report for them in the footer).
    * Empty running_pids -> empty string.
    * Plugin in running_pids with ft > 0 but bytes_received == 0
      is silently omitted (rare transient; nothing useful to show).

    Example expected output for two streaming + 1 waiting:
        ``"[rate-limiter: 16 tok] [software-architecture: 152 tok] [waiting: 1]"``
    """

    @staticmethod
    def _plugin(pid, *, streaming=True):
        return type("P", (), {"id": pid, "supports_streaming": streaming})()

    def test_two_streaming_plugins_and_one_waiting_in_running_pids_order(self):
        """With three streaming-capable plugins in flight -- two
        streaming with bytes and one waiting -- the per-plugin tok
        entries appear in ``running_pids`` insertion order, followed
        by a single ``[waiting: K]`` aggregate. Demonstrates the
        new format end-to-end.
        """
        s = {
            "running_pids": ["rate-limiter", "moe-dense", "wireframes"],
            "rate-limiter_first_tok_ts": 1.5,
            "rate-limiter_bytes_received": 64,     # 16 tok
            # moe-dense has no first_tok_ts -> waiting aggregate
            "wireframes_first_tok_ts": 2.0,
            "wireframes_bytes_received": 128,      # 32 tok
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("moe-dense", streaming=True),
            self._plugin("wireframes", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(
            out,
            "[rate-limiter: 16 tok] [wireframes: 32 tok] [waiting: 1]",
        )

    def test_user_example_output_two_streaming_six_waiting(self):
        """Reproduces the user's exact example verbatim: two
        streaming plugins showing tok counts, followed by the
        ``[waiting: 6]`` aggregate for six other plugins still
        waiting for first token.
        """
        waiting_pids = [f"plugin-waiting-{i}" for i in range(6)]
        s = {
            "running_pids": [
                "rate-limiter",
                "software-architecture",
                *waiting_pids,
            ],
            "rate-limiter_first_tok_ts": 1.5,
            "rate-limiter_bytes_received": 64,            # 16 tok
            "software-architecture_first_tok_ts": 2.0,
            "software-architecture_bytes_received": 608,   # 152 tok
            # waiting_pids get no first_tok_ts -> waiting aggregate
        }
        plugins = [self._plugin(p, streaming=True) for p in s["running_pids"]]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(
            out,
            "[rate-limiter: 16 tok] [software-architecture: 152 tok] [waiting: 6]",
        )

    def test_non_streaming_plugin_in_flight_is_omitted(self):
        """A non-streaming-capable plugin in flight does NOT get a
        bracket indicator here -- the table cell already shows
        ``[in flight]`` per-cell, and the live footer doesn't
        surface a glyph when we cannot observe the transport state.
        """
        s = {
            "running_pids": ["rate-limiter", "structured-output"],
            "rate-limiter_first_tok_ts": 1.5,
            "rate-limiter_bytes_received": 64,        # 16 tok
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("structured-output", streaming=False),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(out, "[rate-limiter: 16 tok]")

    def test_plugin_not_in_running_pids_is_excluded(self):
        """A plugin that completed (not in running_pids) and a
        plugin that never ran (also not in running_pids) are both
        excluded. The output is determined solely by running_pids.
        """
        s = {
            "running_pids": ["rate-limiter"],
            "rate-limiter_first_tok_ts": 0,           # still waiting
            "rate-limiter_bytes_received": 0,
            # These are defined but plugins NOT in running_pids:
            "completed_plugin_first_tok_ts": 5.0,
            "completed_plugin_bytes_received": 200,
            "wireframes_first_tok_ts": 3.0,
            "wireframes_bytes_received": 100,
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("wireframes", streaming=True),
            self._plugin("completed_plugin", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        # rate-limiter is in flight but no first_tok + no bytes ->
        # counted in [waiting: K] aggregate.
        self.assertEqual(out, "[waiting: 1]")

    def test_empty_running_pids_returns_empty_string(self):
        """No in-flight plugins -> empty string. The caller
        (``_render_live_activity``) skips the prefix when the result
        is empty.
        """
        s = {"running_pids": []}
        plugins = [self._plugin("any"), self._plugin("other")]
        self.assertEqual(ai_benchmark._build_live_indicators(s, plugins), "")

    def test_only_waiting_returns_aggregate(self):
        """All in-flight streaming-capable plugins have no first
        token + no bytes yet -> output is just the single
        ``[waiting: K]`` aggregate. No per-plugin entries appear.
        """
        s = {
            "running_pids": ["rate-limiter", "wireframes", "moe-dense"],
            # No first_tok_ts or bytes for any.
        }
        plugins = [self._plugin(p, streaming=True) for p in s["running_pids"]]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(out, "[waiting: 3]")

    def test_streaming_fts_but_no_bytes_omitted_from_output(self):
        """Rare transient: first_tok_ts is set but bytes_received
        is still 0 (just landed first-tok, no delta has accumulated
        yet). The plugin is silently omitted -- emitting
        ``[name: 0 tok]`` would be visually noisy and the per-cell
        [streaming] already conveys "just started streaming" without
        duplicating that into the live footer.
        """
        s = {
            "running_pids": ["rate-limiter", "wireframes"],
            "rate-limiter_first_tok_ts": 1.5,
            "rate-limiter_bytes_received": 64,    # 16 tok
            "wireframes_first_tok_ts": 2.0,        # ft but no bytes
        }
        plugins = [
            self._plugin("rate-limiter", streaming=True),
            self._plugin("wireframes", streaming=True),
        ]
        out = ai_benchmark._build_live_indicators(s, plugins)
        self.assertEqual(out, "[rate-limiter: 16 tok]")

if __name__ == "__main__":
    unittest.main()
