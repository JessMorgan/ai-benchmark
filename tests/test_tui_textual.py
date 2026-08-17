"""Headless tests for the Textual-based live TUI in ``benchmark.cli``.

The Textual renderer rebuilds one full frame as a list of ``(text, style)``
pairs each tick and hands them to a ``Static`` widget. These tests exercise
the pure frame builder, the Rich ``Text`` conversion, and the Textual/plain-
text dispatch without needing a real terminal.
"""
import os
import threading
import time
import unittest
from unittest import mock

from benchmark import cli


class _FakeState:
    def __init__(self):
        self._completed = 0
        self._total = 2

    def snapshot(self):
        return {
            "model-a": {"status": "completed", "source": "Local",
                        "rate-limiter_score": 18},
            "model-b": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }

    @property
    def completed(self):
        return self._completed

    @property
    def total(self):
        return self._total

    def judge_activity_snapshot(self):
        return []

    def judge_progress_snapshot(self):
        return {}

    def recent_log(self, _n):
        return []


def _plugin():
    p = mock.MagicMock()
    p.id = "rate-limiter"
    p.supports_streaming = True
    return p


def _render_lines(state=None, size=(24, 100), num_sources=1):
    state = state or _FakeState()
    plugin = _plugin()
    return cli._build_frame_lines(
        state, [plugin], {"Local": "LC"},
        "  #  S Model  St", "ratSc ratTok ratTm ratTPS",
        num_sources=num_sources, scroll_y=0, scroll_x=0, size=size,
        session_seed=42,
    )


def _frame_text(lines):
    return "\n".join(text for text, _style in lines)


class TestFrameLinesToText(unittest.TestCase):
    def test_joins_lines_with_newlines_and_preserves_content(self):
        text = cli._frame_lines_to_text(
            [("first", "bold"), ("second", None), ("third", "red")]
        )
        self.assertIn("first", text.plain)
        self.assertIn("second", text.plain)
        self.assertIn("third", text.plain)
        self.assertIn("\n", text.plain)

    def test_unknown_style_is_ignored(self):
        # A style name outside the map must not raise; it falls back to
        # the default (unstyled) render.
        text = cli._frame_lines_to_text([("x", "not-a-style")])
        self.assertEqual(text.plain, "x")


class TestBuildFrameLines(unittest.TestCase):
    def test_renders_header_summary_rows_live_and_footer(self):
        with mock.patch("benchmark.cli.get_active_request_count", return_value=2), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines())
        self.assertIn("AI Benchmark", text)
        self.assertIn("Seed: 42", text)
        self.assertIn("Total: 2", text)
        self.assertIn("model-a", text)
        self.assertIn("model-b", text)
        self.assertIn("Live:", text)
        self.assertIn("1 active", text)

    def test_row_style_is_colored_by_status(self):
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines()
        styles = {text: style for text, style in lines}
        # The completed model-a row is green; the running model-b row is
        # yellow (matching the old curses/Rich color coding).
        self.assertIn("green", styles.values())
        self.assertIn("yellow", styles.values())

    def test_small_terminal_limits_visible_rows(self):
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(size=(8, 100)))
        # Height 8 leaves no room for model rows; only the header/summary,
        # divider, live area, and footer remain.
        self.assertNotIn("model-a", text)

    def test_lines_are_truncated_to_terminal_width(self):
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(size=(24, 40))
        for text, _style in lines:
            self.assertLessEqual(cli._display_width(text), 40)


class TestTextualTuiEnabled(unittest.TestCase):
    def test_no_textual_env_disables(self):
        with mock.patch.dict(os.environ, {"AI_BENCHMARK_NO_TEXTUAL": "1"}):
            self.assertFalse(cli._textual_tui_enabled())

    def test_force_textual_env_enables(self):
        with mock.patch.dict(os.environ, {"AI_BENCHMARK_FORCE_TEXTUAL": "1"}):
            self.assertTrue(cli._textual_tui_enabled())

    def test_isatty_default(self):
        with mock.patch.dict(
                os.environ,
                {"AI_BENCHMARK_NO_TEXTUAL": "", "AI_BENCHMARK_FORCE_TEXTUAL": ""}), \
                mock.patch("sys.stdout") as stdout:
            stdout.isatty.return_value = True
            self.assertTrue(cli._textual_tui_enabled())
            stdout.isatty.return_value = False
            self.assertFalse(cli._textual_tui_enabled())


class TestTextualTuiDispatch(unittest.TestCase):
    def _state(self):
        return _FakeState()

    def test_force_textual_runs_textual_path(self):
        stop = threading.Event()
        stop.set()
        with mock.patch.dict(os.environ, {"AI_BENCHMARK_FORCE_TEXTUAL": "1"}), \
                mock.patch("benchmark.cli._tui_main_textual") as textual, \
                mock.patch("benchmark.cli._fallback_tui_loop") as fallback:
            cli.tui_main(self._state(), stop, 1, [_plugin()])
        textual.assert_called_once()
        fallback.assert_not_called()

    def test_default_dispatches_to_fallback(self):
        stop = threading.Event()
        stop.set()
        with mock.patch("benchmark.cli._textual_tui_enabled", return_value=False), \
                mock.patch("benchmark.cli._tui_main_textual") as textual, \
                mock.patch("benchmark.cli._fallback_tui_loop") as fallback:
            cli.tui_main(self._state(), stop, 1, [_plugin()])
        textual.assert_not_called()
        fallback.assert_called_once()

    def test_textual_failure_falls_back_to_fallback(self):
        stop = threading.Event()
        stop.set()
        with mock.patch("benchmark.cli._textual_tui_enabled", return_value=True), \
                mock.patch("benchmark.cli._tui_main_textual",
                           side_effect=RuntimeError("boom")), \
                mock.patch("benchmark.cli._fallback_tui_loop") as fallback:
            cli.tui_main(self._state(), stop, 1, [_plugin()])
        fallback.assert_called_once()


class TestBuildFrameLinesAdvanced(unittest.TestCase):
    def _state(self, snapshot, judge_activities=None, judge_progress=None,
               recent_log=None):
        state = mock.MagicMock()
        state.snapshot.return_value = snapshot
        state.completed = 0
        state.total = 2
        state.judge_activity_snapshot.return_value = judge_activities or []
        state.judge_progress_snapshot.return_value = judge_progress or {}
        state.recent_log.return_value = recent_log or []
        return state

    def test_preloading_and_judge_render(self):
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
            "model-b": {"status": "pending", "source": "Local",
                        "preloading": True,
                        "preload_start_ts": time.monotonic()},
        }
        state = self._state(
            snapshot,
            judge_activities=[{"judge": "judge-a", "target": "model-a",
                                "plugin": "rate-limiter", "elapsed": 3}],
            judge_progress={"judge-a": {"completed": 1, "failed": 0,
                                         "expected": 2}},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(state, num_sources=4))
        self.assertIn("Preloading model", text)
        self.assertIn("Judge judge-a", text)
        self.assertIn("Judging", text)

    def test_429_sleeping_render(self):
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(snapshot)
        backoff = {"sleeping": {"Local|model-a|rate-limiter": {
            "wake_ts": time.time() + 5, "attempts": 1, "max_attempts": 3}}}
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value=backoff):
            text = _frame_text(_render_lines(state, num_sources=4))
        self.assertIn("429 Sleeping:", text)
        self.assertIn("429 1/3", text)

    def test_recent_errors_render(self):
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
            recent_log=[(time.time() - 60, "model-x", "boom")],
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(state))
        self.assertIn("Errors:", text)
        self.assertIn("boom", text)

    def test_all_complete_footer(self):
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(state))
        self.assertIn("All models complete", text)


class TestMainThreadDispatch(unittest.TestCase):
    """The Textual Linux driver needs the main thread (it installs signal
    handlers), so ``main`` must run the orchestrator in a worker thread and
    drive the TUI from the main thread when Textual is enabled."""

    def test_textual_disabled_runs_orchestrator_inline(self):
        with mock.patch("benchmark.cli._textual_tui_enabled", return_value=False), \
                mock.patch("benchmark.cli._run_benchmark") as orchestrator, \
                mock.patch("benchmark.cli.tui_main") as tui:
            cli.main()
        orchestrator.assert_called_once_with()
        tui.assert_not_called()

    def test_textual_enabled_runs_tui_on_main_thread(self):
        captured = {}

        def fake_orchestrator(handoff=None):
            # Simulate setup completing and handing the TUI to the main thread.
            handoff["args"] = ("STATE", "STOP", 1, ["rate-limiter"], None, {})
            handoff["stop_event"] = threading.Event()
            captured["handoff"] = handoff
            handoff["ready"].set()

        with mock.patch("benchmark.cli._textual_tui_enabled", return_value=True), \
                mock.patch("benchmark.cli._run_benchmark",
                           side_effect=fake_orchestrator), \
                mock.patch("benchmark.cli.tui_main") as tui:
            cli.main()
        tui.assert_called_once_with("STATE", "STOP", 1, ["rate-limiter"], None, {})
        self.assertIsNotNone(captured.get("handoff"))

    def test_ctrl_c_sets_stop_event_and_winds_down(self):
        captured = {}

        def fake_orchestrator(handoff=None):
            handoff["args"] = ("STATE", "STOP", 1, ["rate-limiter"], None, {})
            handoff["stop_event"] = threading.Event()
            captured["stop_event"] = handoff["stop_event"]
            handoff["ready"].set()

        with mock.patch("benchmark.cli._textual_tui_enabled", return_value=True), \
                mock.patch("benchmark.cli._run_benchmark",
                           side_effect=fake_orchestrator), \
                mock.patch("benchmark.cli.tui_main",
                           side_effect=KeyboardInterrupt):
            cli.main()
        self.assertTrue(captured["stop_event"].is_set())


if __name__ == "__main__":
    unittest.main()
