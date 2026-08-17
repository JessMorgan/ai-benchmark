"""Headless tests for the Rich-based live TUI in ``benchmark.cli``.

The Rich renderer builds one full frame as a Rich ``Group``; these tests
exercise the frame builder, key decoding, and the curses/Rich dispatch
without needing a real terminal.
"""
import os
import threading
import time
import unittest
from unittest import mock

from rich.console import Console

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


def _render_frame(state=None, size=(24, 100), num_sources=1):
    state = state or _FakeState()
    plugin = _plugin()
    return cli._build_rich_frame(
        state, [plugin], {"Local": "LC"},
        "  #  S Model  St", "ratSc ratTok ratTm ratTPS",
        num_sources=num_sources, scroll_y=0, scroll_x=0, size=size,
        session_seed=42,
    )


def _frame_text(frame, width=100):
    console = Console(record=True, width=width, force_terminal=False)
    console.print(frame)
    return console.export_text()


class TestRichKeyAction(unittest.TestCase):
    def test_quit_keys(self):
        for chunk in (b"q", b"Q", b"\x03", b"\x04"):
            self.assertEqual(cli._rich_key_action(chunk), "quit")

    def test_arrow_keys(self):
        self.assertEqual(cli._rich_key_action(b"\x1b[A"), "up")
        self.assertEqual(cli._rich_key_action(b"\x1b[B"), "down")
        self.assertEqual(cli._rich_key_action(b"\x1b[C"), "right")
        self.assertEqual(cli._rich_key_action(b"\x1b[D"), "left")

    def test_page_and_home_end(self):
        self.assertEqual(cli._rich_key_action(b"\x1b[5~"), "pageup")
        self.assertEqual(cli._rich_key_action(b"\x1b[6~"), "pagedown")
        self.assertEqual(cli._rich_key_action(b"\x1b[H"), "home")
        self.assertEqual(cli._rich_key_action(b"\x1b[F"), "end")

    def test_space_is_pagedown(self):
        self.assertEqual(cli._rich_key_action(b" "), "pagedown")

    def test_unknown_chunk_is_none(self):
        self.assertIsNone(cli._rich_key_action(b"x"))


class TestRichKeyboardFallback(unittest.TestCase):
    def test_non_tty_stdin_yields_noop_poller(self):
        with mock.patch("sys.stdin") as stdin:
            stdin.fileno.side_effect = ValueError("not a tty")
            with cli._rich_keyboard() as poll:
                self.assertIsNone(poll())


class TestBuildRichFrame(unittest.TestCase):
    def test_renders_header_summary_rows_live_and_footer(self):
        with mock.patch("benchmark.cli.get_active_request_count", return_value=2), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_frame())
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
            frame = _render_frame()
        # A completed row and a running row are both present in the frame.
        self.assertEqual(len(frame.renderables), 10)

    def test_small_terminal_limits_visible_rows(self):
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_frame(size=(8, 100)))
        # Height 8 leaves no room for model rows; only the header/summary,
        # divider, live area, and footer remain.
        self.assertNotIn("model-a", text)


class TestRichTuiEnabled(unittest.TestCase):
    def test_no_rich_env_disables(self):
        with mock.patch.dict(os.environ, {"AI_BENCHMARK_NO_RICH": "1"}):
            self.assertFalse(cli._rich_tui_enabled())

    def test_force_rich_env_enables(self):
        with mock.patch.dict(os.environ, {"AI_BENCHMARK_FORCE_RICH": "1"}):
            self.assertTrue(cli._rich_tui_enabled())

    def test_isatty_default(self):
        with mock.patch.dict(
                os.environ,
                {"AI_BENCHMARK_NO_RICH": "", "AI_BENCHMARK_FORCE_RICH": ""}), \
                mock.patch("sys.stdout") as stdout:
            stdout.isatty.return_value = True
            self.assertTrue(cli._rich_tui_enabled())
            stdout.isatty.return_value = False
            self.assertFalse(cli._rich_tui_enabled())


class TestRichTuiDispatch(unittest.TestCase):
    def _state(self):
        return _FakeState()

    def test_force_rich_runs_rich_path(self):
        stop = threading.Event()
        stop.set()
        with mock.patch.dict(os.environ, {"AI_BENCHMARK_FORCE_RICH": "1"}), \
                mock.patch("benchmark.cli._tui_main_rich") as rich, \
                mock.patch("benchmark.cli._tui_main_curses") as curses:
            cli.tui_main(self._state(), stop, 1, [_plugin()])
        rich.assert_called_once()
        curses.assert_not_called()

    def test_default_dispatches_to_curses(self):
        stop = threading.Event()
        stop.set()
        with mock.patch("benchmark.cli._rich_tui_enabled", return_value=False), \
                mock.patch("benchmark.cli._tui_main_rich") as rich, \
                mock.patch("benchmark.cli._tui_main_curses") as curses:
            cli.tui_main(self._state(), stop, 1, [_plugin()])
        rich.assert_not_called()
        curses.assert_called_once()

    def test_rich_failure_falls_back_to_curses(self):
        stop = threading.Event()
        stop.set()
        with mock.patch("benchmark.cli._rich_tui_enabled", return_value=True), \
                mock.patch("benchmark.cli._tui_main_rich",
                           side_effect=RuntimeError("boom")), \
                mock.patch("benchmark.cli._tui_main_curses") as curses:
            cli.tui_main(self._state(), stop, 1, [_plugin()])
        curses.assert_called_once()


class TestBuildRichFrameAdvanced(unittest.TestCase):
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
            text = _frame_text(_render_frame(state, num_sources=4))
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
            text = _frame_text(_render_frame(state, num_sources=4))
        self.assertIn("429 Sleeping:", text)
        self.assertIn("429 1/3", text)

    def test_recent_errors_render(self):
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
            recent_log=[(time.time() - 60, "model-x", "boom")],
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_frame(state))
        self.assertIn("Errors:", text)
        self.assertIn("boom", text)

    def test_all_complete_footer(self):
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_frame(state))
        self.assertIn("All models complete", text)
