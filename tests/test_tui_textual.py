"""Headless tests for the Textual-based live TUI in ``benchmark.cli``.

The Textual renderer rebuilds one full frame as a list of ``(text, style)``
pairs each tick and hands each line to its own ``_FrameRow`` widget, which
repaints only the cells that changed. These tests exercise the pure frame
builder, the cell-diff helpers, the row widget, and the Textual/plain-text
dispatch without needing a real terminal.
"""
import asyncio
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

    def judge_selected_snapshot(self):
        return set()

    def recent_log(self, _n):
        return []

    @property
    def revision(self):
        return 0

    def has_live_work(self):
        return bool(self.snapshot().get("model-b", {}).get("running_pids"))


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


class TestFrameCache(unittest.TestCase):
    def test_frame_alignment_cache_reuses_unchanged_revision(self):
        state = mock.Mock()
        state.snapshot.return_value = {
            "model-a": {
                "status": "completed",
                "source": "Local",
                "judge_models": ["judge"],
                "running_pids": [],
                "p_score": 10.0,
            },
        }
        state.completed = 1
        state.total = 1
        state.judge_activity_snapshot.return_value = []
        state.judge_progress_snapshot.return_value = {}
        state.judge_selected_snapshot.return_value = set()
        state.has_live_work.return_value = False
        state.revision = 1
        state.recent_log.return_value = []
        plugin = mock.Mock(id="p", supports_streaming=True)
        with mock.patch("benchmark.cli._plugin_judge_alignment", return_value=(2, 0, 0)) as alignment:
            cli._build_frame_lines(
                state, [plugin], {"Local": "Loc"}, "#", "p", 1, 0, 0, (20, 80),
            )
            cli._build_frame_lines(
                state, [plugin], {"Local": "Loc"}, "#", "p", 1, 0, 0, (20, 80),
            )
        self.assertEqual(alignment.call_count, 1)



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

    def test_every_style_map_value_is_a_valid_rich_style(self):
        # Regression: a style value Rich cannot parse (e.g. the plain
        # "grey" color name) crashes the live TUI render with
        # ColorParseError. Every mapped value must parse as a Rich style.
        from rich.style import Style

        for key, value in cli._FRAME_STYLE_MAP.items():
            self.assertIsNotNone(
                Style.parse(value),
                f"_FRAME_STYLE_MAP[{key!r}] = {value!r} is not a valid Rich style",
            )


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
               recent_log=None, judge_selected=None):
        state = mock.MagicMock()
        state.snapshot.return_value = snapshot
        state.completed = 0
        state.total = 2
        state.judge_activity_snapshot.return_value = judge_activities or []
        state.judge_progress_snapshot.return_value = judge_progress or {}
        state.judge_selected_snapshot.return_value = judge_selected or set()
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

    def test_live_judge_line_shows_progress_counts(self):
        """An active judge's live line carries its succeeded/failed/total."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(
            snapshot,
            judge_activities=[{"judge": "judge-a", "target": "model-a",
                                "plugin": "rate-limiter", "elapsed": 3}],
            judge_progress={"judge-a": {"completed": 7, "failed": 2,
                                         "expected": 9}},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(state, num_sources=4))
        self.assertIn("Judge judge-a 7\u27052\u274c9\u03a3", text)

    def test_live_judge_line_shows_thinking_and_content_tokens(self):
        """An active judge exposes separate live thinking/content counters."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(
            snapshot,
            judge_activities=[{
                "judge": "judge-a", "target": "model-a",
                "plugin": "rate-limiter", "attempt": 2, "elapsed": 3,
                "thinking_tokens": 123, "content_tokens": 45,
            }],
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(state, num_sources=4))
        self.assertIn("model-a rate-limiter attempt=2 3s thinking=123 content=45", text)

    def test_live_judge_line_without_progress_has_no_counts(self):
        """A judge with no progress record shows only the activity cells."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(
            snapshot,
            judge_activities=[{"judge": "judge-a", "target": "model-a",
                                "plugin": "rate-limiter", "elapsed": 3}],
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            text = _frame_text(_render_lines(state, num_sources=4))
        self.assertIn("Judge judge-a", text)
        self.assertNotIn("\u2705", text)

    def test_many_judges_overflow_to_second_footer_line(self):
        """A large judge roster wraps onto extra footer lines instead of
        being truncated to the terminal width."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        progress = {
            f"judge-{i}": {"completed": i, "failed": 0, "expected": i + 1}
            for i in range(12)
        }
        state = self._state(snapshot, judge_progress=progress)
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            # Narrow terminal: the full judging line can't fit on one row.
            lines = _render_lines(state, size=(40, 50), num_sources=4)
        text = _frame_text(lines)
        self.assertGreaterEqual(text.count("Judging"), 2)
        self.assertIn("judge-11", text)
        # Every judge part is visible (nothing truncated away).
        for i in range(12):
            self.assertIn(f"judge-{i}:", text)

    def test_single_judge_fits_one_footer_line(self):
        state = self._state(
            {"model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0}},
            judge_progress={"judge-a": {"completed": 1, "failed": 0,
                                         "expected": 2}},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        text = _frame_text(lines)
        self.assertEqual(text.count("Judging"), 1)

    def test_stopped_judge_renders_red_on_own_footer_line(self):
        """A 429-halted judge gets its own red footer line; active judges
        stay on the default-styled line."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(
            snapshot,
            judge_progress={
                "judge-active": {"completed": 3, "failed": 0, "expected": 4},
                "judge-stopped": {"completed": 1, "failed": 2, "expected": 3,
                                  "stopped": True},
            },
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        text = _frame_text(lines)
        self.assertIn("Judging [judge-active:", text)
        self.assertIn("Judging [judge-stopped:", text)
        stopped = next(l for l in lines if "judge-stopped:" in l[0])
        active = next(l for l in lines if "judge-active:" in l[0])
        self.assertEqual(stopped[1], "red")
        self.assertIsNone(active[1])

    def test_stopped_judge_alone_renders_red_without_blank_footer(self):
        """With no models running and only a halted judge, the red judge
        line still renders (no stray blank footer row, no crash)."""
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
            judge_progress={
                "judge-stopped": {"completed": 1, "failed": 2, "expected": 3,
                                  "stopped": True},
            },
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        text = _frame_text(lines)
        self.assertIn("Judging [judge-stopped:", text)
        self.assertIn("All models complete", text)
        stopped = next(l for l in lines if "judge-stopped:" in l[0])
        self.assertEqual(stopped[1], "red")

    def test_running_judge_renders_green(self):
        """A judge with an in-flight activity (selected to run) renders on
        its own green footer line."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(
            snapshot,
            judge_activities=[{"judge": "judge-run", "target": "model-a",
                                "plugin": "rate-limiter", "elapsed": 3}],
            judge_progress={
                "judge-run": {"completed": 1, "failed": 0, "expected": 2},
            },
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        text = _frame_text(lines)
        self.assertIn("Judging [judge-run:", text)
        run_line = next(l for l in lines if "judge-run:" in l[0])
        self.assertEqual(run_line[1], "green")

    def test_selected_judge_stays_green_without_active_request(self):
        """A selected judge remains green during a gap between cell requests."""
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
            judge_progress={
                "judge-selected": {"completed": 3, "failed": 0, "expected": 5},
            },
            judge_selected={"judge-selected"},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        selected_line = next(
            line for line in lines if "judge-selected:" in line[0]
        )
        self.assertEqual(selected_line[1], "green")
        self.assertNotIn("All models complete", _frame_text(lines))

    def test_deselected_judge_returns_to_waiting_white(self):
        """Only the newly selected judge stays green after a handoff."""
        state = self._state(
            {"model-a": {"status": "completed", "source": "Local"}},
            judge_progress={
                "judge-old": {"completed": 3, "failed": 0, "expected": 5},
                "judge-new": {"completed": 1, "failed": 0, "expected": 5},
            },
            judge_selected={"judge-new"},
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=0), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        styles = {
            name: next(line[1] for line in lines if f"judge-{name}:" in line[0])
            for name in ("old", "new")
        }
        self.assertEqual(styles, {"old": None, "new": "green"})

    def test_completed_judge_renders_grey(self):
        """A judge whose whole workload is done renders dimmed grey."""
        state = self._state(
            {"model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0}},
            judge_progress={
                "judge-done": {"completed": 5, "failed": 0, "expected": 5},
            },
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        text = _frame_text(lines)
        self.assertIn("Judging [judge-done:", text)
        done_line = next(l for l in lines if "judge-done:" in l[0])
        self.assertEqual(done_line[1], "grey")

    def test_judge_footer_groups_by_state(self):
        """Running/waiting/complete/stopped judges land on separate footer
        lines with green/white/grey/red styles."""
        snapshot = {
            "model-a": {"status": "running", "source": "Local",
                        "running_pids": ["rate-limiter"],
                        "rate-limiter_start_ts": 0},
        }
        state = self._state(
            snapshot,
            judge_activities=[{"judge": "judge-run", "target": "model-a",
                                "plugin": "rate-limiter", "elapsed": 3}],
            judge_progress={
                "judge-run": {"completed": 1, "failed": 0, "expected": 2},
                "judge-wait": {"completed": 0, "failed": 0, "expected": 4},
                "judge-done": {"completed": 4, "failed": 0, "expected": 4},
                "judge-stop": {"completed": 1, "failed": 1, "expected": 2,
                                "stopped": True},
            },
        )
        with mock.patch("benchmark.cli.get_active_request_count", return_value=1), \
                mock.patch("benchmark.cli.get_429_stats", return_value={}):
            lines = _render_lines(state, num_sources=4)
        styles = {
            name: next(l[1] for l in lines if f"judge-{name}:" in l[0])
            for name in ("run", "wait", "done", "stop")
        }
        self.assertEqual(styles, {"run": "green", "wait": None,
                                  "done": "grey", "stop": "red"})

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


class TestLineCells(unittest.TestCase):
    """Per-display-column expansion used by the per-cell TUI repainter."""

    def test_pads_to_width(self):
        self.assertEqual(cli._line_cells("ab", 5), ["a", "b", " ", " ", " "])

    def test_wide_glyph_occupies_two_cells(self):
        self.assertEqual(cli._line_cells("a\U0001f504b", 5),
                         ["a", "\U0001f504", "", "b", " "])

    def test_cjk_wide(self):
        self.assertEqual(cli._line_cells("\u65e5", 2), ["\u65e5", ""])

    def test_truncates_when_wider_than_width(self):
        self.assertEqual(cli._line_cells("abcdef", 3), ["a", "b", "c"])


class TestChangedCellSpans(unittest.TestCase):
    """Diff of two cell lists into changed column spans."""

    def test_identical_returns_empty(self):
        self.assertEqual(cli._changed_cell_spans(list("abc"), list("abc")), [])

    def test_middle_change(self):
        self.assertEqual(cli._changed_cell_spans(list("abcde"), list("abXde")),
                         [(2, 1)])

    def test_multiple_runs(self):
        self.assertEqual(cli._changed_cell_spans(list("abcdef"), list("aXcdeY")),
                         [(1, 1), (5, 1)])

    def test_full_change(self):
        self.assertEqual(cli._changed_cell_spans(list("aaa"), list("bbb")),
                         [(0, 3)])

    def test_wide_glyph_replaced_marks_both_cells(self):
        old = ["a", "\U0001f504", "", "b", " "]
        new = ["a", "b", " ", "c", " "]
        spans = cli._changed_cell_spans(old, new)
        covered = {x + dx for x, w in spans for dx in range(w)}
        self.assertIn(1, covered)
        self.assertIn(2, covered)

    def test_trailing_growth_is_covered(self):
        self.assertEqual(cli._changed_cell_spans(["a", "b"], ["a", "b", "c", " "]),
                         [(2, 2)])

    def test_trailing_padding_repaint(self):
        self.assertEqual(cli._changed_cell_spans(["a", "b", "c", " "], ["a", "b", "c", "d"]),
                         [(3, 1)])


class TestFrameRowWidget(unittest.TestCase):
    """The per-row widget repaints only on change."""

    def test_first_update_stores_padded_cells(self):
        row = cli._FrameRow()
        self.assertTrue(row.update_line("hello", "green", 8))
        self.assertEqual(row._line_cells, list("hello   "))

    def test_identical_update_is_noop(self):
        row = cli._FrameRow()
        row.update_line("hello", None, 8)
        self.assertFalse(row.update_line("hello", None, 8))

    def test_cell_change_repaints(self):
        row = cli._FrameRow()
        row.update_line("hello", None, 8)
        self.assertTrue(row.update_line("heXlo", None, 8))

    def test_resize_repaints_even_when_visible_prefix_is_unchanged(self):
        row = cli._FrameRow()
        row.update_line("hello", None, 8)
        with mock.patch.object(row, "refresh") as refresh:
            row.update_line("hello", None, 4)
        refresh.assert_called_once()

    def test_style_change_repaints(self):
        row = cli._FrameRow()
        row.update_line("hello", None, 8)
        self.assertTrue(row.update_line("hello", "red", 8))

    def test_zero_width_is_noop(self):
        row = cli._FrameRow()
        self.assertFalse(row.update_line("hello", None, 0))

    def test_render_returns_padded_styled_text(self):
        row = cli._FrameRow()
        row.update_line("hi", "bold", 5)
        self.assertEqual(row.render().plain, "hi   ")

    def test_render_styles_are_spans_not_base_style(self):
        """The frame style must ride as a Rich Text SPAN.

        Regression: the per-row rewrite passed the style as ``Text(text,
        style=...)`` (base style). Textual's ``Content.from_rich_text`` then
        kept the raw style string, which Rich parsed with its DEFAULT theme
        - so ``green`` rendered #008000 instead of the app's MONOKAI green
        #98e024 (visible as dimmed colours in the live TUI). Span-styled
        text goes through ``Style.from_rich_style(..., ansi_theme)`` and
        keeps the theme mapping.
        """
        row = cli._FrameRow()
        row.update_line("done", "green", 6)
        rendered = row.render()
        # The style string must be attached as a span (not the base style).
        self.assertTrue(rendered._spans, "expected the style as a span")
        self.assertEqual(rendered.style, "")
        self.assertEqual(rendered._spans[0].style, "green")

    def test_unstyled_rows_render_without_spans(self):
        row = cli._FrameRow()
        row.update_line("plain", None, 6)
        rendered = row.render()
        self.assertFalse(rendered._spans)
        self.assertEqual(rendered.style, "")

    def test_wide_emoji_keeps_count_adjacent(self):
        """The wide scales emoji must not gain a space before its count.

        Regression: the per-cell repainter splits wide glyphs into an atom
        plus an empty continuation cell; joining that empty cell with a
        literal space rendered ``95⚖️1`` as ``95⚖️ 1`` on every repaint.
        """
        row = cli._FrameRow()
        row.update_line("95⚖️1❌3", None, 11)
        self.assertEqual(row.render().plain, "95⚖️1❌3   ")
        self.assertNotIn("⚖️ ", row.render().plain)
        self.assertNotIn("❌ ", row.render().plain)

    def test_wide_emoji_change_differs_only_real_cells(self):
        """Replacing a judge count keeps the wide-emoji atom + continuation
        cell in the diff machinery but never introduces a literal space."""
        row = cli._FrameRow()
        row.update_line("95⚖️1", None, 10)
        self.assertTrue(row.update_line("95⚖️2", None, 10))
        self.assertEqual(row.render().plain, "95⚖️2     ")

class TestBenchmarkTUIAppRows(unittest.IsolatedAsyncioTestCase):
    """End-to-end: the app mounts one row widget per frame line."""

    async def test_rows_follow_the_frame(self):
        stop = threading.Event()
        app = cli._BenchmarkTUIApp(
            _FakeState(), stop, {"Local": "LC"},
            "  #  S Model  St", "ratSc ratTok ratTm ratTPS",
            1, [_plugin()], 0, {"Local": 2},
        )
        async with app.run_test(size=(30, 100)):
            await asyncio.sleep(0.6)
            self.assertGreater(len(app._rows), 3)
            self.assertIn("AI Benchmark", app._rows[0]._line_text)
            self.assertTrue(app._rows[0].id.startswith("row-"))

    async def test_rapid_resize_keeps_rows_and_rendering_valid(self):
        stop = threading.Event()
        app = cli._BenchmarkTUIApp(
            _FakeState(), stop, {"Local": "LC"},
            "  #  S Model  St", "ratSc ratTok ratTm ratTPS",
            1, [_plugin()], 0, {"Local": 2},
        )
        async with app.run_test(size=(30, 100)) as pilot:
            for size in ((8, 32), (30, 100), (10, 40), (30, 100)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
            self.assertTrue(app.is_attached)
            self.assertTrue(app._rows)
            self.assertTrue(all(row.is_attached for row in app._rows))

    async def test_resize_clamps_scroll_offsets(self):
        stop = threading.Event()
        app = cli._BenchmarkTUIApp(
            _FakeState(), stop, {"Local": "LC"},
            "  #  S Model  St", "x" * 300,
            1, [_plugin()], 0, {"Local": 2},
        )
        async with app.run_test(size=(30, 100)) as pilot:
            app.action_scroll_end()
            app.action_scroll_end_x()
            await pilot.resize_terminal(10, 32)
            await pilot.pause()
            self.assertLessEqual(app._scroll_y, app._max_row_offset())
            self.assertLessEqual(app._scroll_x, app._max_col_offset())

    async def test_quit_cancels_requests_and_sets_stop_event(self):
        stop = threading.Event()
        app = cli._BenchmarkTUIApp(
            _FakeState(), stop, {"Local": "LC"},
            "  #  S Model  St", "ratSc ratTok ratTm ratTPS",
            1, [_plugin()], 0, {"Local": 2},
        )
        with mock.patch("benchmark.cli.close_active_requests") as close_requests:
            async with app.run_test(size=(30, 100)) as pilot:
                app.action_quit_tui()
                await pilot.pause()
        self.assertTrue(stop.is_set())
        close_requests.assert_called()


class TestHorizontalScrollActions(unittest.IsolatedAsyncioTestCase):
    """Shift+Left/Right pages horizontally, Ctrl+Left/Right jumps to the
    extremes - mirroring the vertical Page Up/Down and Home/End bindings."""

    WIDE_PLUGIN_HDR = ("rateLim rateTok rateTm rateTPS judgeSc judgeTok "
                       "codeRev debugTr errorRec moeDens multiStp "
                       "multiTurn orchestr prdCreat rateLim2 softArch "
                       "structOut toolCall wirefrms extraCol")

    def _make_app(self, width=100, height=30):
        stop = threading.Event()
        return cli._BenchmarkTUIApp(
            _FakeState(), stop, {"Local": "LC"},
            "  #  S Model  St", self.WIDE_PLUGIN_HDR,
            1, [_plugin()], 0, {"Local": 2},
        )

    async def test_bindings_registered(self):
        app = self._make_app()
        binding_keys = {key for key, _action, _desc in app.BINDINGS}
        self.assertIn("shift+left", binding_keys)
        self.assertIn("shift+right", binding_keys)
        self.assertIn("ctrl+left", binding_keys)
        self.assertIn("ctrl+right", binding_keys)
        by_key = {key: action for key, action, _desc in app.BINDINGS}
        self.assertEqual(by_key["shift+left"], "scroll_page_left")
        self.assertEqual(by_key["shift+right"], "scroll_page_right")
        self.assertEqual(by_key["ctrl+left"], "scroll_home_x")
        self.assertEqual(by_key["ctrl+right"], "scroll_end_x")

    async def test_page_left_right_scrolls_by_visible_width(self):
        app = self._make_app()
        async with app.run_test(size=(100, 30)):
            page = app._visible_cols()
            self.assertGreater(page, 0)
            max_off = app._max_col_offset()
            self.assertGreater(max_off, 0, "test needs a header wider than the view")

            app.action_scroll_right()  # move off the left edge first
            self.assertGreater(app._scroll_x, 0)

            app.action_scroll_page_left()
            self.assertEqual(app._scroll_x, 0)

            app.action_scroll_page_right()
            self.assertEqual(app._scroll_x, min(max_off, 0 + page))
            # No past-the-end overflow.
            app.action_scroll_page_right()
            self.assertEqual(app._scroll_x, min(max_off, page + page))
            self.assertLessEqual(app._scroll_x, max_off)

    async def test_ctrl_left_right_jumps_to_extremes(self):
        app = self._make_app()
        async with app.run_test(size=(100, 30)):
            max_off = app._max_col_offset()
            self.assertGreater(max_off, 0)

            app.action_scroll_end_x()
            self.assertEqual(app._scroll_x, max_off)

            app.action_scroll_home_x()
            self.assertEqual(app._scroll_x, 0)

            # From a mid-page position, home/end still go to the extremes.
            app._scroll_x = max_off // 2
            app.action_scroll_home_x()
            self.assertEqual(app._scroll_x, 0)
            app.action_scroll_end_x()
            self.assertEqual(app._scroll_x, max_off)

    async def test_page_left_clamps_at_zero(self):
        app = self._make_app()
        async with app.run_test(size=(100, 30)):
            app._scroll_x = 1
            app.action_scroll_page_left()
            self.assertEqual(app._scroll_x, 0)


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
