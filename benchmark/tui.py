"""Textual live benchmark application."""
from __future__ import annotations

import os
import sys
import traceback
from typing import ClassVar

from textual.app import App
from textual.geometry import Region
from textual.widgets import Static

from benchmark.http import close_active_requests, get_429_stats
from benchmark.tui_format import (
    FROZEN_VIEW_WIDTH,
    MODEL_NUMBER_COLUMN_WIDTH,
    SCORE_COLUMN_WIDTH,
)

# Runtime rendering hooks are assigned by benchmark.cli after import. Keeping
# them as explicit hooks avoids importing the orchestration module here.
_build_frame_lines = None
_display_width = None
_line_cells = None
_changed_cell_spans = None
_FRAME_STYLE_MAP = {}
_TUI_REFRESH_SECONDS = 0.5
_unique_source_abbrevs = None
_fallback_tui_loop = None

class _FrameRow(Static):  # pragma: no cover - live interactive loop
    """One TUI frame line that repaints only the cells that changed.

    ``Static.update`` marks the entire widget dirty, so a whole-screen frame
    repainted every tick rewrites every cell of the terminal every 0.2s -
    visible as screen-wide cursor flicker on terminals without synchronized
    output. Each row is therefore its own widget, and ``update_line`` marks
    only the changed column spans (via ``refresh(*regions)``) so the
    compositor emits just those cells.
    """

    def __init__(self):
        super().__init__(markup=False)
        self.styles.width = "1fr"
        self.styles.height = 1
        self._line_text = ""
        self._line_style = None
        self._line_cells = None
        self._line_width = 0

    def render(self):
        from rich.text import Text

        text = Text(self._line_text)
        style = _FRAME_STYLE_MAP.get(self._line_style, "")
        if style:
            # Apply the style as a SPAN (not the Text base style) so Textual's
            # ``Content.from_rich_text`` converts it through the app's ANSI
            # theme (``Style.from_rich_style(..., ansi_theme)``). A base-style
            # string stays raw and is parsed with Rich's default theme, which
            # shifts named colors - e.g. ``green`` renders #008000 instead of
            # the app's MONOKAI green #98e024 (this was the colour regression
            # when the per-row rewrite moved from ``append()`` spans to a
            # base style).
            text.stylize(style)
        return text

    def update_line(self, content, style, width):
        """Switch to ``content`` (styled ``style``), repainting changed cells only.

        Returns ``True`` when the row changed (and a repaint was queued). The
        frame is rebuilt every tick but identical rows are left untouched, so
        an idle frame costs no terminal output at all.
        """
        if width <= 0:
            return False
        cells = _line_cells(content, width)
        if (self._line_cells is not None and cells == self._line_cells
                and style == self._line_style and width == self._line_width):
            return False
        spans = []
        if self._line_cells is not None:
            spans = _changed_cell_spans(self._line_cells, cells)
            if (not spans
                    and (style != self._line_style or width != self._line_width)):
                spans = [(0, width)]
        else:
            spans = [(0, width)]
        # Join cell atoms WITHOUT inserting a space for the empty
        # continuation cells that follow wide graphemes (an emoji occupies
        # two display columns but only one atom here). Emitting a literal
        # space there would push the following character apart, e.g.
        # rendering ``95⚖️1`` as ``95⚖️ 1`` on every repaint.
        self._line_text = "".join(cells)
        self._line_style = style
        self._line_cells = cells
        self._line_width = width
        if spans:
            self.refresh(*[Region(x, 0, w, 1) for x, w in spans if w > 0])
        return True


class _BenchmarkTUIApp(App):  # pragma: no cover - live interactive loop
    """Textual app that re-renders the benchmark status frame adaptively.

    The frame is rebuilt at most every ``_TUI_REFRESH_SECONDS`` (2 fps), and
    only when something it displays actually changed: the state revision
    bumped, the terminal resized, the operator scrolled, or live elapsed/
    countdown content is ticking. An idle run rebuilds nothing. Each rebuilt
    frame is delivered to one ``_FrameRow`` widget per line; each row repaints
    only the cells that changed, so a repeated frame produces no terminal
    output.
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "quit_tui", "Quit"),
        ("up", "scroll_up", "Up"),
        ("down", "scroll_down", "Down"),
        ("left", "scroll_left", "Left"),
        ("right", "scroll_right", "Right"),
        ("pageup", "scroll_page_up", "Page Up"),
        ("pagedown", "scroll_page_down", "Page Down"),
        ("home", "scroll_home", "Top"),
        ("end", "scroll_end", "Bottom"),
        ("shift+left", "scroll_page_left", "Page Left"),
        ("shift+right", "scroll_page_right", "Page Right"),
        ("ctrl+left", "scroll_home_x", "Far Left"),
        ("ctrl+right", "scroll_end_x", "Far Right"),
    ]

    def __init__(self, state, stop_event, source_abbrevs, frozen_hdr,
                 plugin_hdr, num_sources, active_plugins, session_seed,
                 model_thread_limits, close_requests=close_active_requests):
        super().__init__()
        self._state = state
        self._stop_event = stop_event
        self._source_abbrevs = source_abbrevs
        self._frozen_hdr = frozen_hdr
        self._plugin_hdr = plugin_hdr
        self._num_sources = num_sources
        self._active_plugins = active_plugins
        self._session_seed = session_seed
        self._model_thread_limits = model_thread_limits
        self._close_requests = close_requests
        self._scroll_y = 0
        self._scroll_x = 0
        self._rows: list[_FrameRow] = []
        self._last_frame_key = None

    def on_mount(self) -> None:
        self.set_interval(_TUI_REFRESH_SECONDS, self._refresh)
        self._refresh()

    def _sync_rows(self, lines) -> None:
        """Mount/unmount row widgets to match the frame, updating changed rows."""
        width = max(0, self.size.width)
        if len(self._rows) < len(lines):
            new_rows = []
            for i in range(len(self._rows), len(lines)):
                row = _FrameRow()
                row.id = f"row-{i}"
                self._rows.append(row)
                new_rows.append(row)
            # Batch the mounts so the initial frame costs a single layout pass
            # instead of one full repaint per row.
            self.mount(*new_rows)
        elif len(self._rows) > len(lines):
            for row in self._rows[len(lines):]:
                row.remove()
            del self._rows[len(lines):]
        for i, (content, style) in enumerate(lines):
            self._rows[i].update_line(content, style, width)

    def on_resize(self, event) -> None:
        """Clamp scroll and invalidate the frame during terminal resizes.

        Textual can deliver intermediate zero/small dimensions while a mobile
        terminal changes orientation. Keep offsets inside the new viewport and
        let the normal timer perform row updates after layout settles; this
        avoids issuing repaint regions against the previous row geometry.
        """
        height = max(0, event.size.height)
        width = max(0, event.size.width)
        visible_rows = max(0, height - 9 - max(3, self._num_sources + 1))
        self._scroll_y = min(
            self._scroll_y,
            max(0, len(self._state.snapshot()) - visible_rows),
        )
        visible_cols = max(0, width - FROZEN_VIEW_WIDTH - 1)
        self._scroll_x = min(
            self._scroll_x,
            max(0, _display_width(self._plugin_hdr) - visible_cols),
        )
        self._last_frame_key = None

    def _visible_rows(self) -> int:
        live_height = max(3, self._num_sources + 1)
        return max(0, self.size.height - 9 - live_height)

    def _visible_cols(self) -> int:
        return max(0, self.size.width - FROZEN_VIEW_WIDTH - 1)

    def _max_row_offset(self) -> int:
        return max(0, len(self._state.snapshot()) - self._visible_rows())

    def _max_col_offset(self) -> int:
        return max(0, _display_width(self._plugin_hdr) - self._visible_cols())

    def _frame_key(self):
        """Identity of the frame's static inputs (excludes ticking content)."""
        return (
            self._state.revision,
            self.size.height, self.size.width,
            self._scroll_y, self._scroll_x,
        )

    def _live_content(self) -> bool:
        """True when time-based frame elements are ticking.

        Elapsed/countdown fields (streaming seconds, judge-activity elapsed,
        preload seconds, 429 sleeps) change over time even when no state
        mutation occurs. While any are visible the frame is rebuilt every
        tick (capped by ``_TUI_REFRESH_SECONDS``) so they stay live; when the
        run is fully idle the frame is skipped entirely.
        """
        return (
            self._state.has_live_work()
            or bool(self._state.judge_activity_snapshot())
            or bool((get_429_stats() or {}).get("sleeping"))
        )

    def _refresh(self) -> None:
        if self._stop_event.is_set():
            self.exit()
            return
        key = self._frame_key()
        if key == self._last_frame_key and not self._live_content():
            # Nothing the frame displays changed since the last tick.
            return
        self._last_frame_key = key
        lines = _build_frame_lines(
            self._state, self._active_plugins, self._source_abbrevs,
            self._frozen_hdr, self._plugin_hdr, self._num_sources,
            self._scroll_y, self._scroll_x, (self.size.height, self.size.width),
            model_thread_limits=self._model_thread_limits,
            session_seed=self._session_seed,
        )
        self._sync_rows(lines)

    def _cancel_requests(self) -> None:
        """Cancel benchmark and judge HTTP requests before leaving the TUI."""
        self._stop_event.set()
        self._close_requests()

    def on_unmount(self) -> None:
        """Ensure app teardown also cancels requests on non-keyboard exits."""
        self._cancel_requests()

    def action_quit_tui(self) -> None:
        self._cancel_requests()
        self.exit()

    def action_scroll_up(self) -> None:
        self._scroll_y = max(0, self._scroll_y - 1)

    def action_scroll_down(self) -> None:
        self._scroll_y = min(self._max_row_offset(), self._scroll_y + 1)

    def action_scroll_left(self) -> None:
        self._scroll_x = max(0, self._scroll_x - 8)

    def action_scroll_right(self) -> None:
        self._scroll_x = min(self._max_col_offset(), self._scroll_x + 8)

    def action_scroll_page_up(self) -> None:
        self._scroll_y = max(0, self._scroll_y - self._visible_rows())

    def action_scroll_page_down(self) -> None:
        self._scroll_y = min(
            self._max_row_offset(), self._scroll_y + self._visible_rows())

    def action_scroll_page_left(self) -> None:
        self._scroll_x = max(0, self._scroll_x - self._visible_cols())

    def action_scroll_page_right(self) -> None:
        self._scroll_x = min(
            self._max_col_offset(), self._scroll_x + self._visible_cols())

    def action_scroll_home(self) -> None:
        self._scroll_y = 0

    def action_scroll_end(self) -> None:
        self._scroll_y = self._max_row_offset()

    def action_scroll_home_x(self) -> None:
        self._scroll_x = 0

    def action_scroll_end_x(self) -> None:
        self._scroll_x = self._max_col_offset()


def _tui_main_textual(state, stop_event, num_sources, active_plugins, session_seed=None,
                      model_thread_limits=None):  # pragma: no cover - live interactive loop
    """Run the live TUI with Textual, re-rendering the full frame each tick."""
    source_abbrevs = _unique_source_abbrevs({s["source"] for s in state.snapshot().values()})
    frozen_cols = [("#", MODEL_NUMBER_COLUMN_WIDTH), ("S", 4), ("Model", 18), ("St", 4)]
    frozen_hdr = " ".join(f"{h:>{w}}" for h, w in frozen_cols)
    plugin_cols = []
    for p in active_plugins:
        plugin_cols.extend([
            (f"{p.id[:3]}Sc", SCORE_COLUMN_WIDTH),
            (f"{p.id[:3]}Tok", 6),
            (f"{p.id[:3]}Tm", 6),
            (f"{p.id[:3]}TPS", 6),
        ])
    plugin_hdr = " ".join(f"{h:>{w}}" for h, w in plugin_cols)

    _BenchmarkTUIApp(
        state, stop_event, source_abbrevs, frozen_hdr, plugin_hdr,
        num_sources, active_plugins, session_seed, model_thread_limits,
        close_requests=close_active_requests,
    ).run()


def _textual_tui_enabled():
    """Return whether the Textual TUI should be used (interactive terminal)."""
    if os.environ.get("AI_BENCHMARK_NO_TEXTUAL"):
        return False
    if os.environ.get("AI_BENCHMARK_FORCE_TEXTUAL"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):  # pragma: no cover
        return False


def tui_main(state, stop_event, num_sources, active_plugins, session_seed=None,
             model_thread_limits=None):
    """Run the live TUI: Textual on interactive terminals, plain text otherwise."""
    if _textual_tui_enabled():
        try:
            _tui_main_textual(state, stop_event, num_sources, active_plugins,
                              session_seed, model_thread_limits)
            return
        except Exception:  # noqa: BLE001 - a Textual failure must fall back to plain text
            try:
                with open("tui_render_errors.log", "a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
            except Exception:  # noqa: BLE001, S110 - logging must not crash the TUI thread
                pass
    _fallback_tui_loop(state, stop_event, session_seed, model_thread_limits)


