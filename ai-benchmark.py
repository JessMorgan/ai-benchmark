#!/usr/bin/env python3
"""
AI Benchmark — Plugin-based benchmark for code generation and reasoning.
Supports arbitrary task plugins, versioned results, and plugin selection.

Configuration: edit benchmark-config.json (or pass --config <path>).
API keys can use ${VAR} or ${VAR:default} syntax for env-var expansion.
"""
import argparse
import curses
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

from benchmark_core import (
    BenchmarkState,
    _apply_http_retry_default,
    _unique_source_abbrevs,
    dump_default_config,
    generate_config_from_api,
    get_target_plugins_blacklist,
    load_config,
    parse_plugin_temperatures,
    resolve_targets,
    run_model,
    _save_outputs,
)
from benchmark_http import (
    close_active_requests,
    get_429_stats,
    get_active_request_count,
)
from plugins import discover_plugins, format_plugin_list
from shell_completion import generate_shell_completion

DEFAULT_CONFIG_PATH = "benchmark-config.json"


def _resolve_config_path(config_path):
    """Resolve a config path, falling back to .yaml/.yml when using the default.

    If ``config_path`` is the default and does not exist, try
    ``benchmark-config.yaml`` then ``benchmark-config.yml``. Returns the
    first existing path, or ``None`` if no fallback exists.
    """
    if os.path.exists(config_path):
        return config_path
    if os.path.basename(config_path) == "benchmark-config.json":
        base, _ = os.path.splitext(config_path)
        for ext in (".yaml", ".yml"):
            fallback = base + ext
            if os.path.exists(fallback):
                return fallback
    return None


def _write_run_info(output_dir, run_info):
    """Persist run metadata to ``run-info.json`` in the output directory."""
    path = os.path.join(output_dir, "run-info.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️  Could not write run-info.json: {e}", file=sys.stderr)


def _wr(stdscr, max_x, max_y, y, x, text, attr=0):
    """Write text to the curses screen, bounded by the terminal size."""
    if not (0 <= y < max_y and 0 <= x < max_x):
        return
    try:
        stdscr.move(y, x)
        stdscr.clrtoeol()
        try:
            stdscr.addstr(y, x, text[:max_x - x], attr)
        except curses.error:
            stdscr.addstr(y, x, text[:max_x - x])
    except curses.error:
        # Window too small or resized since getmaxyx(); skip this frame.
        pass


def _fallback_tui_loop(state, stop_event, session_seed=None):
    """Fallback terminal UI when curses is unavailable."""
    while not stop_event.is_set():
        snap = state.snapshot()
        active = sum(
            1 for s in snap.values()
            if s.get("running_pids") or s["status"] == "queued"
        )
        done = state.completed
        total = state.total
        seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
        http_threads = get_active_request_count()
        backoff_429 = get_429_stats()
        sleeping_count = len(backoff_429.get("sleeping", {}))
        parts = [
            f"{seed_info}🔄 {active} active  |  ✅ {done}/{total} completed"
            f"  |  HTTP: {http_threads}  |  429⏸ {sleeping_count}"
        ]
        for name, s in snap.items():
            if s.get("running_pids"):
                elapsed = (time.time() - s.get("attempt_start", 0)) if s.get("attempt_start") else 0
                err = s.get("last_error", "")
                msg = f"  {name[:30]} {elapsed:.0f}s"
                if err:
                    msg += f"  {err}"
                parts.append(msg)
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.write(" | ".join(parts))
        sys.stdout.flush()
        # Sleep in short increments so Ctrl+C is handled promptly.
        stop_event.wait(0.2)
    print()


def _handle_tui_input(stdscr, scroll_y, scroll_x, max_row_offset, visible_rows, max_x, frozen_width, plugin_hdr_len):
    """Handle keyboard navigation and return updated scroll offsets."""
    try:
        key = stdscr.getch()
    except Exception:
        # getch() can raise on terminal resize or when stdin is interrupted.
        key = -1
    if key == curses.KEY_UP:
        scroll_y = max(0, scroll_y - 1)
    elif key == curses.KEY_DOWN:
        scroll_y = min(max_row_offset, scroll_y + 1)
    elif key == curses.KEY_PPAGE:
        scroll_y = max(0, scroll_y - visible_rows)
    elif key == ord(' ') or key == curses.KEY_NPAGE:
        scroll_y = min(max_row_offset, scroll_y + visible_rows)
    elif key == curses.KEY_HOME:
        scroll_y = 0
    elif key == curses.KEY_END:
        scroll_y = max_row_offset
    elif key == curses.KEY_LEFT:
        scroll_x = max(0, scroll_x - 8)
    elif key == curses.KEY_RIGHT:
        scroll_x = min(max(0, plugin_hdr_len - (max_x - frozen_width)), scroll_x + 8)
    scroll_y = max(0, min(max_row_offset, scroll_y))
    return scroll_y, scroll_x


def _render_header_and_summary(stdscr, max_x, max_y, snap, done, total, running, queued, pending,
                                scroll_y, visible_rows, total_models, session_seed,
                                http_threads, sleeping_count):
    """Render the top header and summary statistics.

    ``http_threads`` is the count of in-flight HTTP responses (the wall-clock
    \"parallelism ceiling\" the network is asked to carry). ``sleeping_count``
    is the number of ``(source, model)`` pairs currently paused in a 429
    backoff window \u2014 seeing this rise is how an operator notices that the
    benchmark is rate-limited rather than making progress.
    """
    from datetime import datetime
    ts = datetime.now().strftime('%H:%M:%S')
    seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
    hdr = f"AI Benchmark \u2014 Parallel  |  {seed_info}{ts}"
    if max_x > len(hdr):
        _wr(stdscr, max_x, max_y, 0, 0, hdr, curses.A_BOLD)

    failed_count = sum(1 for s in snap.values() if s["status"] == "failed")
    err_indicator = f"  |  \u26a0 {failed_count} failed" if failed_count else ""
    summary = (f"Total: {total}  |  "
               f"Done: {done}  |  "
               f"Active: {len(running)}  |  "
               f"Queued: {len(queued + pending)}"
               f"  |  HTTP: {http_threads}"
               f"  |  429\u23f8 {sleeping_count}"
               f"{err_indicator}"
               f"  |  \u2191\u2193 rows {scroll_y + 1}-{min(total_models, scroll_y + visible_rows)}/{total_models}"
               f"  |  \u2190\u2192 cols")
    if max_y > 1 and max_x > len(summary):
        _wr(stdscr, max_x, max_y, 1, 0, summary)

    if max_y > 2:
        _wr(stdscr, max_x, max_y, 2, 0, "\u2500" * min(max_x, 80))


def _render_table_headings(stdscr, max_x, max_y, scroll_x, frozen_cols, plugin_cols, frozen_width):
    """Render the frozen and plugin column headings."""
    frozen_hdr = " ".join(f"{h:>{w}}" for h, w in frozen_cols)
    plugin_hdr_parts = [f"{h:>{w}}" for h, w in plugin_cols]
    plugin_hdr = " ".join(plugin_hdr_parts)
    if max_y > 3:
        visible_plugin_hdr = plugin_hdr[scroll_x:scroll_x + max(0, max_x - frozen_width)]
        _wr(stdscr, max_x, max_y, 3, 0, frozen_hdr + " " + visible_plugin_hdr, curses.A_UNDERLINE)
    return plugin_hdr


# Width of the per-plugin cell block rendered by ``_plugin_cell_block``.
# The standard 5-cell results layout sums to 5+6+6+6+5=28 plus 4 single
# spaces between cells = 32 chars -- so a merged bracket status centred
# in this same 32-char span lines up under the existing sub-headers
# (``RateSc RateTok RateTm RateTPS RateSt``) without reshaping the
# ``plugin_cols`` table.
PLUGIN_BLOCK_WIDTH = 32


def _fmt_value(v, fmt=".1f"):
    """Format a single cell value; ``None`` renders as ``-``.

    Used by ``_plugin_cell_block`` so a missing result reads as ``-``
    rather than as the literal string ``"None"``.
    """
    if v is None:
        return "-"
    try:
        return f"{v:{fmt}}"
    except Exception:
        return str(v)


def _plugin_cell_block(pid, s, p, sleeping_remaining):
    """Render a single per-model cell block for one plugin.

    The block is always ``PLUGIN_BLOCK_WIDTH`` (32) chars wide so it can
    be dropped into ``plugin_str`` in place of the existing 5-cell
    layout. The standard table keeps the existing 5 sub-headers per
    plugin (``RateSc RateTok RateTm RateTPS RateSt``), so a long plugin
    id like ``json-formatter`` truncates its sub-header prefix to the
    first 3 chars; the 32-char cell block deliberately matches the
    span of those 5 sub-headers so a centred bracket status reads as
    belonging to the plugin under whatever prefix the header shows.

    When the plugin is in flight (``pid in running_pids``) OR
    the model is currently in a 429 backoff sleep, the block collapses
    to a single bracket-delimited status centred in 32 chars:
        ``[waiting]``         -- in flight, streaming-capable, no first token yet
        ``[streaming]``       -- in flight, streaming-capable, first token received
        ``[running]``         -- in flight, non-streaming-capable plugin
        ``[429 sleeping Xs]`` -- model is mid-backoff (pauses the plugin task)

    When none of the above applies, the block falls back to the standard
    5-cell results layout (``score tok tm tps st``) with ``st`` set to
    ``-`` because once results are recorded the streaming glyph is no
    longer live.

    ``sleeping_remaining`` is the integer seconds remaining for the
    model's source/model entry in the 429 sleeping map, or ``None`` if
    the model is not currently 429-sleeping. The 429 message takes
    priority over the per-plugin status because the operator cares more
    about the wall-clock backoff than the per-plugin transport detail.
    """
    in_flight = pid in (s.get("running_pids") or [])
    if sleeping_remaining is not None:
        text = f"[429 sleeping {sleeping_remaining}s]"
        return f"{text:^{PLUGIN_BLOCK_WIDTH}}"
    if in_flight:
        if p.supports_streaming:
            ft = s.get(f"{pid}_first_tok_ts", 0) or 0
            text = "[streaming]" if ft else "[waiting]"
        else:
            # ``[running]`` would collide with the model's status column
            # glyph for ``status="running"``; ``[in flight]`` is the
            # unambiguous transport-only label.
            text = "[in flight]"
        return f"{text:^{PLUGIN_BLOCK_WIDTH}}"
    # Standard 5-cell results layout -- widths sum to 5+6+6+6+5=28 with
    # 4 single-space separators between cells = 32 chars, matching the
    # merged status width. The streaming glyph column (``st``) is fixed
    # to ``-`` post-flight because the live stream event is over.
    sc = _fmt_value(s.get(f"{pid}_score"))
    tok = _fmt_value(s.get(f"{pid}_output_tokens"), "d")
    tm = _fmt_value(s.get(f"{pid}_response_time"))
    tps = _fmt_value(s.get(f"{pid}_tps"))
    return f"{sc:>5} {tok:>6} {tm:>6} {tps:>6} {'-':>5}"


def _format_model_row(name, s, display_idx, active_plugins, source_abbrevs,
                      sleeping_remaining=None):
    """Format a single model row into frozen and plugin strings.

    ``sleeping_remaining`` is the integer seconds the (source, model)
    pair is expected to remain in 429 backoff. Passed through to
    ``_plugin_cell_block`` so the entire per-model row reflects the
    same wall-clock status that the live-activity footer reports.
    """
    sv = s["status"]
    status_ch = {"pending": "\u23f3", "queued": "\u23f3",
                 "completed": "\u2705", "failed": "\u274c"}.get(sv, "?")
    if sv == "running" or s.get("running_pids"):
        status_ch = "\U0001f537"

    def fmt_val(v, fmt=".1f"):
        if v is None:
            return "-"
        try:
            return f"{v:{fmt}}"
        except Exception:
            return str(v)

    src_ab = _source_abbr(source_abbrevs, s.get("source"))
    model_disp = name[:16]
    frozen = f"{display_idx:>3}  {src_ab:<3} {model_disp:<18}  {status_ch:<3}"

    # Per-plugin membership in this model's ``running_pids`` list -- decides
    # which streaming columns highlight as live. With parallel plugin threads,
    # multiple plugin ids can sit in ``running_pids`` simultaneously, so each
    # column independently reflects THIS specific plugin's progress while
    # the shared race is gone.
    running_pids_set = set(s.get("running_pids") or [])

    # Each plugin contributes exactly one 32-char block (merged status
    # OR standard 5-cell results) so ``plugin_str`` has the same total
    # length and column geometry as before. Joins the per-plugin blocks
    # with single spaces, matching the existing column-join pattern.
    plugin_parts = [
        _plugin_cell_block(p.id, s, p, sleeping_remaining)
        for p in active_plugins
    ]
    plugin_str = " ".join(plugin_parts)
    return frozen, plugin_str


def _render_model_rows(stdscr, max_x, max_y, snap_items, active_plugins, source_abbrevs,
                       scroll_y, scroll_x, visible_rows, frozen_width, model_top,
                       sleeping_lookup):
    """Render the scrollable model status table.

    ``sleeping_lookup`` maps ``(source_name, api_model) -> sleep info``
    (with ``wake_ts``, ``attempts``, ``max_attempts``) so per-model 429
    backoff state can be folded into each plugin cell via the
    ``[429 sleeping Xs]`` bracket status. Models whose key is absent
    render normally without the indicator.
    """
    total_models = len(snap_items)
    for row_idx in range(visible_rows):
        abs_idx = scroll_y + row_idx
        if abs_idx >= total_models:
            break
        name, s = snap_items[abs_idx]
        display_idx = abs_idx + 1
        src_name = s.get("source")
        api_model = s.get("api_model", name)
        sleep_info = sleeping_lookup.get((src_name, api_model))
        sleeping_remaining = None
        if sleep_info is not None:
            sleeping_remaining = max(0, int(round(sleep_info["wake_ts"] - time.time())))
        frozen, plugin_str = _format_model_row(
            name, s, display_idx, active_plugins, source_abbrevs,
            sleeping_remaining=sleeping_remaining,
        )
        visible_plugin = plugin_str[scroll_x:scroll_x + max(0, max_x - frozen_width - 1)]
        line = frozen + " " + visible_plugin

        attr = 0
        sv = s["status"]
        if sv == "completed":
            try:
                attr = curses.color_pair(1)
            except Exception:
                pass
        elif sv == "failed":
            try:
                attr = curses.color_pair(3)
            except Exception:
                pass
        elif sv == "running" or s.get("running_pids"):
            try:
                attr = curses.color_pair(2)
            except Exception:
                pass
            except Exception:
                pass
        _wr(stdscr, max_x, max_y, model_top + row_idx, 0, line, attr)

    for r in range(model_top + min(visible_rows, max(0, total_models - scroll_y)), model_top + visible_rows):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:
            pass


def _source_abbr(source_abbrevs, source):
    """Return a short abbreviation for a source, with a safe fallback."""
    if source in source_abbrevs:
        return source_abbrevs[source]
    if source is None:
        return "???"
    return str(source)[:3] or "???"


def _render_live_activity(stdscr, max_x, max_y, snap, source_abbrevs, live_models,
                          live_top, live_height, log_top, active_plugins, backoff_429):
    """Render running models + 429-sleeping models in the live area.

    Layout (rows counted from ``live_top`` upward):

      0      ``Live:``  (header)
      1..    one row per running model whose source is NOT in 429 backoff
      ?      ``429 Sleeping:``  (optional header)
      ?..    one row per ``(source, model)`` currently in a 429 backoff sleep

    Models that are simultaneously running and 429-sleeping are rendered
    only in the Sleeping section so a single backoff is never counted twice.
    The streaming/waiting indicator (``(stream)`` / ``(wait)``) is shown for
    any running plugin that supports streaming \u2014 for non-streaming plugins,
    the indicator is omitted so we never lie about a transport detail we
    cannot observe.
    """
    live_row = live_top
    _wr(stdscr, max_x, max_y, live_row, 0, "Live:", curses.A_BOLD)
    live_row += 1

    sleeping_lookup = {}
    for key, info in (backoff_429.get("sleeping") or {}).items():
        src, _, model_id = key.partition("|")
        sleeping_lookup[(src, model_id)] = info

    # Partition live_models into (running-only) and (running + 429 sleeping)
    # in a single pass. Models that are both running and 429-sleeping are
    # rendered only in the Sleeping section so a single backoff is never
    # counted twice on the operator's screen.
    running_rows = []
    sleeping_rows = []
    for nm in live_models:
        s = snap.get(nm) or {}
        src_name = s.get("source")
        api_model = s.get("api_model", nm)
        sleep_info = sleeping_lookup.get((src_name, api_model))
        if sleep_info is not None:
            sleeping_rows.append((src_name, api_model, sleep_info))
        else:
            running_rows.append((nm, s))

    for nm, s in running_rows:
        if live_row >= log_top:
            break
        src_ab = _source_abbr(source_abbrevs, s.get("source"))
        elapsed = (time.time() - s.get("attempt_start", 0)) if s.get("attempt_start") else 0
        err = s.get("last_error", "")
        msg = f" \U0001f537 [{src_ab}] {nm[:36]} {elapsed:5.0f}s"
        running_pids = s.get("running_pids") or []
        # Pick the first streaming-capable plugin in flight for the
        # ``(stream)/(wait)`` indicator. With parallel plugin threads the
        # ``running_pids`` list can carry several ids; choosing the first
        # streaming-capable one is consistent across renders and avoids
        # the last-write-wins highlight flip we had under the old shared
        # ``status="running_<pid>"`` field.
        indicator_pid = next(
            (p.id for p in active_plugins
             if p.id in running_pids and p.supports_streaming),
            None,
        )
        if indicator_pid is not None:
            ft = s.get(f"{indicator_pid}_first_tok_ts", 0) or 0
            msg += "  (stream)" if ft else "  (wait)"
        if err:
            msg += f"  {err}"
        _wr(stdscr, max_x, max_y, live_row, 0, msg)
        live_row += 1

    if sleeping_rows and live_row + 1 < log_top:
        _wr(stdscr, max_x, max_y, live_row, 0, "429 Sleeping:", curses.A_BOLD)
        live_row += 1
        for src_name, api_model, info in sleeping_rows:
            if live_row >= log_top:
                break
            src_ab = _source_abbr(source_abbrevs, src_name)
            wake_ts = info["wake_ts"]
            remaining = max(0, int(round(wake_ts - time.time())))
            msg = (f" \U0001f4a4 [{src_ab}] {api_model[:36]}  "
                   f"[429 {info['attempts']}/{info['max_attempts']} {remaining}s]")
            _wr(stdscr, max_x, max_y, live_row, 0, msg)
            live_row += 1

    for r in range(live_row, log_top):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:
            pass


def _render_recent_errors(stdscr, max_x, max_y, state, log_top, footer_line):
    """Render the recent errors section."""
    from datetime import datetime
    log_row = log_top
    recent_errors = state.recent_log(2)
    if recent_errors:
        _wr(stdscr, max_x, max_y, log_row, 0, "Errors:", curses.A_BOLD)
        log_row += 1
        for ts_entry, model_entry, msg_entry in recent_errors:
            if log_row >= footer_line:
                break
            t_str = datetime.fromtimestamp(ts_entry).strftime('%H:%M:%S')
            err_msg = f"  {t_str} [{model_entry[:20]}]: {msg_entry}"
            _wr(stdscr, max_x, max_y, log_row, 0, err_msg, curses.color_pair(3))
            log_row += 1
    for r in range(log_row, footer_line):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:
            pass


def _render_footer(stdscr, max_x, max_y, live_models, queuing, footer_line):
    """Render the bottom status line."""
    if not live_models and not queuing:
        msg = " All models complete — generating outputs..."
    else:
        q = f"{len(queuing)} queued" if queuing else ""
        a = f"{len(live_models)} active" if live_models else ""
        sep2 = "  |  " if q and a else ""
        msg = f" {a}{sep2}{q}"
    _wr(stdscr, max_x, max_y, footer_line, 0, msg)


def tui_main(state, stop_event, num_sources, active_plugins, session_seed=None):
    """Run ncurses TUI in a daemon thread. Updates every 200ms."""
    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
    except Exception:
        _fallback_tui_loop(state, stop_event, session_seed)
        return

    try:
        LIVE_HEIGHT = max(3, num_sources + 1)
        scroll_y = 0
        scroll_x = 0

        src_snap = {s["source"] for s in state.snapshot().values()}
        source_abbrevs = _unique_source_abbrevs(src_snap)

        frozen_cols = [("#", 4), ("S", 4), ("Model", 18), ("St", 4)]
        frozen_width = sum(w for _h, w in frozen_cols) + len(frozen_cols)

        plugin_cols = []
        for p in active_plugins:
            plugin_cols.extend([
                (f"{p.id[:3]}Sc", 5),
                (f"{p.id[:3]}Tok", 6),
                (f"{p.id[:3]}Tm", 6),
                (f"{p.id[:3]}TPS", 6),
                # Per-plugin streaming indicator (▶ / · / -). Width 5
                # matches the score column so vertical alignment holds.
                (f"{p.id[:3]}St", 5),
            ])

        _last_tui_error_ts = 0.0
        while not stop_event.is_set():
            try:
                max_y, max_x = stdscr.getmaxyx()
                snap = state.snapshot()
                snap_items = list(snap.items())
                done = state.completed
                total = state.total
                # In-flight models are identified by a non-empty ``running_pids`` list
                # (the canonical source-of-truth populated by
                # ``BenchmarkState.start_plugin_run``). The previous
                # ``s["status"].startswith("running_")`` check relied on a
                # legacy pid-suffix status string that the runtime no longer
                # writes (see commit message); reading the list directly is
                # robust to parallel plugin threads and to status mutations
                # from outer callers (e.g. ``status="completed"`` set after
                # the last plugin finishes).
                running = [n for n, s in snap.items() if s.get("running_pids")]
                queued = [n for n, s in snap.items() if s["status"] == "queued"]
                pending = [n for n, s in snap.items() if s["status"] == "pending"]

                FOOTER_LINE = max_y - 1
                MAX_LOG_ROWS = 3
                LOG_TOP = FOOTER_LINE - MAX_LOG_ROWS
                LIVE_TOP = LOG_TOP - LIVE_HEIGHT
                MODEL_BOTTOM = LIVE_TOP - 1
                MODEL_TOP = 4
                VISIBLE_ROWS = max(0, MODEL_BOTTOM - MODEL_TOP)

                http_threads = get_active_request_count()
                backoff_429 = get_429_stats()
                sleeping_count = len(backoff_429.get("sleeping", {}))
                # Per-model 429 lookup keyed by ``(source, api_model)``
                # so each row can fold its backoff countdown into the
                # model-row plugin cells (the ``[429 sleeping Xs]``
                # bracket status). Models whose key is absent render
                # normally without the indicator.
                sleeping_lookup = {}
                for key, info in (backoff_429.get("sleeping") or {}).items():
                    src_name, _, api_model = key.partition("|")
                    sleeping_lookup[(src_name, api_model)] = info

                _render_header_and_summary(
                    stdscr, max_x, max_y, snap, done, total, running, queued, pending,
                    scroll_y, VISIBLE_ROWS, len(snap), session_seed,
                    http_threads, sleeping_count
                )

                plugin_hdr = _render_table_headings(
                    stdscr, max_x, max_y, scroll_x, frozen_cols, plugin_cols, frozen_width
                )

                max_row_offset = max(0, len(snap_items) - VISIBLE_ROWS)
                scroll_y, scroll_x = _handle_tui_input(
                    stdscr, scroll_y, scroll_x, max_row_offset, VISIBLE_ROWS, max_x,
                    frozen_width, len(plugin_hdr)
                )

                _render_model_rows(
                    stdscr, max_x, max_y, snap_items, active_plugins, source_abbrevs,
                    scroll_y, scroll_x, VISIBLE_ROWS, frozen_width, MODEL_TOP,
                    sleeping_lookup,
                )

                if MODEL_BOTTOM >= 0:
                    _wr(stdscr, max_x, max_y, MODEL_BOTTOM, 0, "\u2500" * min(max_x, 60))

                _render_live_activity(
                    stdscr, max_x, max_y, snap, source_abbrevs, running,
                    LIVE_TOP, LIVE_HEIGHT, LOG_TOP, active_plugins, backoff_429
                )

                _render_recent_errors(stdscr, max_x, max_y, state, LOG_TOP, FOOTER_LINE)

                queuing = queued + pending
                _render_footer(stdscr, max_x, max_y, running, queuing, FOOTER_LINE)

                stdscr.refresh()
            except Exception:
                # Don't let a transient curses/render error kill the TUI thread.
                # Log to a file so the screen isn't corrupted and the benchmark
                # workers can keep running. Throttle to avoid a runaway log.
                now = time.time()
                if now - _last_tui_error_ts > 5.0:
                    _last_tui_error_ts = now
                    try:
                        with open("tui_render_errors.log", "a", encoding="utf-8") as f:
                            traceback.print_exc(file=f)
                    except Exception:
                        pass
            time.sleep(0.2)

    finally:
        curses.echo()
        curses.nocbreak()
        try:
            curses.endwin()
        except Exception:
            pass


def _prompt_restart_or_continue(scripted=False):
    """Ask the user whether to restart or continue a run with changed plugins.

    In scripted mode, no input is requested and the run continues automatically,
    running any new plugins on all models while preserving existing results.
    """
    print("\nPlugin set has changed since the last run.")
    if scripted:
        print("   Scripted mode: continuing run and running new plugins on all models.",
              file=sys.stderr)
        return "continue"
    print("[r] Restart run (discard old state)")
    print("[c] Continue run (keep old data, run missing plugins)")
    print("[q] Quit")
    while True:
        try:
            choice = input("Choice [r/c/q]: ").strip().lower()
        except EOFError:
            choice = "q"
        if choice in ("r", "restart"):
            return "restart"
        if choice in ("c", "continue"):
            return "continue"
        if choice in ("q", "quit"):
            return "quit"
        print("Please enter r, c, or q.")


def main():
    try:
        subprocess.run(['stty', 'sane'], stderr=subprocess.DEVNULL,
                       stdin=sys.stdin, timeout=1)
    except Exception:
        pass
    sys.stderr.write('\033[2J\033[H')
    sys.stderr.flush()

    parser = argparse.ArgumentParser(
        description="AI Model Benchmark — Run plugin-based benchmarks across multiple API sources.",
        epilog="Challenge plugins are loaded from plugins/challenges/ and report plugins from plugins/outputs/.\n\n"
               "Examples:\n"
               "  python ai-benchmark.py --restart\n"
               "  python ai-benchmark.py --config my-config.json\n"
               "  python ai-benchmark.py --out /tmp/bench-run --timeout 300\n"
               "  python ai-benchmark.py --plugins-whitelist rate-limiter\n"
               "  python ai-benchmark.py --dump-default-config --base-url http://localhost:11434 > config.json\n"
               "  python ai-benchmark.py --dump-default-config > benchmark-config.json\n\n"
               "Shell completions:\n"
               "  eval \"$(python ai-benchmark.py --generate-shell-completion bash)\"\n"
               "  python ai-benchmark.py --generate-shell-completion zsh > ~/.zsh/completions/_ai-benchmark.py\n"
               "  python ai-benchmark.py --generate-shell-completion fish > ~/.config/fish/completions/ai-benchmark.py.fish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--restart', action='store_true',
                        help='Restart the run from scratch, discarding prior results')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                        help=f'Config file path (default: {DEFAULT_CONFIG_PATH})')
    parser.add_argument('--out', default=None,
                        help='Override output directory from config')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Override request timeout in seconds from config')
    parser.add_argument('--token-levels', type=int, nargs='+', default=None,
                        help='Override token levels (e.g. --token-levels 4096 8192 16384)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Default temperature for all plugins (overrides config; individual --plugin-temperature takes priority)')
    parser.add_argument('--plugin-temperature', type=str, nargs='+', default=None,
                        help='Per-plugin temperatures as id=value (e.g. --plugin-temperature rate-limiter=0.2 moe-dense=0.7)')
    parser.add_argument('--plugin-thread-limit', type=int, default=None,
                        help='Max threads per model for plugin execution. 0 means one thread per plugin (default: 1)')
    parser.add_argument('--plugins-whitelist', type=str, nargs='+', default=None,
                        help='Run only these plugins (e.g. --plugins-whitelist rate-limiter moe-dense)')
    parser.add_argument('--plugins-blacklist', type=str, nargs='+', default=None,
                        help='Run all plugins except these (e.g. --plugins-blacklist moe-dense)')
    parser.add_argument('--list-plugins', action='store_true',
                        help='List discovered challenge plugins (from plugins/challenges/) with their IDs, names, and versions, then exit')
    parser.add_argument('--generate-shell-completion', type=str, default=None,
                        choices=['bash', 'zsh', 'fish'],
                        help='Generate shell completion script for the specified shell and exit')
    parser.add_argument('--dump-default-config', action='store_true',
                        help='Print a default config file to stdout and exit')
    parser.add_argument('--convert-config', type=str, default=None,
                        help='Convert a YAML config to JSON or a JSON config to YAML and print to stdout')
    parser.add_argument('--base-url', default=None,
                        help='Base URL for model discovery via /v1/models API (used with --dump-default-config)')
    parser.add_argument('--api-key', default=None,
                        help='API key for model discovery (used with --dump-default-config --base-url)')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save each model\'s plugin response text to <output_dir>/responses/')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed random seed for all API requests (default: random)')
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument('--retry-on-429', action='store_true', default=True,
                             help='Retry HTTP 429 responses with exponential backoff for any source '
                                  'that did not set its own max_429_retries (default: enabled). Each '
                                  'rate-limited request can sleep up to (max_429_retries x max_backoff_seconds) '
                                  'before failing. Use --no-retry-on-429 if you want the legacy fail-fast '
                                  'behaviour back.')
    retry_group.add_argument('--no-retry-on-429', action='store_false',
                             help='Disable HTTP 429 retries globally. Overrides per-source max_429_retries '
                                  'only when the source did not set its own; explicit per-source values '
                                  'are preserved.')
    parser.add_argument('--no-rerun-failed', action='store_true',
                        help='Do not re-run models that failed in a previous session')
    parser.add_argument('--scripted', action='store_true',
                        help='Non-interactive mode: never prompt for input; default to continuing runs')
    args = parser.parse_args()

    if args.list_plugins:
        print(format_plugin_list(discover_plugins()))
        sys.exit(0)

    if args.generate_shell_completion:
        print(generate_shell_completion(args.generate_shell_completion, discover_plugins()))
        sys.exit(0)

    if args.dump_default_config:
        if args.base_url:
            cfg = generate_config_from_api(args.base_url, args.api_key)
            print(json.dumps(cfg, indent=2))
        else:
            dump_default_config()
        sys.exit(0)

    if args.convert_config:
        if not os.path.exists(args.convert_config):
            print(f"❌ Config file not found: {args.convert_config}", file=sys.stderr)
            sys.exit(1)
        ext = os.path.splitext(args.convert_config)[1].lower()
        if ext not in (".json", ".yaml", ".yml"):
            print(f"❌ Unsupported config format: {ext}. Use .json, .yaml, or .yml.", file=sys.stderr)
            sys.exit(1)
        cfg = load_config(args.convert_config)
        if ext in (".yaml", ".yml"):
            print(json.dumps(cfg, indent=2))
        else:
            import yaml
            print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        sys.exit(0)

    config_path = _resolve_config_path(args.config)
    if config_path is None:
        print(f"❌ Config file not found: {args.config}\n"
              f"   Tried: benchmark-config.json, benchmark-config.yaml, benchmark-config.yml\n"
              f"   Copy benchmark-config.json or create one with --dump-default-config.",
              file=sys.stderr)
        sys.exit(1)
    cfg = load_config(config_path)
    # Apply the global --retry-on-429 / --no-retry-on-429 toggle before any
    # plugin sees the config so per-source defaults are aligned with the flag.
    _apply_http_retry_default(cfg, args.retry_on_429)
    source_config = cfg.get("sources", {})
    models = cfg.get("models", {})
    agents = cfg.get("agents", {})
    collisions = set(models) & set(agents)
    if collisions:
        print(f"❌ Model/agent name collision: {', '.join(sorted(collisions))}", file=sys.stderr)
        sys.exit(1)
    targets = resolve_targets(cfg)
    output_dir = cfg.get("output_dir", "benchmark-results")
    if args.out:
        output_dir = args.out
    state_file = os.path.join(output_dir, "benchmark_state.json")

    timeout = cfg.get("timeout", 600)
    if args.timeout is not None:
        timeout = args.timeout

    token_levels = cfg.get("token_levels", [16384])
    if args.token_levels is not None:
        token_levels = args.token_levels

    whitelist = args.plugins_whitelist or cfg.get("plugins_whitelist") or None
    blacklist = args.plugins_blacklist or cfg.get("plugins_blacklist") or None
    if whitelist and blacklist:
        print("❌ Cannot specify both --plugins-whitelist and --plugins-blacklist.", file=sys.stderr)
        sys.exit(1)
    try:
        active_plugins = discover_plugins(whitelist=whitelist, blacklist=blacklist)
    except Exception as e:
        print(f"❌ Failed to discover plugins: {e}", file=sys.stderr)
        sys.exit(1)

    if not active_plugins:
        print("❌ No plugins selected. Check your whitelist/blacklist.", file=sys.stderr)
        sys.exit(1)

    # Per-plugin temperatures: CLI overrides config. Config keys may use either
    # hyphen or underscore, e.g. "rate-limiter_temperature" or "rate_servererature".
    plugin_temperatures = parse_plugin_temperatures(cfg)
    if args.temperature is not None:
        for plugin in active_plugins:
            plugin_temperatures[plugin.id] = args.temperature
    if args.plugin_temperature:
        for item in args.plugin_temperature:
            if "=" not in item:
                print(f"❌ Invalid --plugin-temperature value: {item}. Expected id=value.", file=sys.stderr)
                sys.exit(1)
            pid, temp_str = item.split("=", 1)
            try:
                plugin_temperatures[pid] = float(temp_str)
            except ValueError:
                print(f"❌ Invalid temperature for {pid}: {temp_str}", file=sys.stderr)
                sys.exit(1)
    cfg["plugin_temperatures"] = plugin_temperatures

    # Apply per-source plugin_thread_limit defaults and CLI override.
    # Top-level plugin_thread_limit is used as a fallback for sources that
    # do not define their own value.
    for src_cfg in source_config.values():
        src_cfg["plugin_thread_limit"] = src_cfg.get(
            "plugin_thread_limit", cfg.get("plugin_thread_limit", 1)
        )
    if args.plugin_thread_limit is not None:
        for src_cfg in source_config.values():
            src_cfg["plugin_thread_limit"] = args.plugin_thread_limit

    print(f"📋 Loaded {len(targets)} targets ({len(models)} models, {len(agents)} agents) "
          f"across {len(source_config)} sources from {config_path}", file=sys.stderr)
    print(f"🔌 Active plugins: {', '.join(p.name for p in active_plugins)} "
          f"(v{', v'.join(p.version for p in active_plugins)})", file=sys.stderr)
    print(f"📂 Output directory: {output_dir}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

    try:
        shutil.copy2(config_path, os.path.join(output_dir, os.path.basename(config_path)))
    except Exception as e:
        print(f"⚠️  Could not copy config file to output directory: {e}", file=sys.stderr)

    state = None
    worker_errors = 0
    interrupted = False
    run_info = {
        "config_file": config_path,
        "cli_args": vars(args),
        "output_dir": output_dir,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "status": "running",
        "total_targets": len(targets),
        "completed_targets": 0,
        "worker_errors": 0,
        "session_seed": None,
        "active_plugins": [p.id for p in active_plugins],
        "targets": list(targets.keys()),
    }

    try:
        if args.restart:
            if os.path.exists(state_file):
                os.remove(state_file)
            for f in glob.glob(os.path.join(output_dir, "results.*")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            logs_dir = os.path.join(output_dir, "logs")
            if os.path.isdir(logs_dir):
                for f in glob.glob(os.path.join(logs_dir, "*.log")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        plugin_ids = [p.id for p in active_plugins]
        plugin_versions = {p.id: p.version for p in active_plugins}

        resumed = False
        if not args.restart and os.path.exists(state_file):
            try:
                with open(state_file) as f:
                    saved_state = json.load(f)
                saved_plugins = saved_state.get("active_plugins", [])

                if set(saved_plugins) != set(plugin_ids):
                    print("\n⚠️  Plugin set has changed.", file=sys.stderr)
                    print(f"   Saved:   {', '.join(saved_plugins) or '(none)'}", file=sys.stderr)
                    print(f"   Current: {', '.join(plugin_ids)}", file=sys.stderr)
                    choice = _prompt_restart_or_continue(scripted=args.scripted)
                    if choice == "restart":
                        os.remove(state_file)
                        state = BenchmarkState(targets, plugin_ids)
                    elif choice == "continue":
                        state = BenchmarkState.load_state(
                            state_file, targets, plugin_ids,
                            rerun_failed=not args.no_rerun_failed)
                        resumed = True
                    else:
                        sys.exit(0)
                else:
                    state = BenchmarkState.load_state(
                        state_file, targets, plugin_ids,
                        rerun_failed=not args.no_rerun_failed)
                    resumed = True

                if resumed:
                    completed = state.completed
                    total = state.total
                    print(f"📂 Resuming — {completed}/{total} models already completed. "
                          f"Failed models/plugins will be re-run.\n"
                          f"   Remove {state_file} or use --restart to start fresh.",
                          file=sys.stderr)

                    if completed == total and total > 0:
                        print(f"\n{'='*70}")
                        print(f"✅ PRIOR RUN COMPLETE — {completed}/{total} successful")
                        print(f"   Results: {output_dir}/")
                        print(f"{'='*70}")
                        sys.exit(0)
            except Exception as e:
                print(f"⚠️  Could not load state file ({e}), starting fresh.",
                      file=sys.stderr)
                state = BenchmarkState(targets, plugin_ids)
        else:
            state = BenchmarkState(targets, plugin_ids)

        # Use the CLI --seed if provided; otherwise preserve the seed from a
        # resumed state so report exports remain consistent.
        if args.seed is not None:
            session_seed = args.seed
        elif getattr(state, "session_seed", None) is not None:
            session_seed = state.session_seed
        else:
            session_seed = random.randint(0, 2**31 - 1)
        state.session_seed = session_seed
        run_info["session_seed"] = session_seed

        stop_event = threading.Event()

        # Run the TUI as a daemon so the process can exit promptly on Ctrl+C.
        # Without this, a stuck curses/fallback UI thread would block interpreter
        # shutdown, forcing the user to press Ctrl+C a second time.
        tui_thread = threading.Thread(
            target=tui_main,
            args=(state, stop_event, len(source_config), active_plugins, session_seed),
            daemon=True,
        )
        tui_thread.start()

        time.sleep(0.3)

        total = state.total

        source_queues = {src: [] for src in set(t["source"] for t in targets.values())}
        for name, info in targets.items():
            snap = state.snapshot().get(name, {})
            if snap.get("status") in ("completed",):
                continue
            source_queues[info["source"]].append(name)

        source_threads = {}
        errors_lock = threading.Lock()
        raw_targets = {}
        raw_targets.update(cfg.get("models", {}))
        raw_targets.update(cfg.get("agents", {}))

        def worker(source, model_names):
            nonlocal worker_errors
            for model_name in model_names:
                if stop_event.is_set():
                    break
                try:
                    model_blacklist = get_target_plugins_blacklist(raw_targets, model_name)
                    model_active_plugins = [p for p in active_plugins if p.id not in model_blacklist]
                    target_info = targets[model_name]
                    run_model(model_name, source, state, model_active_plugins, source_config,
                              timeout, token_levels, output_dir, session_seed=session_seed,
                              global_cfg=cfg, stop_event=stop_event,
                              save_responses=args.save_responses,
                              api_model=target_info["api_model"],
                              system_prompt=target_info["system_prompt"],
                              is_agent=target_info["is_agent"])
                    state.save_state(state_file, plugin_versions=plugin_versions)
                    _save_outputs(state, output_dir, active_plugins)
                except Exception as e:
                    with errors_lock:
                        worker_errors += 1
                    print(f"\n❌ Worker exception ({model_name}): {type(e).__name__}: {e}",
                          file=sys.stderr)

        for source, queue in source_queues.items():
            if not queue:
                continue
            t = threading.Thread(target=worker, args=(source, queue), daemon=True)
            t.start()
            source_threads[source] = t

        def _join_workers(timeout=None):
            """Wait for worker threads with an optional timeout.

            Returns True if all workers finished, False if any are still alive.
            """
            if not source_threads:
                return True
            if timeout is None:
                # Poll with short timeouts so Ctrl+C is handled promptly.
                while any(t.is_alive() for t in source_threads.values()):
                    for t in source_threads.values():
                        t.join(timeout=0.2)
                return True
            for t in source_threads.values():
                t.join(timeout=timeout / max(len(source_threads), 1))
            return not any(t.is_alive() for t in source_threads.values())

        if not source_threads:
            print("✅ All models already completed. Nothing to run.", file=sys.stderr)
        else:
            try:
                _join_workers()
            except KeyboardInterrupt:
                interrupted = True
                run_info["status"] = "interrupted"
                stop_event.set()
                print("\n\n⚠️  Ctrl+C — saving state and shutting down...", file=sys.stderr)
                close_active_requests()
                # Workers are daemon threads, so the process can exit without
                # waiting for them. Give them a brief grace period to finish
                # cleanly, but do not block shutdown on a slow I/O call.
                _join_workers(timeout=1.0)

        stop_event.set()
        # The TUI thread is a daemon, so we don't need to wait for it. A short
        # timeout keeps the terminal tidy if it happens to finish quickly.
        tui_thread.join(timeout=0.5)

        try:
            state.save_state(state_file, plugin_versions=plugin_versions)
        except Exception:
            pass

        if interrupted:
            done = state.completed
            print(f"✅ Saved state ({done}/{total} done). Re-run without --restart to continue.\n",
                  file=sys.stderr)
        else:
            _save_outputs(state, output_dir, active_plugins)
        final_results = state.latest_results()
        ok_count = len([r for r in final_results if r["status"] == "ok"])
        print(f"\n{'='*70}")
        print(f"AI BENCHMARK COMPLETE — {ok_count}/{total} successful "
              f"({worker_errors} worker errors)")
        print(f"Outputs: {output_dir}/")
        for fname in sorted(os.listdir(output_dir)):
            print(f"  - {fname}")
        print(f"{'='*70}")
    except KeyboardInterrupt:
        run_info["status"] = "interrupted"
        raise
    except Exception as exc:
        run_info["status"] = "crashed"
        run_info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run_info["end_time"] = datetime.now().isoformat()
        run_info["completed_targets"] = state.completed if state is not None else 0
        run_info["worker_errors"] = worker_errors
        if run_info["status"] == "running":
            run_info["status"] = "completed"
        _write_run_info(output_dir, run_info)


if __name__ == "__main__":
    main()
