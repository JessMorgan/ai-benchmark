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
import subprocess
import sys
import threading
import time

from benchmark_core import (
    BenchmarkState,
    _unique_source_abbrevs,
    dump_default_config,
    generate_config_from_api,
    get_model_plugins_blacklist,
    load_config,
    parse_plugin_temperatures,
    resolve_model_sources,
    run_model,
    _save_outputs,
)
from benchmark_http import close_active_requests
from plugins import discover_plugins, format_plugin_list
from shell_completion import generate_shell_completion

DEFAULT_CONFIG_PATH = "benchmark-config.json"


def _wr(stdscr, max_x, max_y, y, x, text, attr=0):
    """Write text to the curses screen, bounded by the terminal size."""
    if not (0 <= y < max_y and 0 <= x < max_x):
        return
    stdscr.move(y, x)
    stdscr.clrtoeol()
    try:
        stdscr.addstr(y, x, text[:max_x - x], attr)
    except Exception:
        try:
            stdscr.addstr(y, x, text[:max_x - x])
        except Exception:
            pass


def _fallback_tui_loop(state, stop_event, session_seed=None):
    """Fallback terminal UI when curses is unavailable."""
    while not stop_event.is_set():
        snap = state.snapshot()
        active = sum(1 for s in snap.values() if s["status"].startswith("running_") or s["status"] == "queued")
        done = state.completed
        total = state.total
        seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
        parts = [f"{seed_info}🔄 {active} active  |  ✅ {done}/{total} completed"]
        for name, s in snap.items():
            if s["status"].startswith("running_"):
                elapsed = (time.time() - s.get("attempt_start", 0)) if s.get("attempt_start") else 0
                err = s.get("last_error", "")
                parts.append(f"  {name[:30]}: {s['phase_detail']} "
                             f"att {s['attempt']}/3 tok {s['max_tok']} "
                             f"{elapsed:.0f}s{' '+err if err else ''}")
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.write(" | ".join(parts))
        sys.stdout.flush()
        # Sleep in short increments so Ctrl+C is handled promptly.
        stop_event.wait(0.2)
    print()


def _handle_tui_input(stdscr, scroll_y, scroll_x, max_row_offset, visible_rows, max_x, frozen_width, plugin_hdr_len):
    """Handle keyboard navigation and return updated scroll offsets."""
    key = stdscr.getch()
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
                                scroll_y, visible_rows, total_models, session_seed):
    """Render the top header and summary statistics."""
    from datetime import datetime
    ts = datetime.now().strftime('%H:%M:%S')
    seed_info = f"Seed: {session_seed}  |  " if session_seed is not None else ""
    hdr = f"AI Benchmark — Parallel  |  {seed_info}{ts}"
    if max_x > len(hdr):
        _wr(stdscr, max_x, max_y, 0, 0, hdr, curses.A_BOLD)

    failed_count = sum(1 for s in snap.values() if s["status"] == "failed")
    err_indicator = f"  |  \u26a0 {failed_count} failed" if failed_count else ""
    summary = (f"Total: {total}  |  "
               f"Done: {done}  |  "
               f"Active: {len(running)}  |  "
               f"Queued: {len(queued + pending)}"
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


def _format_model_row(name, s, display_idx, active_plugins, source_abbrevs):
    """Format a single model row into frozen and plugin strings."""
    sv = s["status"]
    status_ch = {"pending": "\u23f3", "queued": "\u23f3",
                 "completed": "\u2705", "failed": "\u274c"}.get(sv, "?")
    if sv.startswith("running_"):
        status_ch = "\U0001f537"

    def fmt_val(v, fmt=".1f"):
        if v is None:
            return "-"
        try:
            return f"{v:{fmt}}"
        except Exception:
            return str(v)

    src_ab = source_abbrevs.get(s["source"], s["source"][:3])
    model_disp = name[:16]
    frozen = f"{display_idx:>3}  {src_ab:<3} {model_disp:<18}  {status_ch:<3}"

    plugin_parts = []
    for p in active_plugins:
        pid = p.id
        sc = fmt_val(s.get(f"{pid}_score"))
        tok = fmt_val(s.get(f"{pid}_output_tokens"), "d")
        tm = fmt_val(s.get(f"{pid}_response_time"))
        tps = fmt_val(s.get(f"{pid}_tps"))
        plugin_parts.extend([sc, tok, tm, tps])
    plugin_str = " ".join(f"{v:>{6 if i % 4 != 0 else 5}}" for i, v in enumerate(plugin_parts))
    return frozen, plugin_str


def _render_model_rows(stdscr, max_x, max_y, snap_items, active_plugins, source_abbrevs,
                       scroll_y, scroll_x, visible_rows, frozen_width, model_top):
    """Render the scrollable model status table."""
    total_models = len(snap_items)
    for row_idx in range(visible_rows):
        abs_idx = scroll_y + row_idx
        if abs_idx >= total_models:
            break
        name, s = snap_items[abs_idx]
        display_idx = abs_idx + 1
        frozen, plugin_str = _format_model_row(name, s, display_idx, active_plugins, source_abbrevs)
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
        elif sv.startswith("running_"):
            try:
                attr = curses.color_pair(2)
            except Exception:
                pass
        _wr(stdscr, max_x, max_y, model_top + row_idx, 0, line, attr)

    for r in range(model_top + min(visible_rows, max(0, total_models - scroll_y)), model_top + visible_rows):
        try:
            stdscr.move(r, 0)
            stdscr.clrtoeol()
        except Exception:
            pass


def _render_live_activity(stdscr, max_x, max_y, snap, source_abbrevs, live_models,
                          live_top, live_height, log_top):
    """Render the live activity section for currently running models."""
    live_row = live_top
    _wr(stdscr, max_x, max_y, live_row, 0, "Live:", curses.A_BOLD)
    live_row += 1
    for nm in live_models[:live_height - 1]:
        if live_row >= log_top:
            break
        s = snap[nm]
        src_ab = source_abbrevs.get(s["source"], s["source"][:3])
        elapsed = (time.time() - s.get("attempt_start", 0)) if s.get("attempt_start") else 0
        err = s.get("last_error", "")
        phase_ch = "\U0001f537"
        msg = (f" {phase_ch} [{src_ab}] {nm[:42]}: {s['phase_detail']} "
               f"Att {s['attempt']}/3  Tok {s['max_tok']}  "
               f"{elapsed:5.0f}s"
               f"{'  '+err if err else ''}")
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
            ])

        while not stop_event.is_set():
            max_y, max_x = stdscr.getmaxyx()
            snap = state.snapshot()
            snap_items = list(snap.items())
            done = state.completed
            total = state.total
            running = [n for n, s in snap.items() if s["status"].startswith("running_")]
            queued = [n for n, s in snap.items() if s["status"] == "queued"]
            pending = [n for n, s in snap.items() if s["status"] == "pending"]

            FOOTER_LINE = max_y - 1
            MAX_LOG_ROWS = 3
            LOG_TOP = FOOTER_LINE - MAX_LOG_ROWS
            LIVE_TOP = LOG_TOP - LIVE_HEIGHT
            MODEL_BOTTOM = LIVE_TOP - 1
            MODEL_TOP = 4
            VISIBLE_ROWS = max(0, MODEL_BOTTOM - MODEL_TOP)

            _render_header_and_summary(
                stdscr, max_x, max_y, snap, done, total, running, queued, pending,
                scroll_y, VISIBLE_ROWS, len(snap), session_seed
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
                scroll_y, scroll_x, VISIBLE_ROWS, frozen_width, MODEL_TOP
            )

            if MODEL_BOTTOM >= 0:
                _wr(stdscr, max_x, max_y, MODEL_BOTTOM, 0, "\u2500" * min(max_x, 60))

            _render_live_activity(
                stdscr, max_x, max_y, snap, source_abbrevs, running,
                LIVE_TOP, LIVE_HEIGHT, LOG_TOP
            )

            _render_recent_errors(stdscr, max_x, max_y, state, LOG_TOP, FOOTER_LINE)

            queuing = queued + pending
            _render_footer(stdscr, max_x, max_y, running, queuing, FOOTER_LINE)

            stdscr.refresh()
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
    parser.add_argument('--base-url', default=None,
                        help='Base URL for model discovery via /v1/models API (used with --dump-default-config)')
    parser.add_argument('--api-key', default=None,
                        help='API key for model discovery (used with --dump-default-config --base-url)')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save each model\'s plugin response text to <output_dir>/responses/')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed random seed for all API requests (default: random)')
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

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}\n"
              f"   Copy benchmark-config.json or create one with --dump-default-config.",
              file=sys.stderr)
        sys.exit(1)
    cfg = load_config(config_path)
    source_config = cfg.get("sources", {})
    models = cfg.get("models", {})
    models_source_map = resolve_model_sources(models)
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

    print(f"📋 Loaded {len(models)} models across {len(source_config)} sources "
          f"from {config_path}", file=sys.stderr)
    print(f"🔌 Active plugins: {', '.join(p.name for p in active_plugins)} "
          f"(v{', v'.join(p.version for p in active_plugins)})", file=sys.stderr)
    print(f"📂 Output directory: {output_dir}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

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
                    state = BenchmarkState(models_source_map, plugin_ids)
                elif choice == "continue":
                    state = BenchmarkState.load_state(
                        state_file, models_source_map, plugin_ids,
                        rerun_failed=not args.no_rerun_failed)
                    resumed = True
                else:
                    sys.exit(0)
            else:
                state = BenchmarkState.load_state(
                    state_file, models_source_map, plugin_ids,
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
            state = BenchmarkState(models_source_map, plugin_ids)
    else:
        state = BenchmarkState(models_source_map, plugin_ids)

    # Use the CLI --seed if provided; otherwise preserve the seed from a
    # resumed state so report exports remain consistent.
    if args.seed is not None:
        session_seed = args.seed
    elif getattr(state, "session_seed", None) is not None:
        session_seed = state.session_seed
    else:
        session_seed = random.randint(0, 2**31 - 1)
    state.session_seed = session_seed

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
    worker_errors = 0
    interrupted = False

    source_queues = {src: [] for src in set(models_source_map.values())}
    for name, src in models_source_map.items():
        info = state.snapshot().get(name, {})
        if info.get("status") in ("completed",):
            continue
        source_queues[src].append(name)

    source_threads = {}
    errors_lock = threading.Lock()

    def worker(source, model_names):
        nonlocal worker_errors
        for model_name in model_names:
            if stop_event.is_set():
                break
            try:
                model_blacklist = get_model_plugins_blacklist(cfg.get("models", {}), model_name)
                model_active_plugins = [p for p in active_plugins if p.id not in model_blacklist]
                run_model(model_name, source, state, model_active_plugins, source_config,
                          timeout, token_levels, output_dir, session_seed=session_seed,
                          global_cfg=cfg, stop_event=stop_event,
                          save_responses=args.save_responses)
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
        return

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


if __name__ == "__main__":
    main()
