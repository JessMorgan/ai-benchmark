#!/usr/bin/env python3
"""
AI Benchmark — Plugin-based benchmark for code generation and reasoning.
Supports arbitrary task plugins, versioned results, and plugin selection.

Configuration: edit benchmark-config.json (or pass --config <path>).
API keys can use ${VAR} or ${VAR:default} syntax for env-var expansion.
"""
import argparse
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
    _source_abbrev,
    dump_default_config,
    generate_config_from_api,
    load_config,
    run_model,
    _save_outputs,
)
from plugins import discover_plugins

DEFAULT_CONFIG_PATH = "benchmark-config.json"


def tui_main(state, stop_event, num_sources, active_plugins):
    """Run ncurses TUI in a daemon thread. Updates every 200ms."""
    from datetime import datetime
    import curses

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
        while not stop_event.is_set():
            snap = state.snapshot()
            active = sum(1 for s in snap.values() if s["status"].startswith("running_") or s["status"] == "queued")
            done = state.completed
            total = state.total
            parts = [f"🔄 {active} active  |  ✅ {done}/{total} completed"]
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
            time.sleep(1)
        print()
        return

    try:
        LIVE_HEIGHT = max(3, num_sources + 1)
        scroll_y = 0
        scroll_x = 0

        src_snap = {s["source"] for s in state.snapshot().values()}
        source_abbrevs = {}
        _used = set()
        for src in sorted(src_snap):
            ab = _source_abbrev(src)
            if ab in _used:
                tokens = []
                for w in src.split():
                    if w.isupper() and 1 < len(w) <= 3:
                        tokens.append(w)
                    else:
                        sub = __import__('re').findall(r'[A-Z]?[a-z]+|[A-Z]+', w)
                        tokens.extend(sub) if sub else tokens.append(w)
                ab = ''.join(t[:2].upper() for t in tokens if t)
                if ab in _used or len(ab) < 2:
                    ab = (src * 2)[:2].upper()
                    if ab in _used:
                        for i in range(2, min(len(src), 6)):
                            ab = src[:i].upper()
                            if ab not in _used:
                                break
            source_abbrevs[src] = ab
            _used.add(ab)

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

        def render_row(name, s, display_idx):
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

        while not stop_event.is_set():
            max_y, max_x = stdscr.getmaxyx()
            snap = state.snapshot()
            snap_items = list(snap.items())
            done = state.completed
            total = state.total
            running = [n for n, s in snap.items() if s["status"].startswith("running_")]
            queued = [n for n, s in snap.items() if s["status"] == "queued"]
            pending = [n for n, s in snap.items() if s["status"] == "pending"]

            def _wr(y, x, text, attr=0):
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

            FOOTER_LINE = max_y - 1
            MAX_LOG_ROWS = 3
            LOG_TOP = FOOTER_LINE - MAX_LOG_ROWS
            LIVE_TOP = LOG_TOP - LIVE_HEIGHT
            MODEL_BOTTOM = LIVE_TOP - 1
            MODEL_TOP = 4
            VISIBLE_ROWS = max(0, MODEL_BOTTOM - MODEL_TOP)

            ts = datetime.now().strftime('%H:%M:%S')
            hdr = f"AI Benchmark — Parallel  |  {ts}"
            if max_x > len(hdr):
                _wr(0, 0, hdr, curses.A_BOLD)

            total_models = len(snap)
            failed_count = sum(1 for s in snap.values() if s["status"] == "failed")
            err_indicator = f"  |  \u26a0 {failed_count} failed" if failed_count else ""
            summary = (f"Total: {total}  |  "
                       f"Done: {done}  |  "
                       f"Active: {len(running)}  |  "
                       f"Queued: {len(queued + pending)}"
                       f"{err_indicator}"
                       f"  |  \u2191\u2193 rows {scroll_y + 1}-{min(total_models, scroll_y + VISIBLE_ROWS)}/{total_models}"
                       f"  |  \u2190\u2192 cols")
            if max_y > 1 and max_x > len(summary):
                _wr(1, 0, summary)

            if max_y > 2:
                _wr(2, 0, "\u2500" * min(max_x, 80))

            frozen_hdr = " ".join(f"{h:>{w}}" for h, w in frozen_cols)
            plugin_hdr_parts = [f"{h:>{w}}" for h, w in plugin_cols]
            plugin_hdr = " ".join(plugin_hdr_parts)
            if max_y > 3:
                visible_plugin_hdr = plugin_hdr[scroll_x:scroll_x + max(0, max_x - frozen_width)]
                _wr(3, 0, frozen_hdr + " " + visible_plugin_hdr, curses.A_UNDERLINE)

            key = stdscr.getch()
            max_row_offset = max(0, total_models - VISIBLE_ROWS)
            if key == curses.KEY_UP:
                scroll_y = max(0, scroll_y - 1)
            elif key == curses.KEY_DOWN:
                scroll_y = min(max_row_offset, scroll_y + 1)
            elif key == curses.KEY_PPAGE:
                scroll_y = max(0, scroll_y - VISIBLE_ROWS)
            elif key == ord(' ') or key == curses.KEY_NPAGE:
                scroll_y = min(max_row_offset, scroll_y + VISIBLE_ROWS)
            elif key == curses.KEY_HOME:
                scroll_y = 0
            elif key == curses.KEY_END:
                scroll_y = max_row_offset
            elif key == curses.KEY_LEFT:
                scroll_x = max(0, scroll_x - 8)
            elif key == curses.KEY_RIGHT:
                scroll_x = min(max(0, len(plugin_hdr) - (max_x - frozen_width)), scroll_x + 8)
            scroll_y = max(0, min(max_row_offset, scroll_y))

            for row_idx in range(VISIBLE_ROWS):
                abs_idx = scroll_y + row_idx
                if abs_idx >= total_models:
                    break
                name, s = snap_items[abs_idx]
                display_idx = abs_idx + 1
                frozen, plugin_str = render_row(name, s, display_idx)
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
                _wr(MODEL_TOP + row_idx, 0, line, attr)

            model_end = MODEL_TOP + min(VISIBLE_ROWS, max(0, total_models - scroll_y))
            for r in range(model_end, MODEL_BOTTOM + 1):
                try:
                    stdscr.move(r, 0)
                    stdscr.clrtoeol()
                except Exception:
                    pass

            if MODEL_BOTTOM >= 0:
                _wr(MODEL_BOTTOM, 0, "\u2500" * min(max_x, 60))

            live_models = running
            live_row = LIVE_TOP
            _wr(live_row, 0, "Live:", curses.A_BOLD)
            live_row += 1
            for nm in live_models[:LIVE_HEIGHT - 1]:
                if live_row >= LOG_TOP:
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
                _wr(live_row, 0, msg)
                live_row += 1
            for r in range(live_row, LOG_TOP):
                try:
                    stdscr.move(r, 0)
                    stdscr.clrtoeol()
                except Exception:
                    pass

            log_row = LOG_TOP
            recent_errors = state.recent_log(2)
            if recent_errors:
                _wr(log_row, 0, "Errors:", curses.A_BOLD)
                log_row += 1
                for ts_entry, model_entry, msg_entry in recent_errors:
                    if log_row >= FOOTER_LINE:
                        break
                    from datetime import datetime
                    t_str = datetime.fromtimestamp(ts_entry).strftime('%H:%M:%S')
                    err_msg = f"  {t_str} [{model_entry[:20]}]: {msg_entry}"
                    _wr(log_row, 0, err_msg, curses.color_pair(3))
                    log_row += 1
            for r in range(log_row, FOOTER_LINE):
                try:
                    stdscr.move(r, 0)
                    stdscr.clrtoeol()
                except Exception:
                    pass

            queuing = queued + pending
            if not live_models and not queuing:
                msg = " All models complete — generating outputs..."
            else:
                q = f"{len(queuing)} queued" if queuing else ""
                a = f"{len(live_models)} active" if live_models else ""
                sep2 = "  |  " if q and a else ""
                msg = f" {a}{sep2}{q}"
            _wr(FOOTER_LINE, 0, msg)

            stdscr.refresh()
            time.sleep(0.2)

    finally:
        curses.echo()
        curses.nocbreak()
        try:
            curses.endwin()
        except Exception:
            pass


def _prompt_restart_or_continue():
    """Ask the user whether to restart or continue a run with changed plugins."""
    print("\nPlugin set has changed since the last run.")
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
        epilog="Examples:\n"
               "  python ai-benchmark.py --restart\n"
               "  python ai-benchmark.py --config my-config.json\n"
               "  python ai-benchmark.py --out /tmp/bench-run --timeout 300\n"
               "  python ai-benchmark.py --plugins-whitelist rate-limiter\n"
               "  python ai-benchmark.py --dump-default-config --base-url http://localhost:11434 > config.json\n"
               "  python ai-benchmark.py --dump-default-config > benchmark-config.json",
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
    parser.add_argument('--plugin-temperature', type=str, nargs='+', default=None,
                        help='Per-plugin temperatures as id=value (e.g. --plugin-temperature rate-limiter=0.2 moe-dense=0.7)')
    parser.add_argument('--plugin-execution-mode', type=str, default=None,
                        choices=['sequential', 'parallel'],
                        help='Run plugins sequentially or in parallel for each model (default: sequential)')
    parser.add_argument('--plugins-whitelist', type=str, nargs='+', default=None,
                        help='Run only these plugins (e.g. --plugins-whitelist rate-limiter moe-dense)')
    parser.add_argument('--plugins-blacklist', type=str, nargs='+', default=None,
                        help='Run all plugins except these (e.g. --plugins-blacklist moe-dense)')
    parser.add_argument('--dump-default-config', action='store_true',
                        help='Print a default config file to stdout and exit')
    parser.add_argument('--base-url', default=None,
                        help='Base URL for model discovery via /v1/models API (used with --dump-default-config)')
    parser.add_argument('--api-key', default=None,
                        help='API key for model discovery (used with --dump-default-config --base-url)')
    args = parser.parse_args()

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

    # Per-plugin temperatures: CLI overrides config. Config keys may use either
    # hyphen or underscore, e.g. "rate-limiter_temperature" or "rate_servererature".
    plugin_temperatures = {}
    for key, value in cfg.items():
        if key.endswith("_temperature"):
            plugin_id = key[:-len("_temperature")].replace("_", "-")
            plugin_temperatures[plugin_id] = value
    # Backward compatibility for legacy config keys
    if "code_temperature" in cfg and "rate-limiter" not in plugin_temperatures:
        plugin_temperatures["rate-limiter"] = cfg["code_temperature"]
    if "general_temperature" in cfg and "moe-dense" not in plugin_temperatures:
        plugin_temperatures["moe-dense"] = cfg["general_temperature"]
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

    plugin_execution_mode = cfg.get("plugin_execution_mode", "sequential")
    if args.plugin_execution_mode:
        plugin_execution_mode = args.plugin_execution_mode
    cfg["plugin_execution_mode"] = plugin_execution_mode

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
                choice = _prompt_restart_or_continue()
                if choice == "restart":
                    os.remove(state_file)
                    state = BenchmarkState(models, plugin_ids)
                elif choice == "continue":
                    state = BenchmarkState.load_state(state_file, models, plugin_ids)
                    resumed = True
                else:
                    sys.exit(0)
            else:
                state = BenchmarkState.load_state(state_file, models, plugin_ids)
                resumed = True

            if resumed:
                completed = state.completed
                total = state.total
                print(f"📂 Resuming — {completed}/{total} models already completed.\n"
                      f"   Remove {state_file} or use --restart to start fresh.",
                      file=sys.stderr)

                if completed == total and total > 0:
                    snap = state.snapshot()
                    ok_count = sum(1 for info in snap.values() if info["status"] == "completed")
                    fail_count = total - ok_count
                    print(f"\n{'='*70}")
                    print(f"✅ PRIOR RUN COMPLETE — {ok_count}/{total} successful"
                          f" ({fail_count} failed)")
                    print(f"   Results: {output_dir}/")
                    print(f"{'='*70}")
                    sys.exit(0)
        except Exception as e:
            print(f"⚠️  Could not load state file ({e}), starting fresh.",
                  file=sys.stderr)
            state = BenchmarkState(models, plugin_ids)
    else:
        state = BenchmarkState(models, plugin_ids)

    stop_event = threading.Event()

    tui_thread = threading.Thread(target=tui_main, args=(state, stop_event, len(source_config), active_plugins))
    tui_thread.start()

    time.sleep(0.3)

    total = state.total
    worker_errors = 0
    interrupted = False

    session_seed = random.randint(0, 2**31 - 1)

    source_queues = {src: [] for src in set(models.values())}
    for name, src in models.items():
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
                run_model(model_name, source, state, active_plugins, source_config,
                          timeout, token_levels, output_dir, session_seed=session_seed,
                          global_cfg=cfg)
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

    if not source_threads:
        print("✅ All models already completed. Nothing to run.", file=sys.stderr)
    else:
        try:
            for t in source_threads.values():
                t.join()
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            print("\n\n⚠️  Ctrl+C — saving state and shutting down...", file=sys.stderr)
            print("   (Press Ctrl+C again to force exit)", file=sys.stderr)
            for t in source_threads.values():
                t.join(timeout=3)

    stop_event.set()
    tui_thread.join(timeout=2)

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
    from benchmark_core import gen_pdf
    pdf_path = gen_pdf(final_results, active_plugins, output_dir)
    ok_count = len([r for r in final_results if r["status"] == "ok"])
    print(f"\n{'='*70}")
    print(f"AI BENCHMARK COMPLETE — {ok_count}/{total} successful "
          f"({worker_errors} worker errors)")
    print(f"Outputs: {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        print(f"  - {fname}")
    if pdf_path:
        print(f"  - {os.path.basename(pdf_path)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
