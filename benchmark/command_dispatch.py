"""Dispatch for CLI commands that do not start a benchmark run."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any

from benchmark.cli_parser import generate_shell_completion
from benchmark.commands import check_sqlite, generate_reports, list_plugins
from benchmark.configuration import dump_default_config, generate_config_from_api, load_config
from benchmark.judge_analysis import write_disagreement_queue
from plugins import discover_plugins


def dispatch_early_command(args: Any) -> bool:
    """Handle independent CLI commands; return whether execution should stop."""
    if args.check_sqlite:
        try:
            report = check_sqlite(args.check_sqlite)
        except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
            print(f"❌ Could not check SQLite integrity: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(json.dumps(report, indent=2, default=str))
        raise SystemExit(0 if report.get("ok") else 1)

    if args.generate_reports:
        if not args.output_format:
            print("❌ --generate-reports requires --output-format with one or more formats.", file=sys.stderr)
            raise SystemExit(2)
        try:
            reports = generate_reports(args.generate_reports, args.output_format, args.revision)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            print(f"❌ Could not generate reports: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        for report_line in reports:
            print(report_line)
        raise SystemExit(0)

    if args.build_judge_queue:
        try:
            path = write_disagreement_queue(
                args.build_judge_queue,
                args.judge_queue_output,
                spread_threshold=None if args.no_judge_spread else args.judge_spread_threshold,
                deviation_threshold=None if args.no_judge_deviation else args.judge_deviation_threshold,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"❌ Could not build judge disagreement queue: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(path)
        raise SystemExit(0)

    if args.list_plugins:
        print(list_plugins())
        raise SystemExit(0)

    if args.generate_shell_completion:
        print(generate_shell_completion(args.generate_shell_completion, discover_plugins()))
        raise SystemExit(0)

    if args.dump_default_config:
        if args.base_url:
            print(json.dumps(generate_config_from_api(args.base_url, args.api_key), indent=2))
        else:
            dump_default_config()
        raise SystemExit(0)

    if args.convert_config:
        if not os.path.exists(args.convert_config):
            print(f"❌ Config file not found: {args.convert_config}", file=sys.stderr)
            raise SystemExit(1)
        ext = os.path.splitext(args.convert_config)[1].lower()
        if ext not in (".json", ".yaml", ".yml"):
            print(f"❌ Unsupported config format: {ext}. Use .json, .yaml, or .yml.", file=sys.stderr)
            raise SystemExit(1)
        cfg = load_config(args.convert_config)
        if ext in (".yaml", ".yml"):
            print(json.dumps(cfg, indent=2))
        else:
            import yaml
            print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        raise SystemExit(0)

    return False
