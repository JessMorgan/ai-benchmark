"""Orchestration helpers extracted from the monolithic ``_run_benchmark``.

Each function here is a pure extraction of a section of the orchestrator,
with explicit parameter lists replacing implicit closure capture.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any

from benchmark.persistence.sqlite_import import LegacySQLiteImporter
from benchmark.persistence.sqlite_reports import SQLiteReportSource
from benchmark.persistence.storage import JsonReportSource, latest_result_rows

# ── _handle_early_command_exits ────────────────────────────────────────────
# The first ~170 lines of ``_run_benchmark`` are independent command-line
# sub-commands that return via ``sys.exit()``.  They share no mutable state
# with the benchmark loop, so extracting them removes dead weight from the
# orchestrator without needing a context object.


def _handle_early_command_exits(args: Any, cfg: dict[str, Any] | None) -> None:
    """Process sub-commands that exit before the benchmark loop starts.

    ``cfg`` may be ``None`` for commands that run before any configuration
    is loaded (``measure_storage``, ``compare_storage``,
    ``dispatch_early_command``).

    Each matching sub-command exits the process via ``sys.exit()``; this
    function never returns to its caller when invoked.
    """
    if args.measure_storage:
        from benchmark.storage_measure import measure_storage

        try:
            print(json.dumps(measure_storage(), indent=2))
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                f"\u274c Could not measure storage: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.exit(0)

    if args.compare_storage:
        from benchmark.storage_validation import compare_read_models

        json_path, sqlite_path = args.compare_storage
        source = None
        try:
            json_results, _plugins, _seed = JsonReportSource().load_results(
                json_path,
            )
            source = SQLiteReportSource.open(sqlite_path)
            sqlite_results, _active, _sqlite_seed, _revision = (
                source.load_results(include_reused=True)
            )
            report = compare_read_models(
                latest_result_rows(json_results), sqlite_results,
            )
            print(json.dumps(report.as_dict(), indent=2, default=str))
        except (
            OSError,
            sqlite3.Error,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(
                f"\u274c Could not compare storage backends: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        finally:
            if source is not None:
                source.close()
        sys.exit(0 if report.equivalent else 1)

    from benchmark.command_dispatch import dispatch_early_command

    dispatch_early_command(args)

    if args.check_sqlite:
        raise AssertionError(
            "dispatch_early_command returned after check_sqlite"
        )

    if args.import_to_sqlite:
        source_path = os.path.abspath(args.import_to_sqlite)
        if not os.path.isfile(source_path):
            print(
                f"\u274c JSON state file not found: {args.import_to_sqlite}",
                file=sys.stderr,
            )
            sys.exit(1)
        output_path = args.sqlite_output or os.path.join(
            os.path.dirname(source_path), "run.sqlite3",
        )
        output_path = os.path.abspath(output_path)
        if os.path.exists(output_path) and not args.overwrite_sqlite:
            print(
                f"\u274c SQLite output already exists: {output_path}\n"
                "   Choose --sqlite-output for a new file or pass "
                "--overwrite-sqlite explicitly.",
                file=sys.stderr,
            )
            sys.exit(2)
        if os.path.exists(output_path) and args.overwrite_sqlite:
            try:
                os.remove(output_path)
            except OSError as exc:
                print(
                    f"\u274c Could not remove existing SQLite file: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            for sidecar in (output_path + "-wal", output_path + "-shm"):
                try:
                    os.remove(sidecar)
                except OSError:
                    pass
        try:
            summary = LegacySQLiteImporter.import_path(
                source_path,
                output_path,
                include_debug_logs=args.import_debug_logs,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            print(
                f"\u274c Could not import JSON to SQLite: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        manifest_path = os.path.join(
            os.path.dirname(output_path), "run-info.json",
        )
        manifest: dict[str, Any] = {}
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                loaded_manifest = json.load(handle)
            if isinstance(loaded_manifest, dict):
                manifest.update(loaded_manifest)
        except (OSError, json.JSONDecodeError, TypeError, UnicodeDecodeError):
            pass
        manifest.update({
            "run_id": summary.run_id,
            "revision_id": summary.revision_id,
            "storage": "sqlite",
            "sqlite_path": output_path,
        })
        # Inline the write to avoid a circular import into benchmark.cli
        info_path = os.path.join(os.path.dirname(output_path), "run-info.json")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            print(f"\u26a0\ufe0f  Could not write run-info.json: {e}", file=sys.stderr)
        print(json.dumps({
            "source": source_path,
            "sqlite": output_path,
            "summary": summary.__dict__,
        }, indent=2, default=str))
        sys.exit(0)

    if args.chatplayground_config:
        from benchmark.chatplayground import (
            generate_config as generate_chatplayground_config,
        )

        try:
            chat_cfg = generate_chatplayground_config()
        except Exception as exc:  # noqa: BLE001
            print(
                f"\u274c Could not enumerate ChatPlayground models: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(json.dumps(chat_cfg, indent=2))
        sys.exit(0)

    # ── sub-commands that require loaded configuration ──
    if cfg is None:
        return  # caller hasn't loaded config yet; continue orchestration

    if args.pi_probe:
        from benchmark.configuration import resolve_targets
        from benchmark.pi import resolve_pi_worker, run_pi_probe

        targets_for_probe = resolve_targets(cfg)
        timeout = (
            args.timeout
            if args.timeout is not None
            else int(cfg.get("timeout", 600))
        )
        probe_results: list[dict[str, Any]] = []
        try:
            node, worker = resolve_pi_worker()
            for target_name, target in targets_for_probe.items():
                result = run_pi_probe(
                    cfg.get("sources", {}),
                    target["source"],
                    target["api_model"],
                    timeout=timeout,
                    pi_config=target.get("pi", {}),
                    node=node,
                    worker=worker,
                )
                result["target"] = target_name
                result["is_agent"] = target.get("is_agent", False)
                probe_results.append(result)
        except (RuntimeError, TypeError, ValueError) as exc:
            probe_results.append({
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        print(json.dumps({
            "probe": "pi-compatibility-v1",
            "scores_affected": False,
            "results": probe_results,
        }, indent=2))
        sys.exit(0)

    if args.schema_sentinel:
        from benchmark.configuration import resolve_targets
        from benchmark.core import run_schema_sentinel

        targets_for_probe = resolve_targets(cfg)
        timeout = (
            args.timeout
            if args.timeout is not None
            else int(cfg.get("timeout", 600))
        )
        probe_results = []
        for target_name, target in targets_for_probe.items():
            result = run_schema_sentinel(
                cfg.get("sources", {}),
                target["source"],
                target["api_model"],
                timeout=timeout,
                drop_params=target.get("drop_params", []),
            )
            result["target"] = target_name
            result["is_agent"] = target.get("is_agent", False)
            probe_results.append(result)
        print(json.dumps({
            "probe": "schema-sentinel-v1",
            "scores_affected": False,
            "results": probe_results,
        }, indent=2))
        sys.exit(0)
