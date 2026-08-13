#!/usr/bin/env python3
"""Recover a benchmark state from its complete ``results.csv`` report.

This is an explicit fallback for state files whose JSON corruption is broader
than the narrowly-audited repair in ``benchmark.state``. It preserves every
CSV result and reconstructs model metadata from the run config where possible.
It is dry-run by default; pass ``--apply`` before replacing a state file.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile

from benchmark.core import load_config, resolve_targets
from benchmark.state import BenchmarkState
from plugins import discover_plugins


def _number(value):
    if value in ("", None):
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _judge_models(value):
    """Parse the display-form judge model list from a CSV row."""
    return [model.strip() for model in (value or "").split(",") if model.strip()]


def _successful_judge_votes(value):
    """Keep only votes that represent a usable completed judge response.

    Failed transport attempts and malformed/empty judge responses are not
    resume evidence. Dropping them here makes the reconstructed state queue
    exactly those judge/model/plugin combinations again while retaining valid
    votes from the original run.
    """
    if not value:
        return []
    try:
        votes = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(votes, list):
        return []
    return [
        vote for vote in votes
        if isinstance(vote, dict)
        and isinstance(vote.get("score"), (int, float))
        and not isinstance(vote.get("score"), bool)
        and vote.get("confidence") in {"high", "medium", "low"}
        and isinstance(vote.get("rationale"), str)
        and vote["rationale"].strip()
        and not vote.get("error")
    ]


def _judge_complete(judge_models, votes, aggregate_score):
    """Return whether a reconstructed cell has complete usable consensus."""
    return (
        bool(judge_models)
        and set(judge_models).issubset({vote.get("model") for vote in votes})
        and isinstance(aggregate_score, (int, float))
        and not isinstance(aggregate_score, bool)
        and 0 <= aggregate_score <= 100
    )


def reconstruct_run_state(run_dir, *, apply=False):
    """Reconstruct ``benchmark_state.json`` from ``results.csv``.

    Returns a report dictionary. With ``apply=False`` no run files are changed;
    with ``apply=True`` the original state is copied to a unique
    ``.pre-repair-*.bak`` file and the reconstructed state is atomically
    installed.
    """
    state_path = os.path.join(run_dir, "benchmark_state.json")
    csv_path = os.path.join(run_dir, "results.csv")
    config_path = os.path.join(run_dir, "benchmark-config.yml")
    if not os.path.isfile(csv_path) or not os.path.isfile(config_path):
        raise FileNotFoundError("run must contain results.csv and benchmark-config.yml")
    if apply and not os.path.isfile(state_path):
        raise FileNotFoundError(state_path)

    with open(csv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("results.csv contains no rows")
    configured_targets = resolve_targets(load_config(config_path))
    discovered_plugins = discover_plugins()
    discovered_by_id = {plugin.id: plugin for plugin in discovered_plugins}
    score_columns = {
        match.group(1): key
        for key in rows[0]
        for match in [re.match(r"^(.*)_Score_\d+$", key)]
        if match and "_Judge_" not in key and match.group(1) != "Overall"
    }
    unknown_plugins = set(score_columns) - set(discovered_by_id)
    if unknown_plugins:
        raise ValueError(
            "results.csv contains unknown plugin score columns: "
            + ", ".join(sorted(unknown_plugins))
        )
    if not score_columns:
        raise ValueError("results.csv contains no plugin score columns")
    # Historical reports legitimately contain fewer plugins than the current
    # checkout. Recover the plugin set recorded by the CSV rather than making
    # old runs unrecoverable whenever a new challenge is added.
    pids = [plugin.id for plugin in discovered_plugins if plugin.id in score_columns]
    plugins = [discovered_by_id[pid] for pid in pids]
    versions = {plugin.id: plugin.version for plugin in plugins}

    state_models = {}
    missing_config_models = set()
    for row in rows:
        model = row["Model"]
        runner = row["Runner"]
        state_key = model if runner == "http" else f"{model} [opencode]"
        if state_key in state_models:
            continue
        target = configured_targets.get(model)
        if target is None:
            missing_config_models.add(model)
            target = {
                "source": row["Source"], "api_model": model,
                "system_prompt": None, "is_agent": False,
                "drop_params": [], "plugins_blacklist": [],
            }
        state_models[state_key] = {**target, "runner": runner}

    state = BenchmarkState(state_models, pids, runner="both")
    for row in rows:
        model = row["Model"]
        runner = row["Runner"]
        state_key = model if runner == "http" else f"{model} [opencode]"
        info = state._model_info[state_key]
        ok = row["Status"] == "OK"
        error = row["Error"] or None
        info.update({
            "status": "completed" if ok else "failed",
            "error": error, "last_error": error or "",
            "elapsed": _number(row["Time_s"]) or 0,
            "ttft": _number(row["TTFT_s"]),
        })
        judge_models = _judge_models(row.get("Judge_Models"))
        result = {
            "model": model, "state_key": state_key, "runner": runner,
            "source": row["Source"], "status": "ok" if ok else "error",
            "error": error, "ttft": _number(row["TTFT_s"]),
            "total_time": _number(row["Time_s"]),
            "plugin_versions": versions,
            "judge_models": judge_models,
        }
        info["judge_models"] = judge_models
        row_has_complete_judging = True
        row_has_judge_vote = False
        for pid in pids:
            values = {
                "response_time": _number(row.get(f"{pid}_Response_s", "")),
                "thinking_tokens": _number(row.get(f"{pid}_Thinking_Tokens", "")),
                "output_tokens": _number(row.get(f"{pid}_Content_Tokens", "")),
                "total_tokens": _number(row.get(f"{pid}_Total_Tokens", "")),
                "tps": _number(row.get(f"{pid}_TPS", "")),
                "empty_reason": row.get(f"{pid}_Empty_Reason") or None,
            }
            score = row.get(score_columns[pid], "")
            values["score"] = "fail" if score == "fail" else _number(score)
            values["stream_ok"] = values["score"] != "fail"
            votes = _successful_judge_votes(row.get(f"{pid}_Judge_Votes", ""))
            aggregate_score = _number(row.get(f"{pid}_Judge_Score_100", ""))
            aggregate_score_valid = (
                isinstance(aggregate_score, (int, float))
                and not isinstance(aggregate_score, bool)
                and 0 <= aggregate_score <= 100
            )
            judge_complete = _judge_complete(
                judge_models, votes, aggregate_score if aggregate_score_valid else None,
            )
            if values["stream_ok"] and judge_models:
                row_has_complete_judging &= judge_complete
                row_has_judge_vote |= bool(votes)
            result[f"{pid}_judge_votes"] = votes
            result[f"{pid}_judge_complete"] = judge_complete
            result[f"{pid}_judge_score"] = aggregate_score if judge_complete else None
            result[f"{pid}_judge_confidence"] = (
                row.get(f"{pid}_Judge_Confidence", "") or None
                if judge_complete else None
            )
            result[f"{pid}_judge_error"] = None if judge_complete else (
                "pending failed or incomplete judge attempts" if votes else None
            )
            result[f"{pid}_judge_rationale"] = None
            for suffix, value in values.items():
                result[f"{pid}_{suffix}"] = value
                if suffix != "stream_ok":
                    info[f"{pid}_{suffix}"] = value
            for suffix in (
                "judge_votes", "judge_complete", "judge_score", "judge_confidence",
                "judge_error", "judge_rationale",
            ):
                info[f"{pid}_{suffix}"] = result[f"{pid}_{suffix}"]
        result["judge_status"] = (
            "complete" if judge_models and row_has_complete_judging
            else "partial" if row_has_judge_vote else None
        )
        state.add_result(result)

    directory = os.path.abspath(run_dir)
    fd, candidate = tempfile.mkstemp(prefix=".benchmark_state.reconstruct-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        state.save_state(candidate, plugin_versions=versions)
        with open(candidate, encoding="utf-8") as handle:
            reconstructed = json.load(handle)
        expected_ids = {
            (
                row["Model"] if row["Runner"] == "http"
                else f"{row['Model']} [opencode]",
                row["Runner"],
            )
            for row in rows
        }
        actual_ids = {
            (result["state_key"], result["runner"])
            for result in reconstructed["results"]
        }
        if actual_ids != expected_ids:
            raise ValueError("reconstructed result identities do not match results.csv")
        by_id = {
            (result["state_key"], result["runner"]): result
            for result in reconstructed["results"]
        }
        score_mismatches = []
        for row in rows:
            key = row["Model"] if row["Runner"] == "http" else f"{row['Model']} [opencode]"
            result = by_id[(key, row["Runner"])]
            for pid, column in score_columns.items():
                expected = "fail" if row[column] == "fail" else _number(row[column])
                if result.get(f"{pid}_score") != expected:
                    score_mismatches.append((key, row["Runner"], pid))
        if score_mismatches:
            raise ValueError(f"reconstructed scores differ from results.csv: {score_mismatches[:3]}")
        loaded = BenchmarkState.load_state(candidate, state_models, pids, rerun_failed=True)
        loaded_snapshot = loaded.snapshot()
        expected_completed = sum(row["Status"] == "OK" for row in rows)
        expected_pending = sum(row["Status"] != "OK" for row in rows)
        if loaded.completed != expected_completed or sum(
            item["status"] == "pending" for item in loaded_snapshot.values()
        ) != expected_pending:
            raise ValueError("reconstructed resume counts do not match results.csv")
        report = {
            "rows": len(rows), "models": len(state_models),
            "completed": sum(row["Status"] == "OK" for row in rows),
            "failed": sum(row["Status"] != "OK" for row in rows),
            "plugins": len(pids),
            "missing_config_models": sorted(missing_config_models),
            "identities_match": True, "score_mismatches": 0,
            "loaded_completed": loaded.completed,
            "loaded_pending": sum(item["status"] == "pending" for item in loaded_snapshot.values()),
            "backup": None, "sha256": None,
        }
        if apply:
            with open(state_path, "rb") as handle:
                existing = handle.read()
            try:
                json.loads(existing)
            except json.JSONDecodeError:
                pass
            else:
                raise ValueError(
                    "refusing to overwrite a valid state; use the utility only for corrupt state"
                )
            backup_fd, backup = tempfile.mkstemp(
                prefix="benchmark_state.json.pre-repair-", suffix=".bak", dir=directory
            )
            os.close(backup_fd)
            shutil.copy2(state_path, backup)
            os.replace(candidate, state_path)
            report["backup"] = backup
            with open(state_path, "rb") as handle:
                report["sha256"] = hashlib.sha256(handle.read()).hexdigest()
            candidate = None
        return report, reconstructed
    finally:
        if candidate and os.path.exists(candidate):
            os.remove(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--apply", action="store_true", help="replace the state after creating a backup")
    args = parser.parse_args()
    report, _ = reconstruct_run_state(args.run_dir, apply=args.apply)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
