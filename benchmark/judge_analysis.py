"""Analysis helpers for persisted model-as-a-judge results.

The benchmark runtime stores one list of judge votes per target/plugin cell. This
module turns that state into reproducible per-judge statistics and a ranked
review queue without changing benchmark execution or resume semantics.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from .judging import JUDGE_CONFIDENCE_WEIGHTS, is_successful_judge_vote
from .outputs import sanitize_filename

DEFAULT_SPREAD_THRESHOLD = 30.0
DEFAULT_DEVIATION_THRESHOLD = 40.0


def _latest_results(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the last persisted result for each state-key/runner pair."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for result in data.get("results", []):
        key = (result.get("state_key", result.get("model")), result.get("runner", "http"))
        latest[key] = result
    return latest


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_threshold(name: str, threshold: float | None) -> None:
    """Reject invalid enabled thresholds while allowing ``None`` to disable."""
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
        raise ValueError(f"{name} must be a finite non-negative number or None")


def _mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def _sample_sd(values: list[float]) -> float | None:
    return round(statistics.stdev(values), 2) if len(values) > 1 else None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if not left_var or not right_var:
        return None
    return round(
        sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
        / math.sqrt(left_var * right_var),
        3,
    )


def _valid_votes_by_judge(latest: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[tuple[str, str, str], dict[str, Any]]]:
    """Return valid votes keyed by judge and cell identity."""
    by_judge: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    failed: dict[str, int] = defaultdict(int)
    plugin_ids = sorted({
        key[:-len("_judge_votes")]
        for result in latest.values()
        for key in result
        if key.endswith("_judge_votes")
    })
    for (state_key, runner), result in latest.items():
        for plugin_id in plugin_ids:
            for vote in result.get(f"{plugin_id}_judge_votes", []) or []:
                if not isinstance(vote, dict) or not vote.get("model"):
                    continue
                cell = (state_key, runner, plugin_id)
                judge = vote["model"]
                if is_successful_judge_vote(vote):
                    by_judge[judge][cell] = vote
                else:
                    failed[judge] += 1
    return plugin_ids, by_judge, failed  # type: ignore[return-value]


def judge_statistics(data: dict[str, Any]) -> dict[str, Any]:
    """Compute per-judge, pairwise, and consensus statistics from state."""
    latest = _latest_results(data)
    plugin_ids, by_judge, failed = _valid_votes_by_judge(latest)
    per_judge = []
    for model in sorted(set(by_judge) | set(failed)):
        votes = list(by_judge.get(model, {}).values())  # type: ignore[attr-defined]
        scores = [vote["score"] for vote in votes]
        conviction = [JUDGE_CONFIDENCE_WEIGHTS[vote["confidence"]] * 100 for vote in votes]
        deterministic: list[float] = []
        judged: list[float] = []
        for (state_key, runner, plugin_id), vote in by_judge.get(model, {}).items():  # type: ignore[attr-defined]
            score = latest[(state_key, runner)].get(f"{plugin_id}_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                deterministic.append(float(score))
                judged.append(float(cast(float, vote["score"])))
        criterion_status_counts: dict[str, int] = {}
        criterion_count = 0
        for vote in votes:
            for criterion in vote.get("criteria", []) or []:
                if not isinstance(criterion, dict):
                    continue
                criterion_count += 1
                status = str(criterion.get("status", "unknown"))
                criterion_status_counts[status] = criterion_status_counts.get(status, 0) + 1
        per_judge.append({
            "model": model,
            "valid_votes": len(votes),
            "criteria": criterion_count,
            "criteria_status_counts": criterion_status_counts,
            "failed_attempts": failed.get(model, 0),  # type: ignore[attr-defined]
            "mean_score": _mean_or_none(scores),
            "sample_sd": _sample_sd(scores),
            "mean_conviction": _mean_or_none(conviction),
            "high_confidence": sum(vote["confidence"] == "high" for vote in votes),
            "medium_confidence": sum(vote["confidence"] == "medium" for vote in votes),
            "low_confidence": sum(vote["confidence"] == "low" for vote in votes),
            "deterministic_cells": len(deterministic),
            "mean_deviation": _mean_or_none([b - a for a, b in zip(deterministic, judged, strict=False)]),
            "deterministic_correlation": _pearson(deterministic, judged),
        })

    pairwise = []
    judges = sorted(by_judge)
    for index, first in enumerate(judges):
        for second in judges[index + 1:]:
            overlap = set(by_judge[first]) & set(by_judge[second])  # type: ignore[index]
            differences = [
                abs(by_judge[first][cell]["score"] - by_judge[second][cell]["score"])  # type: ignore[index,operator]
                for cell in overlap
            ]
            if differences:
                pairwise.append({
                    "judge_a": first,
                    "judge_b": second,
                    "overlap": len(differences),
                    "within_10_points": sum(value <= 10 for value in differences),
                    "within_10_percent": round(
                        100 * sum(value <= 10 for value in differences) / len(differences), 2
                    ),
                    "mean_absolute_difference": round(statistics.mean(differences), 2),
                })
    return {"per_judge": per_judge, "pairwise": pairwise, "plugin_ids": plugin_ids}


def build_disagreement_queue(
    data: dict[str, Any],
    *,
    spread_threshold: float | None = DEFAULT_SPREAD_THRESHOLD,
    deviation_threshold: float | None = DEFAULT_DEVIATION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Build a ranked queue of cells needing human disagreement review.

    A cell is included when it has at least two valid judge votes and at least
    one enabled criterion matches. Pass ``None`` for either threshold to
    disable that criterion; passing ``None`` for both produces an empty queue.
    The ``triggers`` field makes inclusion explicit.
    """
    _validate_threshold("spread_threshold", spread_threshold)
    _validate_threshold("deviation_threshold", deviation_threshold)
    latest = _latest_results(data)
    plugin_ids, _by_judge, _failed = _valid_votes_by_judge(latest)
    queue: list[dict[str, Any]] = []
    for (state_key, runner), result in latest.items():
        for plugin_id in plugin_ids:
            votes_by_judge = {}
            for vote in result.get(f"{plugin_id}_judge_votes", []) or []:
                if is_successful_judge_vote(vote) and vote.get("model"):
                    votes_by_judge[vote["model"]] = vote
            votes = list(votes_by_judge.values())
            if len(votes) < 2:
                continue
            scores = [float(vote["score"]) for vote in votes]
            weights = [JUDGE_CONFIDENCE_WEIGHTS[vote["confidence"]] for vote in votes]
            consensus = sum(score * weight for score, weight in zip(scores, weights, strict=False)) / sum(weights)
            deterministic = result.get(f"{plugin_id}_score")
            deviation = (
                consensus - float(cast(float, deterministic))
                if isinstance(deterministic, (int, float)) and not isinstance(deterministic, bool)
                else None
            )
            spread = max(scores) - min(scores)
            triggers = []
            if spread_threshold is not None and spread >= spread_threshold:
                triggers.append(f"judge-spread>={spread_threshold:g}")
            if (
                deviation_threshold is not None
                and deviation is not None
                and abs(deviation) >= deviation_threshold
            ):
                triggers.append(f"consensus-deviation>={deviation_threshold:g}")
            if not triggers:
                continue
            queue.append({
                "target": result.get("model", state_key),
                "state_key": state_key,
                "runner": runner,
                "source": result.get("source"),
                "plugin": plugin_id,
                "deterministic_score": deterministic,
                "consensus_score": round(consensus, 2),
                "deviation": round(deviation, 2) if deviation is not None else None,
                "judge_spread": round(spread, 2),
                "valid_judges": len(votes),
                "triggers": triggers,
                "judge_response_paths": [
                    str(
                        Path(runner)
                        / "responses"
                        / sanitize_filename(str(result.get("model", state_key)))
                        / f"{plugin_id}.judge.{sanitize_filename(str(vote.get('model')))}.txt"
                    )
                    for vote in votes
                ],
                "judgments": [
                    {
                        "model": vote.get("model"),
                        "judge_contract_id": vote.get("judge_contract_id"),
                        "score": vote.get("score"),
                        "confidence": vote.get("confidence"),
                        "rationale": vote.get("rationale"),
                        "criteria": vote.get("criteria", []),
                        "error": vote.get("error"),
                    }
                    for vote in votes
                ],
            })
    return sorted(
        queue,
        key=lambda item: (-item["judge_spread"], -abs(item["deviation"] or 0), item["target"], item["plugin"]),
    )


def write_disagreement_queue(
    state_path: str | Path,
    output_path: str | Path | None = None,
    *,
    spread_threshold: float | None = DEFAULT_SPREAD_THRESHOLD,
    deviation_threshold: float | None = DEFAULT_DEVIATION_THRESHOLD,
) -> Path:
    """Read a state file and write a JSON disagreement queue beside it."""
    state_path = Path(state_path)
    output_path = Path(output_path) if output_path else state_path.with_name("judge-disagreement-queue.json")
    with state_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    artifact_root = state_path.parent
    entries = build_disagreement_queue(
        data,
        spread_threshold=spread_threshold,
        deviation_threshold=deviation_threshold,
    )
    for entry in entries:
        entry["judge_response_paths"] = [
            str(artifact_root / path) for path in entry["judge_response_paths"]
        ]
    payload = {
        "schema": "judge-disagreement-queue-v1",
        "state_file": str(state_path),
        "spread_threshold": spread_threshold,
        "deviation_threshold": deviation_threshold,
        "enabled_criteria": [
            criterion
            for criterion, threshold in (
                ("judge-spread", spread_threshold),
                ("consensus-deviation", deviation_threshold),
            )
            if threshold is not None
        ],
        "entries": entries,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path
