"""Read the normalized SQLite run store as the legacy report dictionaries."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .sqlite_payloads import SQLitePayloadStore


class SQLiteReportSource:
    """Materialize report-compatible rows from a selected SQLite revision."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.payloads = SQLitePayloadStore(connection)
        self._owned_connection: sqlite3.Connection | None = None
        self._revision_cache: dict[int, dict[str, Any]] = {}

    @classmethod
    def open(cls, path: str) -> SQLiteReportSource:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        from ..sqlite_schema import configure_connection, initialize_schema
        configure_connection(connection)
        initialize_schema(connection)
        source = cls(connection)
        source._owned_connection = connection
        return source

    def close(self) -> None:
        connection: sqlite3.Connection | None = getattr(self, "_owned_connection", None)
        if connection is not None:
            connection.close()
            self._owned_connection = None

    def load_results(
        self, *, revision: int | None = None, run_id: str | None = None,
        include_reused: bool = False,
    ) -> tuple[list[dict[str, Any]], list[str], int | None, int]:
        revision_id = self._resolve_revision(revision, run_id=run_id)
        revision_row = self.connection.execute(
            "SELECT session_seed FROM run_revisions WHERE revision_id = ?",
            (revision_id,),
        ).fetchone()
        if revision_row is None:
            raise ValueError(f"SQLite revision {revision_id} does not exist")
        plugin_rows = self.connection.execute(
            """
            SELECT plugin_id, plugin_version FROM revision_plugins
            WHERE revision_id = ? AND active = 1 ORDER BY rowid
            """,
            (revision_id,),
        ).fetchall()
        active_plugins = [str(row[0]) for row in plugin_rows]
        bulk = self._load_revision_cache(revision_id)
        results: dict[tuple[int, int], dict[str, Any]] = {}
        # Report generation reads only scheduled cells. Resume hydration also
        # needs reused (scheduled=0) cells so their copied selections surface
        # as already-completed scores and are skipped by the runtime.
        reused_clause = " OR s.attempt_id IS NOT NULL" if include_reused else ""
        cells = self.connection.execute(
            f"""
            SELECT c.cell_id, c.target_instance_id, c.plugin_id, c.plugin_version,
                   t.logical_name, t.runner, t.source, t.api_model, t.is_agent,
                   rt.runtime_json, s.attempt_id
            FROM revision_cells rc
            JOIN cells c ON c.cell_id = rc.cell_id
            JOIN target_instances t ON t.target_instance_id = c.target_instance_id
            JOIN revision_targets rt
              ON rt.revision_id = rc.revision_id
             AND rt.target_instance_id = c.target_instance_id
             AND rt.active = 1
            JOIN revision_plugins rp
              ON rp.revision_id = rc.revision_id
             AND rp.plugin_id = c.plugin_id
             AND rp.plugin_version = c.plugin_version
             AND rp.active = 1
            LEFT JOIN benchmark_selections s
              ON s.revision_id = rc.revision_id AND s.cell_id = c.cell_id
            WHERE rc.revision_id = ? AND (rc.scheduled = 1{reused_clause})
            ORDER BY t.logical_name, c.plugin_id
            """,
            (revision_id,),
        ).fetchall()
        for row in cells:
            key = (int(row["target_instance_id"]), int(row["is_agent"]))
            result = results.setdefault(key, {
                "model": row["logical_name"],
                "state_key": row["logical_name"],
                "api_model": row["api_model"],
                "source": row["source"],
                "runner": row["runner"],
                "is_agent": bool(row["is_agent"]),
                "status": "ok",
                "session_seed": revision_row[0],
                "_target_instance_id": int(row["target_instance_id"]),
            })
            runtime_json = self._json_load(row["runtime_json"])
            if isinstance(runtime_json, dict):
                result.setdefault("_runtime_json", {}).update(runtime_json)
            pid = str(row["plugin_id"])
            result[f"{pid}_plugin_version"] = row["plugin_version"]
            attempt_id = row["attempt_id"]
            if attempt_id is None:
                result["status"] = "error"
                result[f"{pid}_score"] = "fail"
                continue
            attempt = bulk["attempts"].get(int(attempt_id))
            if attempt is None:
                result["status"] = "error"
                result[f"{pid}_score"] = "fail"
                continue
            prefix = f"{pid}_"
            result[f"{prefix}score"] = attempt["score"] if attempt["score"] is not None else "fail"
            for column, suffix in (
                ("output_tokens", "output_tokens"),
                ("thinking_tokens", "thinking_tokens"),
                ("total_tokens", "total_tokens"),
                ("tps", "tps"),
                ("response_time", "response_time"),
                ("gen_time", "gen_time"),
                ("finish_reason", "finish_reason"),
                ("response_nature", "response_nature"),
                ("retry_reason", "retry_reason"),
                ("prompt_altered", "prompt_altered"),
                ("truncated", "truncated"),
                ("truncated_due_to_time", "truncated_due_to_time"),
                ("failure_cause", "failure_cause"),
                ("stream_ok", "stream_ok"),
                ("repeating", "repeating"),
                ("empty_reason", "empty_reason"),
                ("error", "error"),
            ):
                value = attempt[column]
                if value is not None:
                    result[f"{prefix}{suffix}"] = bool(value) if suffix in {
                        "truncated", "truncated_due_to_time", "stream_ok", "repeating",
                    } else value
            if attempt["error"] is not None or attempt["score"] is None:
                result["status"] = "error"
            rubric = self._json_load(attempt["rubric_json"])
            if rubric is not None:
                result[f"{prefix}rubric"] = rubric
            diagnostics = self._json_load(attempt["diagnostics_json"])
            if diagnostics is not None:
                result[f"{prefix}diagnostics"] = diagnostics
        rows = list(results.values())
        for result in rows:
            runtime_json = result.pop("_runtime_json", None)
            if isinstance(runtime_json, dict):
                result.update(runtime_json)
            self._attach_model_level(result, revision_id, bulk)
            self._attach_judges(result, revision_id, bulk)
            self._attach_attempt_meta(result, revision_id, bulk)
            result.pop("_target_instance_id", None)
            first_tok_time = None
            for plugin_id in active_plugins:
                if not result.get(f"{plugin_id}_stream_ok"):
                    continue
                response_time = result.get(f"{plugin_id}_response_time")
                if isinstance(response_time, (int, float)) and (
                    first_tok_time is None or response_time < first_tok_time
                ):
                    first_tok_time = response_time
            if first_tok_time is not None:
                result["ttft"] = round(first_tok_time, 3)
        return rows, active_plugins, revision_row[0], revision_id

    def _load_revision_cache(self, revision_id: int) -> dict[str, Any]:
        """Load revision-wide hydration data with bounded bulk queries."""
        cached = self._revision_cache.get(revision_id)
        if cached is not None:
            return cached
        judge_models = [
            str(row["judge_model"])
            for row in self.connection.execute(
                "SELECT judge_model FROM revision_judges WHERE revision_id = ? AND active = 1",
                (revision_id,),
            )
        ]
        cell_rows = self.connection.execute(
            """
            SELECT c.cell_id, c.target_instance_id, c.plugin_id
            FROM revision_cells rc
            JOIN cells c ON c.cell_id = rc.cell_id
            JOIN revision_targets rt
              ON rt.revision_id = rc.revision_id
             AND rt.target_instance_id = c.target_instance_id
             AND rt.active = 1
            JOIN revision_plugins rp
              ON rp.revision_id = rc.revision_id
             AND rp.plugin_id = c.plugin_id
             AND rp.plugin_version = c.plugin_version
             AND rp.active = 1
            WHERE rc.revision_id = ?
            """,
            (revision_id,),
        ).fetchall()
        cells_by_target: dict[int, list[tuple[int, str]]] = {}
        cell_ids: list[int] = []
        for row in cell_rows:
            cell_id = int(row["cell_id"])
            cell_ids.append(cell_id)
            cells_by_target.setdefault(int(row["target_instance_id"]), []).append(
                (cell_id, str(row["plugin_id"])),
            )
        selection_rows = self.connection.execute(
            "SELECT cell_id, attempt_id FROM benchmark_selections WHERE revision_id = ?",
            (revision_id,),
        ).fetchall()
        selected_attempt_ids = [int(row["attempt_id"]) for row in selection_rows]
        attempt_rows: list[sqlite3.Row] = []
        if cell_ids or selected_attempt_ids:
            clauses: list[str] = []
            params: list[Any] = []
            if cell_ids:
                placeholders = ",".join("?" for _ in cell_ids)
                clauses.append(f"(revision_id = ? AND cell_id IN ({placeholders}))")
                params.extend([revision_id, *cell_ids])
            if selected_attempt_ids:
                placeholders = ",".join("?" for _ in selected_attempt_ids)
                clauses.append(f"attempt_id IN ({placeholders})")
                params.extend(selected_attempt_ids)
            attempt_rows = self.connection.execute(
                f"SELECT * FROM benchmark_attempts WHERE {' OR '.join(clauses)}",
                params,
            ).fetchall()
        attempts = {int(row["attempt_id"]): dict(row) for row in attempt_rows}
        attempts_by_cell: dict[int, list[dict[str, Any]]] = {}
        for row in attempt_rows:
            attempts_by_cell.setdefault(int(row["cell_id"]), []).append(dict(row))
        for values in attempts_by_cell.values():
            values.sort(key=lambda row: int(row["attempt_number"]))
        vote_rows = self.connection.execute(
            """
            SELECT c.cell_id, c.judge_model AS model, c.contract_id AS judge_contract_id,
                   v.vote_attempt_id, v.score, v.confidence, v.rationale,
                   v.error, v.usable
            FROM current_judge_votes c
            JOIN judge_vote_attempts v ON v.vote_attempt_id = c.vote_attempt_id
            WHERE c.revision_id = ?
            ORDER BY c.cell_id, c.judge_model
            """,
            (revision_id,),
        ).fetchall()
        votes_by_cell: dict[int, list[dict[str, Any]]] = {}
        vote_ids: list[int] = []
        for row in vote_rows:
            vote = dict(row)
            cell_id = int(vote.pop("cell_id"))
            votes_by_cell.setdefault(cell_id, []).append(vote)
            vote_ids.append(int(vote["vote_attempt_id"]))
        criteria_by_vote: dict[int, list[dict[str, Any]]] = {}
        if vote_ids:
            placeholders = ",".join("?" for _ in vote_ids)
            criteria_rows = self.connection.execute(
                f"""
                SELECT vote_attempt_id, criterion_key, criterion, status, evidence
                FROM judge_criteria
                WHERE vote_attempt_id IN ({placeholders}) ORDER BY vote_attempt_id, ordinal
                """,
                vote_ids,
            ).fetchall()
            for row in criteria_rows:
                criteria_by_vote.setdefault(int(row["vote_attempt_id"]), []).append({
                    "id": row["criterion_key"],
                    "criterion": row["criterion"],
                    "status": row["status"],
                    "evidence": row["evidence"],
                })
        cache = {
            "judge_models": judge_models,
            "cells_by_target": cells_by_target,
            "attempts": attempts,
            "attempts_by_cell": attempts_by_cell,
            "votes_by_cell": votes_by_cell,
            "criteria_by_vote": criteria_by_vote,
        }
        self._revision_cache[revision_id] = cache
        return cache

    def _attach_model_level(self, result: dict[str, Any], revision_id: int,
                            bulk: dict[str, Any]) -> None:
        """Attach model-level judge identities from the cached revision model."""
        del revision_id
        models = bulk["judge_models"]
        if models:
            result["judge_models"] = models
            result.setdefault("judge_status", "pending")

    def _attach_attempt_meta(self, result: dict[str, Any], revision_id: int,
                             bulk: dict[str, Any]) -> None:
        """Attach per-plugin attempt counts and retry reasons.

        Legacy JSON rows carry ``{pid}_attempt_count`` and ``{pid}_retry_reasons``
        computed from the per-attempt list; SQLite stores one row per transport
        attempt, so reconstruct the aggregates from the attempt table.
        """
        target_id = result.get("_target_instance_id")
        if target_id is None:
            return
        for cell_id, pid in bulk["cells_by_target"].get(int(target_id), []):
            attempts = bulk["attempts_by_cell"].get(cell_id, [])
            if not attempts:
                continue
            result[f"{pid}_attempt_count"] = len(attempts)
            retry_reasons = [
                str(row["retry_reason"]) for row in attempts
                if row["retry_reason"] is not None and row["retry_reason"] != "none"
            ]
            if retry_reasons:
                result[f"{pid}_retry_reasons"] = retry_reasons

    def _attach_judges(self, result: dict[str, Any], revision_id: int,
                       bulk: dict[str, Any]) -> None:
        del revision_id
        target_id = result.get("_target_instance_id")
        if target_id is None:
            return
        for cell_id, pid in bulk["cells_by_target"].get(int(target_id), []):
            votes = bulk["votes_by_cell"].get(cell_id, [])
            if not votes:
                continue
            vote_dicts = [dict(vote) for vote in votes]
            # Attach each vote's stored criteria/evidence rows so report
            # helpers (and the resume projection) see the same shape the
            # legacy JSON path produced.
            for vote in vote_dicts:
                vote["criteria"] = bulk["criteria_by_vote"].get(
                    int(vote["vote_attempt_id"]), []
                )
            result[f"{pid}_judge_votes"] = vote_dicts
            usable = [vote["score"] for vote in votes if vote["usable"] and vote["score"] is not None]
            result[f"{pid}_judge_score"] = sum(usable) / len(usable) if usable else None
            result[f"{pid}_judge_models"] = [vote["model"] for vote in votes]
            # Rebuild the flat projection and per-contract consensus from the
            # stored votes, mirroring the legacy in-memory judge path.
            self._attach_judge_projection(result, pid, vote_dicts)

    def _attach_judge_projection(
        self, result: dict[str, Any], pid: str, votes: list[dict[str, Any]]
    ) -> None:
        """Rebuild the flat judge projection from stored votes.

        The legacy JSON path stored ``{pid}_judge_confidence``, ``{pid}_judge_consensus_by_contract``
        and the selected-contract projection alongside the votes. SQLite keeps
        only the vote rows, so recompute the consensus per contract and pick
        the strongest contract as the projected one.
        """
        from ..judging import confidence_weighted_consensus_by_contract

        consensus_by_contract = confidence_weighted_consensus_by_contract(votes)
        if consensus_by_contract:
            result[f"{pid}_judge_consensus_by_contract"] = consensus_by_contract
        # Pick the contract with the most valid judges; ties break on score.
        best_contract = None
        best_valid = -1
        best_score = -1.0
        for contract_id, consensus in consensus_by_contract.items():
            valid = int(consensus.get("valid_judges") or 0)
            score = consensus.get("score")
            score_value = float(score) if isinstance(score, (int, float)) else -1.0
            if valid > best_valid or (valid == best_valid and score_value > best_score):
                best_contract = contract_id
                best_valid = valid
                best_score = score_value
        if best_contract is not None:
            consensus = consensus_by_contract[best_contract]
            result[f"{pid}_judge_selected_contract"] = best_contract
            result[f"{pid}_judge_score"] = consensus.get("score")
            result[f"{pid}_judge_confidence"] = consensus.get("confidence")
            result[f"{pid}_judge_rationale"] = consensus.get("rationale")
            result[f"{pid}_judge_error"] = consensus.get("error")
            result[f"{pid}_judge_criteria"] = consensus.get("criteria", [])
            configured = result.get("judge_models", []) or []
            voted = {vote.get("model") for vote in votes if vote.get("usable")}
            result[f"{pid}_judge_complete"] = bool(
                configured and set(configured).issubset(voted)
            )

    def _resolve_revision(self, revision: int | None, *, run_id: str | None = None) -> int:
        if revision is None:
            if run_id is None:
                row = self.connection.execute(
                    "SELECT current_revision_id FROM runs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            else:
                row = self.connection.execute(
                    "SELECT current_revision_id FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            if row is None or row[0] is None:
                if run_id is None:
                    raise ValueError("SQLite database has no current revision")
                raise ValueError(f"SQLite run {run_id!r} has no current revision")
            return int(row[0])
        row = self.connection.execute(
            """
            SELECT revision_id FROM run_revisions
            WHERE revision_id = ?
               OR (run_id = (
                    SELECT run_id FROM run_revisions WHERE revision_id = ?
                  ) AND revision_number = ?)
            ORDER BY CASE WHEN revision_id = ? THEN 0 ELSE 1 END, revision_id DESC
            LIMIT 1
            """,
            (revision, revision, revision, revision),
        ).fetchone()
        if row is None:
            raise ValueError(f"SQLite revision {revision} does not exist")
        return int(row[0])

    @staticmethod
    def _json_load(value: Any) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None


def sqlite_path_from_report_path(path: str) -> str | None:
    """Resolve a directory/database argument to a SQLite file if present."""
    if os.path.isdir(path):
        for name in ("run.sqlite3", "benchmark.sqlite3", "run.db"):
            probe = os.path.join(path, name)
            if os.path.isfile(probe):
                return probe
        return None
    if os.path.isfile(path) and path.endswith((".sqlite3", ".db")):
        return path
    return None
