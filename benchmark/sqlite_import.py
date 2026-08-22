"""Import legacy benchmark JSON state into the normalized SQLite store."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from .sqlite_benchmarks import SQLiteBenchmarkStore
from .sqlite_judges import SQLiteJudgeStore
from .storage import latest_result_rows
from .sqlite_schema import connect_database


@dataclass(frozen=True)
class ImportSummary:
    run_id: str
    revision_id: int
    imported_targets: int
    imported_cells: int
    imported_attempts: int
    imported_votes: int
    ambiguous_records: int
    skipped_files: int


class LegacySQLiteImporter:
    """Import one legacy state file without loading response payloads wholesale."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    @classmethod
    def import_path(
        cls, source_path: str, database_path: str, *, run_id: str | None = None,
        include_debug_logs: bool = False,
    ) -> ImportSummary:
        connection = connect_database(database_path)
        try:
            importer = cls(connection)
            return importer.import_file(
                source_path, run_id=run_id,
                include_debug_logs=include_debug_logs,
            )
        finally:
            connection.close()

    def import_file(
        self, source_path: str, *, run_id: str | None = None,
        include_debug_logs: bool = False,
    ) -> ImportSummary:
        del include_debug_logs  # Debug-log copying is intentionally opt-in future work.
        source_path = os.path.abspath(source_path)
        source_hash = _sha256_file(source_path)
        with open(source_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError("legacy state must contain a JSON object")
        results = data.get("results", [])
        # JSON resume/report semantics are latest-row-per-(state_key, runner).
        # Import that same projection so an old rerun row cannot collide with
        # the current row's attempt numbers and cause the current result to be
        # silently recorded as ambiguous.
        if isinstance(results, list):
            results = latest_result_rows(results)
        model_info = data.get("model_info", {})
        active_plugins = data.get("active_plugins", [])
        if not isinstance(results, list) or not isinstance(model_info, dict):
            raise TypeError("legacy state has invalid results or model_info")
        if not isinstance(active_plugins, list) or not all(isinstance(p, str) for p in active_plugins):
            raise TypeError("legacy state has invalid active_plugins")

        existing = self.connection.execute(
            "SELECT l.run_id, r.revision_id FROM legacy_import_records l "
            "JOIN run_revisions r ON r.run_id = l.run_id "
            "WHERE l.source_sha256 = ? ORDER BY r.revision_id DESC LIMIT 1",
            (source_hash,),
        ).fetchone()
        if existing is not None:
            return ImportSummary(str(existing[0]), int(existing[1]), 0, 0, 0, 0, 0, 0)

        run_id = run_id or f"legacy-{source_hash[:16]}"
        benchmark = SQLiteBenchmarkStore(self.connection)
        revision_id = benchmark.create_run(
            run_id, score_schema=str(data.get("score_schema", "legacy")),
            storage_profile="portable", runner_mode=str(data.get("runner", "http")),
            config={"source": source_path, "legacy_import": True},
            session_seed=data.get("session_seed"),
        )
        plugin_versions = data.get("plugin_versions", {})
        if not isinstance(plugin_versions, dict):
            plugin_versions = {}
        plugin_ids = [p for p in active_plugins if isinstance(p, str)]
        for plugin_id in plugin_ids:
            version = str(plugin_versions.get(plugin_id, "legacy"))
            benchmark.register_plugin(
                plugin_id, version, name=plugin_id, max_score=100,
                supports_streaming=True,
            )
            benchmark.activate_plugin(revision_id, plugin_id, version)

        target_ids: dict[tuple[str, str], int] = {}
        cell_ids: dict[tuple[str, str, str], int] = {}
        imported_targets = imported_cells = imported_attempts = imported_votes = 0
        ambiguous = 0
        for row_number, result in enumerate(results):
            if not isinstance(result, dict):
                ambiguous += 1
                self._record_legacy(run_id, source_hash, row_number, "result", result, "ambiguous", "not an object")
                continue
            target = result.get("state_key", result.get("model"))
            if not isinstance(target, str) or not target:
                ambiguous += 1
                self._record_legacy(run_id, source_hash, row_number, "result", result, "ambiguous", "missing model")
                continue
            runner = str(result.get("runner", data.get("runner", "http")))
            target_key = (target, runner)
            if target_key not in target_ids:
                target_ids[target_key] = benchmark.register_target(
                    revision_id, run_id=run_id, logical_name=target, runner=runner,
                    source=str(result.get("source", "legacy")),
                    api_model=str(result.get("api_model", target)),
                    target_signature=_target_signature(result),
                    is_agent=bool(result.get("is_agent", False)),
                    system_prompt=result.get("system_prompt"),
                    target_config={"legacy": True}, order_index=len(target_ids),
                )
                imported_targets += 1
            target_id = target_ids[target_key]
            for plugin_id in plugin_ids:
                version = str(plugin_versions.get(plugin_id, "legacy"))
                cell_key = (target, runner, plugin_id)
                if cell_key not in cell_ids:
                    cell_ids[cell_key] = benchmark.ensure_cell(
                        revision_id, target_id, plugin_id, version,
                    )
                    imported_cells += 1
                cell_id = cell_ids[cell_key]
                attempt = _result_attempt(result, plugin_id, model_info.get(target, {}))
                try:
                    attempt_id = benchmark.record_attempt(
                        revision_id, cell_id, attempt, selected=True,
                    )
                    imported_attempts += 1
                except (TypeError, ValueError, sqlite3.IntegrityError) as exc:
                    ambiguous += 1
                    self._record_legacy(
                        run_id, source_hash, row_number, "benchmark", attempt,
                        "ambiguous", str(exc),
                    )
                    continue
                imported_votes += self._import_votes(
                    run_id, source_hash, row_number, revision_id, cell_id,
                    plugin_id, result, attempt_id,
                )
            self._record_legacy(run_id, source_hash, row_number, "result", result, "mapped", None)

        self.connection.commit()
        return ImportSummary(
            run_id, revision_id, imported_targets, imported_cells,
            imported_attempts, imported_votes, ambiguous, 0,
        )

    def _import_votes(
        self, run_id: str, source_hash: str, row_number: int, revision_id: int,
        cell_id: int, plugin_id: str, result: dict[str, Any], attempt_id: int,
    ) -> int:
        votes = result.get(f"{plugin_id}_judge_votes", [])
        if not isinstance(votes, list):
            return 0
        judge_store = SQLiteJudgeStore(self.connection)
        count = 0
        for ordinal, vote in enumerate(votes):
            if not isinstance(vote, dict) or not isinstance(vote.get("model"), str):
                self._record_legacy(run_id, source_hash, row_number, "judge_vote", vote, "ambiguous", "missing judge model")
                continue
            judge_model = str(vote["model"])
            self.connection.execute(
                "INSERT INTO revision_judges(revision_id, judge_model, source, config_json, active) "
                "VALUES (?, ?, 'legacy', NULL, 1) ON CONFLICT DO NOTHING",
                (revision_id, judge_model),
            )
            contract_id = str(vote.get("judge_contract_id") or f"legacy-{plugin_id}")
            contract_hash = hashlib.sha256(contract_id.encode()).hexdigest()
            plugin_version = self.connection.execute(
                "SELECT plugin_version FROM revision_plugins "
                "WHERE revision_id = ? AND plugin_id = ?",
                (revision_id, plugin_id),
            ).fetchone()[0]
            judge_store.register_contract(
                contract_id, plugin_id=plugin_id,
                plugin_version=str(plugin_version), prompt_version=str(vote.get("judge_prompt_version", "legacy")),
                instructions_version=str(vote.get("judge_instructions_version", "legacy")),
                response_schema_hash="legacy", contract={"legacy": True},
                contract_hash=contract_hash,
            )
            judge_store.activate_contract(revision_id, plugin_id, contract_id)
            judge_attempt_id = judge_store.record_attempt(
                revision_id, cell_id, judge_model, contract_id,
                {"attempt_number": ordinal + 1, "raw_response": vote.get("raw_response"),
                 "status": "completed", "error": vote.get("error")},
            )
            vote_id = judge_store.record_vote(judge_attempt_id, vote)
            if vote.get("score") is not None or vote.get("usable"):
                judge_store.select_vote(
                    revision_id, cell_id, judge_model, contract_id, vote_id,
                    reason="legacy-import",
                )
            count += 1
        return count

    def _record_legacy(
        self, run_id: str, source_hash: str, row_number: int, kind: str,
        value: Any, status: str, note: str | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO legacy_import_records(
                run_id, source_file, source_sha256, source_row_number,
                record_kind, raw_json, mapping_status, mapping_note
            ) VALUES (?, '', ?, ?, ?, ?, ?, ?)
            """,
            (run_id, source_hash, row_number, kind, json.dumps(value, default=str), status, note),
        )


def _result_attempt(result: dict[str, Any], plugin_id: str, info: dict[str, Any]) -> dict[str, Any]:
    prefix = f"{plugin_id}_"
    attempts = result.get(f"{prefix}attempts")
    selected_number = result.get(
        f"{prefix}selected_attempt", result.get(f"{prefix}attempt", 1),
    )
    selected = None
    if isinstance(attempts, list):
        for candidate in attempts:
            if isinstance(candidate, dict) and candidate.get("attempt") == selected_number:
                selected = candidate
                break
        if selected is None and attempts:
            candidate = attempts[-1]
            selected = candidate if isinstance(candidate, dict) else None
    attempt = dict(selected) if selected is not None else {}
    merged = dict(attempt)
    fallback_fields = {
        "attempt_number": selected_number,
        "prompt": result.get(f"{prefix}prompt", result.get("prompt", "")),
        "content": result.get(f"{prefix}content", result.get("response", "")),
        "thinking": result.get(f"{prefix}thinking", result.get(f"{prefix}think_text", "")),
        "score": result.get(f"{prefix}score"),
        "output_tokens": result.get(f"{prefix}output_tokens"),
        "thinking_tokens": result.get(f"{prefix}thinking_tokens"),
        "total_tokens": result.get(f"{prefix}total_tokens"),
        "tps": result.get(f"{prefix}tps"),
        "error": result.get(f"{prefix}error", result.get("error")),
        "rubric": result.get(f"{prefix}rubric"),
        "diagnostics": result.get(f"{prefix}diagnostics"),
        "status": "completed" if result.get("status") == "ok" else "failed",
    }
    for key, value in fallback_fields.items():
        if key not in merged or merged[key] in (None, ""):
            merged[key] = value
    if not merged.get("prompt") and isinstance(info, dict):
        merged["prompt"] = info.get(f"{prefix}prompt", "")
    return merged


def _target_signature(result: dict[str, Any]) -> str:
    material = {
        "runner": result.get("runner", "http"),
        "source": result.get("source"),
        "api_model": result.get("api_model", result.get("model")),
        "is_agent": bool(result.get("is_agent", False)),
        "system_prompt": result.get("system_prompt"),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
