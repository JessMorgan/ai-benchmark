"""Import legacy benchmark JSON state into the normalized SQLite store."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logs import AppendOnlyGzipLog, iter_log_members, recover_log
from .outputs import sanitize_filename
from .sqlite_benchmarks import SQLiteBenchmarkStore
from .sqlite_judges import SQLiteJudgeStore
from .sqlite_payloads import build_payload_only_judge_input
from .sqlite_schema import connect_database
from .storage import project_result_rows


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
    imported_artifacts: int = 0
    imported_debug_logs: int = 0


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
                artifact_root=os.path.dirname(os.path.abspath(database_path)),
            )
        finally:
            connection.close()

    def import_file(
        self, source_path: str, *, run_id: str | None = None,
        include_debug_logs: bool = False,
        artifact_root: str | None = None,
    ) -> ImportSummary:
        source_path = os.path.abspath(source_path)
        source_hash = _sha256_file(source_path)
        with open(source_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError("legacy state must contain a JSON object")
        results = data.get("results", [])
        model_info = data.get("model_info", {})
        active_plugins = data.get("active_plugins", [])
        # JSON resume/report semantics are latest-row-per-(state_key, runner),
        # with independent per-plugin recovery. Import that same projection so
        # a shutdown/cancellation row cannot erase scores that were already
        # completed, including scores published to model_info before the row
        # was appended.
        if isinstance(results, list) and isinstance(active_plugins, list):
            results = project_result_rows(results, active_plugins, model_info)
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
        imported_artifacts = imported_debug_logs = skipped_files = 0
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
                votes, artifacts, skipped = self._import_votes(
                    run_id, source_hash, row_number, revision_id, cell_id,
                    plugin_id, result, attempt_id, source_dir=os.path.dirname(source_path),
                )
                imported_votes += votes
                imported_artifacts += artifacts
                skipped_files += skipped
            self._record_legacy(run_id, source_hash, row_number, "result", result, "mapped", None)

        if include_debug_logs:
            imported_debug_logs, log_skipped = self._import_debug_logs(
                run_id, revision_id, os.path.dirname(source_path),
                artifact_root=artifact_root or os.path.dirname(source_path),
            )
            skipped_files += log_skipped
        self.connection.commit()
        return ImportSummary(
            run_id, revision_id, imported_targets, imported_cells,
            imported_attempts, imported_votes, ambiguous, skipped_files,
            imported_artifacts, imported_debug_logs,
        )

    def _import_votes(
        self, run_id: str, source_hash: str, row_number: int, revision_id: int,
        cell_id: int, plugin_id: str, result: dict[str, Any], _attempt_id: int,
        *, source_dir: str,
    ) -> tuple[int, int, int]:
        votes = result.get(f"{plugin_id}_judge_votes", [])
        if not isinstance(votes, list):
            return 0, 0, 0
        judge_store = SQLiteJudgeStore(self.connection)
        count = artifacts = skipped = 0
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
            target = str(result.get("state_key", result.get("model", "target")))
            runner = str(result.get("runner", "http"))
            sidecar = _judge_sidecar_file(
                source_dir, target, runner, plugin_id,
            )
            sidecar_payload = _load_json_file(sidecar)
            raw_response = _load_text_file(
                _judge_response_file(
                    source_dir, target, runner, plugin_id,
                    judge_model, contract_id,
                ),
            )
            if raw_response is None and isinstance(vote.get("raw_response"), str):
                raw_response = vote["raw_response"]
            request_payload = None
            if sidecar_payload is not None:
                request_payload = json.dumps(
                    build_payload_only_judge_input(judge_store.payloads, sidecar_payload),
                    sort_keys=True,
                    ensure_ascii=False,
                )
            if sidecar_payload is not None:
                artifacts += 1
            elif raw_response is None:
                # Response/sidecar files are optional legacy artifacts. Their
                # absence is normal for compact JSON runs, so do not turn it
                # into an ambiguous mapping record; malformed files are still
                # recorded by the loaders' callers when they are discovered.
                pass
            judge_attempt_id = judge_store.record_attempt(
                revision_id, cell_id, judge_model, contract_id,
                {"attempt_number": ordinal + 1, "raw_response": raw_response,
                 "request": request_payload, "status": "completed", "error": vote.get("error")},
                retain_request=request_payload is not None,
            )
            vote_id = judge_store.record_vote(judge_attempt_id, vote)
            if vote.get("score") is not None or vote.get("usable"):
                judge_store.select_vote(
                    revision_id, cell_id, judge_model, contract_id, vote_id,
                    reason="legacy-import",
                )
            count += 1
        return count, artifacts, skipped

    def _import_debug_logs(
        self, run_id: str, revision_id: int, source_dir: str, *, artifact_root: str,
    ) -> tuple[int, int]:
        """Copy legacy logs into compressed append-only files and index them."""
        root = Path(source_dir)
        destination_root = Path(artifact_root) / "logs" / "imported"
        imported = skipped = 0
        candidates = sorted(
            path for path in root.rglob("*")
            if path.is_file()
            and (path.name.endswith(".log") or path.name.endswith(".log.gz"))
            and not path.is_relative_to(destination_root)
        )
        for source in candidates:
            relative = source.relative_to(root)
            safe_parts = [sanitize_filename(part) for part in relative.parts]
            destination_relative = Path("logs", "imported", *safe_parts)
            if not destination_relative.name.endswith(".gz"):
                destination_relative = destination_relative.with_name(
                    destination_relative.name + ".gz"
                )
            destination = Path(artifact_root) / destination_relative
            try:
                writer = AppendOnlyGzipLog(str(destination), sync_policy="final")
                uncompressed = 0
                if source.name.endswith(".gz"):
                    for member in iter_log_members(str(source)):
                        writer.append_record([member])
                        uncompressed += len(member)
                else:
                    with open(source, "rb") as handle:
                        while True:
                            chunk = handle.read(128 * 1024)
                            if not chunk:
                                break
                            writer.append(chunk)
                            uncompressed += len(chunk)
                writer.close()
                recovery = recover_log(str(destination))
                now = int(os.path.getmtime(destination))
                self.connection.execute(
                    """
                    INSERT INTO debug_log_files(
                        run_id, revision_id, path, compression,
                        complete_members, uncompressed_bytes, stored_bytes,
                        truncated_tail, created_at, updated_at
                    ) VALUES (?, ?, ?, 'gzip', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id, revision_id, str(destination_relative),
                        recovery.complete_members, uncompressed,
                        recovery.total_bytes, int(recovery.truncated_tail), now, now,
                    ),
                )
                imported += 1
            except (OSError, EOFError, ValueError) as exc:
                skipped += 1
                self._record_legacy(
                    run_id, _sha256_file(str(source)) if source.exists() else "debug-log",
                    None, "debug_log", str(source), "skipped", str(exc),
                )
        return imported, skipped

    def _record_legacy(
        self, run_id: str, source_hash: str, row_number: int | None, kind: str,
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
        "response_time": result.get(f"{prefix}response_time"),
        "gen_time": result.get(f"{prefix}gen_time"),
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


def _judge_sidecar_file(source_dir: str, target: str, runner: str,
                        plugin_id: str) -> str:
    return os.path.join(
        source_dir, "judge-inputs", runner, sanitize_filename(target),
        f"{plugin_id}.json",
    )


def _judge_response_file(source_dir: str, target: str, runner: str,
                         plugin_id: str, judge_model: str,
                         contract_id: str) -> str:
    suffix = sanitize_filename(judge_model)
    if contract_id:
        suffix += f".{sanitize_filename(contract_id)}"
    return os.path.join(
        source_dir, runner, "responses", sanitize_filename(target),
        f"{plugin_id}.judge.{suffix}.txt",
    )


def _load_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_text_file(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


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
