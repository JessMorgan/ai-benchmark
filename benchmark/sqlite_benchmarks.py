"""Normalized SQLite persistence for benchmark attempts."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any

from .outputs import sanitize_filename
from .sqlite_payloads import SQLitePayloadStore


class SQLiteBenchmarkStore:
    """Persist benchmark cells and immutable attempts for one SQLite database."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.payloads = SQLitePayloadStore(connection)

    def create_run(
        self,
        run_id: str,
        *,
        score_schema: str,
        storage_profile: str,
        runner_mode: str,
        config: dict[str, Any] | str,
        session_seed: int | None = None,
    ) -> int:
        """Create a logical run and its first continuation revision."""
        config_json = config if isinstance(config, str) else json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        now = int(time.time())
        self.connection.execute(
            "INSERT INTO runs(run_id, created_at, status, score_schema, storage_profile) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, now, score_schema, storage_profile),
        )
        cursor = self.connection.execute(
            """
            INSERT INTO run_revisions(
                run_id, revision_number, status, started_at, runner_mode,
                session_seed, config_json, config_sha256, created_at
            ) VALUES (?, 1, 'running', ?, ?, ?, ?, ?, ?)
            """,
            (run_id, now, runner_mode, session_seed, config_json, config_hash, now),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a revision ID")
        revision_id = int(cursor.lastrowid)
        self.connection.execute(
            "UPDATE runs SET current_revision_id = ? WHERE run_id = ?",
            (revision_id, run_id),
        )
        self.connection.commit()
        return revision_id

    def register_plugin(
        self,
        plugin_id: str,
        plugin_version: str,
        *,
        name: str,
        max_score: float,
        supports_streaming: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register one immutable plugin definition."""
        self.connection.execute(
            """
            INSERT INTO plugin_definitions(
                plugin_id, plugin_version, name, max_score,
                supports_streaming, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(plugin_id, plugin_version) DO NOTHING
            """,
            (
                plugin_id, plugin_version, name, max_score,
                int(supports_streaming),
                json.dumps(metadata, sort_keys=True) if metadata is not None else None,
            ),
        )
        self.connection.commit()

    def register_target(
        self,
        revision_id: int,
        *,
        run_id: str,
        logical_name: str,
        runner: str,
        source: str,
        api_model: str,
        target_signature: str,
        is_agent: bool = False,
        system_prompt: str | None = None,
        target_config: dict[str, Any] | str | None = None,
        active: bool = True,
        order_index: int | None = None,
    ) -> int:
        """Register/reuse a target instance and activate it in a revision."""
        target_config_json = (
            target_config if isinstance(target_config, str) else
            json.dumps(target_config, sort_keys=True) if target_config is not None else None
        )
        row = self.connection.execute(
            """
            SELECT target_instance_id FROM target_instances
            WHERE run_id = ? AND logical_name = ? AND runner = ? AND target_signature = ?
            """,
            (run_id, logical_name, runner, target_signature),
        ).fetchone()
        if row is None:
            cursor = self.connection.execute(
                """
                INSERT INTO target_instances(
                    run_id, logical_name, runner, source, api_model, is_agent,
                    system_prompt, target_config_json, target_signature, first_revision_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, logical_name, runner, source, api_model, int(is_agent),
                    system_prompt, target_config_json, target_signature, revision_id,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a target ID")
            target_id = int(cursor.lastrowid)
        else:
            target_id = int(row[0])
        self.connection.execute(
            """
            INSERT INTO revision_targets(revision_id, target_instance_id, active, order_index)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(revision_id, target_instance_id) DO UPDATE SET
                active = excluded.active, order_index = excluded.order_index
            """,
            (revision_id, target_id, int(active), order_index),
        )
        self.connection.commit()
        return target_id

    def activate_plugin(self, revision_id: int, plugin_id: str, plugin_version: str,
                        *, active: bool = True) -> None:
        """Activate one registered plugin version for a revision."""
        self.connection.execute(
            """
            INSERT INTO revision_plugins(revision_id, plugin_id, plugin_version, active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(revision_id, plugin_id) DO UPDATE SET
                plugin_version = excluded.plugin_version, active = excluded.active
            """,
            (revision_id, plugin_id, plugin_version, int(active)),
        )
        self.connection.commit()

    def ensure_cell(self, revision_id: int, target_id: int, plugin_id: str,
                    plugin_version: str, *, scheduled: bool = True,
                    status: str = "pending", queue_reason: str | None = None) -> int:
        """Create/reuse a target/plugin-version cell and revision membership."""
        target_row = self.connection.execute(
            "SELECT run_id FROM target_instances WHERE target_instance_id = ?",
            (target_id,),
        ).fetchone()
        if target_row is None:
            raise KeyError(f"unknown target instance: {target_id}")
        cell = self.connection.execute(
            """
            SELECT cell_id FROM cells
            WHERE target_instance_id = ? AND plugin_id = ? AND plugin_version = ?
            """,
            (target_id, plugin_id, plugin_version),
        ).fetchone()
        if cell is None:
            cursor = self.connection.execute(
                """
                INSERT INTO cells(run_id, target_instance_id, plugin_id, plugin_version, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (target_row[0], target_id, plugin_id, plugin_version, int(time.time())),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a cell ID")
            cell_id = int(cursor.lastrowid)
        else:
            cell_id = int(cell[0])
        self.connection.execute(
            """
            INSERT INTO revision_cells(revision_id, cell_id, scheduled, status, queue_reason, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(revision_id, cell_id) DO UPDATE SET
                scheduled = excluded.scheduled,
                status = excluded.status,
                queue_reason = excluded.queue_reason,
                updated_at = excluded.updated_at
            """,
            (revision_id, cell_id, int(scheduled), status, queue_reason, int(time.time())),
        )
        self.connection.commit()
        return cell_id

    def record_attempt(
        self,
        revision_id: int,
        cell_id: int,
        attempt: dict[str, Any],
        *,
        selected: bool = False,
        selection_reason: str | None = None,
    ) -> int:
        """Insert one immutable benchmark attempt and optionally select it."""
        attempt_number = int(attempt.get("attempt_number", attempt.get("attempt", 1)))
        prompt_id = self._payload_id("benchmark-prompt", attempt.get("prompt"))
        content_id = self._payload_id("benchmark-content", attempt.get("content"))
        thinking_id = self._payload_id("benchmark-thinking", attempt.get("thinking"))
        score = attempt.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            score = None
        values = {
            "started_at": attempt.get("started_at"),
            "ended_at": attempt.get("ended_at"),
            "max_tokens": attempt.get("max_tokens"),
            "output_tokens": attempt.get("output_tokens"),
            "thinking_tokens": attempt.get("thinking_tokens"),
            "total_tokens": attempt.get("total_tokens"),
            "tps": attempt.get("tps"),
            "finish_reason": attempt.get("finish_reason"),
            "response_nature": attempt.get("response_nature"),
            "retry_reason": attempt.get("retry_reason"),
            "prompt_altered": attempt.get("prompt_altered"),
            "truncated": self._bool_value(attempt.get("truncated")),
            "truncated_due_to_time": self._bool_value(attempt.get("truncated_due_to_time")),
            "failure_cause": attempt.get("failure_cause"),
            "stream_ok": self._bool_value(attempt.get("stream_ok")),
            "repeating": self._bool_value(attempt.get("repeating")),
            "empty_reason": attempt.get("empty_reason"),
            "error": attempt.get("error"),
            "score": score,
            "rubric_json": self._json_value(attempt.get("rubric")),
            "diagnostics_json": self._json_value(attempt.get("diagnostics")),
        }
        cursor = self.connection.execute(
            """
            INSERT INTO benchmark_attempts(
                revision_id, cell_id, attempt_number,
                prompt_payload_id, content_payload_id, thinking_payload_id,
                started_at, ended_at, max_tokens, output_tokens, thinking_tokens,
                total_tokens, tps, finish_reason, response_nature, retry_reason,
                prompt_altered, truncated, truncated_due_to_time, failure_cause,
                stream_ok, repeating, empty_reason, error, score, rubric_json,
                diagnostics_json
            ) VALUES (
                :revision_id, :cell_id, :attempt_number,
                :prompt_payload_id, :content_payload_id, :thinking_payload_id,
                :started_at, :ended_at, :max_tokens, :output_tokens, :thinking_tokens,
                :total_tokens, :tps, :finish_reason, :response_nature, :retry_reason,
                :prompt_altered, :truncated, :truncated_due_to_time, :failure_cause,
                :stream_ok, :repeating, :empty_reason, :error, :score, :rubric_json,
                :diagnostics_json
            )
            """,
            {
                "revision_id": revision_id,
                "cell_id": cell_id,
                "attempt_number": attempt_number,
                "prompt_payload_id": prompt_id,
                "content_payload_id": content_id,
                "thinking_payload_id": thinking_id,
                **values,
            },
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return an attempt ID")
        attempt_id = int(cursor.lastrowid)
        if selected:
            self.select_attempt(
                revision_id, cell_id, attempt_id,
                reason=selection_reason or "selected-final-attempt",
                commit=False,
            )
        self.connection.execute(
            "UPDATE revision_cells SET status = ?, updated_at = ? WHERE revision_id = ? AND cell_id = ?",
            ("completed" if score is not None and values["error"] is None else "failed",
             int(time.time()), revision_id, cell_id),
        )
        self.connection.commit()
        return attempt_id

    def select_attempt(self, revision_id: int, cell_id: int, attempt_id: int,
                       *, reason: str = "selected", commit: bool = True) -> None:
        """Select one attempt for a revision-local current result."""
        self.connection.execute(
            """
            INSERT INTO benchmark_selections(
                revision_id, cell_id, attempt_id, selected_at, selection_reason
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(revision_id, cell_id) DO UPDATE SET
                attempt_id = excluded.attempt_id,
                selected_at = excluded.selected_at,
                selection_reason = excluded.selection_reason
            """,
            (revision_id, cell_id, attempt_id, int(time.time()), reason),
        )
        if commit:
            self.connection.commit()

    def should_run_cell(self, revision_id: int, cell_id: int,
                        *, rerun_failed: bool = True) -> bool:
        """Return whether a cell lacks a reusable selected successful attempt."""
        row = self.connection.execute(
            """
            SELECT a.score, a.error
            FROM benchmark_selections s
            JOIN benchmark_attempts a
              ON a.attempt_id = s.attempt_id
             AND a.revision_id = s.revision_id
             AND a.cell_id = s.cell_id
            WHERE s.revision_id = ? AND s.cell_id = ?
            """,
            (revision_id, cell_id),
        ).fetchone()
        if row is None:
            return True
        if row[0] is None or row[1] is not None:
            return rerun_failed
        return False

    def current_attempt(self, revision_id: int, cell_id: int) -> dict[str, Any] | None:
        """Return the selected normalized attempt, including payload IDs."""
        row = self.connection.execute(
            """
            SELECT a.* FROM benchmark_selections s
            JOIN benchmark_attempts a
              ON a.attempt_id = s.attempt_id
             AND a.revision_id = s.revision_id
             AND a.cell_id = s.cell_id
            WHERE s.revision_id = ? AND s.cell_id = ?
            """,
            (revision_id, cell_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def materialize_response(self, revision_id: int, cell_id: int,
                             output_dir: str, target_name: str,
                             plugin_id: str) -> list[str]:
        """Export selected payloads as legacy response files on request."""
        attempt = self.current_attempt(revision_id, cell_id)
        if attempt is None:
            return []
        target_dir = os.path.join(output_dir, "responses", sanitize_filename(target_name))
        os.makedirs(target_dir, exist_ok=True)
        paths: list[str] = []
        for suffix, column in (
            ("prompt.txt", "prompt_payload_id"),
            ("content.txt", "content_payload_id"),
            ("think.txt", "thinking_payload_id"),
        ):
            payload_id = attempt.get(column)
            if payload_id is None:
                continue
            path = os.path.join(target_dir, f"{plugin_id}.{suffix}")
            with open(path, "wb") as handle:
                handle.write(self.payloads.get(payload_id))
            paths.append(path)
        return paths

    @staticmethod
    def _json_value(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _bool_value(value: Any) -> int | None:
        if value is None:
            return None
        return int(bool(value))

    def _payload_id(self, kind: str, value: Any) -> int | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{kind} payload must be text")
        return self.payloads.put_text(kind, value)
