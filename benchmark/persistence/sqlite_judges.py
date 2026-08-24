"""Normalized SQLite persistence for versioned judge attempts and votes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

from .sqlite_payloads import SQLitePayloadStore


class SQLiteJudgeStore:
    """Persist judge history without overwriting prior attempts or contracts."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.payloads = SQLitePayloadStore(connection)

    def register_judge(self, revision_id: int, judge_model: str, *, source: str,
                       config: dict[str, Any] | str | None = None,
                       active: bool = True) -> None:
        config_json = (
            config if isinstance(config, str) else
            json.dumps(config, sort_keys=True) if config is not None else None
        )
        self.connection.execute(
            """
            INSERT INTO revision_judges(revision_id, judge_model, source, config_json, active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(revision_id, judge_model) DO UPDATE SET
                source = excluded.source, config_json = excluded.config_json,
                active = excluded.active
            """,
            (revision_id, judge_model, source, config_json, int(active)),
        )
        self.connection.commit()

    def register_contract(
        self,
        contract_id: str,
        *,
        plugin_id: str,
        plugin_version: str,
        prompt_version: str,
        instructions_version: str,
        response_schema_hash: str,
        contract: dict[str, Any] | str,
        contract_hash: str,
    ) -> None:
        """Register an immutable judge contract by its content hash."""
        contract_json = contract if isinstance(contract, str) else json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        row = self.connection.execute(
            """
            SELECT plugin_id, plugin_version, prompt_version, instructions_version,
                   response_schema_hash, contract_json, contract_hash
            FROM judge_contracts WHERE contract_id = ?
            """,
            (contract_id,),
        ).fetchone()
        if row is None:
            self.connection.execute(
                """
                INSERT INTO judge_contracts(
                    contract_id, plugin_id, plugin_version, prompt_version,
                    instructions_version, response_schema_hash, contract_json, contract_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id, plugin_id, plugin_version, prompt_version,
                    instructions_version, response_schema_hash, contract_json, contract_hash,
                ),
            )
        elif tuple(row) != (
            plugin_id, plugin_version, prompt_version, instructions_version,
            response_schema_hash, contract_json, contract_hash,
        ):
            # A legacy JSON import creates placeholder contracts whose hashes
            # are derived from the contract_id string, not the content. When
            # the runtime provides real content for the same contract_id (or
            # a new version provides updated content), treat the newer
            # content as authoritative and update the row.
            self.connection.execute(
                """
                UPDATE judge_contracts SET
                    plugin_id = ?, plugin_version = ?, prompt_version = ?,
                    instructions_version = ?, response_schema_hash = ?,
                    contract_json = ?, contract_hash = ?
                WHERE contract_id = ?
                """,
                (
                    plugin_id, plugin_version, prompt_version, instructions_version,
                    response_schema_hash, contract_json, contract_hash, contract_id,
                ),
            )
        self.connection.commit()

    def activate_contract(self, revision_id: int, plugin_id: str, contract_id: str,
                          *, active: bool = True) -> None:
        self.connection.execute(
            """
            INSERT INTO revision_judge_contracts(revision_id, plugin_id, contract_id, active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(revision_id, plugin_id) DO UPDATE SET
                contract_id = excluded.contract_id, active = excluded.active
            """,
            (revision_id, plugin_id, contract_id, int(active)),
        )
        self.connection.commit()

    def record_attempt(
        self,
        revision_id: int,
        cell_id: int,
        judge_model: str,
        contract_id: str,
        attempt: dict[str, Any],
        *,
        retain_request: bool = False,
    ) -> int:
        """Persist one transport attempt, optionally retaining its request."""
        raw_response_id = self._payload_id(
            "judge-response", attempt.get("raw_response", attempt.get("response")),
        )
        request_id = (
            self._payload_id("judge-request", attempt.get("request"))
            if retain_request else None
        )
        cursor = self.connection.execute(
            """
            INSERT INTO judge_attempts(
                revision_id, cell_id, judge_model, contract_id, attempt_number,
                started_at, ended_at, max_tokens, raw_response_payload_id,
                request_payload_id, response_usage_json, diagnostics_json,
                finish_reason, error, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id, cell_id, judge_model, contract_id,
                int(attempt.get("attempt_number", attempt.get("attempt", 1))),
                attempt.get("started_at"), attempt.get("ended_at"),
                attempt.get("max_tokens"), raw_response_id, request_id,
                self._json_value(attempt.get("usage")),
                self._json_value(attempt.get("diagnostics")),
                attempt.get("finish_reason"), attempt.get("error"),
                attempt.get("status", "completed"),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a judge attempt ID")
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_vote(self, judge_attempt_id: int, vote: dict[str, Any]) -> int:
        """Persist one parsed vote and its criteria/evidence rows."""
        score = vote.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            score = None
        usable = bool(vote.get("usable", score is not None and not vote.get("error")))
        cursor = self.connection.execute(
            """
            INSERT INTO judge_vote_attempts(
                judge_attempt_id, score, confidence, rationale, error, usable, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                judge_attempt_id, score, vote.get("confidence"), vote.get("rationale"),
                vote.get("error"), int(usable), int(time.time()),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a vote attempt ID")
        vote_id = int(cursor.lastrowid)
        for ordinal, criterion in enumerate(vote.get("criteria", []) or []):
            if not isinstance(criterion, dict):
                continue
            self.connection.execute(
                """
                INSERT INTO judge_criteria(
                    vote_attempt_id, ordinal, criterion_key, criterion,
                    status, evidence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    vote_id,
                    ordinal,
                    str(criterion.get("id", criterion.get("key", ordinal))),
                    str(criterion.get("criterion", "")),
                    str(criterion.get("status", "unknown")),
                    str(criterion.get("evidence", "")),
                ),
            )
        self.connection.commit()
        return vote_id

    def select_vote(self, revision_id: int, cell_id: int, judge_model: str,
                    contract_id: str, vote_attempt_id: int,
                    *, reason: str = "latest-usable") -> None:
        """Select one parsed vote for the revision-local projection."""
        self.connection.execute(
            """
            INSERT INTO current_judge_votes(
                revision_id, cell_id, judge_model, contract_id,
                vote_attempt_id, selected_at, selection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(revision_id, cell_id, judge_model, contract_id) DO UPDATE SET
                vote_attempt_id = excluded.vote_attempt_id,
                selected_at = excluded.selected_at,
                selection_reason = excluded.selection_reason
            """,
            (
                revision_id, cell_id, judge_model, contract_id,
                vote_attempt_id, int(time.time()), reason,
            ),
        )
        self.connection.commit()

    def current_votes(self, revision_id: int, cell_id: int,
                      contract_id: str | None = None) -> list[dict[str, Any]]:
        """Return current parsed votes, optionally for one contract."""
        query = """
            SELECT v.vote_attempt_id, v.judge_attempt_id, a.judge_model,
                   a.contract_id, v.score, v.confidence, v.rationale,
                   v.error, v.usable
            FROM current_judge_votes c
            JOIN judge_vote_attempts v ON v.vote_attempt_id = c.vote_attempt_id
            JOIN judge_attempts a ON a.judge_attempt_id = v.judge_attempt_id
            WHERE c.revision_id = ? AND c.cell_id = ?
        """
        params: list[Any] = [revision_id, cell_id]
        if contract_id is not None:
            query += " AND c.contract_id = ?"
            params.append(contract_id)
        query += " ORDER BY a.judge_model"
        return [dict(row) for row in self.connection.execute(query, params)]

    def vote_set_hash(self, revision_id: int, cell_id: int, contract_id: str) -> str:
        """Hash the current vote identities and values for cache invalidation."""
        votes = self.current_votes(revision_id, cell_id, contract_id)
        material = json.dumps(votes, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def cache_consensus(self, revision_id: int, cell_id: int, contract_id: str,
                        *, score: float | None, confidence: str | None,
                        attempts: int | None = None) -> str:
        """Store a consensus projection keyed by the current vote-set hash."""
        votes = self.current_votes(revision_id, cell_id, contract_id)
        vote_hash = self.vote_set_hash(revision_id, cell_id, contract_id)
        valid_judges = sum(1 for vote in votes if vote["usable"])
        self.connection.execute(
            """
            INSERT INTO consensus_cache(
                revision_id, cell_id, contract_id, score, confidence,
                valid_judges, attempts, vote_set_hash, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(revision_id, cell_id, contract_id) DO UPDATE SET
                score = excluded.score, confidence = excluded.confidence,
                valid_judges = excluded.valid_judges, attempts = excluded.attempts,
                vote_set_hash = excluded.vote_set_hash,
                calculated_at = excluded.calculated_at
            """,
            (
                revision_id, cell_id, contract_id, score, confidence,
                valid_judges, len(votes) if attempts is None else attempts,
                vote_hash, int(time.time()),
            ),
        )
        self.connection.commit()
        return vote_hash

    def cached_consensus(self, revision_id: int, cell_id: int,
                         contract_id: str) -> dict[str, Any] | None:
        """Return a consensus cache only when its vote set is still current."""
        row = self.connection.execute(
            """
            SELECT * FROM consensus_cache
            WHERE revision_id = ? AND cell_id = ? AND contract_id = ?
            """,
            (revision_id, cell_id, contract_id),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result["vote_set_hash"] != self.vote_set_hash(revision_id, cell_id, contract_id):
            return None
        return result

    def criteria(self, vote_attempt_id: int) -> list[dict[str, Any]]:
        """Return normalized criteria/evidence for a parsed vote."""
        return [
            dict(row) for row in self.connection.execute(
                """
                SELECT ordinal, criterion_key, criterion, status, evidence
                FROM judge_criteria WHERE vote_attempt_id = ? ORDER BY ordinal
                """,
                (vote_attempt_id,),
            )
        ]

    def _payload_id(self, kind: str, value: Any) -> int | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{kind} payload must be text")
        return self.payloads.put_text(kind, value)

    @staticmethod
    def _json_value(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
