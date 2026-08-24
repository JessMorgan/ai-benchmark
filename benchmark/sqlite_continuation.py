"""Continuation and lifecycle operations for the normalized SQLite run store."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sqlite_benchmarks import SQLiteBenchmarkStore


@dataclass(frozen=True)
class TargetSpec:
    """Execution identity for one configured benchmark target."""

    logical_name: str
    runner: str
    source: str
    api_model: str
    target_signature: str
    is_agent: bool = False
    system_prompt: str | None = None
    target_config: Mapping[str, Any] | str | None = None
    order_index: int | None = None


@dataclass(frozen=True)
class PluginSpec:
    """Immutable plugin definition selected by a revision."""

    plugin_id: str
    plugin_version: str
    name: str
    max_score: float
    supports_streaming: bool
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class JudgeSpec:
    """Configured judge identity for one continuation revision."""

    judge_model: str
    source: str
    config: Mapping[str, Any] | str | None = None


@dataclass(frozen=True)
class ContractSpec:
    """Versioned judge contract for one plugin version."""

    contract_id: str
    plugin_id: str
    plugin_version: str
    prompt_version: str
    instructions_version: str
    response_schema_hash: str
    contract: Mapping[str, Any] | str
    contract_hash: str


@dataclass(frozen=True)
class ContinuationSummary:
    """Identifiers and counts produced by a continuation."""

    run_id: str
    revision_id: int
    reused_targets: int
    added_targets: int
    removed_targets: int
    reused_plugins: int
    added_plugins: int
    removed_plugins: int
    reused_cells: int
    scheduled_cells: int
    reused_votes: int


@dataclass(frozen=True)
class PurgeSummary:
    """Counts for a revision-local purge that retains immutable history."""

    revision_id: int
    benchmark_selections: int
    judge_votes: int
    cells_reset: int


class SQLiteContinuationStore:
    """Manage immutable continuation revisions and current projections.

    The store never deletes benchmark attempts, judge attempts, parsed votes,
    payloads, or old memberships during normal continuation. A new revision
    receives its own membership and projection rows and may point at compatible
    attempts/votes from an earlier revision.
    """

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def current_revision(self, run_id: str) -> int:
        row = self.connection.execute(
            "SELECT current_revision_id FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown SQLite run: {run_id}")
        if row[0] is None:
            raise RuntimeError(f"run has no current revision: {run_id}")
        return int(row[0])

    def create_continuation(
        self,
        run_id: str,
        *,
        config: Mapping[str, Any] | str,
        runner_mode: str,
        targets: Sequence[TargetSpec],
        plugins: Sequence[PluginSpec],
        judges: Sequence[JudgeSpec] = (),
        contracts: Mapping[str, ContractSpec] | None = None,
        session_seed: int | None = None,
        rerun_failed: bool = True,
    ) -> ContinuationSummary:
        """Create a revision and resolve its target/plugin/judge memberships."""
        old_revision = self.current_revision(run_id)
        old_number = self._revision_number(old_revision)
        config_json = self._json_text(config)
        now = int(time.time())
        contracts = contracts or {}
        target_specs = self._unique(targets, lambda item: (item.logical_name, item.runner))
        plugin_specs = self._unique(plugins, lambda item: item.plugin_id)
        judge_specs = self._unique(judges, lambda item: item.judge_model)

        try:
            self.connection.execute("BEGIN")
            cursor = self.connection.execute(
                """
                INSERT INTO run_revisions(
                    run_id, revision_number, status, started_at, runner_mode,
                    session_seed, config_json, config_sha256, created_at
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, old_number + 1, now, runner_mode, session_seed,
                    config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest(), now,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a continuation revision ID")
            revision_id = int(cursor.lastrowid)
            self.connection.execute(
                "UPDATE run_revisions SET status = 'continued', ended_at = ? "
                "WHERE revision_id = ? AND status = 'running'",
                (now, old_revision),
            )
            self.connection.execute(
                "UPDATE runs SET current_revision_id = ?, status = 'running' WHERE run_id = ?",
                (revision_id, run_id),
            )

            old_targets = self._copy_target_membership(old_revision, revision_id)
            target_ids: dict[tuple[str, str], int] = {}
            reused_targets = added_targets = 0
            for spec in target_specs:
                target_id, existed = self._get_or_create_target(revision_id, run_id, spec)
                target_ids[(spec.logical_name, spec.runner)] = target_id
                self._set_target_membership(revision_id, target_id, True, spec.order_index)
                if existed:
                    reused_targets += 1
                else:
                    added_targets += 1
            active_target_ids = set(target_ids.values())
            removed_targets = 0
            for target_id in old_targets:
                if target_id not in active_target_ids:
                    self._set_target_membership(revision_id, target_id, False, None)
                    self.connection.execute(
                        "UPDATE target_instances SET retired_revision_id = "
                        "COALESCE(retired_revision_id, ?) WHERE target_instance_id = ?",
                        (revision_id, target_id),
                    )
                    removed_targets += 1

            old_plugins = self._copy_plugin_membership(old_revision, revision_id)
            plugin_versions: dict[str, str] = {}
            reused_plugins = added_plugins = 0
            for spec in plugin_specs:
                self._ensure_plugin(spec)
                self._set_plugin_membership(revision_id, spec.plugin_id, spec.plugin_version, True)
                plugin_versions[spec.plugin_id] = spec.plugin_version
                if spec.plugin_id in old_plugins and old_plugins[spec.plugin_id] == spec.plugin_version:
                    reused_plugins += 1
                else:
                    added_plugins += 1
            removed_plugins = 0
            for plugin_id, version in old_plugins.items():
                if plugin_id not in plugin_versions:
                    self._set_plugin_membership(revision_id, plugin_id, version, False)
                    removed_plugins += 1

            old_contracts = self._copy_contract_membership(old_revision, revision_id)
            active_contracts: dict[str, str] = {}
            for plugin_id in plugin_versions:
                contract = contracts.get(plugin_id)
                if contract is None:
                    contract_id = old_contracts.get(plugin_id)
                else:
                    self._ensure_contract(contract)
                    contract_id = contract.contract_id
                if contract_id is not None:
                    self._set_contract_membership(revision_id, plugin_id, contract_id, True)
                    active_contracts[plugin_id] = contract_id
            for plugin_id, contract_id in old_contracts.items():
                if plugin_id not in active_contracts:
                    self._set_contract_membership(revision_id, plugin_id, contract_id, False)

            old_judges = self._copy_judge_membership(old_revision, revision_id)
            active_judges: dict[str, JudgeSpec] = {}
            for spec in judge_specs:
                self._set_judge_membership(revision_id, spec, True)
                active_judges[spec.judge_model] = spec
            for judge_model, old_spec in old_judges.items():
                if judge_model not in active_judges:
                    self._set_judge_membership(
                        revision_id,
                        JudgeSpec(judge_model, old_spec[0], old_spec[1]),
                        False,
                    )

            self._copy_revision_cells(old_revision, revision_id)
            reused_cells = scheduled_cells = 0
            active_cells: list[tuple[int, str, str]] = []
            for target_id in active_target_ids:
                for plugin_id, plugin_version in plugin_versions.items():
                    cell_id = self._ensure_cell(
                        revision_id, target_id, plugin_id, plugin_version,
                    )
                    active_cells.append((cell_id, plugin_id, plugin_version))
                    old_selection = self._selection(old_revision, cell_id)
                    reusable = self._selection_is_reusable(old_selection, rerun_failed)
                    if reusable:
                        if old_selection is None:
                            raise AssertionError("reusable selection unexpectedly missing")
                        old_attempt_id, old_success = old_selection
                        self._copy_selection(revision_id, cell_id, old_attempt_id)
                        self._set_cell_state(
                            revision_id, cell_id,
                            "completed" if old_success else "failed",
                            False,
                            "reused-compatible-attempt",
                        )
                        reused_cells += 1
                    else:
                        self._set_cell_state(
                            revision_id, cell_id, "pending", True,
                            "new-cell" if old_selection is None else "rerun-failed",
                        )
                        scheduled_cells += 1

            reused_votes = self._copy_compatible_votes(
                old_revision, revision_id, active_cells, active_judges,
                active_contracts,
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

        return ContinuationSummary(
            run_id=run_id,
            revision_id=revision_id,
            reused_targets=reused_targets,
            added_targets=added_targets,
            removed_targets=removed_targets,
            reused_plugins=reused_plugins,
            added_plugins=added_plugins,
            removed_plugins=removed_plugins,
            reused_cells=reused_cells,
            scheduled_cells=scheduled_cells,
            reused_votes=reused_votes,
        )

    def stop_revision(self, revision_id: int, *, reason: str = "stopped") -> None:
        """Mark in-flight attempts abandoned and interrupt a revision."""
        now = int(time.time())
        try:
            self.connection.execute("BEGIN")
            self.connection.execute(
                """
                UPDATE benchmark_attempts
                SET status = 'abandoned', response_nature = COALESCE(response_nature, 'abandoned'),
                    failure_cause = COALESCE(failure_cause, 'abandoned'),
                    error = COALESCE(error, ?), ended_at = COALESCE(ended_at, ?)
                WHERE revision_id = ? AND status IN ('running', 'in_flight')
                """,
                (reason, now, revision_id),
            )
            self.connection.execute(
                """
                UPDATE judge_attempts
                SET status = 'abandoned', error = COALESCE(error, ?),
                    ended_at = COALESCE(ended_at, ?)
                WHERE revision_id = ? AND status IN ('running', 'in_flight')
                """,
                (reason, now, revision_id),
            )
            self.connection.execute(
                "UPDATE run_revisions SET status = 'interrupted', ended_at = ? WHERE revision_id = ?",
                (now, revision_id),
            )
            self.connection.execute(
                "UPDATE runs SET status = 'interrupted' WHERE current_revision_id = ?",
                (revision_id,),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def purge_revision(
        self,
        revision_id: int,
        *,
        cell_ids: Sequence[int] | None = None,
    ) -> PurgeSummary:
        """Clear current selections while retaining all immutable history."""
        params: list[Any] = [revision_id]
        cell_filter = ""
        if cell_ids:
            placeholders = ",".join("?" for _ in cell_ids)
            cell_filter = f" AND cell_id IN ({placeholders})"
            params.extend(cell_ids)
        try:
            self.connection.execute("BEGIN")
            benchmark_count = self.connection.execute(
                f"DELETE FROM benchmark_selections WHERE revision_id = ?{cell_filter}", params
            ).rowcount
            judge_count = self.connection.execute(
                f"DELETE FROM current_judge_votes WHERE revision_id = ?{cell_filter}", params
            ).rowcount
            self.connection.execute(
                f"DELETE FROM consensus_cache WHERE revision_id = ?{cell_filter}", params
            )
            cells_reset = self.connection.execute(
                f"UPDATE revision_cells SET scheduled = 1, status = 'pending', "
                f"queue_reason = 'purged', updated_at = ? WHERE revision_id = ?{cell_filter}",
                [int(time.time()), *params],
            ).rowcount
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return PurgeSummary(revision_id, benchmark_count, judge_count, cells_reset)

    def restart_run(
        self,
        new_run_id: str,
        *,
        score_schema: str,
        storage_profile: str,
        runner_mode: str,
        config: Mapping[str, Any] | str,
        session_seed: int | None = None,
    ) -> int:
        """Create a fresh logical run in the same database without deleting history."""
        return SQLiteBenchmarkStore(self.connection).create_run(
            new_run_id,
            score_schema=score_schema,
            storage_profile=storage_profile,
            runner_mode=runner_mode,
            config=config if isinstance(config, str) else dict(config),
            session_seed=session_seed,
        )

    def _revision_number(self, revision_id: int) -> int:
        row = self.connection.execute(
            "SELECT revision_number FROM run_revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown revision: {revision_id}")
        return int(row[0])

    def _copy_target_membership(self, old_revision: int, revision_id: int) -> set[int]:
        rows = self.connection.execute(
            "SELECT target_instance_id FROM revision_targets WHERE revision_id = ? AND active = 1",
            (old_revision,),
        )
        active = {int(row[0]) for row in rows}
        for row in self.connection.execute(
            "SELECT target_instance_id, order_index FROM revision_targets WHERE revision_id = ?",
            (old_revision,),
        ):
            self._set_target_membership(revision_id, int(row[0]), False, row[1])
        return active

    def _get_or_create_target(self, revision_id: int, run_id: str, spec: TargetSpec) -> tuple[int, bool]:
        target_config = self._json_optional(spec.target_config)
        row = self.connection.execute(
            """
            SELECT target_instance_id FROM target_instances
            WHERE run_id = ? AND logical_name = ? AND runner = ? AND target_signature = ?
            """,
            (run_id, spec.logical_name, spec.runner, spec.target_signature),
        ).fetchone()
        if row is None:
            # Legacy JSON imports used a SHA-256 target signature, while the
            # live SQLite runtime uses ``source/api_model``. Match that one
            # known representation change by identity, but only when the old
            # signature is exactly a hexadecimal digest and all execution
            # identity fields agree. Other signature changes remain distinct
            # targets and therefore intentionally do not reuse attempts.
            legacy_rows = self.connection.execute(
                """
                SELECT target_instance_id, target_signature
                FROM target_instances
                WHERE run_id = ? AND logical_name = ? AND runner = ?
                  AND source = ? AND api_model = ?
                """,
                (run_id, spec.logical_name, spec.runner, spec.source, spec.api_model),
            ).fetchall()
            matching_legacy = [
                candidate for candidate in legacy_rows
                if isinstance(candidate[1], str)
                and len(candidate[1]) == 64
                and all(char in "0123456789abcdefABCDEF" for char in candidate[1])
            ]
            if len(matching_legacy) == 1:
                row = matching_legacy[0]

        if row is not None:
            return int(row[0]), True
        cursor = self.connection.execute(
            """
            INSERT INTO target_instances(
                run_id, logical_name, runner, source, api_model, is_agent,
                system_prompt, target_config_json, target_signature, first_revision_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, spec.logical_name, spec.runner, spec.source, spec.api_model,
                int(spec.is_agent), spec.system_prompt, target_config,
                spec.target_signature, revision_id,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a target ID")
        return int(cursor.lastrowid), False

    def _set_target_membership(self, revision_id: int, target_id: int,
                               active: bool, order_index: int | None) -> None:
        self.connection.execute(
            """
            INSERT INTO revision_targets(revision_id, target_instance_id, active, order_index, runtime_json)
            VALUES (?, ?, ?, ?, '{}')
            ON CONFLICT(revision_id, target_instance_id) DO UPDATE SET
                active = excluded.active, order_index = excluded.order_index
            """,
            (revision_id, target_id, int(active), order_index),
        )

    def _ensure_plugin(self, spec: PluginSpec) -> None:
        metadata = self._json_optional(spec.metadata)
        row = self.connection.execute(
            "SELECT name, max_score, supports_streaming, metadata_json FROM plugin_definitions "
            "WHERE plugin_id = ? AND plugin_version = ?",
            (spec.plugin_id, spec.plugin_version),
        ).fetchone()
        if row is not None:
            if (row[0], float(row[1]), bool(row[2]), row[3]) != (
                spec.name, float(spec.max_score), bool(spec.supports_streaming), metadata,
            ):
                # Plugin properties changed — update the stored definition so
                # the continuation reflects the current code.  Earlier schema
                # versions treated plugin definitions as immutable, which broke
                # resume after any plugin edit.  Updating here is safe because
                # benchmark cells and selections reference plugin_id+version,
                # not the mutable properties.
                self.connection.execute(
                    "UPDATE plugin_definitions SET name = ?, max_score = ?, "
                    "supports_streaming = ?, metadata_json = ? "
                    "WHERE plugin_id = ? AND plugin_version = ?",
                    (spec.name, spec.max_score, int(spec.supports_streaming),
                     metadata, spec.plugin_id, spec.plugin_version),
                )
            return
        self.connection.execute(
            "INSERT INTO plugin_definitions(plugin_id, plugin_version, name, max_score, "
            "supports_streaming, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (spec.plugin_id, spec.plugin_version, spec.name, spec.max_score,
             int(spec.supports_streaming), metadata),
        )

    def _copy_plugin_membership(self, old_revision: int, revision_id: int) -> dict[str, str]:
        rows = self.connection.execute(
            "SELECT plugin_id, plugin_version, active FROM revision_plugins WHERE revision_id = ?",
            (old_revision,),
        )
        old: dict[str, str] = {}
        for row in rows:
            if row[2]:
                old[str(row[0])] = str(row[1])
            self._set_plugin_membership(revision_id, str(row[0]), str(row[1]), False)
        return old

    def _set_plugin_membership(self, revision_id: int, plugin_id: str,
                               plugin_version: str, active: bool) -> None:
        self.connection.execute(
            """
            INSERT INTO revision_plugins(revision_id, plugin_id, plugin_version, active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(revision_id, plugin_id) DO UPDATE SET
                plugin_version = excluded.plugin_version, active = excluded.active
            """,
            (revision_id, plugin_id, plugin_version, int(active)),
        )

    def _copy_contract_membership(self, old_revision: int, revision_id: int) -> dict[str, str]:
        old: dict[str, str] = {}
        rows = self.connection.execute(
            "SELECT plugin_id, contract_id, active FROM revision_judge_contracts WHERE revision_id = ?",
            (old_revision,),
        )
        for row in rows:
            if row[2]:
                old[str(row[0])] = str(row[1])
            self._set_contract_membership(revision_id, str(row[0]), str(row[1]), False)
        return old

    def _ensure_contract(self, spec: ContractSpec) -> None:
        contract_json = self._json_text(spec.contract)
        row = self.connection.execute(
            "SELECT contract_hash, contract_json FROM judge_contracts WHERE contract_id = ?",
            (spec.contract_id,),
        ).fetchone()
        if row is not None:
            if row[0] == spec.contract_hash and row[1] == contract_json:
                return
            # Legacy JSON imports use a deliberately minimal placeholder
            # because the original contract body is not present in the state
            # file, and their ID-derived hash ≠ the canonical content hash.
            # When the runtime provides real content for the same contract_id
            # (or a new version provides updated content), treat the newer
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
                    spec.plugin_id, spec.plugin_version, spec.prompt_version,
                    spec.instructions_version, spec.response_schema_hash,
                    contract_json, spec.contract_hash, spec.contract_id,
                ),
            )
            return
        existing = self.connection.execute(
            "SELECT contract_id FROM judge_contracts WHERE contract_hash = ?",
            (spec.contract_hash,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != spec.contract_id:
                # The content identity is authoritative; callers may use the
                # existing ID by passing the same hash on a later continuation.
                raise ValueError(
                    f"contract hash already belongs to {existing[0]}, not {spec.contract_id}"
                )
            return
        self.connection.execute(
            """
            INSERT INTO judge_contracts(
                contract_id, plugin_id, plugin_version, prompt_version,
                instructions_version, response_schema_hash, contract_json, contract_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.contract_id, spec.plugin_id, spec.plugin_version,
                spec.prompt_version, spec.instructions_version,
                spec.response_schema_hash, contract_json, spec.contract_hash,
            ),
        )

    def _set_contract_membership(self, revision_id: int, plugin_id: str,
                                 contract_id: str, active: bool) -> None:
        self.connection.execute(
            """
            INSERT INTO revision_judge_contracts(revision_id, plugin_id, contract_id, active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(revision_id, plugin_id) DO UPDATE SET
                contract_id = excluded.contract_id, active = excluded.active
            """,
            (revision_id, plugin_id, contract_id, int(active)),
        )

    def _copy_judge_membership(self, old_revision: int, revision_id: int) -> dict[str, tuple[str, str | None]]:
        old: dict[str, tuple[str, str | None]] = {}
        rows = self.connection.execute(
            "SELECT judge_model, source, config_json, active FROM revision_judges WHERE revision_id = ?",
            (old_revision,),
        )
        for row in rows:
            if row[3]:
                old[str(row[0])] = (str(row[1]), row[2])
            self._set_judge_membership(
                revision_id, JudgeSpec(str(row[0]), str(row[1]), row[2]), False,
            )
        return old

    def _set_judge_membership(self, revision_id: int, spec: JudgeSpec, active: bool) -> None:
        self.connection.execute(
            """
            INSERT INTO revision_judges(revision_id, judge_model, source, config_json, active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(revision_id, judge_model) DO UPDATE SET
                source = excluded.source, config_json = excluded.config_json,
                active = excluded.active
            """,
            (revision_id, spec.judge_model, spec.source,
             self._json_optional(spec.config), int(active)),
        )

    def _copy_revision_cells(self, old_revision: int, revision_id: int) -> None:
        for row in self.connection.execute(
            "SELECT cell_id, scheduled, status, queue_reason FROM revision_cells WHERE revision_id = ?",
            (old_revision,),
        ):
            self.connection.execute(
                """
                INSERT INTO revision_cells(
                    revision_id, cell_id, scheduled, status, queue_reason, updated_at
                ) VALUES (?, ?, 0, 'inactive', 'not-active-in-revision', ?)
                """,
                (revision_id, int(row[0]), int(time.time())),
            )

    def _ensure_cell(self, revision_id: int, target_id: int,
                     plugin_id: str, plugin_version: str) -> int:
        row = self.connection.execute(
            "SELECT run_id FROM target_instances WHERE target_instance_id = ?", (target_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown target instance: {target_id}")
        cell = self.connection.execute(
            "SELECT cell_id FROM cells WHERE target_instance_id = ? AND plugin_id = ? "
            "AND plugin_version = ?", (target_id, plugin_id, plugin_version),
        ).fetchone()
        if cell is None:
            cursor = self.connection.execute(
                "INSERT INTO cells(run_id, target_instance_id, plugin_id, plugin_version, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (row[0], target_id, plugin_id, plugin_version, int(time.time())),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a cell ID")
            cell_id = int(cursor.lastrowid)
        else:
            cell_id = int(cell[0])
        self.connection.execute(
            """
            INSERT INTO revision_cells(revision_id, cell_id, scheduled, status, queue_reason, updated_at)
            VALUES (?, ?, 1, 'pending', 'new-cell', ?)
            ON CONFLICT(revision_id, cell_id) DO UPDATE SET
                scheduled = 1, status = 'pending', queue_reason = 'active-cell', updated_at = excluded.updated_at
            """,
            (revision_id, cell_id, int(time.time())),
        )
        return cell_id

    def _selection(self, revision_id: int, cell_id: int) -> tuple[int, bool] | None:
        row = self.connection.execute(
            """
            SELECT s.attempt_id, (a.score IS NOT NULL AND a.error IS NULL
                                  AND a.status NOT IN ('abandoned', 'failed'))
            FROM benchmark_selections s
            JOIN benchmark_attempts a ON a.attempt_id = s.attempt_id
            WHERE s.revision_id = ? AND s.cell_id = ?
            """,
            (revision_id, cell_id),
        ).fetchone()
        return (int(row[0]), bool(row[1])) if row is not None else None

    @staticmethod
    def _selection_is_reusable(selection: tuple[int, bool] | None,
                               rerun_failed: bool) -> bool:
        return selection is not None and (selection[1] or not rerun_failed)

    def _copy_selection(self, revision_id: int, cell_id: int, attempt_id: int) -> None:
        self.connection.execute(
            """
            INSERT INTO benchmark_selections(
                revision_id, cell_id, attempt_id, selected_at, selection_reason
            ) VALUES (?, ?, ?, ?, 'reused-compatible-attempt')
            """,
            (revision_id, cell_id, attempt_id, int(time.time())),
        )

    def _set_cell_state(self, revision_id: int, cell_id: int, status: str,
                        scheduled: bool, reason: str) -> None:
        self.connection.execute(
            """
            UPDATE revision_cells SET status = ?, scheduled = ?, queue_reason = ?, updated_at = ?
            WHERE revision_id = ? AND cell_id = ?
            """,
            (status, int(scheduled), reason, int(time.time()), revision_id, cell_id),
        )

    def _copy_compatible_votes(
        self,
        old_revision: int,
        revision_id: int,
        active_cells: Sequence[tuple[int, str, str]],
        active_judges: Mapping[str, JudgeSpec],
        active_contracts: Mapping[str, str],
    ) -> int:
        count = 0
        for cell_id, plugin_id, _plugin_version in active_cells:
            contract_id = active_contracts.get(plugin_id)
            if contract_id is None:
                continue
            new_selection = self._selection(revision_id, cell_id)
            old_selection = self._selection(old_revision, cell_id)
            if new_selection is None or old_selection is None or new_selection[0] != old_selection[0]:
                continue
            for judge_model, spec in active_judges.items():
                old = self.connection.execute(
                    "SELECT source, config_json FROM revision_judges WHERE revision_id = ? "
                    "AND judge_model = ? AND active = 1", (old_revision, judge_model),
                ).fetchone()
                if old is None or (old[0], old[1]) != (spec.source, self._json_optional(spec.config)):
                    continue
                vote = self.connection.execute(
                    """
                    SELECT vote_attempt_id FROM current_judge_votes
                    WHERE revision_id = ? AND cell_id = ? AND judge_model = ? AND contract_id = ?
                    """,
                    (old_revision, cell_id, judge_model, contract_id),
                ).fetchone()
                if vote is None:
                    continue
                self.connection.execute(
                    """
                    INSERT INTO current_judge_votes(
                        revision_id, cell_id, judge_model, contract_id,
                        vote_attempt_id, selected_at, selection_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, 'reused-compatible-vote')
                    """,
                    (revision_id, cell_id, judge_model, contract_id, int(vote[0]), int(time.time())),
                )
                count += 1
        return count

    @staticmethod
    def _unique(items: Sequence[Any], key: Callable[[Any], Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[Any] = set()
        for item in items:
            item_key = key(item)
            if item_key in seen:
                raise ValueError(f"duplicate continuation membership: {item_key}")
            seen.add(item_key)
            result.append(item)
        return result

    @staticmethod
    def _json_text(value: Mapping[str, Any] | str) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _json_optional(value: Mapping[str, Any] | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
