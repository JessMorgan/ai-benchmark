"""Backend-neutral persistence façade for benchmark runs."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .runtime_records import (
    BenchmarkAttemptRecord,
    JudgeAttemptRecord,
    JudgeVoteRecord,
    PluginRecord,
    TargetRecord,
)
from .sqlite_benchmarks import SQLiteBenchmarkStore
from .sqlite_continuation import ContractSpec, JudgeSpec, PluginSpec, TargetSpec
from .sqlite_judges import SQLiteJudgeStore
from .sqlite_writer import SQLiteWriteQueue


@dataclass(frozen=True)
class RunIdentity:
    """Logical run and continuation revision identity."""

    run_id: str
    revision_id: int | None


@runtime_checkable
class RunStore(Protocol):
    """Persistence façade used by benchmark and judge runtime code."""

    backend_name: str

    def start_run(self, identity: RunIdentity, **metadata: Any) -> None:
        """Initialize or attach the current run/revision."""
        ...

    def record_result(self, result: dict[str, Any]) -> None:
        """Record one benchmark result read-model update."""
        ...

    def update_model(self, model_name: str, **fields: Any) -> None:
        """Update live model state or its backend projection."""
        ...

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        """Record a judge projection update."""
        ...

    def save_snapshot(self, path: str | None = None,
                      plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        """Flush/persist a backend snapshot."""
        ...

    def latest_results(self) -> list[dict[str, Any]]:
        """Return the current result read model."""
        ...

    def flush(self, timeout: float | None = None) -> None:
        """Wait for queued persistence work."""
        ...

    def close(self, timeout: float | None = None) -> bool:
        """Close the backend and return whether shutdown completed."""
        ...

    def prepare_run(self, targets: Iterable[TargetRecord],
                    plugins: Iterable[PluginRecord]) -> None:
        ...

    def register_target(self, target: TargetRecord) -> int | None:
        ...

    def register_plugin(self, plugin: PluginRecord) -> None:
        ...

    def ensure_cell(self, target: TargetRecord, plugin: PluginRecord) -> int | None:
        ...

    def record_benchmark_attempt(self, cell_id: int, attempt: BenchmarkAttemptRecord,
                                 *, selected: bool = False) -> Any:
        ...

    def record_judge_attempt(self, cell_id: int, attempt: JudgeAttemptRecord,
                             vote: JudgeVoteRecord | None = None) -> Any:
        ...

    def get_cell_id(self, target_name: str, runner: str, plugin_id: str) -> int | None:
        """Return the cell ID for a (target, runner, plugin) pair, if registered."""
        ...

    def register_judge(self, judge_model: str, source: str,
                       config: dict[str, Any] | None = None) -> None:
        ...

    def register_contract(self, contract_id: str, *, plugin_id: str,
                          plugin_version: str, prompt_version: str,
                          instructions_version: str) -> None:
        ...


@runtime_checkable
class PayloadStore(Protocol):
    """Canonical storage for large prompt and response payloads."""

    def put(self, kind: str, data: bytes) -> int:
        ...

    def get(self, payload_id: int) -> bytes:
        ...


@runtime_checkable
class DebugLogStore(Protocol):
    """Append-only diagnostic log contract."""

    def append(self, path: str, data: str | bytes) -> None:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class ReportSource(Protocol):
    """Read-only source used to construct derived reports."""

    def load_results(self, path: str) -> tuple[list[dict[str, Any]], list[str], int | None]:
        ...


class JsonRunStore:
    """Facade adapter preserving the existing JSON ``BenchmarkState`` behavior."""

    backend_name = "json"

    def __init__(self, state: Any):
        self._state = state
        self.identity: RunIdentity | None = None

    def start_run(self, identity: RunIdentity, **metadata: Any) -> None:
        del metadata
        self.identity = identity

    def record_result(self, result: dict[str, Any]) -> None:
        self._state.add_result(result)

    def update_model(self, model_name: str, **fields: Any) -> None:
        # BenchmarkState owns the live JSON mutation; this method is only a
        # façade hook for non-JSON backends.
        del model_name, fields

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        self._state.update_judge_result(state_key, runner, plugin_id, **fields)

    def save_snapshot(self, path: str | None = None,
                      plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        if path is None:
            raise ValueError("JSON snapshots require a state path")
        return self._state.compact_journal(
            path, plugin_versions=plugin_versions,
            raise_on_error=raise_on_error,
        )

    def latest_results(self) -> list[dict[str, Any]]:
        return self._state.latest_results()

    def flush(self, timeout: float | None = None) -> None:
        del timeout

    def close(self, timeout: float | None = None) -> bool:
        del timeout
        return True

    def prepare_run(self, targets: Iterable[TargetRecord],
                    plugins: Iterable[PluginRecord]) -> None:
        del targets, plugins

    def register_target(self, target: TargetRecord) -> int | None:
        del target
        return None

    def register_plugin(self, plugin: PluginRecord) -> None:
        del plugin

    def ensure_cell(self, target: TargetRecord, plugin: PluginRecord) -> int | None:
        del target, plugin
        return None

    def record_benchmark_attempt(self, cell_id: int, attempt: BenchmarkAttemptRecord,
                                 *, selected: bool = False) -> Any:
        del cell_id, attempt, selected
        return None

    def record_judge_attempt(self, cell_id: int, attempt: JudgeAttemptRecord,
                             vote: JudgeVoteRecord | None = None) -> Any:
        del cell_id, attempt, vote
        return None

    def get_cell_id(self, target_name: str, runner: str, plugin_id: str) -> int | None:
        del target_name, runner, plugin_id
        return None

    def register_judge(self, judge_model: str, source: str,
                       config: dict[str, Any] | None = None) -> None:
        del judge_model, source, config

    def register_contract(self, contract_id: str, *, plugin_id: str,
                          plugin_version: str, prompt_version: str,
                          instructions_version: str) -> None:
        del contract_id, plugin_id, plugin_version, prompt_version, instructions_version


class SQLiteRunStore:
    """Facade over the SQLite schema stores and one background writer thread.

    The façade deliberately accepts backend operations as callables so callers
    do not share SQLite connections. Each operation runs on the writer thread
    and commits with the configured batch transaction. The normalized benchmark
    and judge stores remain available through ``benchmark`` and ``judges`` for
    migration code and advanced callers.
    """

    backend_name = "sqlite"

    def __init__(
        self,
        path: str,
        *,
        batch_size: int = 64,
        flush_interval: float = 0.25,
        synchronous: str = "NORMAL",
        failure_callback: Any = None,
    ):
        self.path = path
        self.writer = SQLiteWriteQueue(
            path, batch_size=batch_size, flush_interval=flush_interval,
            synchronous=synchronous, failure_callback=failure_callback,
        )
        self.identity: RunIdentity | None = None
        self._metadata: dict[str, Any] = {}
        self._results: dict[tuple[Any, Any], dict[str, Any]] = {}
        self._connection: sqlite3.Connection | None = None
        self.benchmark: SQLiteBenchmarkStore | None = None
        self.judges: SQLiteJudgeStore | None = None
        self._revision_id: int | None = None
        self._is_new_run: bool = False
        self._target_ids: dict[tuple[str, str, str], int] = {}
        self._target_records: dict[tuple[str, str], TargetRecord] = {}
        self._plugin_records: dict[str, PluginRecord] = {}
        self._judge_records: dict[str, tuple[str, dict[str, Any] | str | None]] = {}
        self._contract_records: dict[str, tuple[str, str, str, str]] = {}
        self._cell_ids: dict[tuple[str, str, str], int] = {}

    @property
    def revision_id(self) -> int | None:
        """The authoritative SQLite revision rowid for the current run."""
        return self._revision_id

    def start_run(self, identity: RunIdentity, **metadata: Any) -> None:
        """Start the writer and resolve/create the run's current revision.

        ``identity.revision_id`` is the CLI's continuation counter (run-info
        bookkeeping), not the SQLite revision rowid. The SQLite rowid is
        resolved from the database: a missing run creates revision 1, an
        existing run reuses its current revision. A continuation creates a
        fresh revision via ``continue_run``.
        """
        self.writer.start()
        self.writer.flush(timeout=10)
        def operation(connection):
            row = connection.execute(
                "SELECT run_id, current_revision_id FROM runs WHERE run_id = ?",
                (identity.run_id,),
            ).fetchone()
            # Recover databases created by the pre-manifest importer before
            # honoring a non-imported shadow UUID. The output directory is a
            # single logical run, and legacy_import_records is the durable
            # evidence that identifies the history the operator intended to
            # continue.
            imported = connection.execute(
                """
                SELECT DISTINCT r.run_id, r.current_revision_id
                FROM legacy_import_records l
                JOIN runs r ON r.run_id = l.run_id
                WHERE r.current_revision_id IS NOT NULL
                ORDER BY r.created_at, r.run_id
                """
            ).fetchall()
            if len(imported) == 1 and str(imported[0][0]) != identity.run_id:
                return str(imported[0][0]), int(imported[0][1]), False

            if row is not None and row[1] is not None:
                return str(row[0]), int(row[1]), False

            # JSON-to-SQLite import predates the SQLite run manifest, so the
            # CLI may present a newly generated UUID even though this database
            # already contains the imported logical run. Attach to a sole
            # existing run instead of silently creating an empty shadow run.
            # With multiple logical runs there is no safe inference; fail
            # rather than resuming the wrong history.
            existing = connection.execute(
                "SELECT run_id, current_revision_id FROM runs "
                "WHERE current_revision_id IS NOT NULL ORDER BY created_at, run_id"
            ).fetchall()
            if len(existing) == 1:
                return str(existing[0][0]), int(existing[0][1]), False
            if len(existing) > 1:
                raise RuntimeError(
                    "SQLite database contains multiple logical runs and the "
                    f"requested run ID {identity.run_id!r} is unknown; "
                    "provide the matching run-info.json or use a fresh database"
                )
            revision_id = SQLiteBenchmarkStore(connection).create_run(
                identity.run_id,
                score_schema=str(metadata.get("score_schema", "v1")),
                storage_profile=str(metadata.get("storage_profile", "compact")),
                runner_mode=str(metadata.get("runner_mode", "http")),
                config=metadata.get("config", {}),
                session_seed=metadata.get("session_seed"),
            )
            return identity.run_id, revision_id, True
        actual_run_id, self._revision_id, self._is_new_run = self._connection_operation(operation)
        self.identity = RunIdentity(actual_run_id, identity.revision_id)
        self._metadata = dict(metadata)

    def continue_run(self, *, config: Any, runner_mode: str, session_seed: int | None,
                     rerun_failed: bool = True) -> None:
        """Create a continuation revision reusing compatible prior attempts.

        Uses the target/plugin records registered by ``prepare_run``. Cells are
        stable across revisions, so existing ``_cell_ids`` remain valid for the
        new revision. Judge votes are not carried forward here; they are
        re-derived on resume from the retained benchmark selections.
        """
        from .sqlite_continuation import SQLiteContinuationStore
        if self.identity is None or self._revision_id is None:
            raise RuntimeError("SQLite run must be started before continuing")

        targets = [
            TargetSpec(
                record.logical_name, record.runner, record.source, record.api_model,
                record.target_signature, record.is_agent, record.system_prompt,
                record.target_config, record.order_index,
            )
            for record in self._target_records.values()
        ]
        plugins = [
            PluginSpec(
                record.plugin_id, record.plugin_version, record.name,
                record.max_score, record.supports_streaming, record.metadata,
            )
            for record in self._plugin_records.values()
        ]
        judges = [
            JudgeSpec(judge_model, source, config)
            for judge_model, (source, config) in self._judge_records.items()
        ]
        contracts = {
            plugin_id: ContractSpec(
                contract_id, plugin_id, plugin_version, prompt_version,
                instructions_version, response_schema_hash,
                {"contract_id": contract_id}, response_schema_hash,
            )
            for plugin_id, (
                contract_id, plugin_version, prompt_version,
                instructions_version,
            ) in self._contract_records.items()
            for response_schema_hash in [hashlib.sha256(contract_id.encode("utf-8")).hexdigest()]
        }

        # ``create_continuation`` manages its own BEGIN/COMMIT transaction,
        # which conflicts with the writer's batch transaction. It runs on a
        # dedicated connection at startup, after ``prepare_run`` has flushed
        # the writer, so no writer work is in flight.
        self.flush(timeout=30)
        from .sqlite_schema import configure_connection, connect_database
        connection = connect_database(self.path)
        try:
            configure_connection(connection)
            store = SQLiteContinuationStore(connection)
            self._revision_id = store.create_continuation(
                self.identity.run_id,
                config=config,
                runner_mode=runner_mode,
                targets=targets,
                plugins=plugins,
                judges=judges,
                contracts=contracts,
                session_seed=session_seed,
                rerun_failed=rerun_failed,
            ).revision_id
        finally:
            connection.close()

    def _connection_operation(self, operation: Any) -> Any:
        future = self.writer.submit(operation)
        return future.result(timeout=30)

    def prepare_run(self, targets: Iterable[TargetRecord],
                    plugins: Iterable[PluginRecord]) -> None:
        target_list = list(targets)
        plugin_list = list(plugins)
        for plugin in plugin_list:
            self.register_plugin(plugin)
        for target in target_list:
            self.register_target(target)
        for target in target_list:
            for plugin in plugin_list:
                self.ensure_cell(target, plugin)

    def register_target(self, target: TargetRecord) -> int | None:
        if self.identity is None or self._revision_id is None:
            raise RuntimeError("SQLite run must be started before registering targets")
        from .sqlite_continuation import TargetSpec
        spec = TargetSpec(
            target.logical_name, target.runner, target.source, target.api_model,
            target.target_signature, target.is_agent, target.system_prompt,
            target.target_config, target.order_index,
        )
        def operation(connection):
            store = SQLiteBenchmarkStore(connection)
            return store.register_target(
                self._revision_id, run_id=self.identity.run_id,
                logical_name=spec.logical_name, runner=spec.runner,
                source=spec.source, api_model=spec.api_model,
                target_signature=spec.target_signature, is_agent=spec.is_agent,
                system_prompt=spec.system_prompt, target_config=spec.target_config,
                order_index=spec.order_index,
            )
        target_id = self._connection_operation(operation)
        self._target_ids[(target.logical_name, target.runner, target.target_signature)] = target_id
        self._target_records[(target.logical_name, target.runner)] = target
        return target_id

    def register_plugin(self, plugin: PluginRecord) -> None:
        if self.identity is None or self._revision_id is None:
            raise RuntimeError("SQLite run must be started before registering plugins")
        def operation(connection):
            store = SQLiteBenchmarkStore(connection)
            store.register_plugin(
                plugin.plugin_id, plugin.plugin_version, name=plugin.name,
                max_score=plugin.max_score,
                supports_streaming=plugin.supports_streaming,
                metadata=plugin.metadata,
            )
            store.activate_plugin(
                self._revision_id, plugin.plugin_id, plugin.plugin_version,
            )
        self._connection_operation(operation)
        self._plugin_records[plugin.plugin_id] = plugin

    def ensure_cell(self, target: TargetRecord, plugin: PluginRecord) -> int | None:
        target_id = self._target_ids.get(
            (target.logical_name, target.runner, target.target_signature),
        )
        if target_id is None or self.identity is None or self._revision_id is None:
            raise RuntimeError("target/run must be registered before creating a cell")
        def operation(connection):
            return SQLiteBenchmarkStore(connection).ensure_cell(
                self._revision_id, target_id,
                plugin.plugin_id, plugin.plugin_version,
            )
        cell_id = self._connection_operation(operation)
        self._cell_ids[(target.logical_name, target.runner, plugin.plugin_id)] = cell_id
        return cell_id

    def record_benchmark_attempt(self, cell_id: int, attempt: BenchmarkAttemptRecord,
                                 *, selected: bool = False) -> Any:
        if self.identity is None or self._revision_id is None:
            raise RuntimeError("SQLite run must be started before recording attempts")
        def operation(connection):
            return SQLiteBenchmarkStore(connection).record_attempt(
                self._revision_id, cell_id, attempt.as_dict(), selected=selected,
            )
        return self._submit_async(operation)

    def record_judge_attempt(self, cell_id: int, attempt: JudgeAttemptRecord,
                             vote: JudgeVoteRecord | None = None) -> Any:
        if self.identity is None or self._revision_id is None:
            raise RuntimeError("SQLite run must be started before recording judge attempts")
        def operation(connection):
            judges = SQLiteJudgeStore(connection)
            judge_attempt_id = judges.record_attempt(
                self._revision_id, cell_id, attempt.judge_model,
                attempt.contract_id, attempt.as_dict(),
            )
            if vote is not None:
                vote_id = judges.record_vote(judge_attempt_id, vote.as_dict())
                if vote.usable:
                    judges.select_vote(
                        self._revision_id, cell_id, attempt.judge_model,
                        attempt.contract_id, vote_id,
                    )
            return judge_attempt_id
        return self._submit_async(operation)

    def get_cell_id(self, target_name: str, runner: str, plugin_id: str) -> int | None:
        """Return the cell ID for a (target, runner, plugin) pair, if registered."""
        return self._cell_ids.get((target_name, runner, plugin_id))

    def register_judge(self, judge_model: str, source: str,
                       config: dict[str, Any] | None = None) -> None:
        """Activate one judge model in the current revision."""
        if self._revision_id is None:
            raise RuntimeError("SQLite run must be started before registering judges")
        def operation(connection):
            SQLiteJudgeStore(connection).register_judge(
                self._revision_id, judge_model, source=source, config=config,
            )
        self._connection_operation(operation)
        self._judge_records[judge_model] = (source, config)

    def register_contract(self, contract_id: str, *, plugin_id: str,
                          plugin_version: str, prompt_version: str,
                          instructions_version: str) -> None:
        """Register and activate one judge contract for a plugin version."""
        if self._revision_id is None:
            raise RuntimeError("SQLite run must be started before registering contracts")
        contract_hash = hashlib.sha256(contract_id.encode("utf-8")).hexdigest()
        def operation(connection):
            judges = SQLiteJudgeStore(connection)
            judges.register_contract(
                contract_id, plugin_id=plugin_id, plugin_version=plugin_version,
                prompt_version=prompt_version,
                instructions_version=instructions_version,
                response_schema_hash=contract_hash,
                contract={"contract_id": contract_id},
                contract_hash=contract_hash,
            )
            judges.activate_contract(self._revision_id, plugin_id, contract_id)
        self._connection_operation(operation)
        self._contract_records[plugin_id] = (
            contract_id, plugin_version, prompt_version, instructions_version,
        )

    def _submit_async(self, operation: Any) -> Future[Any]:
        """Queue an operation without blocking the caller.

        The writer already reports commit failures through ``failure_callback``
        and ``writer.failures``; retrieving the future's exception here merely
        prevents an un-retrieved-exception warning when the future is collected.
        """
        future = self.writer.submit(operation)

        def _consume(done: Future[Any]) -> None:
            try:
                done.exception()
            except Exception:  # noqa: BLE001, S110 - already reported by the writer
                pass

        future.add_done_callback(_consume)
        return future

    def submit(self, operation: Any) -> Future[Any]:
        """Submit a normalized SQLite operation to the background writer."""
        return self.writer.submit(operation)

    def record_result(self, result: dict[str, Any]) -> None:
        key = (result.get("state_key", result.get("model")), result.get("runner", "http"))
        self._results[key] = dict(result)

    def update_model(self, model_name: str, **fields: Any) -> None:
        del model_name, fields
        # Live model state remains owned by BenchmarkState. Normalized model
        # projections are written by the benchmark store in the next runtime
        # integration layer; this no-op keeps the façade safe during migration.

    def record_judge_result(self, state_key: str, runner: str, plugin_id: str,
                            **fields: Any) -> None:
        key = (state_key, runner)
        row = self._results.setdefault(key, {
            "model": state_key, "state_key": state_key, "runner": runner,
        })
        row.update({f"{plugin_id}_{name}": value for name, value in fields.items()})

    def save_snapshot(self, path: str | None = None,
                      plugin_versions: dict[str, str] | None = None,
                      *, raise_on_error: bool = False) -> bool:
        del path, plugin_versions, raise_on_error
        try:
            self.writer.flush(timeout=10)
        except (OSError, RuntimeError, TimeoutError):
            return False
        return not self.writer.failures

    def latest_results(self) -> list[dict[str, Any]]:
        """Return the authoritative read model for the current revision.

        Reads the normalized SQLite tables rather than the in-memory cache so
        resume and report generation observe durable state. The writer is
        flushed first so queued attempt/vote writes are visible.
        """
        self.flush(timeout=30)
        if self._revision_id is None:
            return list(self._results.values())
        if self.identity is None:
            raise RuntimeError("SQLite run identity is not initialized")
        from .sqlite_reports import SQLiteReportSource
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            from .sqlite_schema import configure_connection
            configure_connection(connection)
            source = SQLiteReportSource(connection)
            rows, _active_plugins, _seed, _revision = source.load_results(
                revision=self._revision_id, run_id=self.identity.run_id,
                include_reused=True,
            )
            return rows
        finally:
            connection.close()

    def flush(self, timeout: float | None = None) -> None:
        self.writer.flush(timeout=timeout)

    def close(self, timeout: float | None = None) -> bool:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        return self.writer.close(timeout=timeout)


class JsonReportSource:
    """Read the legacy JSON state format for report-only generation."""

    def load_results(self, path: str) -> tuple[list[dict[str, Any]], list[str], int | None]:
        state_path = path
        if os.path.isdir(path):
            state_path = os.path.join(path, "benchmark_state.json")
        if state_path.endswith((".sqlite3", ".db")):
            raise RuntimeError("Use SQLiteReportSource for SQLite report loading")
        with open(state_path, encoding="utf-8") as handle:
            data = json.load(handle)
        results = data.get("results")
        if not isinstance(results, list):
            raise TypeError(f"{state_path} does not contain a results list")
        active_plugins = data.get("active_plugins") or []
        if not isinstance(active_plugins, list) or not all(
            isinstance(plugin_id, str) for plugin_id in active_plugins
        ):
            raise TypeError(f"{state_path} contains invalid active_plugins metadata")
        return results, active_plugins, data.get("session_seed")


class JsonPayloadStore:
    """Compatibility placeholder for JSON's embedded payload representation."""

    def put(self, kind: str, data: bytes) -> int:
        del kind, data
        raise NotImplementedError("JSON storage has no separate payload table")

    def get(self, payload_id: int) -> bytes:
        del payload_id
        raise NotImplementedError("JSON storage has no separate payload table")


class JsonDebugLogStore:
    """Compatibility adapter for existing plaintext JSON-run diagnostics."""

    def append(self, path: str, data: str | bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = data.encode("utf-8") if isinstance(data, str) else data
        with open(path, "ab") as handle:
            handle.write(payload)

    def close(self) -> None:
        return None


def project_result_rows(
    results: Iterable[dict[str, Any]],
    plugin_ids: Iterable[str],
    model_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project the latest rows without losing usable per-plugin values.

    A shutdown/cancellation row can be the newest row for a model while
    containing ``fail`` for plugins that already completed. Preserve the most
    recent usable value for each plugin independently, then use ``model_info``
    as a fallback for values that were published there but never reached a
    result row.
    """
    plugin_ids = list(plugin_ids)
    latest: dict[tuple[Any, Any], dict[str, Any]] = {}
    history: dict[tuple[Any, Any], dict[str, Any]] = {}
    for result in results:
        identity = (
            result.get("state_key", result.get("model")),
            result.get("runner", "http"),
        )
        latest[identity] = dict(result)
        per_plugin = history.setdefault(identity, {})
        for plugin_id in plugin_ids:
            prefix = f"{plugin_id}_"
            for key, value in result.items():
                if key.startswith(prefix) and value not in (None, "fail"):
                    per_plugin[key] = value

    projected: list[dict[str, Any]] = []
    for identity, row in latest.items():
        state_key, runner = identity
        for key, value in history.get(identity, {}).items():
            if row.get(key) in (None, "fail"):
                row[key] = value
        info = (model_info or {}).get(state_key, {})
        if isinstance(info, dict):
            for plugin_id in plugin_ids:
                prefix = f"{plugin_id}_"
                for key, value in info.items():
                    if key.startswith(prefix) and value not in (None, "fail"):
                        if row.get(key) in (None, "fail"):
                            row[key] = value
        projected.append(row)
    return projected


def latest_result_rows(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the last row for each state-key/runner identity."""
    return project_result_rows(results, ())
