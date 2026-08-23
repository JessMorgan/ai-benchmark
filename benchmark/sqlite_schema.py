"""SQLite schema and migration primitives for run storage.

This stage creates the normalized, revision-aware schema. Runtime persistence
and payload/debug-log writers are implemented in later stages; keeping schema
creation here makes migrations independently testable and keeps the database
contract stable as those writers are added.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

SQLITE_SCHEMA_VERSION = 3
_DEFAULT_BUSY_TIMEOUT_MS = 5_000


def connect_database(
    path: str,
    *,
    timeout: float = 5.0,
    synchronous: str = "NORMAL",
) -> sqlite3.Connection:
    """Open a configured SQLite connection and initialize its schema."""
    connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    configure_connection(connection, synchronous=synchronous)
    initialize_schema(connection)
    return connection


def configure_connection(connection: sqlite3.Connection, *, synchronous: str = "NORMAL") -> None:
    """Apply SQLite settings required for concurrent run persistence."""
    synchronous = synchronous.upper()
    if synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
        raise ValueError("synchronous must be OFF, NORMAL, FULL, or EXTRA")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {_DEFAULT_BUSY_TIMEOUT_MS}")
    mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    database_file = connection.execute("PRAGMA database_list").fetchone()[2]
    if database_file not in ("", ":memory:") and str(mode).lower() != "wal":
        raise RuntimeError(f"SQLite WAL could not be enabled (got {mode!r})")
    connection.execute(f"PRAGMA synchronous = {synchronous}")


def _create_schema_v1(connection: sqlite3.Connection) -> None:
    """Create the complete revision-aware schema for version 1."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS payloads (
            payload_id          INTEGER PRIMARY KEY,
            sha256              TEXT NOT NULL UNIQUE,
            kind                TEXT NOT NULL,
            compression         TEXT NOT NULL DEFAULT 'gzip',
            uncompressed_bytes  INTEGER NOT NULL CHECK (uncompressed_bytes >= 0),
            stored_bytes        INTEGER NOT NULL CHECK (stored_bytes >= 0),
            data                BLOB NOT NULL,
            created_at          INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id             TEXT PRIMARY KEY,
            created_at         INTEGER NOT NULL,
            status             TEXT NOT NULL,
            score_schema       TEXT NOT NULL,
            storage_profile    TEXT NOT NULL,
            current_revision_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS run_revisions (
            revision_id       INTEGER PRIMARY KEY,
            run_id            TEXT NOT NULL REFERENCES runs(run_id),
            revision_number   INTEGER NOT NULL,
            status            TEXT NOT NULL,
            started_at        INTEGER,
            ended_at          INTEGER,
            runner_mode       TEXT NOT NULL,
            session_seed      INTEGER,
            config_json       TEXT NOT NULL,
            config_sha256     TEXT NOT NULL,
            created_at        INTEGER NOT NULL,
            UNIQUE(run_id, revision_number)
        );

        CREATE TABLE IF NOT EXISTS target_instances (
            target_instance_id  INTEGER PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES runs(run_id),
            logical_name        TEXT NOT NULL,
            runner              TEXT NOT NULL,
            source              TEXT NOT NULL,
            api_model           TEXT NOT NULL,
            is_agent            INTEGER NOT NULL DEFAULT 0 CHECK (is_agent IN (0, 1)),
            system_prompt       TEXT,
            target_config_json  TEXT,
            target_signature    TEXT NOT NULL,
            first_revision_id   INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            retired_revision_id INTEGER REFERENCES run_revisions(revision_id),
            UNIQUE(run_id, logical_name, runner, target_signature),
            UNIQUE(run_id, target_instance_id)
        );

        CREATE TABLE IF NOT EXISTS revision_targets (
            revision_id        INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            target_instance_id INTEGER NOT NULL REFERENCES target_instances(target_instance_id),
            active              INTEGER NOT NULL CHECK (active IN (0, 1)),
            order_index         INTEGER,
            PRIMARY KEY(revision_id, target_instance_id)
        );

        CREATE TABLE IF NOT EXISTS plugin_definitions (
            plugin_id          TEXT NOT NULL,
            plugin_version      TEXT NOT NULL,
            name                TEXT NOT NULL,
            max_score           REAL NOT NULL CHECK (max_score > 0),
            supports_streaming INTEGER NOT NULL CHECK (supports_streaming IN (0, 1)),
            metadata_json       TEXT,
            PRIMARY KEY(plugin_id, plugin_version)
        );

        CREATE TABLE IF NOT EXISTS revision_plugins (
            revision_id   INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            plugin_id     TEXT NOT NULL,
            plugin_version TEXT NOT NULL,
            active        INTEGER NOT NULL CHECK (active IN (0, 1)),
            PRIMARY KEY(revision_id, plugin_id),
            FOREIGN KEY(plugin_id, plugin_version)
                REFERENCES plugin_definitions(plugin_id, plugin_version)
        );

        CREATE TABLE IF NOT EXISTS cells (
            cell_id             INTEGER PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES runs(run_id),
            target_instance_id  INTEGER NOT NULL,
            plugin_id           TEXT NOT NULL,
            plugin_version      TEXT NOT NULL,
            created_at          INTEGER NOT NULL,
            UNIQUE(target_instance_id, plugin_id, plugin_version),
            FOREIGN KEY(run_id, target_instance_id)
                REFERENCES target_instances(run_id, target_instance_id),
            FOREIGN KEY(plugin_id, plugin_version)
                REFERENCES plugin_definitions(plugin_id, plugin_version)
        );

        CREATE TABLE IF NOT EXISTS revision_cells (
            revision_id INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            cell_id     INTEGER NOT NULL REFERENCES cells(cell_id),
            scheduled   INTEGER NOT NULL CHECK (scheduled IN (0, 1)),
            status      TEXT NOT NULL,
            queue_reason TEXT,
            updated_at  INTEGER NOT NULL,
            PRIMARY KEY(revision_id, cell_id)
        );

        CREATE TABLE IF NOT EXISTS benchmark_attempts (
            attempt_id             INTEGER PRIMARY KEY,
            revision_id            INTEGER NOT NULL,
            cell_id                INTEGER NOT NULL,
            attempt_number         INTEGER NOT NULL CHECK (attempt_number > 0),
            prompt_payload_id      INTEGER REFERENCES payloads(payload_id),
            content_payload_id     INTEGER REFERENCES payloads(payload_id),
            thinking_payload_id    INTEGER REFERENCES payloads(payload_id),
            started_at             INTEGER,
            ended_at               INTEGER,
            response_time          REAL,
            gen_time               REAL,
            max_tokens             INTEGER CHECK (max_tokens IS NULL OR max_tokens > 0),
            output_tokens          INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
            thinking_tokens        INTEGER CHECK (thinking_tokens IS NULL OR thinking_tokens >= 0),
            total_tokens           INTEGER CHECK (total_tokens IS NULL OR total_tokens >= 0),
            tps                    REAL,
            finish_reason          TEXT,
            response_nature        TEXT,
            retry_reason           TEXT,
            prompt_altered        TEXT,
            truncated              INTEGER CHECK (truncated IS NULL OR truncated IN (0, 1)),
            truncated_due_to_time INTEGER CHECK (truncated_due_to_time IS NULL OR truncated_due_to_time IN (0, 1)),
            failure_cause          TEXT,
            stream_ok              INTEGER CHECK (stream_ok IS NULL OR stream_ok IN (0, 1)),
            repeating              INTEGER CHECK (repeating IS NULL OR repeating IN (0, 1)),
            empty_reason           TEXT,
            error                  TEXT,
            score                  REAL,
            rubric_json            TEXT,
            diagnostics_json       TEXT,
            status                 TEXT NOT NULL DEFAULT 'completed',
            UNIQUE(revision_id, cell_id, attempt_number),
            UNIQUE(attempt_id, revision_id, cell_id),
            FOREIGN KEY(revision_id, cell_id)
                REFERENCES revision_cells(revision_id, cell_id)
        );

        CREATE TABLE IF NOT EXISTS benchmark_selections (
            revision_id      INTEGER NOT NULL,
            cell_id          INTEGER NOT NULL,
            attempt_id       INTEGER NOT NULL,
            selected_at      INTEGER NOT NULL,
            selection_reason TEXT,
            PRIMARY KEY(revision_id, cell_id),
            FOREIGN KEY(attempt_id, revision_id, cell_id)
                REFERENCES benchmark_attempts(attempt_id, revision_id, cell_id)
        );

        CREATE TABLE IF NOT EXISTS revision_judges (
            revision_id INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            judge_model TEXT NOT NULL,
            source      TEXT NOT NULL,
            config_json TEXT,
            active      INTEGER NOT NULL CHECK (active IN (0, 1)),
            PRIMARY KEY(revision_id, judge_model)
        );

        CREATE TABLE IF NOT EXISTS judge_contracts (
            contract_id          TEXT PRIMARY KEY,
            plugin_id            TEXT NOT NULL,
            plugin_version       TEXT NOT NULL,
            prompt_version       TEXT NOT NULL,
            instructions_version TEXT NOT NULL,
            response_schema_hash TEXT NOT NULL,
            contract_json        TEXT NOT NULL,
            contract_hash        TEXT NOT NULL UNIQUE,
            FOREIGN KEY(plugin_id, plugin_version)
                REFERENCES plugin_definitions(plugin_id, plugin_version)
        );

        CREATE TABLE IF NOT EXISTS revision_judge_contracts (
            revision_id INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            plugin_id   TEXT NOT NULL,
            contract_id TEXT NOT NULL REFERENCES judge_contracts(contract_id),
            active      INTEGER NOT NULL CHECK (active IN (0, 1)),
            PRIMARY KEY(revision_id, plugin_id)
        );

        CREATE TABLE IF NOT EXISTS judge_attempts (
            judge_attempt_id        INTEGER PRIMARY KEY,
            revision_id             INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            cell_id                 INTEGER NOT NULL REFERENCES cells(cell_id),
            judge_model             TEXT NOT NULL,
            contract_id             TEXT NOT NULL REFERENCES judge_contracts(contract_id),
            attempt_number          INTEGER NOT NULL CHECK (attempt_number > 0),
            started_at              INTEGER,
            ended_at                INTEGER,
            max_tokens              INTEGER CHECK (max_tokens IS NULL OR max_tokens > 0),
            raw_response_payload_id INTEGER REFERENCES payloads(payload_id),
            request_payload_id     INTEGER REFERENCES payloads(payload_id),
            response_usage_json     TEXT,
            diagnostics_json        TEXT,
            finish_reason           TEXT,
            error                   TEXT,
            status                  TEXT NOT NULL DEFAULT 'completed',
            UNIQUE(revision_id, cell_id, judge_model, contract_id, attempt_number),
            UNIQUE(judge_attempt_id, revision_id, cell_id, judge_model, contract_id)
        );

        CREATE TABLE IF NOT EXISTS judge_vote_attempts (
            vote_attempt_id INTEGER PRIMARY KEY,
            judge_attempt_id INTEGER NOT NULL UNIQUE REFERENCES judge_attempts(judge_attempt_id),
            score           REAL,
            confidence      TEXT,
            rationale       TEXT,
            error           TEXT,
            usable          INTEGER NOT NULL CHECK (usable IN (0, 1)),
            created_at      INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS judge_criteria (
            criterion_id  INTEGER PRIMARY KEY,
            vote_attempt_id INTEGER NOT NULL REFERENCES judge_vote_attempts(vote_attempt_id),
            ordinal       INTEGER NOT NULL CHECK (ordinal >= 0),
            criterion_key TEXT NOT NULL,
            criterion     TEXT NOT NULL,
            status        TEXT NOT NULL,
            evidence      TEXT NOT NULL,
            UNIQUE(vote_attempt_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS current_judge_votes (
            revision_id      INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            cell_id          INTEGER NOT NULL REFERENCES cells(cell_id),
            judge_model      TEXT NOT NULL,
            contract_id      TEXT NOT NULL REFERENCES judge_contracts(contract_id),
            vote_attempt_id  INTEGER NOT NULL REFERENCES judge_vote_attempts(vote_attempt_id),
            selected_at      INTEGER NOT NULL,
            selection_reason TEXT,
            PRIMARY KEY(revision_id, cell_id, judge_model, contract_id)
        );

        CREATE TABLE IF NOT EXISTS consensus_cache (
            revision_id   INTEGER NOT NULL REFERENCES run_revisions(revision_id),
            cell_id       INTEGER NOT NULL REFERENCES cells(cell_id),
            contract_id   TEXT NOT NULL REFERENCES judge_contracts(contract_id),
            score         REAL,
            confidence    TEXT,
            valid_judges  INTEGER NOT NULL CHECK (valid_judges >= 0),
            attempts      INTEGER NOT NULL CHECK (attempts >= 0),
            vote_set_hash TEXT NOT NULL,
            calculated_at INTEGER NOT NULL,
            PRIMARY KEY(revision_id, cell_id, contract_id)
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            event_id         INTEGER PRIMARY KEY,
            run_id           TEXT NOT NULL REFERENCES runs(run_id),
            revision_id      INTEGER REFERENCES run_revisions(revision_id),
            event_type       TEXT NOT NULL,
            cell_id          INTEGER REFERENCES cells(cell_id),
            attempt_id       INTEGER REFERENCES benchmark_attempts(attempt_id),
            judge_attempt_id INTEGER REFERENCES judge_attempts(judge_attempt_id),
            vote_attempt_id  INTEGER REFERENCES judge_vote_attempts(vote_attempt_id),
            created_at       INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS debug_log_files (
            log_id             INTEGER PRIMARY KEY,
            run_id             TEXT NOT NULL REFERENCES runs(run_id),
            revision_id        INTEGER REFERENCES run_revisions(revision_id),
            path               TEXT NOT NULL,
            compression        TEXT NOT NULL,
            complete_members   INTEGER NOT NULL DEFAULT 0,
            uncompressed_bytes INTEGER NOT NULL DEFAULT 0,
            stored_bytes       INTEGER NOT NULL DEFAULT 0,
            truncated_tail     INTEGER NOT NULL DEFAULT 0,
            created_at         INTEGER NOT NULL,
            updated_at         INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS legacy_import_records (
            legacy_record_id  INTEGER PRIMARY KEY,
            run_id            TEXT NOT NULL REFERENCES runs(run_id),
            source_file       TEXT NOT NULL,
            source_sha256     TEXT NOT NULL,
            source_row_number INTEGER,
            record_kind       TEXT NOT NULL,
            raw_json          TEXT NOT NULL,
            mapping_status    TEXT NOT NULL,
            mapping_note      TEXT,
            UNIQUE(source_sha256, source_row_number, record_kind)
        );

        CREATE INDEX IF NOT EXISTS idx_revision_cells_status
            ON revision_cells(revision_id, status, scheduled);
        CREATE INDEX IF NOT EXISTS idx_attempts_cell_revision
            ON benchmark_attempts(revision_id, cell_id, attempt_number);
        CREATE INDEX IF NOT EXISTS idx_judge_attempts_cell
            ON judge_attempts(revision_id, cell_id, judge_model, contract_id);
        CREATE INDEX IF NOT EXISTS idx_vote_attempts_judge
            ON judge_vote_attempts(judge_attempt_id);
        CREATE INDEX IF NOT EXISTS idx_criteria_vote
            ON judge_criteria(vote_attempt_id);
        CREATE INDEX IF NOT EXISTS idx_payload_kind
            ON payloads(kind);
        CREATE INDEX IF NOT EXISTS idx_legacy_import_source
            ON legacy_import_records(source_sha256, source_row_number);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_contract_per_plugin
            ON revision_judge_contracts(revision_id, plugin_id)
            WHERE active = 1;

        CREATE TRIGGER IF NOT EXISTS validate_current_revision_insert
        BEFORE INSERT ON runs
        WHEN NEW.current_revision_id IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM run_revisions
                 WHERE revision_id = NEW.current_revision_id
                   AND run_id = NEW.run_id
             )
        BEGIN
            SELECT RAISE(ABORT, 'current revision does not belong to run');
        END;

        CREATE TRIGGER IF NOT EXISTS validate_current_revision_update
        BEFORE UPDATE OF current_revision_id ON runs
        WHEN NEW.current_revision_id IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM run_revisions
                 WHERE revision_id = NEW.current_revision_id
                   AND run_id = NEW.run_id
             )
        BEGIN
            SELECT RAISE(ABORT, 'current revision does not belong to run');
        END;

        CREATE TRIGGER IF NOT EXISTS validate_current_judge_vote
        BEFORE INSERT ON current_judge_votes
        WHEN NOT EXISTS (
            SELECT 1
            FROM judge_vote_attempts v
            JOIN judge_attempts a ON a.judge_attempt_id = v.judge_attempt_id
            WHERE v.vote_attempt_id = NEW.vote_attempt_id
              AND a.revision_id = NEW.revision_id
              AND a.cell_id = NEW.cell_id
              AND a.judge_model = NEW.judge_model
              AND a.contract_id = NEW.contract_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'current judge vote identity mismatch');
        END;

        CREATE TRIGGER IF NOT EXISTS validate_current_judge_vote_update
        BEFORE UPDATE OF revision_id, cell_id, judge_model, contract_id, vote_attempt_id
            ON current_judge_votes
        WHEN NOT EXISTS (
            SELECT 1
            FROM judge_vote_attempts v
            JOIN judge_attempts a ON a.judge_attempt_id = v.judge_attempt_id
            WHERE v.vote_attempt_id = NEW.vote_attempt_id
              AND a.revision_id = NEW.revision_id
              AND a.cell_id = NEW.cell_id
              AND a.judge_model = NEW.judge_model
              AND a.contract_id = NEW.contract_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'current judge vote identity mismatch');
        END;
        """
    )


def _set_schema_version(connection: sqlite3.Connection, version: int) -> None:
    connection.execute(
        "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(version),),
    )


def _read_schema_version(connection: sqlite3.Connection) -> int | None:
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return None
    try:
        version = int(row[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SQLite schema_version is not an integer") from exc
    if version < 0:
        raise RuntimeError("SQLite schema_version cannot be negative")
    return version


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    """Return columns for a table while keeping migrations introspectable."""
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _migrate_v2_continuation_reuse(connection: sqlite3.Connection) -> None:
    """Allow current projections to reuse compatible prior-revision history.

    Stage 1 of the schema used same-revision composite foreign keys for
    selections and triggers. Continuation revisions deliberately keep the
    immutable attempt in its original revision while selecting it from the
    new revision, so replace those constraints with cell-identity validation.
    Also add lifecycle status to legacy databases so stopping can mark
    genuinely in-flight rows as abandoned without rewriting completed data.
    """
    if "status" not in _column_names(connection, "benchmark_attempts"):
        connection.execute(
            "ALTER TABLE benchmark_attempts ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'"
        )
    if "status" not in _column_names(connection, "judge_attempts"):
        connection.execute(
            "ALTER TABLE judge_attempts ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'"
        )

    connection.execute("DROP TRIGGER IF EXISTS validate_current_judge_vote")
    connection.execute("DROP TRIGGER IF EXISTS validate_current_judge_vote_update")
    connection.execute("DROP TRIGGER IF EXISTS validate_benchmark_selection")
    connection.execute("DROP TRIGGER IF EXISTS validate_benchmark_selection_update")

    connection.execute("ALTER TABLE benchmark_selections RENAME TO benchmark_selections_v1")
    connection.execute(
        """
        CREATE TABLE benchmark_selections (
            revision_id      INTEGER NOT NULL,
            cell_id          INTEGER NOT NULL,
            attempt_id       INTEGER NOT NULL,
            selected_at      INTEGER NOT NULL,
            selection_reason TEXT,
            PRIMARY KEY(revision_id, cell_id),
            FOREIGN KEY(revision_id, cell_id)
                REFERENCES revision_cells(revision_id, cell_id),
            FOREIGN KEY(attempt_id) REFERENCES benchmark_attempts(attempt_id)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO benchmark_selections(
            revision_id, cell_id, attempt_id, selected_at, selection_reason
        )
        SELECT revision_id, cell_id, attempt_id, selected_at, selection_reason
        FROM benchmark_selections_v1
        """
    )
    connection.execute("DROP TABLE benchmark_selections_v1")

    connection.executescript(
        """
        CREATE TRIGGER validate_benchmark_selection
        BEFORE INSERT ON benchmark_selections
        WHEN NOT EXISTS (
            SELECT 1 FROM benchmark_attempts a
            WHERE a.attempt_id = NEW.attempt_id
              AND a.cell_id = NEW.cell_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'benchmark selection identity mismatch');
        END;

        CREATE TRIGGER validate_benchmark_selection_update
        BEFORE UPDATE OF revision_id, cell_id, attempt_id ON benchmark_selections
        WHEN NOT EXISTS (
            SELECT 1 FROM benchmark_attempts a
            WHERE a.attempt_id = NEW.attempt_id
              AND a.cell_id = NEW.cell_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'benchmark selection identity mismatch');
        END;

        CREATE TRIGGER validate_current_judge_vote
        BEFORE INSERT ON current_judge_votes
        WHEN NOT EXISTS (
            SELECT 1
            FROM judge_vote_attempts v
            JOIN judge_attempts a ON a.judge_attempt_id = v.judge_attempt_id
            WHERE v.vote_attempt_id = NEW.vote_attempt_id
              AND a.cell_id = NEW.cell_id
              AND a.judge_model = NEW.judge_model
              AND a.contract_id = NEW.contract_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'current judge vote identity mismatch');
        END;

        CREATE TRIGGER validate_current_judge_vote_update
        BEFORE UPDATE OF revision_id, cell_id, judge_model, contract_id, vote_attempt_id
            ON current_judge_votes
        WHEN NOT EXISTS (
            SELECT 1
            FROM judge_vote_attempts v
            JOIN judge_attempts a ON a.judge_attempt_id = v.judge_attempt_id
            WHERE v.vote_attempt_id = NEW.vote_attempt_id
              AND a.cell_id = NEW.cell_id
              AND a.judge_model = NEW.judge_model
              AND a.contract_id = NEW.contract_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'current judge vote identity mismatch');
        END;
        """
    )


def _migrate_v3_attempt_timing(connection: sqlite3.Connection) -> None:
    """Persist per-attempt response/generation timing on benchmark attempts.

    The runtime measures ``response_time`` (TTFT for streaming, full request
    time for non-streaming) and ``gen_time`` per transport attempt, and the
    legacy JSON read model surfaces ``{plugin}_response_time`` in results.
    Earlier schema revisions had ``started_at``/``ended_at`` epoch columns but
    no runtime path populated them, so timing was silently dropped on the
    SQLite path. Add explicit REAL duration columns so continuations and
    reports can reproduce the timing each attempt took.
    """
    for column, definition in (
        ("response_time", "REAL"),
        ("gen_time", "REAL"),
    ):
        if column not in _column_names(connection, "benchmark_attempts"):
            connection.execute(
                f"ALTER TABLE benchmark_attempts ADD COLUMN {column} {definition}"
            )


Migration = Callable[[sqlite3.Connection], None]
MIGRATIONS: dict[int, Migration] = {
    1: _create_schema_v1,
    2: _migrate_v2_continuation_reuse,
    3: _migrate_v3_attempt_timing,
}


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create or forward-migrate a database to the current schema version."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    version = _read_schema_version(connection)
    if version is None:
        version = 0
    if version > SQLITE_SCHEMA_VERSION:
        raise RuntimeError(
            f"SQLite schema version {version} is newer than supported version "
            f"{SQLITE_SCHEMA_VERSION}"
        )
    while version < SQLITE_SCHEMA_VERSION:
        next_version = version + 1
        migration = MIGRATIONS.get(next_version)
        if migration is None:
            raise RuntimeError(
                f"No migration is registered for SQLite schema version {next_version}"
            )
        try:
            if connection.in_transaction:
                connection.commit()
            connection.execute("BEGIN")
            migration(connection)
            _set_schema_version(connection, next_version)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        version = next_version


def schema_version(connection: sqlite3.Connection) -> int:
    """Return the initialized schema version."""
    version = _read_schema_version(connection)
    if version is None:
        raise RuntimeError("SQLite schema has not been initialized")
    return version


def table_names(connection: sqlite3.Connection) -> set[str]:
    """Return user table names, useful for migration diagnostics and tests."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    return {row[0] for row in rows}


def foreign_keys_enabled(connection: sqlite3.Connection) -> bool:
    """Return whether SQLite foreign-key enforcement is active."""
    return bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])


def sqlite_options(connection: sqlite3.Connection) -> dict[str, Any]:
    """Return relevant connection options for run metadata and diagnostics."""
    return {
        "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
        "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
        "foreign_keys": foreign_keys_enabled(connection),
        "busy_timeout_ms": connection.execute("PRAGMA busy_timeout").fetchone()[0],
    }
