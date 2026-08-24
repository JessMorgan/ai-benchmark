# SQLite Run Storage Plan

**Status:** Runtime implementation complete; empirical rollout validation remains open
**Last updated:** 2026-08-23
**Primary goals:** fast writes, small output directories, reduced duplication, reliable resume, and reproducible reports.

## 1. Goals and decisions

The current file layout is human-readable and resilient, but duplicates prompts,
responses, judge metadata, consensus projections, report data, and journal
payloads. The target design is a SQLite-first run store with canonical payload
references and derived reports.

### Decisions

- SQLite is the authoritative store for new runs.
- A logical run contains immutable continuation revisions. Stopping and
  continuing a run never mutates historical revisions.
- Benchmark and judge attempts are immutable records. Current selections are
  revision-scoped pointers, not copied result objects.
- Prompts, candidate responses, thinking text, and raw judge responses use one
  representation: a payload ID referencing one compressed SQLite BLOB. There is
  no parallel inline-text/hash representation for the same field.
- Short, frequently queried judge text (`rationale`, `criterion`, and `evidence`)
  uses one representation: SQLite `TEXT` columns. It is not optionally split
  into inline text and payload hashes.
- Full judge request/response transcripts are debug-only. They are not written
  during ordinary compact runs.
- Debug transcripts use gzip concatenated members and append without
  decompressing existing data. The detailed design and recovery tests are in
  [the gzip append-log plan](gzip-append-log-plan.md).
- Reports are generated only when requested.
- JSON remains available as an import/export and compatibility backend during
  migration.
- SQLite WAL and one background writer replace the separate full-state JSONL
  journal for normal SQLite runs. An audit-event table is optional and never
  contains full payloads.

## 2. Storage profiles

### `compact` (default)

- SQLite metadata and result tables.
- Deduplicated compressed payload BLOBs.
- No full successful judge request/response transcript.
- Failure metadata and compact transport diagnostics retained.
- No purge backups in the normal run directory.
- Reports generated only when `--output-format` is supplied.

### `debug`

Includes everything in `compact`, plus compressed full judge transcripts,
HTTP diagnostics, and OpenCode/Pi stdout/stderr artifacts. Debug log behavior
is specified in [gzip-append-log-plan.md](gzip-append-log-plan.md).

### `portable`

- SQLite database.
- Selected reports.
- Optionally materialized prompt/response files.
- Redacted configuration and diagnostics by default.
- No credentials or authorization headers in exported artifacts.

## 3. Proposed output layout

```text
<output_dir>/
├── run.sqlite3                 # authoritative run store
├── run.sqlite3-wal             # temporary while the run is active
├── run.sqlite3-shm             # temporary WAL support file
├── run-info.json               # small compatibility/run manifest
├── results.csv                 # optional generated export
├── results.md                  # optional generated export
├── results.html                # optional generated export
├── results.pdf                 # optional generated export
└── logs/                       # only in debug mode or for failures
    ├── judge-<model>.log.gz
    ├── <target>/<plugin>.log.gz
    └── <runner>/<target>/<plugin>.stdout.gz
```

After a clean shutdown, SQLite checkpoints its WAL. The database is the main
portable artifact; reports and debug logs are derived or optional artifacts.

## 4. On-demand report generation

Reports should not be generated automatically for every run.

For a benchmark run:

```sh
python ai-benchmark.py \
  --config benchmark-config.yml \
  --output-format csv html
```

`--output-format` accepts one or more values:

```text
csv | md | html | pdf
```

If omitted, the run stores results but does not generate reports.

For an existing run:

```sh
python ai-benchmark.py \
  --generate-reports runs/2026-08-17-nas-and-more-test-changes \
  --output-format csv html
```

`--generate-reports PATH` is report-only mode. It loads SQLite or a legacy JSON
run and schedules no model, benchmark, or judge work.

CLI rules:

- `--output-format` requires a benchmark run or `--generate-reports PATH`.
- Duplicate formats are removed while preserving request order.
- Unknown formats fail before model loading.
- Existing reports are replaced atomically.
- Report failures identify the format and do not corrupt the database.
- `--save-responses` controls materialized response files; it does not enable
  reports.

The first SQLite implementation should expose the existing result-dictionary
read model to output plugins. Reports remain derived artifacts rather than a
second authoritative result store.

## 5. Representation and deduplication policy

There is deliberately one representation per logical field.

| Data | Representation |
|---|---|
| Task prompt | `payload_id` → compressed BLOB in `payloads` |
| Candidate content | `payload_id` → compressed BLOB in `payloads` |
| Candidate thinking | `payload_id` → compressed BLOB in `payloads` |
| Raw judge response | `payload_id` → compressed BLOB in `payloads` |
| Judge rationale | SQLite `TEXT` |
| Criterion description | SQLite `TEXT` |
| Criterion evidence | SQLite `TEXT` |
| Stable metadata | Typed SQLite columns |
| Provider-specific metadata | JSON stored in one `TEXT` column |
| Full debug transcript | Append-only `.gz` file, never a second payload representation |

The `payloads` table stores compressed BLOBs only. It does not have an
alternative `external_path` representation. Debug logs are the exception:
they are append-only diagnostic files governed by the separate gzip plan.

Payload identity is the SHA-256 of the uncompressed bytes. Identical prompts,
responses, or judge outputs receive one payload row.

## 6. SQLite schema

### Schema metadata and payloads

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE payloads (
    payload_id          INTEGER PRIMARY KEY,
    sha256              TEXT NOT NULL UNIQUE,
    kind                TEXT NOT NULL,
    compression         TEXT NOT NULL DEFAULT 'gzip',
    uncompressed_bytes  INTEGER NOT NULL,
    stored_bytes        INTEGER NOT NULL,
    data                BLOB NOT NULL,
    created_at          INTEGER NOT NULL
);
```

All payload-bearing fields use an integer `payload_id`; they never switch
between a text column and a payload reference. Short queryable text remains
plain `TEXT` instead of requiring a payload join.

### Logical runs and continuation revisions

A database may contain one logical run with many continuation revisions. A
revision is one invocation of the benchmark using one configuration snapshot.
The `runs.current_revision_id` foreign key is intentionally cyclic with
`run_revisions.run_id`; schema creation must either create the tables in a
foreign-key-compatible order or add the current-revision foreign key in a
migration after both tables exist.

Plugin definitions, payloads, and judge contract definitions are immutable
within the database and may be reused by multiple logical runs. Run-specific
activation belongs in the revision membership tables.

```sql
CREATE TABLE runs (
    run_id             TEXT PRIMARY KEY,
    created_at         INTEGER NOT NULL,
    status              TEXT NOT NULL,
    score_schema        TEXT NOT NULL,
    storage_profile     TEXT NOT NULL,
    current_revision_id INTEGER REFERENCES run_revisions(revision_id)
);

CREATE TABLE run_revisions (
    revision_id        INTEGER PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs(run_id),
    revision_number    INTEGER NOT NULL,
    status              TEXT NOT NULL,
    started_at          INTEGER,
    ended_at            INTEGER,
    runner_mode        TEXT NOT NULL,
    session_seed       INTEGER,
    config_json        TEXT NOT NULL,
    config_sha256      TEXT NOT NULL,
    created_at         INTEGER NOT NULL,
    UNIQUE(run_id, revision_number)
);
```

`config_json` is a redacted configuration snapshot. It is one `TEXT` field,
not a choice between inline text and a hash. `config_sha256` identifies the
snapshot without replacing it.

### Target definitions and revision membership

A target instance is immutable. If a model's source, API model, runner, agent
prompt, or execution-affecting configuration changes, a new target instance is
created.

```sql
CREATE TABLE target_instances (
    target_instance_id INTEGER PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs(run_id),
    logical_name       TEXT NOT NULL,
    runner             TEXT NOT NULL,
    source             TEXT NOT NULL,
    api_model          TEXT NOT NULL,
    is_agent           INTEGER NOT NULL DEFAULT 0,
    system_prompt      TEXT,
    target_config_json TEXT,
    target_signature    TEXT NOT NULL,
    first_revision_id   INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    retired_revision_id INTEGER REFERENCES run_revisions(revision_id),
    UNIQUE(run_id, logical_name, runner, target_signature)
);

CREATE TABLE revision_targets (
    revision_id        INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    target_instance_id INTEGER NOT NULL REFERENCES target_instances(target_instance_id),
    active              INTEGER NOT NULL,
    order_index         INTEGER,
    PRIMARY KEY(revision_id, target_instance_id)
);
```

### Plugin definitions and revision membership

```sql
CREATE TABLE plugin_definitions (
    plugin_id          TEXT NOT NULL,
    plugin_version      TEXT NOT NULL,
    name                TEXT NOT NULL,
    max_score           INTEGER NOT NULL,
    supports_streaming INTEGER NOT NULL,
    metadata_json       TEXT,
    PRIMARY KEY(plugin_id, plugin_version)
);

CREATE TABLE revision_plugins (
    revision_id         INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    plugin_id           TEXT NOT NULL,
    plugin_version      TEXT NOT NULL,
    active               INTEGER NOT NULL,
    PRIMARY KEY(revision_id, plugin_id),
    FOREIGN KEY(plugin_id, plugin_version)
        REFERENCES plugin_definitions(plugin_id, plugin_version)
);
```

### Cells and revision-specific selection

A cell identifies a target instance and plugin version across the logical run.
It does not contain a mutable current score or current attempt pointer.

```sql
CREATE TABLE cells (
    cell_id             INTEGER PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    target_instance_id  INTEGER NOT NULL REFERENCES target_instances(target_instance_id),
    plugin_id           TEXT NOT NULL,
    plugin_version      TEXT NOT NULL,
    created_at          INTEGER NOT NULL,
    UNIQUE(target_instance_id, plugin_id, plugin_version),
    FOREIGN KEY(plugin_id, plugin_version)
        REFERENCES plugin_definitions(plugin_id, plugin_version)
);

CREATE TABLE revision_cells (
    revision_id         INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    cell_id             INTEGER NOT NULL REFERENCES cells(cell_id),
    scheduled            INTEGER NOT NULL,
    status               TEXT NOT NULL,
    queue_reason         TEXT,
    updated_at           INTEGER NOT NULL,
    PRIMARY KEY(revision_id, cell_id)
);

CREATE TABLE benchmark_selections (
    revision_id         INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    cell_id             INTEGER NOT NULL REFERENCES cells(cell_id),
    attempt_id          INTEGER NOT NULL,
    selected_at         INTEGER NOT NULL,
    selection_reason    TEXT,
    PRIMARY KEY(revision_id, cell_id),
    FOREIGN KEY(attempt_id, revision_id, cell_id)
        REFERENCES benchmark_attempts(attempt_id, revision_id, cell_id)
);
```

The selection table avoids mutable selection fields on cells and makes
historical report selection revision-specific. The implementation should create
`benchmark_attempts` before `benchmark_selections`, then use a composite foreign
key (or an equivalent validation trigger) to ensure the selected attempt belongs
to the same revision and cell. The DDL snippets are organized conceptually;
the migration must create tables in dependency order.

### Benchmark attempts

```sql
CREATE TABLE benchmark_attempts (
    attempt_id             INTEGER PRIMARY KEY,
    revision_id            INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    cell_id                INTEGER NOT NULL REFERENCES cells(cell_id),
    attempt_number         INTEGER NOT NULL,
    prompt_payload_id      INTEGER REFERENCES payloads(payload_id),
    content_payload_id     INTEGER REFERENCES payloads(payload_id),
    thinking_payload_id    INTEGER REFERENCES payloads(payload_id),
    started_at             INTEGER,
    ended_at               INTEGER,
    max_tokens             INTEGER,
    output_tokens          INTEGER,
    thinking_tokens        INTEGER,
    total_tokens           INTEGER,
    tps                    REAL,
    finish_reason          TEXT,
    response_nature        TEXT,
    retry_reason           TEXT,
    prompt_altered        TEXT,
    truncated              INTEGER,
    truncated_due_to_time INTEGER,
    failure_cause          TEXT,
    stream_ok              INTEGER,
    repeating              INTEGER,
    empty_reason           TEXT,
    error                  TEXT,
    score                  REAL,
    rubric_json            TEXT,
    diagnostics_json       TEXT,
    UNIQUE(revision_id, cell_id, attempt_number),
    UNIQUE(attempt_id, revision_id, cell_id)
);
```

An attempt is immutable. A stopped or abandoned attempt is never selected as a
successful result.

### Judge revisions, contracts, and attempts

```sql
CREATE TABLE revision_judges (
    revision_id  INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    judge_model  TEXT NOT NULL,
    source       TEXT NOT NULL,
    config_json  TEXT,
    active       INTEGER NOT NULL,
    PRIMARY KEY(revision_id, judge_model)
);

CREATE TABLE judge_contracts (
    contract_id              TEXT PRIMARY KEY,
    plugin_id                TEXT NOT NULL,
    plugin_version           TEXT NOT NULL,
    prompt_version           TEXT NOT NULL,
    instructions_version     TEXT NOT NULL,
    response_schema_hash     TEXT NOT NULL,
    contract_json             TEXT NOT NULL,
    contract_hash             TEXT NOT NULL UNIQUE,
    FOREIGN KEY(plugin_id, plugin_version)
        REFERENCES plugin_definitions(plugin_id, plugin_version)
);

CREATE TABLE revision_judge_contracts (
    revision_id  INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    plugin_id    TEXT NOT NULL,
    contract_id  TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    active       INTEGER NOT NULL,
    PRIMARY KEY(revision_id, plugin_id)
);

CREATE TABLE judge_attempts (
    judge_attempt_id         INTEGER PRIMARY KEY,
    revision_id              INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    cell_id                  INTEGER NOT NULL REFERENCES cells(cell_id),
    judge_model              TEXT NOT NULL,
    contract_id              TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    attempt_number           INTEGER NOT NULL,
    started_at               INTEGER,
    ended_at                 INTEGER,
    max_tokens               INTEGER,
    raw_response_payload_id  INTEGER REFERENCES payloads(payload_id),
    request_payload_id       INTEGER REFERENCES payloads(payload_id),
    response_usage_json      TEXT,
    diagnostics_json         TEXT,
    finish_reason            TEXT,
    error                    TEXT,
    UNIQUE(revision_id, cell_id, judge_model, contract_id, attempt_number)
);
```

`request_payload_id` is populated only in debug mode or for an explicitly
retained failure transcript. It is still one field with one type; it is simply
nullable when compact storage omits the full request.

### Parsed judge votes and criteria

Every judge attempt gets at most one parsed vote record. This preserves failed,
invalid, and superseded parsed outcomes without overwriting history.

```sql
CREATE TABLE judge_vote_attempts (
    vote_attempt_id  INTEGER PRIMARY KEY,
    judge_attempt_id  INTEGER NOT NULL UNIQUE
        REFERENCES judge_attempts(judge_attempt_id),
    score             INTEGER,
    confidence        TEXT,
    rationale         TEXT,
    error             TEXT,
    usable            INTEGER NOT NULL,
    created_at        INTEGER NOT NULL
);

CREATE TABLE judge_criteria (
    criterion_id     INTEGER PRIMARY KEY,
    vote_attempt_id  INTEGER NOT NULL REFERENCES judge_vote_attempts(vote_attempt_id),
    ordinal          INTEGER NOT NULL,
    criterion_key    TEXT NOT NULL,
    criterion        TEXT NOT NULL,
    status           TEXT NOT NULL,
    evidence         TEXT NOT NULL,
    UNIQUE(vote_attempt_id, ordinal)
);

CREATE TABLE current_judge_votes (
    revision_id      INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    cell_id          INTEGER NOT NULL REFERENCES cells(cell_id),
    judge_model      TEXT NOT NULL,
    contract_id      TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    vote_attempt_id  INTEGER NOT NULL REFERENCES judge_vote_attempts(vote_attempt_id),
    selected_at      INTEGER NOT NULL,
    selection_reason TEXT,
    PRIMARY KEY(revision_id, cell_id, judge_model, contract_id)
);
```

`current_judge_votes` is a projection. Historical parsed votes remain in
`judge_vote_attempts`; criteria and evidence remain queryable as `TEXT`.

### Consensus cache

```sql
CREATE TABLE consensus_cache (
    revision_id       INTEGER NOT NULL REFERENCES run_revisions(revision_id),
    cell_id           INTEGER NOT NULL REFERENCES cells(cell_id),
    contract_id       TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    score             REAL,
    confidence        TEXT,
    valid_judges      INTEGER NOT NULL,
    attempts          INTEGER NOT NULL,
    vote_set_hash     TEXT NOT NULL,
    calculated_at     INTEGER NOT NULL,
    PRIMARY KEY(revision_id, cell_id, contract_id)
);
```

Rationale and criteria are derived from the current vote rows rather than copied
into consensus. The cache is invalidated whenever `vote_set_hash` changes.

### Optional audit and debug-log metadata

SQLite WAL is sufficient for crash recovery. Audit events are optional and
contain IDs, never full payloads:

```sql
CREATE TABLE audit_events (
    event_id          INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    revision_id       INTEGER REFERENCES run_revisions(revision_id),
    event_type        TEXT NOT NULL,
    cell_id           INTEGER REFERENCES cells(cell_id),
    attempt_id        INTEGER REFERENCES benchmark_attempts(attempt_id),
    judge_attempt_id  INTEGER REFERENCES judge_attempts(judge_attempt_id),
    vote_attempt_id   INTEGER REFERENCES judge_vote_attempts(vote_attempt_id),
    created_at        INTEGER NOT NULL
);

CREATE TABLE debug_log_files (
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

CREATE TABLE legacy_import_records (
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
```

`legacy_import_records` is populated only when importing legacy JSON data that
cannot be mapped confidently to a normalized attempt, vote, or revision. It
prevents an importer from silently dropping or inventing historical data.

The implementation and recovery tests for `debug_log_files` are defined in
[gzip-append-log-plan.md](gzip-append-log-plan.md), especially its writer,
recovery, and integration-test stages.

## 7. Continuation and configuration-change semantics

A normal continuation creates a new `run_revisions` row. It never deletes or
mutates prior revision membership, attempts, votes, contracts, or reports.
The current revision is selected through `runs.current_revision_id`.

### Models/targets

- **Unchanged target:** resolve to the same `target_signature` and reuse the
  existing `target_instance` and cells. A selected successful attempt can be
  reused.
- **Added target:** create a target instance, cells, and current-revision rows;
  schedule only its pending cells.
- **Removed target:** mark it inactive in `revision_targets` and its cells
  unscheduled in `revision_cells`; retain all historical data.
- **Re-added target with the same signature:** reuse its previous instance and
  selected results unless explicitly purged.
- **Changed source, API model, runner, agent prompt, or execution config:**
  create a new target instance and new cells. The old target instance remains
  historical and is not silently mixed with the new execution.

### Plugins

- **Added plugin:** register its immutable definition, add it to
  `revision_plugins`, create cells for active targets, and schedule only those
  cells.
- **Removed plugin:** deactivate it for the new revision; retain old cells and
  results for historical reports.
- **Re-added plugin with the same version:** reuse the existing plugin
  definition and cell identity where the target signature is unchanged.
- **Changed plugin version:** register a new plugin definition and create new
  cells. The old plugin-version cells remain available and are not overwritten.

The current revision's plugin set controls default reports and scheduling; the
historical set controls historical reports. A report query must always include
`revision_id`; it must never infer current membership from whether a target,
plugin, or judge row still exists.

### Judges and contracts

- **Added judge model:** add it to `revision_judges`; queue missing votes for
  the active contract.
- **Removed judge model:** mark it inactive for the new revision; retain its
  previous attempts and votes.
- **Re-added judge model:** reuse old votes only when the active contract and
  input identity match; otherwise queue a new attempt.
- **Changed judge prompt, instructions, schema, or plugin guidance:** create a
  new immutable `judge_contracts` row and select it through
  `revision_judge_contracts`. Do not rerun benchmark tasks solely because the
  judge contract changed.
- **Judge contract change:** preserve all old contract votes and consensus;
  calculate new consensus independently for the new contract.

### Stopping and continuing

When a run stops:

1. mark the current revision `interrupted`;
2. mark in-flight benchmark/judge attempts `abandoned` or equivalent;
3. do not select abandoned attempts or votes as successful results;
4. commit all completed attempts and votes before shutdown;
5. retain the current revision as historical.

On continuation:

1. create a new revision with the new configuration snapshot;
2. resolve target/plugin/judge memberships;
3. reuse compatible selected attempts and votes;
4. schedule only missing, invalidated, or newly added cells;
5. leave removed models/plugins/judges available historically but inactive.

### Restart semantics

`--restart` should create a new logical `run_id` in a database rather than
silently mixing fresh results with an old logical run. The output manifest can
mark the new run as current. A separate explicit prune operation may remove old
runs; restart itself should not destroy historical evidence.

## 8. Indexes and constraints

At minimum:

```sql
CREATE INDEX idx_revision_cells_status
    ON revision_cells(revision_id, status, scheduled);

CREATE INDEX idx_attempts_cell_revision
    ON benchmark_attempts(revision_id, cell_id, attempt_number);

CREATE INDEX idx_judge_attempts_cell
    ON judge_attempts(revision_id, cell_id, judge_model, contract_id);

CREATE INDEX idx_vote_attempts_judge
    ON judge_vote_attempts(judge_attempt_id);

CREATE INDEX idx_criteria_vote
    ON judge_criteria(vote_attempt_id);

CREATE INDEX idx_payload_kind
    ON payloads(kind);

CREATE INDEX idx_legacy_import_source
    ON legacy_import_records(source_sha256, source_row_number);

CREATE UNIQUE INDEX one_active_contract_per_plugin
    ON revision_judge_contracts(revision_id, plugin_id)
    WHERE active = 1;
```

Use SQLite checks for known statuses, confidence values, and non-negative token
counts. Add validation triggers or composite foreign keys for relationships that
must share a run/revision/cell, including current revision ownership, revision
membership ownership, benchmark selection ownership, and current judge-vote
ownership. Store timestamps as integer UTC epoch values for smaller rows and
faster ordering; format them at report boundaries.

## 9. Migration stages

### Stage 0 — storage contract and CLI

- [x] Add `--storage json|sqlite`.
- [x] Add `--storage-profile compact|debug|portable`.
- [x] Add `--debug-logs` as an explicit debug-transcript override.
- [x] Add `--output-format csv md html pdf` with one-or-more selection.
- [x] Add report-only `--generate-reports PATH` mode.
- [x] Define no-report behavior when no format is selected.
- [x] Define redaction rules for configs, headers, and credentials.
- [x] Record run ID, revision ID, storage profile, and report formats in
      `run-info.json`.

### Stage 1 — backend abstraction

- [x] Define the shared `RunStore` run/session façade plus `PayloadStore`,
      `DebugLogStore`, and `ReportSource`.
- [x] Implement `JsonRunStore` without changing JSON behavior.
- [x] Implement `SQLiteRunStore` over the background writer and normalized
      benchmark/judge stores.
- [x] Route current state result/judge mutation hooks through the façade.
- [x] Provide common flush/close lifecycle methods with bounded shutdown.
- [x] Preserve existing JSON behavior and tests.

### Stage 2 — SQLite schema and migrations

- [x] Implement schema creation and version checking.
- [x] Add forward-only migrations.
- [x] Create revision-aware run/target/plugin membership tables.
- [x] Create immutable benchmark and judge attempt tables.
- [x] Create parsed judge vote-attempt and current-vote projection tables.
- [x] Enforce plugin-version and cross-revision relationships with foreign
      keys, composite keys, or validation triggers.
- [x] Create tables in dependency order and add the cyclic current-revision
      foreign key through a migration if necessary.
- [x] Enable foreign keys, WAL, busy timeout, and configured synchronous mode.
- [x] Add schema, index, constraint, and continuation tests.

### Stage 3 — background SQLite writer

- [x] Implement one persistence writer thread.
- [x] Batch operations by count and time.
- [x] Ensure workers never serialize a full state snapshot.
- [x] Add writer queue failure reporting.
- [x] Add shutdown timeout and synchronous final commit.
- [x] Add transaction crash-injection tests.

### Stage 4 — payload store and deduplication

- [x] Implement SHA-256 payload identity.
- [x] Store payload-bearing fields only as payload IDs.
- [x] Store short queryable judge text only as `TEXT`.
- [x] Add gzip compression for payload BLOBs.
- [x] Add payload integrity and round-trip tests.
- [x] Replace embedded judge-input prompt/response copies with payload IDs.

### Stage 5 — append-only compressed debug logs

- [x] Add the debug-only full-transcript policy.
- [x] Stop creating full successful judge logs in compact mode.
- [x] Implement the writer and recovery behavior from
      [gzip-append-log-plan.md](gzip-append-log-plan.md).
- [x] Add gzip concatenated-member, truncation, corruption, append-after-recovery,
      concurrency, redaction, and policy integration tests.
- [x] Add a reproducible gzip-vs-bz2 measurement helper and representative-log test.
- [ ] Run the measurement against a production-sized judge log before deciding
      whether zstd is worth a dependency; do not use 7z for active append-only logs.

### Stage 6 — benchmark task persistence

- [x] Persist one immutable row per benchmark attempt.
- [x] Persist revision-specific cell membership and selection.
- [x] Store prompt/content/thinking through payload IDs.
- [x] Preserve rubric, diagnostics, token, retry, and failure metadata.
- [x] Reproduce current resume and failed-task semantics.
- [x] Export legacy response files when `--save-responses` is selected.

### Stage 7 — judge persistence

- [x] Persist immutable judge contracts.
- [x] Persist revision-specific judge model and contract membership.
- [x] Persist every judge transport attempt.
- [x] Persist every parsed vote attempt without overwriting history.
- [x] Normalize criteria and evidence as `TEXT`.
- [x] Maintain current-vote projections per revision and contract.
- [x] Cache consensus using a vote-set hash.
- [x] Preserve historical contracts side-by-side.
- [x] Keep full assembled request payloads debug-only.

### Stage 8 — SQLite continuation, resume, and purge

- [x] Implement revision creation for every continuation invocation.
- [x] Reuse compatible target instances and cells.
- [x] Schedule added models/plugins/judges only.
- [x] Retire removed models/plugins/judges without deleting history.
- [x] Create new cells for changed target signatures or plugin versions.
- [x] Create new contracts for changed judge prompt/instruction/schema versions.
- [x] Reuse immutable global contract definitions when the contract hash matches.
- [x] Implement interrupted/abandoned attempt handling.
- [x] Reproduce `--no-rerun-failed` behavior.
- [x] Update `purge-results` to clear revision selections while retaining history.
- [x] Define and test `--restart` as a new logical run.

### Stage 9 — report generation

- [x] Generate reports only when `--output-format` is selected.
- [x] Implement report-only loading from SQLite.
- [x] Support legacy JSON runs in report-only mode.
- [x] Default reports to the current revision.
- [x] Add an explicit historical-revision report option if needed (`--revision`).
- [x] Write each report atomically.
- [x] Preserve current report content through the compatibility read model.
- [x] Add no-report, one-format, multi-format, and changed-config tests.

### Stage 10 — existing-run importer

- [x] Import legacy runs into a new logical SQLite run.
- [x] Preserve the imported state snapshot as a revision; ambiguous continuation
      snapshots remain explicit legacy records for later import.
- [x] Import target/plugin membership and version snapshots.
- [x] Deduplicate model-info/result-row judge data by cell/judge/contract identity.
- [x] Import benchmark attempts and nested attempt history.
- [x] Import judge-input sidecars and raw judge responses by payload hash.
- [x] Store ambiguous/unmappable source rows in a legacy-import table.
- [x] Import debug logs only when requested.
- [x] Exclude purge backups by default (the importer accepts only the requested
      state file, never directory-wide backup discovery).
- [x] Make imports idempotent and restartable using the source SHA-256.
- [x] Expose JSON-to-SQLite conversion through `--import-to-sqlite`, with
      `--sqlite-output` for an explicit destination.
- [x] Never modify the source JSON; refuse existing SQLite destinations unless
      `--overwrite-sqlite` is explicitly supplied.
- [x] Read source files in bounded chunks for hashing; JSON decoding remains the
      compatibility limitation for the legacy state container.

### Stage 11 — validation and rollout

- [ ] Add optional JSON/SQLite dual-write shadow mode. Runtime wiring remains
      intentionally gated until all task and judge persistence paths use the
      backend abstraction.
- [x] Compare scores, attempts, judge votes, consensus, revisions, and resume
      queues through the standalone semantic read-model validator.
- [x] Test model/plugin additions and removals across interrupted continuations.
- [x] Test plugin-version and judge-contract changes across continuations.
- [ ] Benchmark write latency and TUI responsiveness. `--measure-storage` now
      provides a reproducible synthetic persistence baseline; real TUI measurement
      still requires an operator run.
- [ ] Measure storage against the 2026-08-17 run shape. Use
      `ai-benchmark --measure-storage` for a synthetic baseline and compare the
      resulting profile with a copied run directory before marking this complete.
- [ ] Verify compact storage targets at least 60% savings. This remains an
      empirical acceptance criterion, not a design assumption.
- [ ] Verify debug compressed storage remains materially smaller than plaintext.
- [x] Make SQLite compact the default for new runs.
- [x] Retain JSON import/export as a fallback.
- [x] Update `AGENTS.md`, README, CLI docs, configuration docs, and architecture docs.

## 10. Acceptance criteria

### Correctness

- [ ] SQLite and JSON produce equivalent current scores and statuses.
- [ ] Every benchmark attempt remains queryable after retries and continuations.
- [ ] Every judge transport and parsed vote attempt remains queryable.
- [ ] Current judge votes never overwrite historical vote attempts.
- [ ] New judge contracts coexist with old contracts.
- [ ] Added and removed targets/plugins/judges are handled without deleting history.
- [ ] Changed target signatures and plugin versions do not reuse incompatible results.
- [ ] Purge and resume select the same cells for rerun.
- [ ] Reports from both backends are semantically equivalent.
- [ ] Interrupted runs recover without duplicated votes or attempts.

### Performance

- [ ] Worker threads do not block on full-state serialization.
- [ ] Persistence p95 latency remains below the configured batch interval.
- [ ] TUI refresh remains responsive during judge-heavy runs.
- [ ] SQLite startup/resume is no slower than JSON for normal runs.
- [ ] Report-only generation does not load models or schedule requests.

### Storage

Using the measured `2026-08-17-nas-and-more-test-changes` run as the reference:

- [ ] Compact SQLite output target: approximately 200–400 MiB with reports.
- [ ] Compact mode has no full successful judge transcript logs.
- [ ] Debug gzip logs are appendable without decompressing existing data.
- [ ] Reports are absent unless explicitly requested.
- [ ] Payload-bearing fields have one representation and identical payloads are
      stored once per database.
- [ ] Purge backups are outside the normal output directory or explicitly opt-in.

## 11. Recommended implementation order

1. [x] CLI/storage and continuation contract.
2. [x] `RunStore`/`PayloadStore`/`DebugLogStore` abstractions.
3. [x] Revision-aware SQLite schema and migrations.
4. [x] Background SQLite writer.
5. [x] Single-representation payload store.
6. [x] Debug-only gzip writer from `gzip-append-log-plan.md`.
7. [x] Benchmark attempt persistence.
8. [x] Judge attempt/vote persistence.
9. [x] Revision-aware resume, continuation, and purge.
10. [x] On-demand report generation.
11. [x] Existing-run importer.
12. [x] Semantic read-model validation.
13. [x] Read-only JSON/SQLite semantic comparison (`--compare-storage`).
14. [x] SQLite compact default.
15. [ ] Runtime dual-write shadow mode.

The key invariant is:

> Store immutable attempts, immutable configuration revisions, and canonical
> payload references. Derive current projections, consensus, TUI state, and
> reports from those records instead of independently copying the same data
> into multiple durable structures.
