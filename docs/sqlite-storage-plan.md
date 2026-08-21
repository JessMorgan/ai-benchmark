# SQLite Run Storage Plan

**Status:** Planned
**Last updated:** 2026-08-20
**Primary goals:** fast writes, small output directories, reduced duplication, reliable resume, and reproducible reports.

## 1. Goals and decisions

The current file layout is human-readable and resilient, but duplicates prompts,
responses, judge metadata, consensus projections, report data, and journal
payloads. The target design is a SQLite-first run store with canonical payload
references and derived reports.

### Decisions

- SQLite is the authoritative store for new runs.
- Benchmark and judge attempts are immutable records; current selections are
  references, not copied result objects.
- Prompts, candidate responses, thinking text, and raw judge responses are
  content-addressed and stored once.
- Full judge request/response transcripts are **debug-only**. They are not
  written during ordinary compact runs.
- Debug transcripts are compressed while being written and support append-only
  operation without decompressing the existing file.
- Reports are generated only when requested.
- Users select one or more report formats with a CLI argument.
- JSON remains available as an import/export and compatibility backend during
  migration.
- SQLite WAL and one background writer replace the separate full-state JSONL
  journal for normal SQLite runs.

## 2. Storage profiles

### `compact` (default)

- SQLite metadata and result tables.
- Deduplicated compressed payloads.
- Raw benchmark and judge responses retained as payloads.
- No full successful judge request/response transcript.
- Failure metadata and compact transport diagnostics retained.
- No purge backups in the normal run directory.
- Reports generated only when `--output-format` is supplied.

### `debug`

Includes everything in `compact`, plus:

- Full judge request and response transcripts.
- HTTP request/response diagnostics.
- OpenCode and Pi stdout/stderr artifacts.
- Attempt-level transport details useful for reproducing failures.

Debug transcripts are compressed append-only files. They are not duplicated in
SQLite unless the operator explicitly requests database embedding.

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
├── payloads/                   # optional external large payloads
└── logs/                       # only in debug mode or for failures
    ├── judge-<model>.log.gz
    ├── <target>/<plugin>.log.gz
    └── <runner>/<target>/<plugin>.stdout.gz
```

After a clean shutdown, SQLite checkpoints its WAL. The database is the main
portable artifact; reports and debug logs are derived or optional artifacts.

## 4. Append-only compressed debug logs

### 4.1 Preferred format: gzip member streams

The first implementation should use Python's standard-library `gzip` module.
Gzip is the best initial choice because it is:

- available without a new dependency;
- fast enough for diagnostic text;
- widely available for operators and shell tooling;
- streamable;
- compatible with append-only concatenated members.

A log writer can append a new gzip member to an existing file:

```python
with gzip.open(path, "ab") as stream:
    stream.write(chunk)
```

Readers concatenate gzip members transparently. The existing compressed log is
never decompressed or rewritten merely to append new data.

The implementation should buffer logical log entries or chunks, rather than
creating a gzip member for every tiny character delta. A target member size of
roughly 64–256 KiB is a reasonable starting point. Each member should contain
complete UTF-8 text boundaries where practical.

### 4.2 Crash behavior

A process crash can leave an incomplete final gzip member. Readers must:

- return all complete preceding members;
- tolerate an incomplete final member;
- report the truncated tail in debug metadata;
- never rewrite the file during normal reading.

The single persistence/log writer should own each log file. If more than one
thread can write the same log, retain the existing log lock or use one queue
per file.

### 4.3 Alternatives considered

| Format | Advantages | Disadvantages | Decision |
|---|---|---|---|
| gzip | Standard library, streaming, concatenated members, ubiquitous | Worse ratio than newer codecs | **Initial implementation** |
| bz2 | Better ratio for some text; standard library | Slower compression/decompression; less convenient for frequent small appends | Optional later profile |
| xz/lzma | Excellent ratio | Slow, high memory, poor fit for live append | Do not use for live logs |
| zstd | Excellent speed/ratio; supports independent frames | Requires adding/verifying a dependency; packaging complexity | Consider after baseline |
| 7z | Strong compression and archive features | Not a streaming log format; append often rewrites archive/metadata; external dependency | Reject for append-only logs |
| tar.gz | Convenient archive of many files | Appending requires tar records and complicates random file access | Use only for final packaging |

The plan should not add 7z for active logs. A future `zstd` profile may be
added if measurements justify a dependency, but gzip is the default baseline.

### 4.4 Log policy

Normal compact runs should store only structured diagnostics:

- request ID and target/cell identity;
- source/model/runner;
- start/end time;
- status and error;
- HTTP status and retry counts;
- response hash;
- token/finish metadata.

Full request bodies and responses should be written only when:

- `--storage-profile debug` is selected;
- a dedicated `--debug-logs` option is selected;
- an operator explicitly requests failure transcripts;
- a provider/schema failure requires a retained reproduction artifact.

Successful judge requests must not create full plaintext `judge-*.log` files in
compact mode.

## 5. On-demand report generation

Reports should no longer be generated automatically for every run. This avoids
writing large CSV/HTML/PDF artifacts that may never be used.

### 5.1 Proposed CLI

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

`--generate-reports PATH` is report-only mode. It loads the SQLite store, or a
legacy JSON run during migration, and writes only the selected formats.

### 5.2 CLI validation

- `--output-format` requires either a benchmark run or
  `--generate-reports PATH`.
- `--generate-reports PATH` must not schedule model, benchmark, or judge work.
- Duplicate formats are removed while preserving the first requested order.
- Unknown formats fail before loading a model.
- Existing output files are replaced atomically.
- Report generation failures identify the format and do not corrupt the run
  database.
- `--save-responses` controls materialized human-readable response files; it
  does not implicitly enable reports.

### 5.3 Report compatibility

The first SQLite implementation should expose the existing result dictionary
read model to output plugins. This keeps report content equivalent while the
storage backend changes.

Later, output plugins may query SQLite directly for large reports. Reports must
remain derived artifacts and must not become a second authoritative result
store.

## 6. SQLite schema

### Schema metadata

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

### Runs and targets

```sql
CREATE TABLE runs (
    run_id              TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    ended_at            TEXT,
    status              TEXT NOT NULL,
    score_schema        TEXT NOT NULL,
    runner_mode         TEXT NOT NULL,
    session_seed        INTEGER,
    config_payload_id   INTEGER,
    app_version         TEXT,
    storage_profile     TEXT NOT NULL
);

CREATE TABLE targets (
    target_id           INTEGER PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    target_name         TEXT NOT NULL,
    source              TEXT NOT NULL,
    api_model           TEXT NOT NULL,
    runner              TEXT NOT NULL,
    is_agent            INTEGER NOT NULL DEFAULT 0,
    system_prompt_id    INTEGER REFERENCES payloads(payload_id),
    target_config_id    INTEGER REFERENCES payloads(payload_id),
    UNIQUE(run_id, target_name, runner)
);
```

### Canonical payloads

```sql
CREATE TABLE payloads (
    payload_id          INTEGER PRIMARY KEY,
    sha256               TEXT NOT NULL UNIQUE,
    kind                 TEXT NOT NULL,
    compression          TEXT NOT NULL DEFAULT 'gzip',
    uncompressed_bytes   INTEGER NOT NULL,
    stored_bytes         INTEGER NOT NULL,
    data                 BLOB,
    external_path        TEXT,
    created_at           TEXT NOT NULL,
    CHECK (data IS NOT NULL OR external_path IS NOT NULL)
);
```

Normal prompts, responses, thinking text, judge responses, and JSON metadata
should be compressed payloads. Very large debug transcripts may instead be
stored as external compressed files referenced by `external_path`.

### Benchmark cells and attempts

```sql
CREATE TABLE cells (
    cell_id                  INTEGER PRIMARY KEY,
    run_id                   TEXT NOT NULL REFERENCES runs(run_id),
    target_id                INTEGER NOT NULL REFERENCES targets(target_id),
    plugin_id                TEXT NOT NULL,
    plugin_version           TEXT NOT NULL,
    status                   TEXT NOT NULL DEFAULT 'pending',
    selected_attempt_id      INTEGER,
    deterministic_score      REAL,
    selected_attempt_number  INTEGER,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    UNIQUE(target_id, plugin_id)
);

CREATE TABLE benchmark_attempts (
    attempt_id              INTEGER PRIMARY KEY,
    cell_id                 INTEGER NOT NULL REFERENCES cells(cell_id),
    attempt_number          INTEGER NOT NULL,
    prompt_payload_id       INTEGER REFERENCES payloads(payload_id),
    content_payload_id      INTEGER REFERENCES payloads(payload_id),
    thinking_payload_id     INTEGER REFERENCES payloads(payload_id),
    started_at              TEXT,
    ended_at                TEXT,
    max_tokens              INTEGER,
    output_tokens           INTEGER,
    thinking_tokens         INTEGER,
    total_tokens            INTEGER,
    tps                     REAL,
    finish_reason           TEXT,
    response_nature         TEXT,
    retry_reason            TEXT,
    prompt_altered         TEXT,
    truncated               INTEGER,
    truncated_due_to_time  INTEGER,
    failure_cause           TEXT,
    stream_ok               INTEGER,
    repeating               INTEGER,
    empty_reason            TEXT,
    error                   TEXT,
    score                   REAL,
    rubric_json             TEXT,
    diagnostics_json        TEXT,
    selected                INTEGER NOT NULL DEFAULT 0,
    UNIQUE(cell_id, attempt_number)
);
```

Retries are represented as rows instead of a nested `{plugin}_attempts` JSON
array. A cell has one selected attempt, while all historical attempts remain
available.

### Judge contracts and attempts

```sql
CREATE TABLE judge_contracts (
    contract_id              TEXT PRIMARY KEY,
    run_id                   TEXT NOT NULL REFERENCES runs(run_id),
    plugin_id                TEXT NOT NULL,
    plugin_version            TEXT NOT NULL,
    prompt_version            TEXT NOT NULL,
    instructions_version     TEXT NOT NULL,
    response_schema_hash      TEXT NOT NULL,
    contract_payload_id       INTEGER REFERENCES payloads(payload_id),
    active                    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE judge_attempts (
    judge_attempt_id         INTEGER PRIMARY KEY,
    cell_id                  INTEGER NOT NULL REFERENCES cells(cell_id),
    judge_model              TEXT NOT NULL,
    judge_source              TEXT,
    contract_id              TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    attempt_number            INTEGER NOT NULL,
    started_at                TEXT,
    ended_at                  TEXT,
    max_tokens                INTEGER,
    raw_response_payload_id   INTEGER REFERENCES payloads(payload_id),
    request_payload_id        INTEGER REFERENCES payloads(payload_id),
    response_usage_json       TEXT,
    diagnostics_json          TEXT,
    finish_reason             TEXT,
    error                     TEXT,
    selected                  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(cell_id, judge_model, contract_id, attempt_number)
);

CREATE TABLE judge_votes (
    vote_id                  INTEGER PRIMARY KEY,
    cell_id                  INTEGER NOT NULL REFERENCES cells(cell_id),
    judge_attempt_id         INTEGER NOT NULL REFERENCES judge_attempts(judge_attempt_id),
    judge_model              TEXT NOT NULL,
    contract_id              TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    score                    INTEGER,
    confidence               TEXT,
    rationale_payload_id     INTEGER REFERENCES payloads(payload_id),
    error                    TEXT,
    usable                   INTEGER NOT NULL DEFAULT 0,
    created_at               TEXT NOT NULL,
    UNIQUE(cell_id, judge_model, contract_id)
);

CREATE TABLE judge_criteria (
    criterion_id             INTEGER PRIMARY KEY,
    vote_id                  INTEGER NOT NULL REFERENCES judge_votes(vote_id),
    ordinal                  INTEGER NOT NULL,
    criterion_key            TEXT NOT NULL,
    criterion_payload_id     INTEGER REFERENCES payloads(payload_id),
    status                   TEXT NOT NULL,
    evidence_payload_id      INTEGER REFERENCES payloads(payload_id),
    UNIQUE(vote_id, ordinal)
);
```

The full assembled judge prompt is reconstructed from the contract and payload
references by default. `request_payload_id` is populated only in debug mode or
for explicitly retained failure transcripts.

### Consensus and audit records

```sql
CREATE TABLE consensus (
    cell_id                  INTEGER NOT NULL REFERENCES cells(cell_id),
    contract_id              TEXT NOT NULL REFERENCES judge_contracts(contract_id),
    score                    REAL,
    confidence               TEXT,
    valid_judges             INTEGER NOT NULL,
    attempts                 INTEGER NOT NULL,
    vote_set_hash            TEXT NOT NULL,
    calculated_at            TEXT NOT NULL,
    PRIMARY KEY(cell_id, contract_id)
);

CREATE TABLE events (
    event_id                 INTEGER PRIMARY KEY,
    sequence                 INTEGER NOT NULL UNIQUE,
    event_type               TEXT NOT NULL,
    cell_id                  INTEGER REFERENCES cells(cell_id),
    attempt_id               INTEGER REFERENCES benchmark_attempts(attempt_id),
    judge_attempt_id         INTEGER REFERENCES judge_attempts(judge_attempt_id),
    vote_id                  INTEGER REFERENCES judge_votes(vote_id),
    created_at               TEXT NOT NULL
);
```

The `events` table contains references and scalar metadata, not full prompt,
response, vote, or consensus objects. SQLite WAL provides crash recovery for
normal operation.

## 7. Migration stages

### Stage 0 — storage contract and CLI

- [ ] Add `--storage json|sqlite`.
- [ ] Add `--storage-profile compact|debug|portable`.
- [ ] Add `--output-format csv md html pdf` with one-or-more selection.
- [ ] Add report-only `--generate-reports PATH` mode.
- [ ] Define behavior when no output format is selected: no reports generated.
- [ ] Define redaction rules for configs, headers, and credentials.
- [ ] Record storage profile and selected report formats in `run-info.json`.

### Stage 1 — backend abstraction

- [ ] Define the `RunStore` interface.
- [ ] Wrap the existing JSON persistence in `JsonRunStore`.
- [ ] Route task and judge result callers through the interface.
- [ ] Preserve existing JSON behavior and tests.

### Stage 2 — SQLite schema and migrations

- [ ] Implement schema creation and version checking.
- [ ] Add forward-only migration helpers.
- [ ] Enable foreign keys, WAL, busy timeout, and configured synchronous mode.
- [ ] Add schema/index/constraint tests.

### Stage 3 — background SQLite writer

- [ ] Implement a single persistence writer thread.
- [ ] Batch operations by count and time.
- [ ] Ensure worker threads never serialize a full state snapshot.
- [ ] Add writer queue failure reporting.
- [ ] Add shutdown timeout and synchronous final commit.
- [ ] Add crash-injection tests around transaction boundaries.

### Stage 4 — payload store and deduplication

- [ ] Implement SHA-256 payload identity.
- [ ] Deduplicate identical prompts, responses, thinking text, and judge outputs.
- [ ] Add gzip compression for ordinary payloads.
- [ ] Add an external compressed-payload path for very large data.
- [ ] Add payload integrity and round-trip tests.
- [ ] Replace embedded judge-input prompt/response copies with references.

### Stage 5 — append-only compressed debug logs

- [ ] Add a debug-only full-transcript policy.
- [ ] Stop creating successful full judge logs in compact mode.
- [ ] Implement buffered gzip member appends.
- [ ] Add a per-log writer lock or writer-queue ownership rule.
- [ ] Make readers tolerate an incomplete final gzip member.
- [ ] Store log compression, byte counts, and truncation status in metadata.
- [ ] Add optional bz2 benchmark support for comparison.
- [ ] Evaluate zstd as a future optional dependency.
- [ ] Explicitly reject 7z for active append-only logs.
- [ ] Add tests proving append works without decompressing old data.
- [ ] Add crash/truncated-tail reader tests.
- [ ] Add compact-mode tests proving full successful judge transcripts are absent.

### Stage 6 — benchmark task persistence

- [ ] Persist one row per benchmark attempt.
- [ ] Store selected attempt through a cell reference.
- [ ] Store prompt/content/thinking through payload references.
- [ ] Preserve rubric, diagnostics, token, retry, and failure metadata.
- [ ] Reproduce current resume and failed-task semantics.
- [ ] Export legacy response files when `--save-responses` is selected.

### Stage 7 — judge persistence

- [ ] Persist versioned judge contracts.
- [ ] Persist every judge attempt separately.
- [ ] Store only one current vote per cell/judge/contract.
- [ ] Normalize criteria and evidence.
- [ ] Cache consensus using a vote-set hash.
- [ ] Preserve historical contracts side-by-side.
- [ ] Keep full assembled request payloads debug-only.
- [ ] Ensure retries remain visible as judge-attempt rows.

### Stage 8 — SQLite resume and purge

- [ ] Implement resume queries for pending/failed cells.
- [ ] Keep HTTP, OpenCode, and Pi runner identities separate.
- [ ] Reproduce `--no-rerun-failed` behavior.
- [ ] Reproduce plugin-version and judge-contract invalidation.
- [ ] Update `purge-results` to operate on SQLite.
- [ ] Retain historical attempts while clearing the selected current attempt.
- [ ] Add resume/purge regression tests.

### Stage 9 — report generation

- [ ] Generate reports only when `--output-format` is selected.
- [ ] Implement report-only loading from SQLite.
- [ ] Support legacy JSON runs in report-only mode.
- [ ] Validate one or more requested formats.
- [ ] Write each report atomically.
- [ ] Preserve current report content through the compatibility read model.
- [ ] Add tests for no-report, one-format, and multi-format runs.

### Stage 10 — existing-run importer

- [ ] Import `benchmark_state.json` into SQLite.
- [ ] Import benchmark config and run metadata.
- [ ] Import judge-input sidecars as canonical payloads.
- [ ] Import response artifacts and judge raw responses by hash.
- [ ] Import historical result rows without duplicating latest projections.
- [ ] Import debug logs only when requested.
- [ ] Exclude purge backups by default.
- [ ] Emit missing/ambiguous-data warnings.
- [ ] Make imports idempotent.

### Stage 11 — validation and rollout

- [ ] Add optional JSON/SQLite dual-write shadow mode.
- [ ] Compare scores, attempts, judge votes, consensus, and resume queues.
- [ ] Benchmark write latency and TUI responsiveness.
- [ ] Measure storage against the 2026-08-17 run shape.
- [ ] Verify compact storage targets at least 60% savings.
- [ ] Verify debug compressed storage remains materially smaller than plaintext.
- [ ] Make SQLite compact the default for new runs.
- [ ] Retain JSON import/export as a fallback.
- [ ] Update `AGENTS.md`, README, CLI docs, configuration docs, and architecture docs.

## 8. Acceptance criteria

### Correctness

- [ ] SQLite and JSON produce equivalent selected scores and statuses.
- [ ] Benchmark retry history is preserved.
- [ ] Judge retry history and contract versioning are preserved.
- [ ] Purge and resume select the same cells for rerun.
- [ ] Reports from both backends are semantically equivalent.
- [ ] Interrupted runs recover without duplicated votes or attempts.

### Performance

- [ ] Worker threads do not block on full-state serialization.
- [ ] Persistence p95 latency remains below the configured batch interval.
- [ ] TUI refresh remains responsive during judge-heavy runs.
- [ ] SQLite startup/resume is no slower than JSON for normal runs.

### Storage

Using the measured `2026-08-17-nas-and-more-test-changes` run as the reference:

- [ ] Compact SQLite output target: approximately 200–400 MiB.
- [ ] Compact mode has no full successful judge transcript logs.
- [ ] Debug gzip logs are appendable without decompressing existing data.
- [ ] Reports are absent unless explicitly requested.
- [ ] Prompt/response payloads are stored once per unique hash.
- [ ] Purge backups are outside the normal output directory or explicitly opt-in.

## 9. Recommended implementation order

1. [ ] CLI/storage contract.
2. [ ] `RunStore` abstraction.
3. [ ] SQLite schema and migrations.
4. [ ] Background writer.
5. [ ] Payload deduplication.
6. [ ] Debug-only gzip log writer.
7. [ ] Benchmark attempt persistence.
8. [ ] Judge attempt/vote persistence.
9. [ ] SQLite resume and purge.
10. [ ] On-demand report generation.
11. [ ] Existing-run importer.
12. [ ] Dual-write validation.
13. [ ] SQLite compact default.

The most important invariant is:

> Store immutable attempts and canonical payload references; derive current
> projections, consensus, TUI state, and reports from those records instead of
> independently copying the same data into multiple durable structures.
