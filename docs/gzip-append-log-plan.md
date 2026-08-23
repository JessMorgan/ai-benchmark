# Gzip Concatenated-Member Log Plan

**Status:** Core implementation and subprocess rollout complete; codec benchmarking remains open
**Primary use:** compressed debug and failure transcripts for benchmark, judge, OpenCode, and Pi execution.

## Goals

The log writer must:

- compress while writing;
- append without decompressing or rewriting existing data;
- preserve all complete log data after a crash;
- tolerate an incomplete final gzip member;
- keep memory bounded for large streams;
- avoid blocking benchmark and judge workers;
- redact credentials before compression;
- expose recovery failures prominently.

This design is intentionally separate from SQLite WAL. SQLite remains the
authoritative state store; compressed logs are optional diagnostics.

## Format decision

Use gzip concatenated members for the first implementation. A gzip file may
contain:

```text
gzip member 1 | gzip member 2 | gzip member 3 | ...
```

Appending a new member does not require reading or decompressing previous
members. Python's standard-library `gzip` module is sufficient.

```python
with open(path, "ab") as raw:
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        compressed.write(data)
```

Use `mtime=0` and omit the original filename from the gzip header for more
reproducible output.

## Format comparison

| Format | Strengths | Weaknesses | Decision |
|---|---|---|---|
| gzip | Standard library, streaming, concatenated members, ubiquitous | Lower ratio than newer codecs | **Initial implementation** |
| bz2 | Often better ratio; standard library | Slower and less suitable for frequent appends | Benchmark as an optional profile |
| xz/lzma | Excellent ratio | Slow, memory-heavy, poor live-logging fit | Reject for active logs |
| zstd | Excellent speed/ratio and independent frames | Requires a dependency and packaging decision | Consider after gzip measurements |
| 7z | Strong archive compression | Not a streaming log format; append may rewrite archive metadata; external dependency | Reject for active logs |
| tar.gz | Useful final archive format | Complicates append and random access | Use only for final packaging |

A gzip member is the unit of crash recovery. The existing file is never
decompressed just to append a new member.

## Writer API

Create `benchmark/logs.py` with an API similar to:

```python
class AppendOnlyGzipLog:
    def __init__(
        self,
        path: str,
        *,
        member_target_bytes: int = 128 * 1024,
        flush_interval: float = 0.5,
        sync_policy: str = "batch",
        recover_tail: bool = True,
    ):
        ...

    def append(self, data: str | bytes) -> None:
        ...

    def append_record(self, chunks) -> None:
        ...

    def flush(self, *, sync: bool = False) -> None:
        ...

    def recover(self) -> "LogRecovery":
        ...

    def close(self, *, sync: bool = True) -> None:
        ...
```

```python
@dataclass(frozen=True)
class LogRecovery:
    complete_members: int
    valid_bytes: int
    total_bytes: int
    truncated_tail: bool
    invalid_tail: bool
    discarded_bytes: int
```

Text input is encoded as UTF-8. Bytes input is written unchanged.

## Member boundaries and buffering

Use different boundaries for different streams:

| Stream | Recommended member boundary |
|---|---|
| Judge request/response transcript | One logical request attempt |
| HTTP request log | One complete request/response entry |
| OpenCode/Pi stdout/stderr | Buffered chunks around 64–256 KiB |
| Token/event streams | Buffered chunks, never one member per token |

For a logical request entry, `append_record()` should finish one member at the
end of the entry. A crash then loses at most the active request record.

For high-volume subprocess output, accumulate a bounded buffer and finish a
member when it reaches the target size or the flush timer fires. The writer
must not retain an entire long response in memory.

## Synchronization and durability

A single writer queue should own each log file. If direct multi-threaded access
is supported, use a per-file lock. Do not depend on multiple gzip writers
coordinating their offsets.

Recommended synchronization policies:

```text
none   - fastest; a crash may lose the OS-buffered tail
batch  - periodic flush/fsync; recommended debug default
final  - durable close at shutdown, minimal periodic syncing
```

Debug logs are not authoritative state, so an `fsync()` is not required for
every member. A final shutdown flush should use synchronous durability.

Write or fsync failures should:

- not unexpectedly terminate benchmark workers;
- be recorded as persistence/log failures;
- be printed prominently during shutdown;
- be included in `run-info.json`.

## Crash recovery

A crash can leave:

```text
complete member 1
complete member 2
partial member 3
```

The tolerant reader must return members 1 and 2 and identify member 3 as an
incomplete tail.

### Recovery scanner

Use a streaming `zlib.decompressobj(16 + zlib.MAX_WBITS)` scanner to process
one gzip member at a time. Track the raw byte offset at which each complete
member ends. Do not load a multi-hundred-megabyte log into memory just to scan
its tail.

The scanner should distinguish:

1. empty file;
2. complete file;
3. incomplete final member;
4. corruption in a non-final member;
5. unrelated trailing bytes.

Only an incomplete final member is automatically repairable.

### Repair behavior

When opening for append:

1. scan the existing tail;
2. if the tail is provably an incomplete final member, truncate to the last
   complete-member offset;
3. record a recovery warning and continue appending;
4. if corruption is non-final or ambiguous, do not silently truncate;
5. preserve the file and rotate to a new recovery segment or fail the debug
   log operation visibly.

Normal reads must not repair files. Repair belongs to the append/open path or
an explicit repair command.

A process killed after writing a gzip header but before writing a footer must
be treated the same as any other incomplete final member.

## Reader API

Provide a tolerant reader:

```python
def iter_log_members(
    path: str,
    *,
    tolerate_truncated_tail: bool = True,
):
    ...
```

It should yield complete member contents and expose a `LogRecovery` result.
Standard `gzip.open()` may raise when a final member is incomplete, so recovery
and diagnostics code must use the tolerant reader.

## Redaction

Redact before compression. Debug logs can contain request headers, curl
commands, prompts, candidate answers, and judge responses.

At minimum redact:

```text
Authorization
API-KEY
Cookie
Set-Cookie
password
secret
api_key
```

Preserve the header/key name and replace the value with `[REDACTED]`. Add a
metadata flag indicating that redaction occurred. Never rely on compression
as a security boundary.

## Integration policy

### Compact mode

Do not create full successful judge transcripts. Store structured metadata:

- request/cell identity;
- source, model, and runner;
- start/end times;
- status and error;
- HTTP status and retry counts;
- response hash;
- token and finish metadata.

Failure transcripts may be retained as compressed records if explicitly
configured.

### Debug mode

Create compressed files such as:

```text
logs/judge-<safe-model>.log.gz
logs/<target>/<plugin>.log.gz
logs/<runner>/<target>/<plugin>.stdout.gz
```

Preserve the current human-readable request/response block format inside each
member, including attempt number, prompt version, and contract ID.

New runs should not create plaintext `judge-*.log` files. Readers should still
support legacy plaintext logs.

## Test plan

Create `tests/test_compressed_logs.py`.

### Round-trip and append tests

- [ ] One-member round trip preserves exact bytes.
- [x] Multiple concatenated members are read in order.
- [x] Appending does not rewrite existing member bytes.
- [x] UTF-8, emoji, multiline, and empty records round-trip.
- [ ] Large streamed input remains bounded-memory.
- [ ] Python gzip readers and the tolerant reader see the same complete data.

### Crash and corruption tests

- [x] Gzip header-only final member is detected as truncated.
- [ ] Every truncation point within a final compressed member preserves all
      preceding members.
- [ ] Truncated footer is detected correctly.
- [ ] Abrupt child-process termination leaves earlier members readable.
- [ ] Recovery truncates only a provably incomplete final member.
- [ ] Appending after recovery yields all prior complete records plus the new
      record.
- [ ] Corruption in a non-final member is reported and never auto-truncated.
- [ ] Trailing non-gzip bytes are reported as invalid tail data.

### Concurrency and failure tests

- [x] Concurrent append calls do not interleave records.
- [ ] Concurrent recovery/open operations are serialized.
- [ ] Write failures are surfaced without crashing benchmark workers.
- [ ] `fsync()` failures are reported through persistence diagnostics.
- [ ] Final shutdown flush is durable when configured.

### Policy integration tests

- [x] Compact mode creates no full successful judge transcript.
- [x] Debug mode creates `.log.gz` and not plaintext judge logs.
- [ ] Debug log content includes attempt/version/contract metadata.
- [x] Credentials never appear in compressed or exported logs.
- [ ] Legacy plaintext logs remain readable.
- [ ] Incomplete gzip tails are reported in run metadata.

## Implementation order

1. [x] Implement `AppendOnlyGzipLog` and member metadata.
2. [x] Implement bounded-memory tolerant member scanning.
3. [x] Implement safe final-tail repair.
4. [x] Add round-trip, truncation, corruption, and append-after-recovery tests.
5. [x] Add per-file writer ownership and failure reporting.
6. [x] Add redaction tests.
7. [x] Integrate judge debug logs.
8. [x] Integrate HTTP diagnostic streams.
9. [x] Integrate OpenCode/Pi diagnostic streams.
10. [ ] Benchmark gzip against bz2 on representative judge logs.
11. [ ] Reconsider zstd only if gzip measurements justify a dependency.
