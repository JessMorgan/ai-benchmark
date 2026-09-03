# Pi-agent runner support plan

**Status:** Implemented (initial isolated-worker runner); the compatibility smoke test remains opt-in because CI does not provide a real provider.

## Decisions captured

- Target the current package name `@earendil-works/pi-coding-agent` (the
  repository formerly associated with `@mariozechner/pi-coding-agent`).
- Use the Pi SDK through a small Node/TypeScript adapter subprocess.
- Start one isolated adapter subprocess per model/plugin cell, rather than
  sharing a long-lived Pi session between benchmark tasks.
- Design against the planned scalar `max_tokens` contract and one-retry policy,
  not the current list-based `token_levels` implementation.
- Add Pi as a runner and add an explicit multi-runner selection mechanism.
  Preserve the existing meaning of `--runner both` for backward compatibility
  instead of silently adding Pi to it.
- Plain model targets receive no Pi tools by default. Target configuration may
  explicitly enable selected Pi tools.

## Goal

Add a Pi-agent execution path comparable to the existing OpenCode path while
preserving benchmark isolation, reproducibility, streaming telemetry, timeout
behavior, cancellation, retry metadata, and runner-specific resume semantics.

The benchmark should be able to run a configured target through Pi as an
alternative to the direct OpenAI-compatible HTTP path and OpenCode path, then
score the extracted final answer with the same task plugins.

## Non-goals

- Do not make Pi the default runner.
- Do not reuse one Pi conversation across unrelated plugins.
- Do not make Pi tools part of the plain-model benchmark contract implicitly.
- Do not make the benchmark depend on Pi's interactive terminal UI.
- Do not silently translate legacy `token_levels`; the separate max-token
  migration plan requires that configuration key to be removed and rejected.
- Do not make Pi-specific output incomparable by injecting a hidden benchmark
  system prompt beyond the documented neutral/tool policy.

## 1. Proposed architecture

### Python benchmark adapter

Add a dedicated `benchmark/pi.py` module, keeping Pi separate from
`benchmark/opencode.py` and the direct HTTP transport. It should provide the
same conceptual boundary as the OpenCode adapter:

- Resolve and validate the Pi adapter/runtime.
- Build a per-cell request payload.
- Launch one worker subprocess.
- Stream normalized events from the worker.
- Enforce timeout and cancellation.
- Retain raw worker output and diagnostics.
- Return a runner-neutral process result to `benchmark/core.py`.

The Python adapter should not import JavaScript or depend on an embedded JS
runtime. It should communicate with the worker using a documented NDJSON
protocol over stdin/stdout.

### Node/TypeScript worker

Add a small project-local Pi worker, for example under a dedicated tool or
adapter directory, that:

1. Imports the pinned `@mariozechner/pi-coding-agent` SDK.
2. Reads one JSON request from stdin, or accepts a well-defined request stream
   if the implementation later supports multiple requests per process.
3. Creates exactly one Pi agent/session for that cell.
4. Configures the requested provider/model, system prompt, tools, and
   `max_tokens`.
5. Emits normalized NDJSON events as Pi events arrive.
6. Emits a terminal success/error event and exits.
7. Flushes output after each event so the Python parent can update live state.

The first version should use one request per worker process even if the SDK can
support multiple sessions. This keeps session isolation and failure semantics
obvious. A future pooled-worker optimization can be considered only after
contract tests prove that state cannot leak across cells.

### Why use a worker instead of invoking Pi's interactive CLI?

- The Python benchmark already owns scheduling, state, retry policy, and
  reporting.
- A worker can use the SDK's structured event API without scraping terminal
  output.
- Prompts and credentials travel through a structured request/config channel
  rather than a process argument.
- The worker can normalize SDK-version-specific event names in one place.
- The Python side remains independent of the TypeScript SDK's internal object
  model.

## 2. Runner integration

### Runner identity

Introduce `pi` as a runner identity everywhere runner identity is persisted or
reported:

- `BenchmarkState` model keys and result dictionaries.
- Runner-specific output directories.
- Resume and queue construction.
- Response/log artifact paths.
- CSV/HTML/Markdown runner columns.
- `run-info.json`, including Pi adapter and SDK versions.

A Pi result must never satisfy an HTTP or OpenCode result during resume.

### CLI selection

Add a Pi runner option while preserving existing behavior:

```sh
python ai-benchmark.py --runner pi
```

Keep the current `--runner both` meaning unchanged for compatibility. Add an
explicit arbitrary multi-runner option, preferably with a clear name such as:

```sh
python ai-benchmark.py --runners http,opencode,pi
```

The exact spelling should follow the existing argparse/completion conventions.
The implementation should reject duplicate or unknown runner names and should
produce deterministic runner ordering.

The scheduler must treat each `(target, runner)` pair as an independent leg.
For multi-runner mode, preserve the existing per-source concurrency limits and
avoid accidental overlap when a source is configured for one model slot.

### Configuration

Extend resolved target configuration with Pi-specific optional fields, without
requiring them for HTTP/OpenCode targets. Proposed fields:

```yaml
models:
  model-a:
    source: Local Server
    api_model: model-a
    pi:
      tools: []
      system_prompt: null
      max_tokens: 4096
```

The final shape should be validated centrally and documented. Prefer a small
allowlist of Pi settings over passing arbitrary SDK internals through the
benchmark config.

At minimum, the Pi target configuration needs to express:

- Provider/model mapping.
- Optional system prompt or agent persona.
- Selected Pi tools, defaulting to none.
- Tool permission/approval behavior.
- The scalar `max_tokens` value.
- Optional provider-specific request settings that are known to be safe and
  reproducible.

## 3. Provider and model mapping

The benchmark sources currently describe OpenAI-compatible endpoints, headers,
API keys, and model IDs. The Pi adapter should map those values into Pi's
provider/model configuration rather than asking operators to duplicate all
credentials in a second config format.

The mapping layer should:

- Derive a deterministic provider ID from the benchmark source name.
- Preserve the configured API base URL and model ID.
- Translate bearer/API-key headers carefully.
- Preserve non-secret custom headers where Pi supports them.
- Avoid writing credentials into logs or event payloads.
- Record a redacted provider/model identity in metadata.
- Fail clearly when a source cannot be represented by Pi's provider interface.

Because the exact SDK provider registration API may vary by pinned Pi version,
implementation must begin with a small provider compatibility probe. The probe
should verify that a configured OpenAI-compatible source can be registered and
that one minimal request reaches the intended endpoint/model.

## 4. Prompt and tool contract

### Plain model targets

The default Pi target should receive:

- The benchmark plugin prompt as the user task.
- A neutral system instruction, if Pi requires one to avoid its normal coding
  agent persona.
- No tools and no tool definitions in the model context.

This keeps Pi comparable to the direct HTTP and current neutral OpenCode paths.

### Configurable Pi tools

A target may explicitly enable a selected, named subset of Pi tools. The plan
should define a stable benchmark-level tool policy instead of exposing every
SDK tool option directly. For example:

```yaml
pi:
  tools:
    - read
    - grep
  permissions:
    read: allow
    grep: allow
    edit: deny
    bash: deny
```

Tool behavior must be recorded in metadata so a Pi result is not mistaken for
a no-tool model result. The metadata should include the requested tool names,
resolved tool names, permission policy, and whether any tool was actually
called.

Interactive approval must be disabled or made deterministic. A benchmark task
cannot hang waiting for a human to approve a tool call. Unsupported,
interactive-only, networked, or host-mutating tools should be denied by
 default unless explicitly designed and tested.

### Agent/persona targets

If a configured target is an agent persona, its system prompt may be passed to
Pi, subject to the same prompt-version and artifact-recording rules used by
OpenCode. The persona's tool policy must still be explicit; an agent target
must not silently receive the full Pi tool set.

## 5. Normalized event protocol

The Node worker should emit a versioned, runner-neutral event envelope. A
possible shape is:

```json
{
  "protocol": "pi-worker-v1",
  "event": "text_delta",
  "attempt": 1,
  "timestamp": 1730000000.123,
  "data": {
    "text": "partial answer"
  }
}
```

Proposed event types:

- `worker_started`
- `session_started`
- `prompt_started`
- `reasoning_delta`
- `text_delta`
- `tool_started`
- `tool_input_delta`
- `tool_finished`
- `usage`
- `step`
- `finish`
- `error`
- `worker_finished`

The worker should preserve the original Pi event type and a sanitized raw
payload where practical, but the Python adapter must rely only on the stable
normalized fields.

The terminal `finish` event should include, when available:

- Finish reason.
- Content text length and token usage.
- Reasoning/thinking token usage.
- Total token usage.
- Whether the response was truncated.
- Whether any tool was called.
- Provider/model identifiers.

If the SDK does not provide exact usage, the adapter should use the repository's
existing estimates and identify the source as estimated.

## 6. Response extraction and artifacts

The Python side should assemble:

- Final answer text from `text_delta` events.
- Thinking text from `reasoning_delta` events, when Pi exposes it.
- Tool-call summaries from normalized tool events.
- Raw worker stdout/stderr for diagnostics.

Artifacts should live under a runner-specific namespace, for example:

```text
<output_dir>/pi/responses/<target>/<plugin>/response.txt
<output_dir>/pi/responses/<target>/<plugin>/think.txt
<output_dir>/pi/responses/<target>/<plugin>/content.txt
<output_dir>/pi/responses/<target>/<plugin>/meta.json
<output_dir>/pi/logs/<target>/<plugin>.stdout.ndjson
<output_dir>/pi/logs/<target>/<plugin>.stderr.txt
```

The metadata should identify:

- `runner: "pi"`
- Pi SDK version.
- Worker protocol version.
- Adapter version.
- Target and source identity.
- Provider/model identity, redacted as necessary.
- Tool policy and actual tool activity.
- Prompt hash and system-prompt/persona identity.
- Selected attempt and complete attempt history.
- Response nature, finish reason, timeout/cancellation state, and token
  accounting.

## 7. Max-token and retry behavior

Pi support should use the shared max-token redesign rather than reintroducing
runner-specific token-level lists.

The Python benchmark core should own the retry policy. The Pi worker should run
one attempt and report enough information for the core to classify it.

For each cell:

- Use one scalar `max_tokens`.
- Permit at most one benchmark retry.
- On retryable transport failure, repeat with the original prompt and same
  budget.
- On token exhaustion, alter the prompt according to thinking usage:
  - `>= 80%`: `thinking_50_percent`.
  - `> 50%` and `< 80%`: `thinking_30_percent`.
  - `<= 50%` or unavailable: `response_under_budget`.
- On useful repetition abort, retry with `avoid_repetition`.
- Do not retry timeout or cancellation.
- Preserve all attempts and select the best usable attempt.
- Make top-level metadata describe the selected attempt only.

The worker request should include an explicit attempt number and prompt
alteration enum. The altered prompt should be passed as structured input to the
worker, not concatenated into an opaque shell command.

## 8. Timeout, cancellation, and runaway protection

The Python adapter should monitor worker stdout and stderr continuously and
should support:

- Overall task timeout.
- No-output/staleness timeout.
- Parent `stop_event` cancellation.
- Process-tree termination on POSIX and the supported Windows strategy.
- Graceful worker shutdown followed by forced termination if necessary.
- Retention of all partial output received before termination.

The worker should also attempt to abort the Pi session using the SDK's native
abort/cancel mechanism before exiting. Native abort is an optimization, not a
correctness requirement; the parent process must still be able to kill the
worker if the SDK hangs.

Runaway protections should include:

- Maximum tool/agent steps.
- Repetition detection over reasoning/content where event granularity allows.
- Maximum tool-call duration or inactivity if the SDK exposes those events.
- A deterministic policy for tool calls that request approval or external
  interaction.

Timeout and cancellation metadata must be distinct from transport errors and
must never accidentally consume the benchmark retry slot.

## 9. Installation and versioning

The repository is Python/`uv`-managed and currently has no Node package
workspace. The plan should avoid an unpinned global Pi installation.

Recommended approach:

- Keep the Pi worker and its package manifest in a dedicated project-local
  adapter directory.
- Pin the Pi SDK and Node runtime assumptions with a lockfile.
- Resolve the worker from a project-local path first.
- Provide a preflight command that checks Node, the worker, the SDK version,
  provider registration, event protocol, and required capabilities.
- Record the worker/SDK versions in `run-info.json`.
- Do not auto-download arbitrary runtime code during a benchmark run without an
  explicit install step or operator opt-in.

Whether the project uses npm, pnpm, or another Node package manager should be
settled during implementation based on the supported Pi SDK installation
instructions and CI environment. The Python benchmark should fail with an
actionable message when the worker is unavailable.

## 10. Scheduling and resume behavior

Initially, Pi should reuse the existing runner-aware scheduling model:

- Each `(target, runner)` is a separate execution identity.
- Per-source model and plugin concurrency limits remain authoritative.
- A Pi worker occupies one model slot and one plugin slot for its cell.
- Multi-runner mode must not exceed the source's configured concurrency.
- Resume skips only the completed Pi leg for the same target/plugin contract.
- HTTP, OpenCode, and Pi results remain separately reportable.

If explicit multi-runner mode runs the same target through several runners,
results should preserve a stable runner ordering and should not overwrite one
another's artifacts or state fields.

## 11. Compatibility probes

Add a separately runnable Pi compatibility probe, analogous to the existing
schema sentinel and OpenCode preflight. It should verify without scoring a
benchmark result that:

1. The worker starts.
2. The pinned SDK imports.
3. The configured provider/model can be registered.
4. A minimal request can be sent.
5. Text events are emitted and extracted.
6. Reasoning events are either emitted or explicitly reported unsupported.
7. `max_tokens` reaches the provider as expected.
8. Tool-denied mode emits no tool definitions/calls.
9. A configured tool policy is applied deterministically.
10. Cancellation terminates the session and worker.
11. The worker emits a valid terminal event.

The probe should output diagnostic JSON and must not modify benchmark scores or
resume state.

## 12. Testing plan

### Worker tests

- Request parsing and validation.
- Provider/model registration.
- Header and credential redaction.
- Neutral no-tool mode.
- Configured tool allowlist/denylist.
- System prompt/persona handling.
- NDJSON event normalization.
- Text and reasoning extraction.
- Tool event normalization.
- Usage and finish-reason normalization.
- Malformed SDK events and worker errors.
- Graceful abort.

### Python adapter tests

- Correct worker command and environment.
- Structured prompt transport rather than command-line prompt injection.
- Incremental event parsing across chunk boundaries.
- Partial output retention.
- Timeout and cancellation process-tree handling.
- Staleness, step, and repetition guards.
- SDK/worker version metadata.
- Retry classification and prompt-alteration metadata.
- Selected-attempt top-level metadata.
- Redacted logs.

### Core/scheduler tests

- `pi` runner is accepted and unknown runners are rejected.
- Explicit multi-runner combinations are parsed deterministically.
- Existing `--runner both` behavior remains unchanged.
- Pi state keys/results do not satisfy HTTP or OpenCode resume checks.
- Per-source concurrency limits are preserved.
- Multi-runner output paths and report columns remain distinct.
- Failed Pi legs can resume independently.

### Integration tests

Use a fake Node worker or fake Pi event stream so CI does not require model
credentials. Add an opt-in real-provider smoke test outside the normal suite
for the pinned SDK/provider combination.

## 13. Documentation and reporting

Update:

- CLI reference and shell completions.
- Configuration reference.
- Architecture documentation.
- Runner comparison documentation.
- Development instructions for the Node worker.
- State/resume schema documentation.
- Output/report documentation.

Reports should be able to distinguish:

- HTTP, OpenCode, and Pi execution.
- No-tool versus tool-enabled Pi targets.
- Requested versus actual tool calls.
- Pi SDK/worker versions.
- Text and reasoning token usage.
- Retry cause and prompt alteration.
- Timeout, cancellation, transport failure, token exhaustion, repetition, and
  normal completion.

## Implementation sequence

1. Confirm the pinned Pi SDK API and install/runtime strategy with a minimal
   standalone worker.
2. Define and test the `pi-worker-v1` event protocol.
3. Implement provider/model mapping and the no-tool compatibility probe.
4. Implement the Python subprocess adapter with streaming, timeout, and
   cancellation handling.
5. Integrate Pi into `_run_plugin_task()` and the shared max-token retry path.
6. Add Pi runner identity to scheduling, state, resume, artifacts, and reports.
7. Add explicit multi-runner CLI parsing while preserving `--runner both`.
8. Add configurable per-target Pi tools and deterministic permissions.
9. Add unit, fake-worker integration, and compatibility-probe tests.
10. Update documentation and run the complete Python and Node verification
    suites.

## Completion criteria

Pi support is complete when a configured target can run every eligible plugin
through an isolated Pi SDK worker, stream and persist normalized text/reasoning
and tool events, enforce timeout/cancellation and one-retry semantics, resume
without cross-runner contamination, report its SDK/tool/prompt metadata, and
coexist with HTTP/OpenCode under explicit multi-runner scheduling.
