# OpenCode Runner Feature Specification

**Status:** Implemented (with version-sensitive OpenCode adapter assumptions documented below)
**Scope:** Feature specification and implementation record. The implementation is in `opencode_runner.py` and the runner integration points described here.
**Short name:** OpenCode runner

## 1. Summary

Add an optional OpenCode-backed execution runner to AI Benchmark. The user selects the runner at invocation time with a CLI option rather than configuring it in the benchmark config file. The runner controls whether configured models and agents execute through the existing OpenAI-compatible HTTP path, through a locally pre-installed `opencode` CLI process, or through both paths.

The feature must dynamically generate an OpenCode configuration from the loaded ai-benchmark configuration. OpenCode is an external prerequisite, but when it is missing (or too old for the required CLI contract) the benchmark downloads the official release into a project-local directory (`.tools/opencode/`) and uses that binary; see the "Preflight behavior" section for the full resolution order and the `--no-install-opencode` opt-out.

## 2. Decisions captured from the interview

- Runner selection is a CLI concern, not an ai-benchmark config-file setting.
- The new option is a choice-style `--runner` argument.
- Canonical values are:
  - `http`: use the existing direct model/API execution path.
  - `opencode`: use OpenCode for the configured targets.
  - `both`: run both runner variants.
- Omitting `--runner` preserves current behavior: `http` is the default.
- Existing models and agents are eligible for the selected runner(s); there is no separate OpenCode target declaration in the config.
- OpenCode uses the already resolved `api_model` value, not the display/target key.
- The OpenCode model identifier is constructed literally as:

  ```text
  {strictly_slugified_source}/{api_model}
  ```

  The source is the benchmark source map key, not an independently configured OpenCode provider ID.
- Source normalization uses strict slugification: lowercase; replace every run of non-alphanumeric characters with `-`; collapse duplicate separators; trim leading/trailing `-`.
- Existing provider prefixes in `api_model` are not removed. The mapping always applies the formula literally, even when `api_model` already contains `/`.
- OpenCode is resolved during startup. The selected OpenCode mode is a hard prerequisite; failure is reported before benchmark work is scheduled. Resolution order: (1) an on-PATH install that passes the capability preflight; (2) a previously auto-installed local copy under `.tools/opencode/`; (3) a fresh auto-install of the latest release into `.tools/opencode/` when not disabled by `--no-install-opencode`.
- In `both` mode, OpenCode runs before the HTTP runner.
- Both runner variants may be represented in the same overall benchmark run, but their artifacts use separate `http/` and `opencode/` namespaces.
- Resume is runner-aware: a result can be reused only when target, runner, and plugin identity match.
- OpenCode receives an existing agent's system prompt separately from the plugin prompt; the prompts must not be silently discarded or flattened unless the OpenCode adapter requires a documented equivalent.
- One generated OpenCode config is created per benchmark run and reused by that run's OpenCode subprocesses.
- The generated config is retained as an exact artifact, including resolved credentials. This is an explicit user decision and must be clearly documented as a security trade-off.
- The generated config should project the full benchmark configuration where OpenCode has a meaningful equivalent, including provider endpoint/auth, selected model information, timeout, temperature, token limits, and compatible request parameters. Unsupported fields must be documented and must not be silently misrepresented.
- OpenCode stdout and stderr are stored separately. stdout supplies the final response for scoring; stderr is diagnostic/progress output.

## 3. Existing architecture to preserve

The current project:

- Loads JSON or YAML benchmark configs through `benchmark_core.load_config`.
- Resolves `models` and `agents` into a unified target map through `resolve_targets`.
- Distinguishes a target's display name from its resolved `api_model`, source, system prompt, and `is_agent` metadata.
- Runs each target through `run_model`, which dispatches plugins through `_run_plugin_task`.
- Uses OpenAI-compatible HTTP functions for streaming and non-streaming plugins.
- Stores target results and live status in `BenchmarkState` and writes report/output artifacts under the configured output directory.
- Uses target/model names as important identity keys today, so runner separation must be designed explicitly rather than relying on accidental filename differences.

The implementation should add a runner abstraction or an equivalently narrow adapter boundary around target/plugin execution. It should not rewrite the existing HTTP request behavior or change default `http` semantics.

## 4. CLI interface

### Required option

Add a choice-style option equivalent to:

```text
--runner {http,opencode,both}
```

Help text must explain:

- `http` is the existing OpenAI-compatible execution path.
- `opencode` invokes the pre-installed `opencode` CLI.
- `both` executes both runner variants.
- The default is `http`.
- OpenCode mode generates a run-scoped config from the benchmark config.

The exact argparse spelling should be `--runner` unless project conventions require a compatible alias. The accepted values should be validated by argparse rather than manually accepting arbitrary strings.

### Preflight behavior

- For `--runner http`, do not require or probe OpenCode.
- For `--runner opencode` and `--runner both`, resolve the OpenCode binary before scheduling target workers or making HTTP requests. Resolution order:
  1. `shutil.which("opencode")` — an existing on-PATH install that passes the capability preflight (`opencode run --help` advertising `--model`/`--format`/`--agent`/`--pure`/`--thinking` and the `json` format choice).
  2. A previously auto-installed local copy at `<project root>/.tools/opencode/opencode` if it still passes the preflight.
  3. When `--no-install-opencode` is NOT given, download the official latest release into `.tools/opencode/` and validate it. The download mirrors the official installer's platform detection (`opencode-<os>-<arch>[-baseline][-musl].tar.gz` on Linux, `.zip` on macOS/Windows) and writes a `version.txt` marker next to the binary. An on-PATH install that exists but fails the preflight is replaced by the local copy or a fresh install automatically.
- If no usable binary can be resolved, exit non-zero with an actionable message. With `--no-install-opencode`, the message states that OpenCode must be installed and available on `PATH` (or that the on-PATH binary is incompatible) and that the opt-out prevented the automatic download; the benchmark never silently falls back to HTTP.
- The preflight must happen before any target is run, including in `both` mode, so a missing dependency cannot produce a partial mixed run.
- The selected mode and the resolved binary path should be recorded in run metadata, such as `run-info.json` (`runner`, `opencode_binary`).

### Ordering

When `--runner both` is selected:

1. Resolve and validate all targets.
2. Validate OpenCode availability and generated-config inputs.
3. Create one execution worker per source.
4. Have that worker run each target's OpenCode variant followed by its HTTP variant before advancing to the next target.

Each source has exactly one active model run at a time: OpenCode and HTTP never overlap on the same source. Sources may run concurrently with one another. On resume, a target with completed OpenCode state proceeds directly to its pending HTTP step. Single-runner modes retain their existing source-level worker behavior.

## 5. Target and runner identity

### Target selection

The runner applies to every resolved model and agent target from the loaded benchmark config. Existing per-target plugin blacklists remain effective for both runner variants.

For each target:

- `target_name` remains the configured display/identity name.
- `api_model` remains the resolved model value already produced by `resolve_targets`.
- `source` remains the original benchmark source name in result metadata.
- `system_prompt` remains the existing target system prompt, if any.
- `is_agent` remains the existing boolean.

### Runner metadata

Every result and persisted state record produced by this feature must include a runner discriminator, at minimum:

```json
"runner": "http" | "opencode"
```

OpenCode results should additionally record the computed model string, for example:

```json
"opencode_model": "local-server-1/my-model"
```

The original source and resolved `api_model` must also remain available. Do not overwrite `source` with the literal string `opencode`; the source is still useful benchmark metadata.

### Output namespaces

Use stable runner namespaces:

```text
<output_dir>/http/...
<output_dir>/opencode/...
```

The namespaces apply to runner-specific response files, logs, metadata, and other per-runner artifacts. The exact top-level report layout may remain shared if the report format gains a runner column, but runner-specific artifact links must be unambiguous.

For `both`, the report must make it possible to compare the two results for the same target and plugin without relying on filename guessing. Preferred representation:

- same logical target label;
- explicit `runner` column or label;
- separate response/log links under `http/` and `opencode/`.

If existing output formats require unique row keys, derive an internal composite identity such as `(target_name, runner)` while preserving the human-readable target name in display fields.

## 6. OpenCode model mapping

For every OpenCode target, compute:

```text
opencode_model = slugify(source) + "/" + api_model
```

Where `slugify(source)` is deterministic and defined as:

1. Convert the source name to lowercase.
2. Replace every contiguous run of characters outside `[a-z0-9]` with `-`.
3. Collapse repeated `-` characters.
4. Strip leading and trailing `-` characters.

Examples:

| Benchmark source | `api_model` | OpenCode model |
|---|---|---|
| `Local Server 1` | `qwen3:8b` | `local-server-1/qwen3:8b` |
| `Remote/OpenAI` | `gpt-4.1` | `remote-openai/gpt-4.1` |
| ` Gaming PC ` | `vendor/model-x` | `gaming-pc/vendor/model-x` |

The `api_model` portion is opaque and must not be altered. In particular, an existing slash is preserved, and the implementation must not strip an existing provider prefix or attempt to infer a different provider.

Validation requirements:

- A source that slugifies to an empty string is invalid for OpenCode mode.
- An empty `api_model` is invalid for an OpenCode target.
- Mapping collisions should be detected before execution. If two source/model pairs produce the same OpenCode model string, fail with a message identifying the conflicting targets rather than silently choosing one.

## 7. Dynamic OpenCode configuration

### Generation timing and scope

- Generate one config for the benchmark run when an OpenCode phase is selected.
- It must represent all source/provider mappings needed by the targets in that run.
- Reuse the generated config for every OpenCode subprocess in the phase(s) of that run.
- Do not require users to create an OpenCode config manually.
- Do not modify the repository's project-level OpenCode configuration.

### Location and artifact policy

The generated file should be placed in a deterministic runner-specific location under the output directory, for example:

```text
<output_dir>/opencode/opencode.generated.json
```

The exact filename may vary, but it must be recorded in run metadata and be discoverable from the output directory. Since the user explicitly chose to retain exact copies, the file must not be deleted automatically after normal completion or cancellation.

Because this file may contain resolved credentials:

- create parent directories securely;
- write the file with restrictive permissions where supported;
- do not print its contents in normal CLI output;
- do not include credential values in exception messages;
- document that retaining the generated file stores secrets on disk;
- if logs or reports reference it, reference the path, not its contents.

### Source/provider projection

For each benchmark `sources` entry used by an OpenCode target, project all meaningful settings that OpenCode supports, including:

- the source's endpoint/base URL;
- authentication derived from the source headers;
- provider/model registration needed to resolve the computed `{source}/{api_model}` identifier;
- timeout settings;
- compatible temperature and token-limit settings;
- compatible request parameters and headers.

The projection must use an explicit mapping function rather than blindly copying arbitrary benchmark keys into OpenCode JSON. Fields without a valid OpenCode equivalent must be either:

- omitted with a documented warning/metadata note; or
- rejected during preflight when omission could change the meaning or safety of the benchmark.

The generated config must select or register the mapped OpenCode model so that `opencode run` can resolve the computed model identifier.

### Credentials

The selected policy is to write resolved credential values temporarily/per-run into the retained generated config. This is intentionally different from the safer environment-reference approach. The implementation must:

- ensure environment expansion has already occurred consistently with the loaded benchmark config;
- preserve the relevant authorization/header semantics when translating to OpenCode;
- never expose secret values in command-line arguments, logs, result reports, or error strings;
- set restrictive file permissions;
- document the security implications.

If a source cannot be translated cleanly to an OpenCode provider definition, fail during preflight before scheduling any target. Do not run some OpenCode targets and silently skip others.

### Full benchmark projection caveat

The benchmark and OpenCode do not necessarily share identical request models. The implementation spec must include a field-by-field projection table in code comments or documentation covering at least:

- source API URL/base URL;
- headers/auth;
- model;
- timeout;
- temperature;
- max tokens/token levels;
- seed and `drop_params`;
- streaming/non-streaming behavior;
- retry/backoff settings;
- system prompts;
- plugin-specific request semantics.

Unsupported fields must have explicit behavior. In particular, an OpenCode subprocess may not expose the same streaming telemetry or HTTP 429 controls as the direct path; those differences must be reflected in result metadata rather than fabricated.

## 8. OpenCode invocation

### Process granularity

Run one fresh OpenCode process per target/plugin invocation. This keeps plugin prompts isolated and prevents context from one benchmark task affecting another. The generated run-level config is shared, but sessions/processes are not.

### Command behavior

Use OpenCode's non-interactive command form (`opencode run`) rather than launching the TUI. The implementation must verify the exact installed-version CLI syntax during implementation, because flags and output formats are version-sensitive.

The invocation must:

- select the computed `opencode_model`;
- use the generated config through OpenCode's supported config override mechanism, such as `OPENCODE_CONFIG` or the documented equivalent;
- pass the plugin prompt as the task/user prompt;
- preserve the existing agent system prompt separately for agent targets;
- avoid interactive confirmation prompts and TUI output;
- capture stdout and stderr separately;
- use an argument list, not a shell command string, to avoid quoting/injection issues.

The installed OpenCode CLI's supported machine-readable mode should be checked during implementation. The benchmark uses `--format json` and extracts the final assistant text from the NDJSON event stream without including progress/event wrappers in the score input; stderr is retained as diagnostics. Every invocation also uses `--pure` so external OpenCode plugins cannot change the benchmark's tools, prompts, permissions, or event stream.

Thinking content: every invocation passes `--thinking`, because non-interactive `opencode run` defaults the `thinking` option to false and without it the CLI never emits `reasoning` NDJSON events — thinking-capable models' chain-of-thought would be silently dropped even though the provider returned it. The adapter joins `reasoning` events into the same `think_text` the HTTP runner accumulates from `reasoning_content`, so OpenCode results produce the same `{plugin}.think.txt` and `<thinking>…</thinking>`-wrapped `{plugin}.txt` sidecars as HTTP under `--save-responses`. Preflight therefore requires the installed CLI to advertise `--thinking`.

### Prompt handling

- For ordinary model targets, the plugin prompt is the OpenCode task prompt.
- For existing ai-benchmark agent targets, the `system_prompt` must be supplied as a separate system-level context using the OpenCode-supported mechanism.
- The plugin prompt must remain the user/task content and must not be silently replaced by the agent prompt.
- Prompt serialization must preserve Unicode and newlines.

### Timeout and cancellation

Each subprocess is governed by the existing benchmark timeout policy. The implementation must:

- enforce a hard timeout per plugin process;
- terminate the process cleanly when the timeout expires;
- terminate the process tree or child processes as needed so OpenCode cannot remain running in the background;
- honor the benchmark stop event/interrupt path;
- capture any partial stdout available before termination;
- mark the result as failed/truncated with an actionable error rather than hanging the worker;
- ensure generated config and process resources are closed even on timeout, cancellation, or worker exceptions.

OpenCode-specific process timing should be recorded separately from direct HTTP timing where possible. Do not claim direct-path TTFT or streaming metrics when OpenCode only supplies a final buffered response.

## 9. Scoring and result semantics

The final stdout assistant response is passed through the existing plugin evaluator unchanged. Existing rubric and score behavior must not be duplicated for OpenCode.

For every OpenCode plugin result, preserve the existing common metrics where they are meaningful:

- score and rubric;
- response time;
- output token estimate;
- error/failure status;
- truncation/repetition indicators when determinable;
- runner metadata;
- original target/source/api model metadata.

Metrics that cannot be measured through OpenCode must be explicit (`null`, unavailable, or a documented approximation). Do not infer a first-token timestamp from process completion and label it as TTFT.

Capture separate artifacts:

```text
<output_dir>/opencode/responses/<target>/<plugin>.txt
<output_dir>/opencode/logs/<target>/<plugin>.stdout.txt
<output_dir>/opencode/logs/<target>/<plugin>.stderr.txt
<output_dir>/opencode/... metadata as supported by existing save-response behavior
```

The exact layout should follow existing sanitization conventions and must prevent target/plugin names from escaping the runner directory.

## 10. Resume and state behavior

Resume must be runner-aware.

- A successful `http` result must never satisfy an `opencode` target/plugin on resume.
- A successful `opencode` result must never satisfy an `http` target/plugin on resume.
- In `both` mode, each runner variant is independently reusable.
- If a prior run contains only one runner variant and the next run requests `both`, schedule only the missing variant plus any failed/missing plugins.
- If the same target/plugin has changed runner-specific mapping or generated-config inputs, the implementation should invalidate/re-run the OpenCode result rather than reuse stale output. At minimum record a runner/config fingerprint or equivalent metadata to support this decision.
- `--restart` continues to discard all prior state as it does today.
- Existing `--no-rerun-failed` semantics must apply independently to each runner variant.

The state schema and `latest_results()` logic must be extended carefully because current state identity is primarily keyed by model/target name. A runner field or composite internal identity is required to prevent cross-runner result reuse and collisions.

## 11. Validation and error handling

Preflight must validate all conditions that can be checked without running a model:

1. `--runner` is a recognized value.
2. OpenCode is on `PATH` for `opencode`/`both`.
3. All resolved targets have usable source and `api_model` values for OpenCode.
4. Source slugification produces non-empty, valid identifiers.
5. Computed OpenCode model mappings do not collide.
6. Every required source can be projected into the generated OpenCode config.
7. The generated config is valid JSON/JSONC and can be written to the selected output location.
8. The OpenCode command/config override mechanism is available for the supported CLI version.

Preflight errors must be actionable and must occur before target scheduling. Error messages should identify the target/source field involved, but must redact credentials.

Runtime errors should distinguish at least:

- executable launch failure;
- config generation/lookup failure;
- OpenCode non-zero exit;
- timeout/cancellation;
- empty stdout;
- malformed structured output, if structured output is used internally;
- plugin evaluation failure.

These errors should flow through the existing named result/error contract and be persisted in runner-specific metadata/logs.

## 12. Tests required

Add or update tests without weakening current HTTP behavior.

### CLI and preflight

- default runner is `http` and preserves existing behavior;
- `--runner http`, `--runner opencode`, and `--runner both` parse correctly;
- invalid runner values are rejected;
- OpenCode is not checked for `http` mode;
- missing `opencode` fails before any target worker starts for `opencode` and `both`;
- OpenCode preflight does not make an HTTP request before failure;
- selected runner is recorded in run metadata.

### Mapping

- strict slugification for spaces, punctuation, repeated separators, leading/trailing separators, and mixed case;
- literal preservation of `api_model`, including an existing provider slash;
- expected `{source}/{api_model}` construction;
- empty-source/empty-model rejection;
- mapping collision detection.

### Config generation

- all required source/provider mappings are generated;
- endpoint and auth/header projection is correct;
- full-projection fields are mapped or explicitly rejected/warned as specified;
- generated config is valid and written in the runner namespace;
- exact generated config is retained;
- secrets are not printed to command output/log messages;
- permission/cleanup behavior is tested as far as the platform permits.

### Invocation and process handling

- subprocess is invoked with an argument list and the mapped model;
- `OPENCODE_CONFIG` (or final supported override) points to the generated config;
- plugin prompt is passed correctly;
- system prompt is preserved separately for agents;
- stdout becomes the scored response and stderr is stored separately;
- non-zero exit, empty output, timeout, cancellation, and partial output are handled;
- child process cleanup is attempted on timeout/cancellation.

### Runner isolation and resume

- `http` and `opencode` artifacts use separate namespaces;
- `both` uses one source worker and runs each target OpenCode then HTTP with no same-source overlap;
- both variants appear distinctly in results/reports;
- one runner's result cannot satisfy the other runner on resume;
- requesting `both` after an HTTP-only run schedules the missing OpenCode variant;
- failed/missing plugin rerun behavior remains correct per runner.

### Regression coverage

Run the existing full test suite and the relevant type/compile/lint checks. Existing direct HTTP tests must continue to pass unchanged except where assertions intentionally gain runner metadata.

## 13. Documentation requirements

Update user-facing documentation during implementation, not in this spec-only change:

- CLI reference with `--runner` values, default, ordering, and preflight behavior;
- configuration/architecture docs explaining that no OpenCode config block is required;
- OpenCode installation prerequisite and supported version expectations;
- source-to-OpenCode mapping examples;
- generated config location and credential-retention warning;
- output/report namespaces and runner-aware resume behavior;
- limitations around OpenCode streaming, TTFT, retries, and unsupported source settings.

The `--dump-default-config` output should not add an OpenCode configuration block because runner selection is intentionally CLI-only.

## 14. Version-sensitive adapter assumptions

The adapter uses the OpenCode contract available to the implementation: `OPENCODE_CONFIG` selects the retained generated config, `opencode run` performs a non-interactive invocation, `--pure` disables external plugins, `--model` selects the mapped provider/model, `--agent` selects generated agent context, and `--format json` supplies an NDJSON event stream from which final text is extracted for scoring. Operators should verify these flags against their installed OpenCode release when upgrading OpenCode; no automatic installation or upgrade is performed.

The benchmark's source configuration is projected into OpenCode's OpenAI-compatible provider shape. HTTP-only controls such as direct streaming telemetry and 429 retry bookkeeping are not fabricated for subprocess results; OpenCode response time and estimated output tokens are recorded instead.

## 15. Open questions to resolve during implementation

These items were intentionally left version-sensitive or implementation-dependent:

1. Which OpenCode release/version range is supported?
2. What exact non-interactive `opencode run` flags are available in that support range?
3. What exact config override mechanism is stable (`OPENCODE_CONFIG`, inline content, or another supported option)?
4. How does the installed OpenCode version represent providers, custom base URLs, headers, model registration, permissions, timeout, and token limits?
5. What exact mechanism preserves a separate system prompt for a one-shot `opencode run` invocation?
6. Does the chosen OpenCode version offer a reliable plain-final-text mode, or must a JSON/event stream be parsed before scoring?
7. Does OpenCode support source names normalized into arbitrary provider IDs, or will the generated config need provider aliases that make the requested `{source}/{model}` mapping resolvable?
8. How should direct and OpenCode result rows be represented in each output plugin (Markdown, CSV, HTML, PDF) while retaining backwards compatibility?
9. What process-group termination strategy is portable across supported operating systems?
10. Should a retained exact generated config be opt-in despite the interview decision, or is the security trade-off intentionally part of the required feature?

These questions must be answered by checking the exact OpenCode version/documentation used by the implementation rather than guessed from generic CLI conventions.

## 16. Definition of done

The feature is complete when:

- a user can run the existing benchmark unchanged with no new flags;
- a user can choose `--runner opencode` and run all configured models/agents through a pre-installed OpenCode CLI;
- a user can choose `--runner both` and receive OpenCode-first plus HTTP results without collisions;
- OpenCode configuration is generated from the loaded benchmark config at runtime;
- model mapping follows the exact normalized-source/API-model rule;
- missing OpenCode fails before benchmark work starts;
- prompts, credentials, logs, responses, timing, failures, and runner metadata follow this spec;
- resume never cross-reuses HTTP and OpenCode results;
- generated config artifacts and their security implications are documented;
- automated tests cover preflight, mapping, generation, invocation, lifecycle, namespaces, and resume behavior;
- the full existing test suite and relevant quality checks pass.
