# OpenCode runner architecture comparison

## Current implementation: one `opencode run` process per task

The benchmark currently launches an isolated subprocess for each model/plugin cell:

```text
opencode run --pure --model ... --format json --thinking --agent ... <prompt>
```

It captures stdout and stderr, parses OpenCode's NDJSON event stream, extracts text and reasoning, scores the answer, and terminates the process.

### Advantages

- **Strong isolation:** each cell gets a fresh OpenCode session. Conversation history, tool state, and agent state cannot leak from one benchmark task into another.
- **Simple lifecycle:** start one process, wait for it, collect output, and terminate it.
- **Easy failure containment:** the benchmark can kill the entire process tree on timeout, cancellation, staleness, step loops, or repeated text.
- **Reproducibility:** `--pure`, an explicit generated config, explicit model, and explicit agent make the invocation self-contained.
- **Good fit for benchmark semantics:** each plugin is naturally an independent task.
- **No daemon management:** there is no server to start, monitor, clean up, or accidentally leave running.
- **Already machine-readable:** `--format json` provides a structured event stream; using an API would not fundamentally improve the event format.

### Disadvantages

- **Process startup overhead:** every plugin task starts a new OpenCode process and initializes its session.
- **Repeated setup:** config loading, agent setup, and session initialization happen for every cell.
- **More difficult high concurrency:** many simultaneous cells mean many subprocesses and potentially many OpenCode runtimes.
- **CLI contract dependency:** the adapter must preflight flags such as `--pure`, `--thinking`, `--agent`, and `--format json`, and CLI releases can change those flags.
- **Limited mid-run control:** after starting the process, the benchmark mostly observes stdout/stderr and can kill the process. It cannot easily send a structured follow-up, change options, inspect session state, or request a graceful abort.
- **Prompt as a process argument:** the prompt is passed on the command line. The code uses an argument list, so shell interpretation is avoided, but very large prompts can encounter OS argument-length limits and the prompt may be visible in process listings or `/proc` while the command runs.
- **Custom monitoring burden:** the benchmark has to implement its own NDJSON reader, staleness detection, step limits, repetition detection, and process-tree cleanup.

## `opencode serve` plus API

With the server model, the benchmark would start one long-lived OpenCode server and submit tasks through its HTTP API. Each cell would likely create a fresh session, send a prompt, subscribe to events, and explicitly abort or delete the session when finished.

### Advantages

- **Much lower per-task startup cost:** one server can handle many cells without launching a new process each time.
- **Structured request transport:** prompts and options go in JSON request bodies instead of command-line arguments.
- **Fewer prompt-size concerns:** large prompts are not constrained by OS command-line argument limits or exposed in process listings.
- **Better interactive control:** an API can potentially support session creation, explicit model and agent selection, event subscriptions, status inspection, graceful cancellation/abort, follow-up messages, and server health checks.
- **Potentially better streaming integration:** instead of parsing a child process's stdout, the benchmark could consume the server's event stream directly.
- **Efficient concurrency:** a server can manage a bounded pool of sessions and connections more efficiently than repeatedly spawning processes.
- **Centralized diagnostics:** server-level logs and session identifiers could make it easier to correlate requests, events, errors, and aborts.

### Disadvantages

- **Session isolation becomes the benchmark's responsibility:** using one session for multiple plugins would be wrong because task prompts and answers could contaminate each other. The benchmark would need a new session per cell, or a rigorously reset session.
- **Long-lived state can leak:** OpenCode may retain conversation history, tool state, caches, agent state, or provider state unless sessions are explicitly isolated and cleaned up.
- **More complicated lifecycle management:** the benchmark must start the server, wait for readiness, handle startup failure, detect crashes, stop it on exit, and avoid orphaned servers.
- **Harder recovery semantics:** if the server crashes halfway through a task, the benchmark must determine whether the session can be resumed, whether the response is partial, and whether a new server/session would change the result.
- **API compatibility risk:** the benchmark becomes dependent on OpenCode's server endpoints, request schemas, event schemas, session semantics, and abort behavior. These may evolve independently of the CLI.
- **Unclear option mapping:** CLI flags such as `--pure` and `--thinking` may not map one-to-one to server request fields or config options. Model, agent, output limits, tool permissions, and reasoning-event behavior would need explicit compatibility tests.
- **Concurrency interactions:** OpenCode's own scheduler and session limits could interact with the benchmark's `model_thread_limit` and `plugin_thread_limit`. The benchmark would need to prevent double-queuing or hidden serialization.
- **Potential cross-cell impact from a bad session:** a leaked context or unclosed session could affect many later tasks, whereas the subprocess design confines the damage to one cell.
- **More difficult debugging:** failures could occur in the benchmark, the HTTP connection, the OpenCode server, the session, the provider, or the server's internal event broker.

## Important distinction

The current implementation is already using OpenCode's structured event output:

```text
opencode run --format json
```

Therefore, the primary difference is not unstructured CLI output versus a structured API. It is the transport and lifecycle model:

| Concern | Current subprocess | `opencode serve` API |
|---|---|---|
| Task isolation | Automatic | Must be designed explicitly |
| Startup cost | High per cell | Low per cell after server startup |
| Prompt transport | CLI argument | JSON request |
| Streaming | Child stdout NDJSON | API event stream |
| Cancellation | Kill process tree | API abort plus server cleanup |
| Session reuse | None | Available, but dangerous for benchmarks |
| Lifecycle | Simple per task | More complex long-lived daemon |
| Failure scope | Usually one cell | Potentially many cells |
| Concurrency | Process-based | Server/session-based |
| Contract surface | CLI flags plus event format | Server endpoints, schemas, and events |

## Recommendation for this benchmark

The current subprocess approach is the safer correctness-first design because benchmark cells are independent and isolation is valuable. It also makes timeout and runaway-loop handling unambiguous.

A server API becomes attractive if OpenCode startup overhead materially affects benchmark duration or if richer live control is needed. If adopted, the recommended design is:

1. Start a dedicated server for a benchmark run, or at least one server per source/config.
2. Create a fresh session for every model/plugin cell.
3. Never reuse a session across plugins.
4. Record server version, session ID, request ID, and API event status in metadata.
5. Enforce benchmark concurrency explicitly rather than relying on OpenCode defaults.
6. Add health checks and guaranteed shutdown.
7. Add contract tests for model selection, agent selection, thinking events, max-output limits, cancellation, and session isolation.
8. Keep the subprocess runner available as a fallback.

A hybrid implementation could use the API for normal execution but fall back to the current isolated subprocess path when the server API lacks a required capability or fails a compatibility probe.
