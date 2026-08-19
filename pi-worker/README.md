# AI Benchmark Pi worker

This directory contains the isolated Node worker used by `--runner pi`.

## Install

The worker requires Node.js **22.19 or newer** and the pinned Pi SDK packages
in `package.json`:

```sh
npm install
```

The Python benchmark runs `worker.mjs --preflight` at startup to verify that
Node can parse the worker and import the SDK. It then starts one fresh worker
process per `(target, plugin, attempt)` and sends one JSON request on stdin.
Prompts are not placed in command-line arguments.

## Protocol

Normal stdout is newline-delimited `pi-worker-v1` events. The worker emits
`worker_started`, session/prompt lifecycle events, reasoning and text deltas,
tool lifecycle events, usage, `finish`, and `worker_finished` (or `error`).
Diagnostics go to stderr. The Python adapter owns timeout, cancellation,
process-group termination, retry policy, scoring, and persistence.

Plain targets have an empty tool list. Tools are enabled only by an explicit
validated target `pi.tools` allowlist, and `pi.permissions` accepts only
`allow`/`deny`. No interactive approval is possible during a benchmark run.
