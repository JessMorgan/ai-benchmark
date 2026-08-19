# Development Guide

This guide covers how to set up a development environment, run tests, write plugins, and contribute to AI Benchmark.

## Development Setup

Install the project and its dev dependencies (pytest, mypy, ruff, ...) from
`pyproject.toml` — `uv sync` installs the project editable plus the `dev`
dependency group, resolving exactly the tree pinned in `uv.lock`:

```sh
uv sync
```

Run any project tool through the managed environment with `uv run`, e.g.
`uv run pytest`, `uv run mypy`, or `uv run ai-benchmark`.

To refresh the lock file against newer releases: `uv lock --upgrade`, then
`uv sync` (see the `README` for the test commands).

## Project Layout

```
.
├── ai-benchmark.py      # Thin launcher -> benchmark.cli.main
├── benchmark/           # Core library package
│   ├── cli.py           # CLI entry point, argparse, TUI
│   ├── core.py          # Core benchmark logic, state, output generators
│   ├── http.py          # Streaming / non-streaming HTTP request helpers
│   ├── plugin.py        # Abstract base classes for plugins
│   ├── outputs.py       # Markdown/CSV/HTML/PDF report generators
│   ├── completions.py   # Shell completion script generation
│   └── opencode.py      # Optional OpenCode subprocess runner
├── plugins/             # Built-in benchmark task plugins
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Running Tests

Use `pytest`:

```sh
uv run pytest tests/ plugins/challenges/ plugins/outputs/ -q
```

To run a specific test file:

```sh
uv run pytest tests/test_cli.py -v
```

## Type Checking

Run `mypy` on the core modules:

```sh
uv run mypy benchmark/ ai-benchmark.py
```

## Linting

Run `ruff`:

```sh
ruff check .
```

## Investigating EOF / Stream-Abort Failures

A `litellm.APIConnectionError: Ollama_chatException - EOF` (or a stream that
ends without `[DONE]`/`finish_reason`) is the proxy reporting that the model
backend dropped the connection mid-generation. The benchmark already surfaces
this as `stream_error` in the plugin's `meta.json` and records the replayable
curl + response body in `logs/<model>.log`, so the first step is to reproduce
one failure in isolation. The EOF itself is a symptom; pin the layer that owns
it before changing benchmark code:

1. **Which layer raised it?** Re-run the logged curl directly against the
   model server (bypassing litellm). If raw Ollama returns cleanly, the proxy
   is the failing layer; if Ollama itself reports `{'error': 'EOF'}`, the model
   process is dying or being killed.
2. **Ollama server logs.** On the model host check `journalctl -u ollama -n 200`
   (or the equivalent service log) for OOM kills, `context length exceeded`,
   panic, or loader errors around the failure timestamps.
3. **Correlate with the request shape.** Note whether EOFs cluster on
   long-context prompts, thinking-capable models, `response_format`/judge
   requests, or specific plugins. A shape-specific cluster points at
   context-window or provider-parameter problems rather than hardware.
4. **Concurrency / VRAM pressure.** Re-run the failing request serially
   (`plugin_thread_limit: 1`, `model_thread_limit: 1`). EOFs that disappear
   under serial load point at resource exhaustion, not the model itself.
5. **Timing.** Compare `response_time` and `stream_error` in `meta.json`;
   a 2–3 s abort is a near-instant backend failure, while a long delay before
   EOF suggests a timeout or watchdog close.

Record the outcome of each probe in the run's notes so a repeated EOF can be
attributed to the proxy, the model server, or the benchmark request itself.

## Writing a Plugin

1. Create a new module in `plugins/challenges/` (for example, `plugins/challenges/my_task.py`).
2. Define a class that inherits from `BenchmarkTaskPlugin`.
3. Implement the required properties and methods.
4. Optionally override `supports_streaming`; challenge modules are discovered automatically, so do not add manual registration.

Minimal example:

```python
"""My custom benchmark task."""
from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult


class MyTaskPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "my-task"

    @property
    def version(self):
        return "1.0.0"

    @property
    def name(self):
        return "My Task"

    @property
    def max_score(self):
        return 10.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return "Write a Python function that..."

    def get_temperature(self, global_config):
        return global_config.get("my-task_temperature", 0.2)

    @property
    def judge_instructions_version(self):
        return "1.0.0"

    def get_judge_instructions(self):
        return ""

    def evaluate(self, response_text):
        s = 0.0
        if "def " in response_text:
            s += 5.0
        return EvaluationResult(min(s, self.max_score), [])

    def score(self, response_text):
        return self.evaluate(response_text).score

    # The benchmark core converts this native score to an integer percentage
    # for public results and reports. Persisted rubric entries use points/total.
```

## Plugin Temperature

Plugins can read a temperature from `global_config`. The convention is:

```json
"my-task_temperature": 0.2
```

Use `global_config.get("my-task_temperature")` in `get_temperature()`.

## Testing Plugins

Add tests under `plugins/challenges/test_*.py` or `tests/`. Every challenge should have empty, partial, adversarial, and complete fixtures. For document tasks, assert section-local and contradiction behavior. For code-generation tasks, define an exact API and run pytest-compatible assertion tests through `run_python_check`; Podman is preferred and `local-restricted` fallback is explicitly recorded. Do not use dated benchmark-run paths as fixtures.

Example:

```python
def test_my_task_scores_function():
    from plugins.my_task import MyTaskPlugin
    plugin = MyTaskPlugin()
    assert plugin.score("def foo(): pass") == 5.0
```

## Adding Tests for Core Changes

Core changes in `benchmark/core.py` or `benchmark/cli.py` should include tests in `tests/test_cli.py` or `tests/test_output.py`.

## Pre-Commit Hooks

The project uses `pre-commit`. Install hooks with:

```sh
pre-commit install
```

## Contribution Checklist

- [ ] Tests pass (`uv run pytest tests/ plugins/challenges/ plugins/outputs/ -q`)
- [ ] Type checks pass (`uv run mypy benchmark/ ai-benchmark.py`)
- [ ] Lint passes (`ruff check .`)
- [ ] New plugins include documentation in `docs/plugins/`
- [ ] Code-generation plugins have executable API tests and adversarial contract fixtures
- [ ] No source or test directly references a dated historical benchmark directory
- [ ] README or docs updated if user-facing behavior changed
