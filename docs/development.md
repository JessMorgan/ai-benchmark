# Development Guide

This guide covers how to set up a development environment, run tests, write plugins, and contribute to AI Benchmark.

## Development Setup

Install dependencies:

```sh
pip install -r requirements.txt
```

For linting and type checking:

```sh
pip install ruff mypy
```

## Project Layout

```
.
├── ai-benchmark.py      # CLI entry point and TUI
├── benchmark_core.py    # Core benchmark logic, state, output generators
├── benchmark_plugin.py  # Abstract base class for plugins
├── shell_completion.py  # Shell completion script generation
├── plugins/             # Built-in benchmark task plugins
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Running Tests

Use `pytest`:

```sh
python -m pytest tests/ -v
```

To run a specific test file:

```sh
python -m pytest tests/test_cli.py -v
```

## Type Checking

Run `mypy` on the core modules:

```sh
python -m mypy benchmark_core.py ai-benchmark.py
```

## Linting

Run `ruff`:

```sh
ruff check .
```

## Writing a Plugin

1. Create a new file in `plugins/` (e.g. `plugins/my_task.py`).
2. Define a class that inherits from `BenchmarkTaskPlugin`.
3. Implement the required properties and methods.
4. Optionally override `supports_streaming`.

Minimal example:

```python
"""My custom benchmark task."""
from benchmark_plugin import BenchmarkTaskPlugin


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

    def score(self, response_text):
        s = 0.0
        if "def " in response_text:
            s += 5.0
        return min(s, self.max_score)
```

## Plugin Temperature

Plugins can read a temperature from `global_config`. The convention is:

```json
"my-task_temperature": 0.2
```

Use `global_config.get("my-task_temperature")` in `get_temperature()`.

## Testing Plugins

Add tests in `tests/test_scoring.py` or create a new test file. Use the plugin's `score()` method directly with sample responses.

Example:

```python
def test_my_task_scores_function():
    from plugins.my_task import MyTaskPlugin
    plugin = MyTaskPlugin()
    assert plugin.score("def foo(): pass") == 5.0
```

## Adding Tests for Core Changes

Core changes in `benchmark_core.py` or `ai-benchmark.py` should include tests in `tests/test_cli.py` or `tests/test_output.py`.

## Pre-Commit Hooks

The project uses `pre-commit`. Install hooks with:

```sh
pre-commit install
```

## Contribution Checklist

- [ ] Tests pass (`python -m pytest tests/ -v`)
- [ ] Type checks pass (`python -m mypy ...`)
- [ ] Lint passes (`ruff check .`)
- [ ] New plugins include documentation in `docs/plugins/`
- [ ] README or docs updated if user-facing behavior changed
