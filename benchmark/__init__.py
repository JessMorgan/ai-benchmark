"""Core library package for the AI Benchmark project.

Modules that used to live in the repository root (``benchmark_core.py``,
``benchmark_http.py``, ...) now live here with short names::

    from benchmark.core import run_model, load_config
    from benchmark.http import stream_request
    from benchmark.state import BenchmarkState
    from benchmark.plugin import BenchmarkTaskPlugin
    from benchmark.opencode import run_process

The CLI entry point remains ``ai-benchmark.py`` at the repository root.
"""
