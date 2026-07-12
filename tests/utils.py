"""Shared utilities for tests."""
import importlib.util


def load_benchmark_module():
    """Load ai-benchmark.py as a module using importlib.

    The file has a hyphen in its name, so it cannot be imported directly.
    """
    spec = importlib.util.spec_from_file_location("ai_benchmark", "ai-benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
