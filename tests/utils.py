"""Shared utilities for tests."""


def load_benchmark_module():
    """Load benchmark_core.py as a module.

    benchmark_core.py is importable directly, so this just imports it.
    Kept as a helper in case the import path ever changes.
    """
    import benchmark_core
    return benchmark_core
