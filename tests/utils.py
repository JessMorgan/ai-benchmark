"""Shared utilities for tests."""


class MockResponse:
    """Minimal mock for requests.Response used in tests."""

    def __init__(self, text="", status_code=200):
        self.status_code = status_code
        self._text = text

    def iter_lines(self, decode_unicode=False):
        return []

    def close(self):
        pass

    @property
    def text(self):
        return self._text


def load_benchmark_module():
    """Load benchmark_core.py as a module.

    benchmark_core.py is importable directly, so this just imports it.
    Kept as a helper in case the import path ever changes.
    """
    import benchmark_core
    return benchmark_core
