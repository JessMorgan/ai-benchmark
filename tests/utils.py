"""Shared utilities for tests."""
import json


class MockResponse:
    """Minimal mock for requests.Response used in tests."""

    def __init__(self, text="", status_code=200):
        self.status_code = status_code
        if text:
            self._text = text
        else:
            self._text = json.dumps({
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "usage": {},
            })

    def iter_lines(self, decode_unicode=False):
        return []

    def iter_content(self, chunk_size=8192):
        return [self.content]

    def close(self):
        pass

    @property
    def text(self):
        return self._text

    @property
    def content(self):
        return self._text.encode("utf-8")


def load_benchmark_module():
    """Load benchmark_core.py as a module.

    benchmark_core.py is importable directly, so this just imports it.
    Kept as a helper in case the import path ever changes.
    """
    import benchmark_core
    return benchmark_core
