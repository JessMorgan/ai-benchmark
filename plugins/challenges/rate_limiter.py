"""Executable concurrent rate-limiter challenge."""
from __future__ import annotations

import ast
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions


class RateLimiterPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "rate-limiter"

    @property
    def version(self) -> str:
        return "1.2.0"

    @property
    def name(self) -> str:
        return "Rate Limiter"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Implement these exact Python classes using only the standard library:\n\n"
            "TokenBucket(limit: int, window_seconds: float), SlidingWindowLog(limit: int, "
            "window_seconds: float), and FixedWindow(limit: int, window_seconds: float).\n"
            "Each class must expose:\n"
            "- allow_request(client_id: str, now: float) -> bool\n"
            "- get_usage_stats(client_id: str) -> dict\n"
            "- cleanup(now: float) -> int\n\n"
            "The first limit requests in a window are allowed and later requests are denied; "
            "time advances through the supplied `now` argument so behavior is deterministic. "
            "Support independent client IDs, zero/negative configuration validation, stale-entry "
            "cleanup, and thread safety. Include docstrings and type hints. Return only the code."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("rate_limiter_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _classes(tree: ast.AST | None) -> set[str]:
        return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)} if tree else set()

    @staticmethod
    def _inherits(node: ast.ClassDef, base_name: str) -> bool:
        """Return whether a class inherits the named base class directly."""
        return any(
            isinstance(base, ast.Name) and base.id == base_name
            for base in node.bases
        )

    def evaluate(self, response_text: str) -> EvaluationResult:
        rubric = Rubric(self.max_score)
        if not response_text or not response_text.strip():
            return rubric.results()
        text = response_text.strip()
        validation = parse_python(text)
        rubric.record_validation(validation)
        classes = self._classes(validation.value)
        required = {"TokenBucket", "SlidingWindowLog", "FixedWindow"}
        present = classes & required
        rubric.add_criterion(
            "Strategy API contract", 3.0, 3.0 * len(present) / 3.0,
            evidence=[{"kind": "class", "name": name} for name in sorted(present)],
            negative_findings=[{"finding": f"missing strategy: {name}"} for name in sorted(required - present)],
        )

        methods = {
            node.name
            for node in ast.walk(validation.value)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        } if validation.valid else set()
        method_hits = sum(
            method in methods
            for method in ("allow_request", "get_usage_stats", "cleanup")
        )
        rubric.add_criterion("Shared method contract", 1.0, 1.0 * method_hits / 3.0)

        class_nodes = {
            node.name: node
            for node in ast.walk(validation.value)
            if isinstance(node, ast.ClassDef)
        } if validation.valid else {}
        base_text = ast.unparse(class_nodes["_Base"]) if "_Base" in class_nodes else ""
        for name in ("TokenBucket", "SlidingWindowLog", "FixedWindow"):
            node = class_nodes.get(name)
            class_text = ast.unparse(node) if node is not None else ""
            strategy_text = class_text + (
                base_text if node is not None and self._inherits(node, "_Base") else ""
            )
            points = 1.0 if name in present else 0.0
            if name == "TokenBucket":
                points += 1.0 if re.search(r"refill|token|capacity", strategy_text, re.IGNORECASE) else 0.0
            elif name == "SlidingWindowLog":
                points += 1.0 if re.search(r"deque|timestamp|bisect|window", strategy_text, re.IGNORECASE) else 0.0
            else:
                points += 1.0 if re.search(r"counter|window|reset", strategy_text, re.IGNORECASE) else 0.0
            rubric.add_criterion(name, 1.0, points / 2.0)

        thread_hits = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in (
            r"threading\.(?:Lock|RLock)", r"with\s+\w+|\.acquire\(",
        ))
        rubric.add_criterion("Thread safety", 1.0, min(1.0, float(thread_hits) / 2.0))
        cleanup_hits = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in (
            r"cleanup|clean_up|remove_stale|expire|ttl", r"pop|del\s+|remove",
        ))
        rubric.add_criterion("Cleanup/memory management", 1.0, min(1.0, float(cleanup_hits) / 2.0))
        quality_hits = sum(bool(re.search(pattern, text)) for pattern in (r"->\s*(?:bool|dict|int)", r"\"\"\""))
        rubric.add_criterion("Types and documentation", 1.0, min(1.0, float(quality_hits) / 2.0))

        stubs = stub_definitions(validation.value, required) if validation.valid else []
        stubs = [
            name for name in stubs
            if name not in class_nodes or not self._inherits(class_nodes[name], "_Base")
        ]
        if stubs:
            for criterion in ("TokenBucket", "SlidingWindowLog", "FixedWindow"):
                if criterion in stubs:
                    rubric.penalize_criterion(criterion, 1.0, f"{criterion} is a placeholder")

        source = extract_python_source(text)
        if source:
            harness = r'''
import inspect
import threading

_classes = [TokenBucket, SlidingWindowLog, FixedWindow]
for _cls in _classes:
    _instance = _cls(limit=2, window_seconds=10.0)
    assert isinstance(_instance.allow_request("a", 0.0), bool)
    assert _instance.allow_request("a", 0.0) is True
    assert _instance.allow_request("a", 0.0) is False
    assert _instance.allow_request("b", 0.0) is True
    assert isinstance(_instance.get_usage_stats("a"), dict)
    assert isinstance(_instance.cleanup(100.0), int)
    try:
        _cls(limit=0, window_seconds=10.0)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("invalid limit must be rejected")

_instance = TokenBucket(limit=100, window_seconds=10.0)
_results = []
def _call():
    _results.append(_instance.allow_request("concurrent", 0.0))
_threads = [threading.Thread(target=_call) for _ in range(16)]
for _thread in _threads:
    _thread.start()
for _thread in _threads:
    _thread.join(timeout=1)
    assert not _thread.is_alive()
assert len(_results) == 16
'''
            execution = run_python_check(source, harness)
            rubric.add_criterion(
                "Behavioral strategy tests", 10.0,
                10.0 if execution.status == "passed" else 0.0,
                evidence=[{"kind": "execution", "status": execution.status, "isolation": execution.isolation}],
                negative_findings=[] if execution.status == "passed" else [{"finding": execution.error or execution.status}],
            )
        else:
            rubric.add_criterion("Behavioral strategy tests", 10.0, 0.0, negative_findings=[{"finding": "no executable source"}])

        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
