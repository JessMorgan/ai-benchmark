"""Executable idempotent event-processing challenge."""
from __future__ import annotations

import ast
import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions


class EventProcessorPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "event-processor"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "Concurrent Event Processor"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "Implement exactly `class EventProcessor` using only the standard library.\n\n"
            "Constructor: `EventProcessor(handler: Callable[[dict], None], max_workers: int = 4, "
            "max_retries: int = 2)`.\n"
            "Method: `process(events: list[dict]) -> dict` where each event has a unique-looking "
            "string `id`. Return `{processed: list[str], duplicates: list[str], failed: list[str]}` "
            "with IDs in their original first-seen order. Invoke the handler concurrently for "
            "unique events, retry a failed handler up to max_retries, invoke it at most once "
            "successfully per ID, and put permanently failed IDs in `failed`. Duplicate IDs "
            "must never invoke the handler again. Protect shared state and validate malformed "
            "events and invalid constructor arguments. Include type hints and docstrings. Return code only."
        )

    def get_temperature(self, global_config):
        return global_config.get("event_processor_temperature")

    def evaluate(self, response_text):
        rubric = Rubric(self.max_score)
        if not response_text or not response_text.strip():
            return rubric.results()
        text = response_text.strip()
        validation = parse_python(text)
        rubric.record_validation(validation)
        tree = validation.value
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)} if tree is not None else set()
        has_class = "EventProcessor" in classes
        rubric.add_criterion("Exact EventProcessor API", 3.0, 3.0 if has_class else 0.0,
                             negative_findings=[] if has_class else [{"finding": "EventProcessor class missing"}])
        concepts = [
            r"ThreadPoolExecutor|concurrent\.futures|asyncio",
            r"retry|max_retries|attempt",
            r"duplicate|dedup|seen|idempot",
            r"failed|dead.?letter|error",
            r"Lock|RLock|threading|synchron",
        ]
        concept_hits = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in concepts)
        rubric.add_criterion("Concurrent idempotent design", 2.0, 2.0 * concept_hits / len(concepts))
        type_hits = sum(bool(re.search(pattern, text)) for pattern in (r"->\s*dict", r"\"\"\""))
        rubric.add_criterion("Types and documentation", 1.0, float(type_hits))
        stubs = stub_definitions(validation.value, {"EventProcessor"}) if validation.valid else []
        rubric.add_criterion("Non-stub implementation", 2.0, 2.0 if has_class and not stubs else 0.0,
                             negative_findings=[{"finding": f"stub: {name}"} for name in stubs])

        source = extract_python_source(text)
        if source:
            harness = r'''
import threading
import time

_calls = []
_lock = threading.Lock()
_attempts = {}
def _handler(event):
    with _lock:
        _calls.append(event["id"])
        _attempts[event["id"]] = _attempts.get(event["id"], 0) + 1
    if event["id"] == "retry" and _attempts[event["id"]] == 1:
        raise RuntimeError("transient")
    if event["id"] == "bad":
        raise RuntimeError("permanent")

for _bad_args in ((0, 2), (4, -1)):
    try:
        EventProcessor(_handler, max_workers=_bad_args[0], max_retries=_bad_args[1])
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("invalid constructor arguments must be rejected")

_processor = EventProcessor(_handler, max_workers=4, max_retries=2)
try:
    _processor.process([{"value": 1}])
except (ValueError, TypeError):
    pass
else:
    raise AssertionError("malformed events must be rejected")

_result = _processor.process([
    {"id": "a", "value": 1}, {"id": "a", "value": 1},
    {"id": "retry", "value": 2}, {"id": "bad", "value": 3},
])
assert _result["processed"] == ["a", "retry"]
assert _result["duplicates"] == ["a"]
assert _result["failed"] == ["bad"]
assert _attempts["retry"] == 2
assert _attempts["bad"] == 3
assert _calls.count("a") == 1
assert _result["processed"] == ["a", "retry"]
'''
            execution = run_python_check(source, harness)
            rubric.add_criterion("Behavioral event tests", 12.0,
                12.0 if execution.status == "passed" else 0.0,

                evidence=[{"kind": "execution", "status": execution.status, "isolation": execution.isolation}],
                negative_findings=[] if execution.status == "passed" else [{"finding": execution.error or execution.status}],
            )
        else:
            rubric.add_criterion("Behavioral event tests", 12.0, 0.0, negative_findings=[{"finding": "no executable source"}])
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
