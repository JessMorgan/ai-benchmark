"""Executable idempotent event-processing challenge."""
from __future__ import annotations

import ast
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions


class EventProcessorPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "event-processor"

    @property
    def version(self) -> str:
        return "0.5.0"

    @property
    def name(self) -> str:
        return "Concurrent Event Processor"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Implement exactly `class EventProcessor` using only the standard library.\n\n"
            "Constructor: `EventProcessor(handler: Callable[[dict], None], max_workers: int = 4, "
            "max_retries: int = 2)`.\n"
            "Method: `process(events: list[dict]) -> dict` where each event has a unique-looking "
            "string `id`. Return `{processed: list[str], duplicates: list[str], failed: list[str]}` "
            "with IDs in their original first-seen order. Invoke the handler concurrently for "
            "unique events, retry a failed handler up to max_retries, invoke it at most once "
            "successfully per ID, and put permanently failed IDs in `failed`. Duplicate IDs "
            "must never invoke the handler again. Idempotency is instance-scoped: an ID that "
            "has already been processed successfully by this instance is reported in "
            "`duplicates` on any later `process()` call and must not re-invoke the handler. "
            "Protect shared state.\n\n"
            "Validation semantics (must raise, never silently skip): malformed events — an "
            "element of `events` that is not a dict, or a dict whose `id` is missing or is not "
            "a non-empty string — must raise `TypeError` or `ValueError`. Invalid constructor "
            "arguments must raise `TypeError` for wrong types (including a `handler` that is "
            "not callable) or `ValueError` for out-of-range values (`max_workers < 1`, "
            "including negatives, `max_retries < 0`). Do not skip, ignore, or drop malformed "
            "input.\n\n"
            "Include type hints and docstrings. Return code only."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("event_processor_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def evaluate(self, response_text: str) -> EvaluationResult:
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
            validation_harness = r'''
import threading

_calls = []
_lock = threading.Lock()
def _handler(event):
    with _lock:
        _calls.append(event["id"])

_reject = []
def _must_raise(label, fn, expected):
    try:
        fn()
    except expected:
        return
    except Exception as exc:
        _reject.append(label + ": raised " + type(exc).__name__ + " instead of " + "/".join(e.__name__ for e in expected))
        return
    _reject.append(label + ": accepted invalid input")

_processor = EventProcessor(_handler, max_workers=4, max_retries=2)
_must_raise("non-dict event", lambda: _processor.process(["not-a-dict"]), (TypeError, ValueError))
_must_raise("non-string id", lambda: _processor.process([{"id": 42}]), (TypeError, ValueError))
_must_raise("empty id", lambda: _processor.process([{"id": ""}]), (TypeError, ValueError))
_must_raise("missing id", lambda: _processor.process([{"value": 1}]), (TypeError, ValueError))
_must_raise("max_workers=0", lambda: EventProcessor(_handler, max_workers=0, max_retries=2), (ValueError,))
_must_raise("max_workers=-2", lambda: EventProcessor(_handler, max_workers=-2, max_retries=2), (ValueError,))
_must_raise("max_retries=-1", lambda: EventProcessor(_handler, max_workers=4, max_retries=-1), (ValueError,))
_must_raise("max_workers wrong type", lambda: EventProcessor(_handler, max_workers="4", max_retries=2), (TypeError,))
_must_raise("non-callable handler", lambda: EventProcessor(None, max_workers=4, max_retries=2), (TypeError,))
assert not _reject, "; ".join(_reject)
'''
            validation_exec = run_python_check(source, validation_harness)
            # Gate on harness_ok (passed AND sentinel), not status == "passed": a
            # response that exits 0 before the harness runs must not pass.
            rubric.add_criterion("Validation semantics", 4.0,
                4.0 if validation_exec.harness_ok else 0.0,
                evidence=[validation_exec.as_evidence()],
                negative_findings=[] if validation_exec.harness_ok else [{"finding": validation_exec.error or "harness did not complete (missing completion sentinel)"}],
            )

            behavioral_harness = r'''
import threading

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

_processor = EventProcessor(_handler, max_workers=4, max_retries=2)
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

_second = _processor.process([{"id": "a", "value": 1}, {"id": "fresh", "value": 9}])
assert _second["duplicates"] == ["a"]
assert "a" not in _second["processed"]
assert "fresh" in _second["processed"]
assert _calls.count("a") == 1
'''
            behavioral_exec = run_python_check(source, behavioral_harness)
            rubric.add_criterion("Behavioral event tests", 8.0,
                8.0 if behavioral_exec.harness_ok else 0.0,
                evidence=[behavioral_exec.as_evidence()],
                negative_findings=[] if behavioral_exec.harness_ok else [{"finding": behavioral_exec.error or "harness did not complete (missing completion sentinel)"}],
            )
        else:
            rubric.add_criterion("Validation semantics", 4.0, 0.0, negative_findings=[{"finding": "no executable source"}])
            rubric.add_criterion("Behavioral event tests", 8.0, 0.0, negative_findings=[{"finding": "no executable source"}])
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
