"""Executable resilient multi-provider API challenge."""
from __future__ import annotations

import ast
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions


class ErrorRecoveryPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "error-recovery"

    @property
    def version(self) -> str:
        return "1.2.0"

    @property
    def name(self) -> str:
        return "Error Recovery"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Implement an async resilient weather fetcher with this exact API:\n\n"
            "class AllProvidersFailedError(Exception): ...\n"
            "class WeatherClient:\n"
            "    async def fetch(self, provider: str, city: str) -> dict: ...\n"
            "async def get_weather_resilient(city: str, client: WeatherClient) -> dict\n\n"
            "The function must attempt WeatherAPI, OpenMeteo, and VisualCrossing; treat an "
            "exception, timeout, malformed response, or a 200 response containing an error "
            "field as failure; return the first successful response unchanged; log every "
            "failure with provider and reason; and raise exactly AllProvidersFailedError "
            "with provider details when all fail. Provider calls must be concurrent.\n\n"
            "Also provide `async def demo()` showing all-success, partial-failure, and all-failure "
            "scenarios. Use only the standard library and include type hints/docstrings."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("error_recovery_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _classes(tree: ast.AST) -> set[str]:
        return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

    def evaluate(self, response_text: str) -> EvaluationResult:
        if not response_text or not response_text.strip():
            return EvaluationResult(0.0, [])
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        validation = parse_python(text)
        rubric.record_validation(validation)
        tree = validation.value if validation.valid else None
        classes = self._classes(tree) if tree is not None else set()
        functions = (
            {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if tree is not None else set()
        )
        required = {"AllProvidersFailedError", "WeatherClient", "get_weather_resilient", "demo"}
        present = required & (classes | functions)
        rubric.add_criterion(
            "Required API contract", 3.0, 3.0 * len(present) / len(required),
            evidence=[{"kind": "definition", "name": name} for name in sorted(present)],
            negative_findings=[{"finding": f"missing required definition: {name}"} for name in sorted(required - present)],
        )

        signature_hits = 0
        if tree is not None:
            async_defs = {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, ast.AsyncFunctionDef)
            }
            gr = async_defs.get("get_weather_resilient")
            gr_sig = bool(
                gr is not None
                and [arg.arg for arg in gr.args.args] == ["city", "client"]
                and all(
                    isinstance(arg.annotation, ast.Name)
                    and arg.annotation.id == expected
                    for arg, expected in zip(gr.args.args, ("str", "WeatherClient"), strict=False)
                )
            )
            signature_hits = sum([
                "AllProvidersFailedError" in classes,
                "WeatherClient" in classes,
                "fetch" in async_defs,
                gr_sig,
            ])
        rubric.add_criterion("Typed injectable signatures", 2.0, 2.0 * signature_hits / 4.0)

        concepts = {
            "concurrent provider calls": r"asyncio\.(?:gather|create_task|as_completed)|TaskGroup",
            "fallback/error handling": r"try\s*:|except\s+|fallback|next provider",
            "timeouts": r"wait_for|timeout",
            "error payload validation": r"(?:error\s*['\"]?\s*:|error.*payload|malformed|schema)",
            "logging": r"logging|logger\.(?:error|warning|exception)",
        }
        concept_hits = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in concepts.values())
        rubric.add_criterion(
            "Recovery design", 2.0, 2.0 * concept_hits / len(concepts),
            evidence=[{"kind": "concept", "name": name} for name, pattern in concepts.items() if re.search(pattern, text, re.IGNORECASE)],
        )

        stubs = stub_definitions(tree, required) if tree is not None else []
        rubric.add_criterion(
            "Non-stub implementation", 1.0, 1.0 if not stubs and present == required else 0.0,
            negative_findings=[{"finding": f"stub definition: {name}"} for name in stubs],
        )

        demo_markers = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in (
            r"all\s+(?:providers\s+)?succeed", r"one|partial|fallback", r"all\s+(?:providers\s+)?fail",
        ))
        rubric.add_criterion(
            "Demo scenarios", 1.0, 1.0 * demo_markers / 3.0)

        quality_hits = sum(bool(re.search(pattern, text)) for pattern in (r"->\s*(?:dict|None|Any)", r"\"\"\""))
        rubric.add_criterion("Type hints and docstrings", 1.0, min(1.0, float(quality_hits) / 2.0))

        source = extract_python_source(text)
        if source:
            harness = r'''
import asyncio
import inspect

assert isinstance(AllProvidersFailedError, type)
assert issubclass(AllProvidersFailedError, Exception)
assert inspect.iscoroutinefunction(get_weather_resilient)

class FakeClient:
    providers = {"WeatherAPI", "OpenMeteo", "VisualCrossing"}
    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.started = []
    async def fetch(self, provider, city):
        self.started.append(provider)
        await asyncio.sleep(0.02)
        self.calls.append(provider)
        if self.mode == "all-fail":
            raise RuntimeError(provider + " unavailable")
        if self.mode == "partial" and provider == "WeatherAPI":
            raise RuntimeError("primary unavailable")
        if self.mode == "payload" and provider == "WeatherAPI":
            return {"error": "rate limited"}
        return {"city": city, "temperature": 21}

def assert_all_providers_were_attempted(client):
    assert set(client.calls) == client.providers
    assert set(client.started) == client.providers

async def run_checks():
    for mode in ("all-success", "partial", "payload"):
        client = FakeClient(mode)
        value = await asyncio.wait_for(get_weather_resilient("Paris", client), 1)
        assert value == {"city": "Paris", "temperature": 21}
        assert_all_providers_were_attempted(client)

    client = FakeClient("all-fail")
    try:
        await asyncio.wait_for(get_weather_resilient("Paris", client), 1)
    except AllProvidersFailedError as exc:
        assert all(provider in str(exc) for provider in client.providers)
        assert_all_providers_were_attempted(client)
    else:
        raise AssertionError("all failures must raise AllProvidersFailedError")

asyncio.run(run_checks())
'''
            execution = run_python_check(source, harness)
            behavior_points = 10.0 if execution.status == "passed" else 0.0
            rubric.add_criterion(
                "Behavioral provider tests", 10.0, behavior_points,
                evidence=[{"kind": "execution", "status": execution.status, "isolation": execution.isolation}],
                negative_findings=[] if execution.status == "passed" else [{"finding": execution.error or execution.status}],
            )
        else:
            rubric.add_criterion("Behavioral provider tests", 10.0, 0.0, negative_findings=[{"finding": "no executable source"}])

        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
