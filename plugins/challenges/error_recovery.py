"""Executable resilient multi-provider API challenge."""
from __future__ import annotations

import ast
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions

# The four behavioral modes the harness exercises, paired with the rubric
# criterion each one scores. One execution run reports one marker per mode,
# so a response correct in three modes keeps 7.5 of the 10-pt behavioral
# block instead of losing it all (event-processor split precedent).
_BEHAVIORAL_MODES: tuple[tuple[str, str], ...] = (
    ("all-success", "Behavioral all-success mode"),
    ("partial", "Behavioral partial-failure mode"),
    ("payload", "Behavioral error-payload mode"),
    ("all-fail", "Behavioral all-failure mode"),
)

_MODE_RESULT_RE = re.compile(r"MODE_RESULT (?P<mode>[A-Za-z0-9-]+) (?P<result>PASS|FAIL)(?P<detail>.*)")

# Lexical concept patterns for the "Recovery design" criterion. The
# concurrent pattern accepts bare gather/ensure_future calls (from-asyncio
# import style), not only the asyncio.-prefixed forms.
_CONCEPT_PATTERNS: dict[str, str] = {
    "concurrent provider calls": r"asyncio\.(?:gather|create_task|as_completed|ensure_future)|\b(?:gather|ensure_future)\s*\(|TaskGroup",
    "fallback/error handling": r"try\s*:|except\s+|fallback|next provider",
    "timeouts": r"wait_for|timeout",
    "error payload validation": r"(?:error\s*['\"]?\s*:|error.*payload|malformed|schema)",
    "logging": r"logging|logger\.(?:error|warning|exception)",
}

# The demo may run at module level (``if __name__ == "__main__"`` fires
# inside the combined check script), and a response that makes live network
# calls there would hang the local-restricted check until the 5s execution
# timeout (Podman already runs with --network=none). Block the network-facing
# socket entry points before the response source runs so demo side effects
# fail fast with a clear marker and the harness still executes. Every stdlib
# HTTP client (urllib / http.client) routes connects through
# ``create_connection`` and DNS through ``getaddrinfo``; the event loop's
# self-pipe uses the local ``socketpair`` (no network), so the
# ``socket.socket`` class itself is left intact. A raw
# ``socket.socket().connect(raw_ip)`` bypass is out of scope here: the
# local-restricted path is not a security boundary (Podman's --network=none is).
_SOCKET_BLOCK_PREAMBLE = (
    "import socket as _benchmark_socket\n"
    "def _benchmark_network_blocked(*_args, **_kwargs):\n"
    "    raise RuntimeError('network access is disabled in the benchmark sandbox')\n"
    "_benchmark_socket.create_connection = _benchmark_network_blocked\n"
    "_benchmark_socket.getaddrinfo = _benchmark_network_blocked\n"
)


class ErrorRecoveryPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "error-recovery"

    @property
    def version(self) -> str:
        return "1.5.0"

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

    @staticmethod
    def _annotation_name(node: ast.expr | None) -> str | None:
        # String-literal forward references (``client: "WeatherClient"``) parse
        # as a string Constant, not a Name; accept both so either form earns
        # the signature credit.
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    @staticmethod
    def _mode_results(output: str) -> dict[str, tuple[bool, str]]:
        # The harness runs after the response source, so the LAST marker per
        # mode is the harness's own verdict; markers a response prints at
        # module level are overridden by the real result.
        results: dict[str, tuple[bool, str]] = {}
        for match in _MODE_RESULT_RE.finditer(output):
            detail = match.group("detail").strip()
            results[match.group("mode")] = (match.group("result") == "PASS", detail)
        return results

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
                    self._annotation_name(arg.annotation) == expected
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

        concepts = _CONCEPT_PATTERNS
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

        # The prompt labels the demo scenarios "all-success", "partial-failure",
        # and "all-failure"; accept hyphen/underscore separators and the noun
        # forms (success/failure) in addition to the verb forms (succeed/fail).
        demo_markers = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in (
            r"all[\s\-_]+(?:providers[\s\-_]+)?(?:succeed|success)",
            r"one|partial|fallback",
            r"all[\s\-_]+(?:providers[\s\-_]+)?fail(?:ure|s|ed)?",
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

async def run_mode(mode):
    client = FakeClient(mode)
    if mode == "all-fail":
        try:
            await asyncio.wait_for(get_weather_resilient("Paris", client), 1)
        except AllProvidersFailedError as exc:
            assert all(provider in str(exc) for provider in client.providers)
        else:
            raise AssertionError("all failures must raise AllProvidersFailedError")
    else:
        value = await asyncio.wait_for(get_weather_resilient("Paris", client), 1)
        assert value == {"city": "Paris", "temperature": 21}
    assert_all_providers_were_attempted(client)

async def run_checks():
    for mode in ("all-success", "partial", "payload", "all-fail"):
        try:
            await run_mode(mode)
        except Exception as exc:
            print("MODE_RESULT " + mode + " FAIL " + type(exc).__name__ + ": " + str(exc))
        else:
            print("MODE_RESULT " + mode + " PASS")

asyncio.run(run_checks())
'''
            execution = run_python_check(_SOCKET_BLOCK_PREAMBLE + "\n" + source, harness)
            # A response can print fake MODE_RESULT markers at module level and
            # then exit early (sys.exit / os._exit) so the harness never runs;
            # gate the behavioral credit on the harness actually completing
            # (the completion sentinel is printed only after the harness ends).
            harness_completed = execution.harness_ok
            mode_results = self._mode_results(execution.output) if harness_completed else {}
            for mode, criterion_name in _BEHAVIORAL_MODES:
                if harness_completed:
                    passed, detail = mode_results.get(
                        mode, (False, f"mode was not reported by the harness ({execution.status})")
                    )
                else:
                    passed = False
                    detail = execution.error or "harness did not complete (missing completion sentinel)"
                rubric.add_criterion(
                    criterion_name, 2.5, 2.5 if passed else 0.0,
                    evidence=[execution.as_evidence()],
                    negative_findings=[] if passed else [{"finding": f"{mode}: {detail}"}],
                )
        else:
            for _mode, criterion_name in _BEHAVIORAL_MODES:
                rubric.add_criterion(criterion_name, 2.5, 0.0, negative_findings=[{"finding": "no executable source"}])

        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
