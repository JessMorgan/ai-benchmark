"""Error recovery / robustness benchmark task.

Tests the model's ability to:
1. Handle simulated API failures gracefully
2. Design fallback strategies when primary services fail
3. Implement retry logic with exponential backoff
4. Provide meaningful error messages to end users
5. Design circuit breakers and degradation strategies
"""
import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import find_definitions, parse_python, stub_definitions


class ErrorRecoveryPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "error-recovery"

    @property
    def version(self):
        return "0.4.0"

    @property
    def name(self):
        return "Error Recovery"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are building a resilient microservice that fetches weather data "
            "from three external providers: WeatherAPI, OpenMeteo, and VisualCrossing. "
            "Your service must:\n\n"
            "1. Query all three providers simultaneously\n"
            "2. Return the first successful response (or aggregate)\n"
            "3. Handle partial failures gracefully\n\n"
            "SCENARIO: Write a Python function `get_weather_resilient(city: str) -> dict` "
            "that queries these three providers and:\n\n"
            "ARCHITECTURE REQUIREMENTS:\n"
            "- Use asyncio for concurrent queries\n"
            "- If one provider fails (HTTP error, timeout, bad data), fall through to the next\n"
            "- If all three fail, raise a custom `AllProvidersFailedError` with details\n"
            "- Log each failure with provider name and error reason\n"
            "- Return the FIRST successful response as-is (you may mock the actual HTTP calls)\n\n"
            "EDGE CASES TO HANDLE:\n"
            "- A provider returns 200 OK but with an error payload (e.g., {\"error\": \"rate limited\"})\n"
            "- A provider times out (simulate with a delay)\n"
            "- A provider returns data in a different format than expected\n"
            "- Two providers succeed but return conflicting data (different temperatures)\n\n"
            "OUTPUT FORMAT:\n"
            "```python\n"
            "import asyncio\n"
            "import logging\n"
            "from typing import Any\n\n"
            "class AllProvidersFailedError(Exception):\n"
            "    ...\n\n"
            "class WeatherClient:\n"
            "    \"\"\"Mock client demonstrating resilient multi-provider pattern.\"\"\"\n"
            "    ...\n\n"
            "async def get_weather_resilient(city: str) -> dict:\n"
            "    ...\n\n"
            "async def demo():\n"
            "    \"\"\"Demonstrate error handling scenarios.\"\"\"\n"
            "    ...\n"
            "```\n\n"
            "Include docstrings and type hints. The demo function should show at least "
            "3 scenarios: all providers succeed, one provider fails, all providers fail."
        )

    def get_temperature(self, global_config):
        if "error_recovery_temperature" in global_config:
            return global_config["error_recovery_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        rubric = Rubric(self.max_score)
        python_validation = parse_python(t, require_block=True)
        rubric.record_validation(python_validation)
        definitions = find_definitions(python_validation.value) if python_validation.valid else {}

        # Asyncio / concurrent design
        earned = 0.0
        if any(name in definitions for name in ("get_weather_resilient", "demo")):
            earned += 1.0
        if re.search(r"(?:async def|asyncio|await|gather|create_task)", t):
            earned += 2.0
        if re.search(r"(?:asyncio\.gather|as_completed|wait|TaskGroup)", t):
            earned += 1.0
        rubric.add_criterion("Concurrent / asyncio design", 3.0, earned)

        # Retry / fallback logic
        earned = 0.0
        if re.search(r"(?:fallback|fall through|next provider|alternate|secondary)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:retry|backoff|exponential|timeout|attempt)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:try:|except|Exception|finally)", t):
            earned += 1.0
        rubric.add_criterion("Fallback / retry logic", 3.0, earned)

        # Error handling
        earned = 0.0
        if re.search(r"class\s+AllProvidersFailedError", t):
            earned += 1.0
        if re.search(r"(?:raise|raise.*Error|raise.*Exception)", t):
            earned += 1.0
        if re.search(r"(?:logging|logger|log\.error|log\.warning|print\()", t):
            earned += 1.0
        rubric.add_criterion("Error types / custom exception / logging", 3.0, earned)

        # Edge case handling
        earned = 0.0
        if re.search(r"(?:rate\s*limit|error.*payload|bad.*data|invalid)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:format|schema|parse|unexpected|mismatch)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:conflict|disagree|differ|contradict|discrepancy)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:timeout|delay|time\.sleep|asyncio\.wait_for)", t):
            earned += 1.0
        rubric.add_criterion("Edge case coverage", 3.0, earned)

        # Code quality
        earned = 0.0
        if re.search(r'->\s*(?:dict|dict\[|Optional|Any|dict\[str)', t):
            earned += 1.0
        if re.search(r':\s*(?:str|int|bool|float|dict|list|Optional)', t):
            earned += 1.0
        if '"""' in t or 'docstring' in t:  # check for docstrings
            earned += 1.0
        rubric.add_criterion("Type hints & docstrings", 3.0, earned)

        # Demo / usage example
        earned = 0.0
        if re.search(r"(?:demo|example|usage|if __name__)", t):
            earned += 1.0
        if re.search(r"(?:all succeed|all fail|one fail|partial|scenario)", t, re.IGNORECASE):
            earned += 1.0
        if re.search(r"(?:print|return|result|output)", t):
            earned += 1.0
        rubric.add_criterion("Demo scenarios", 3.0, earned)

        # Structure / completeness
        earned = 1.0
        if re.search(r"class\s+AllProvidersFailedError", t):
            earned += 0.5
        if re.search(r"class\s+WeatherClient", t):
            earned += 0.5
        if re.search(r"async\s+def\s+get_weather_resilient", t):
            earned += 0.5
        if re.search(r"async\s+def\s+demo", t):
            earned += 0.5
        rubric.add_criterion("Structure / completeness", 2.0, earned)

        placeholders = re.findall(r"(?m)^\s*\.\.\.\s*$", t)
        if python_validation.valid:
            placeholders.extend(stub_definitions(
                python_validation.value,
                {"WeatherClient", "get_weather_resilient", "demo"},
            ))
        if placeholders:
            rubric.penalize_criterion(
                "Structure / completeness", 2.0,
                f"response contains {len(placeholders)} placeholder or pass-only line(s)",
            )
            rubric.penalize_criterion(
                "Demo scenarios", 1.0,
                "placeholder implementation cannot demonstrate the requested scenarios",
            )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
