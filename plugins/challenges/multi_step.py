"""Multi-step instruction following benchmark task."""
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import find_definitions, parse_python, stub_definitions


class MultiStepPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "multi-step"

    @property
    def version(self):
        return "0.6.0"

    @property
    def name(self):
        return "Multi-Step Instructions"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "Follow the multi-step instructions below exactly. "
            "Your final response must include all requested artifacts in the order specified.\n\n"
            "Step 1: Write a Python function named `greet_user` that takes one argument `name` (a string) "
            "and returns a greeting string in the exact format: 'Hello, <name>! Welcome.'\n\n"
            "Step 2: Write a Python function named `validate_name` that takes one argument `name` (a string) "
            "and returns True if the name is non-empty, contains only alphabetic characters and spaces, "
            "and is at most 50 characters long; otherwise return False.\n\n"
            "Step 3: Write a Python function named `format_greeting` that takes two arguments, `greeting` "
            "(a string) and `times` (an integer), and returns the greeting repeated `times` times, "
            "each on its own line. If `times` is less than 1, return an empty string.\n\n"
            "Step 4: At the end of your response, add a line exactly in this format "
            "(including the square brackets and the trailing period):\n"
            "[SUMMARY: <total_lines> lines, <total_functions> functions, completed all steps].\n\n"
            "Important constraints:\n"
            "- Do not write a main block or example usage.\n"
            "- Do not add explanatory text outside the code blocks and summary line.\n"
            "- Each function must be in its own fenced Python code block.\n"
            "- The summary line must match the format exactly."
        )

    def get_temperature(self, global_config):
        if "multi_step_temperature" in global_config:
            return global_config["multi_step_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        if not t or not t.strip():
            return EvaluationResult(0.0, [])

        rubric = Rubric(self.max_score)
        python_validation = parse_python(t, require_block=True)
        rubric.record_validation(python_validation)
        definitions = find_definitions(python_validation.value) if python_validation.valid else {}

        earned = 0.0
        if "greet_user" in definitions and not re.search(r"def\s+greet_user", t):
            earned += 1.0
        if re.search(r"def\s+greet_user\s*\(\s*name\s*:\s*str\s*\)", t):
            earned += 1.0
        elif re.search(r"def\s+greet_user\s*\(\s*name\s*\)", t):
            earned += 0.5
        if re.search(r"['\"]Hello,\s*\{?name\}?[!\.]?\s*Welcome\.['\"]", t):
            earned += 2.0
        elif re.search(r"Hello,\s*.*Welcome", t):
            earned += 1.0
        if re.search(r"return\s+['\"]", t):
            earned += 1.0
        if re.search(r"def\s+greet_user", t):
            earned += 1.0
        rubric.add_criterion("greet_user function", 5.0, earned)

        earned = 0.0
        if "validate_name" in definitions and not re.search(r"def\s+validate_name", t):
            earned += 1.0
        if re.search(r"def\s+validate_name\s*\(\s*name\s*:\s*str\s*\)", t):
            earned += 1.0
        elif re.search(r"def\s+validate_name\s*\(\s*name\s*\)", t):
            earned += 0.5
        if re.search(r"len\(\s*name\s*\)\s*<=?\s*50", t):
            earned += 1.5
        if re.search(r"name\s*\.\s*(isalpha|isalnum)\s*\(\)", t):
            earned += 1.5
        if re.search(r"return\s+(True|False)", t):
            earned += 1.0
        rubric.add_criterion("validate_name function", 5.0, earned)

        earned = 0.0
        if "format_greeting" in definitions and not re.search(r"def\s+format_greeting", t):
            earned += 1.0
        if re.search(r"def\s+format_greeting\s*\(\s*greeting\s*:\s*str\s*,\s*times\s*:\s*int\s*\)", t):
            earned += 1.0
        elif re.search(r"def\s+format_greeting\s*\(\s*greeting\s*,\s*times\s*\)", t):
            earned += 0.5
        if re.search(r"if\s+times\s*<\s*1", t):
            earned += 1.5
        if re.search(r"return\s+.*join|join\s*\(\s*\[\s*greeting\s*\]\s*\*\s*times\s*\)", t):
            earned += 1.5
        if re.search(r"greeting\s*\*\s*times|for\s+.*in\s+range\(\s*times\s*\)", t):
            earned += 1.0
        rubric.add_criterion("format_greeting function", 5.0, earned)

        earned = 0.0
        summary_match = re.search(r"\[SUMMARY:\s*(\d+)\s*lines?,\s*(\d+)\s*functions?,\s*completed all steps\]\.", t)
        if summary_match:
            earned = 3.0
        elif re.search(r"\[SUMMARY:.*completed all steps\]", t):
            earned = 1.5
        elif re.search(r"completed all steps", t, re.IGNORECASE):
            earned = 0.5
        rubric.add_criterion("Summary line format", 3.0, earned)

        earned = 2.0
        if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", t):
            earned -= 1.0
        code_blocks = re.findall(r"```[\s\S]*?```", t)
        if len(code_blocks) < 3:
            earned -= 1.0
        earned = round(max(earned, 0.0), 1)
        rubric.add_criterion("No extra prose/main block", 2.0, earned)
        if re.search(r"(?m)^\s*if\s+__name__\s*==", t):
            rubric.penalize_criterion("No extra prose/main block", 1.0, "response includes a forbidden main block")
        if python_validation.valid and stub_definitions(
                python_validation.value,
                {"greet_user", "validate_name", "format_greeting"},
        ):
            rubric.penalize_criterion("No extra prose/main block", 0.5, "response contains a required-function stub")

        source = extract_python_source(t)
        if source:
            checks = (
                (
                    "greet_user function",
                    'assert greet_user("Ada") == "Hello, Ada! Welcome."',
                ),
                (
                    "validate_name function",
                    (
                        "assert validate_name(\"Ada Lovelace\") is True\n"
                        "assert validate_name(\"\") is False\n"
                        "assert validate_name(\"Ada123\") is False\n"
                        "assert validate_name(\"x\" * 51) is False"
                    ),
                ),
                (
                    "format_greeting function",
                    (
                        'assert format_greeting("Hi", 3) == "Hi\\nHi\\nHi"\n'
                        'assert format_greeting("Hi", 0) == ""'
                    ),
                ),
            )
            for criterion, harness in checks:
                execution = run_python_check(source, harness)
                rubric.record_execution(
                    execution,
                    criterion=criterion,
                    penalty=1.0,
                    failure_reason=f"isolated {criterion} harness failed",
                )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
