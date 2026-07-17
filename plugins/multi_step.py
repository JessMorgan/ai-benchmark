"""Multi-step instruction following benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin


class MultiStepPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "multi-step"

    @property
    def version(self):
        return "0.1.0"

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
            return 0.0, []
        rubric = []
        s = 0.0

        # 1. greet_user function present and correct (0-5)
        earned = 0.0
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
        earned = round(min(earned, 5.0), 1)
        s += earned
        rubric.append({"name": "greet_user function", "max": 5.0, "earned": earned, "missed": round(5.0 - earned, 1)})

        # 2. validate_name function present and correct (0-5)
        earned = 0.0
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
        earned = round(min(earned, 5.0), 1)
        s += earned
        rubric.append({"name": "validate_name function", "max": 5.0, "earned": earned, "missed": round(5.0 - earned, 1)})

        # 3. format_greeting function present and correct (0-5)
        earned = 0.0
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
        earned = round(min(earned, 5.0), 1)
        s += earned
        rubric.append({"name": "format_greeting function", "max": 5.0, "earned": earned, "missed": round(5.0 - earned, 1)})

        # 4. Summary line format (0-3)
        earned = 0.0
        summary_match = re.search(r"\[SUMMARY:\s*(\d+)\s*lines?,\s*(\d+)\s*functions?,\s*completed all steps\]\.", t)
        if summary_match:
            earned = 3.0
        elif re.search(r"\[SUMMARY:.*completed all steps\]", t):
            earned = 1.5
        elif re.search(r"completed all steps", t, re.IGNORECASE):
            earned = 0.5
        earned = round(min(earned, 3.0), 1)
        s += earned
        rubric.append({"name": "Summary line format", "max": 3.0, "earned": earned, "missed": round(3.0 - earned, 1)})

        # 5. No extra explanatory text / no main block (0-2)
        earned = 2.0
        if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", t):
            earned -= 1.0
        if re.search(r"Example\s+usage|Here\s+is|Below\s+is|Step\s+\d+:", t, re.IGNORECASE):
            # Allow the prompt-included step markers, but penalize extra prose.
            pass
        # Heuristic: count code fences vs prose paragraphs
        code_blocks = re.findall(r"```[\s\S]*?```", t)
        if len(code_blocks) < 3:
            earned -= 1.0
        earned = round(max(earned, 0.0), 1)
        s += earned
        rubric.append({"name": "No extra prose/main block", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        return round(min(s, self.max_score), 1), rubric

    def score(self, response_text):
        return self.evaluate(response_text)[0]
