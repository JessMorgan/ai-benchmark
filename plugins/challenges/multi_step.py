"""Executable multi-step function-generation challenge."""
from __future__ import annotations

import ast
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._analysis import fenced_blocks, text_without_fences
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions


class MultiStepPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "multi-step"

    @property
    def version(self):
        return "1.0.0"

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
            "Implement exactly the three functions below. Return exactly three fenced Python "
            "code blocks followed by the summary line; do not include prose or a main block.\n\n"
            "1. `greet_user(name: str) -> str` returns exactly `Hello, <name>! Welcome.`\n"
            "2. `validate_name(name: str) -> bool` returns True only when name is non-empty, "
            "contains alphabetic characters and spaces only, and has at most 50 characters.\n"
            "3. `format_greeting(greeting: str, times: int) -> str` returns greeting repeated "
            "times with newline separators, or an empty string when times < 1.\n\n"
            "Each function must be in its own fenced Python block. End with exactly:\n"
            "[SUMMARY: 3 functions, 3 code blocks, completed all steps]."
        )

    def get_temperature(self, global_config):
        return global_config.get("multi_step_temperature")

    @staticmethod
    def _definitions(tree: ast.AST) -> set[str]:
        return {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def evaluate(self, response_text):
        if not response_text or not response_text.strip():
            return EvaluationResult(0.0, [])
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        blocks = fenced_blocks(text, "python")
        validation = parse_python(text)
        rubric.record_validation(validation)
        tree = validation.value if validation.valid else None
        definitions = self._definitions(tree) if tree is not None else set()

        expected = {"greet_user", "validate_name", "format_greeting"}
        present = expected & definitions
        rubric.add_criterion(
            "Required function contract", 4.0,
            4.0 if present == expected else 4.0 * len(present) / len(expected),
            evidence=[{"kind": "definition", "name": name} for name in sorted(present)],
            negative_findings=(
                [{"finding": f"missing required function: {name}"} for name in sorted(expected - present)]
            ),
        )

        signatures = {
            "greet_user": re.compile(r"def\s+greet_user\s*\(\s*name\s*:\s*str\s*\)\s*->\s*str"),
            "validate_name": re.compile(r"def\s+validate_name\s*\(\s*name\s*:\s*str\s*\)\s*->\s*bool"),
            "format_greeting": re.compile(r"def\s+format_greeting\s*\(\s*greeting\s*:\s*str\s*,\s*times\s*:\s*int\s*\)\s*->\s*str"),
        }
        signature_hits = sum(bool(pattern.search(text)) for pattern in signatures.values())
        rubric.add_criterion(
            "Typed signatures", 1.0, 1.0 * signature_hits / 3.0,
            evidence=[{"kind": "signature", "name": name} for name, pattern in signatures.items() if pattern.search(text)],
        )

        summary = re.fullmatch(
            r"\[SUMMARY:\s*3\s+functions,\s*3\s+code\s+blocks,\s*completed all steps\]\.\s*",
            text.splitlines()[-1] if text.splitlines() else "",
        )
        block_contract = False
        if len(blocks) == 3 and summary:
            block_names = []
            block_contract = True
            for block in blocks:
                try:
                    block_tree = ast.parse(block)
                except SyntaxError:
                    block_contract = False
                    break
                block_defs = [
                    node.name for node in block_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if len(block_defs) != 1:
                    block_contract = False
                    break
                block_names.append(block_defs[0])
            block_contract = block_contract and block_names == [
                "greet_user", "validate_name", "format_greeting"
            ]
        rubric.add_criterion(
            "Exact response contract", 2.0,
            3.0 if block_contract and text_without_fences(text).strip() == text.splitlines()[-1].strip() else 0.0,
            negative_findings=(
                [{"finding": "each required function must occupy its own Python block, followed only by the exact summary"}]
                if not block_contract else []
            ),
        )

        stubs = stub_definitions(tree, expected) if tree is not None else []
        rubric.add_criterion(
            "Non-stub implementation", 1.0,
            2.0 if not stubs and present == expected else 0.0,
            negative_findings=[{"finding": f"stub definition: {name}"} for name in stubs],
        )

        forbidden = []
        outside = text_without_fences(text)
        if re.search(r"(?m)^\s*if\s+__name__\s*==", outside):
            forbidden.append("main block")
        if re.search(r"(?m)^\s*(?:Here|Explanation|The following|This code)\b", outside, re.IGNORECASE):
            forbidden.append("explanatory prose")
        rubric.add_criterion(
            "No forbidden prose or main block", 1.0,
            2.0 if not forbidden else 0.0,
            negative_findings=[{"finding": value} for value in forbidden],
        )

        source = extract_python_source(text)
        execution = None
        if source:
            checks = """
assert greet_user("Ada") == "Hello, Ada! Welcome."
assert validate_name("Ada Lovelace") is True
assert validate_name("") is False
assert validate_name("Ada123") is False
assert validate_name("x" * 51) is False
assert format_greeting("Hi", 3) == "Hi\\nHi\\nHi"
assert format_greeting("Hi", 0) == ""
"""
            execution = run_python_check(source, checks)
            rubric.record_execution(
                execution,
                criterion="Non-stub implementation",
                penalty=1.0,
                failure_reason="required function behavior failed its isolated API tests",
            )
            if execution.status == "passed":
                rubric.credit_criterion("Non-stub implementation", 2.0, "all API tests passed")
        else:
            rubric.add_criterion("Behavioral API tests", 11.0, 0.0, negative_findings=[{"finding": "no executable Python source"}])
        if execution is not None and execution.status != "passed":
            # Keep partial lexical credit, but make the behavioral failure visible.
            rubric.add_criterion("Behavioral API tests", 11.0, 0.0, negative_findings=[{"finding": execution.error or execution.status}])
        elif execution is not None:
            rubric.add_criterion("Behavioral API tests", 11.0, 11.0, evidence=[{"kind": "execution", "status": execution.status}])

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
