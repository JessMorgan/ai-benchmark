"""Executable multi-step function-generation challenge."""
from __future__ import annotations

import ast
import re
from typing import Any

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import fenced_blocks, text_without_fences
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions


class MultiStepPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "multi-step"

    @property
    def version(self) -> str:
        return "1.3.0"

    @property
    def name(self) -> str:
        return "Multi-Step Instructions"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
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

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("multi_step_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    @staticmethod
    def _definitions(tree: ast.AST) -> set[str]:
        return {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    @staticmethod
    def _signature_matches(node: Any, args: tuple[tuple[str, str], ...], returns: str) -> bool:
        """Check a FunctionDef's positional argument names/types and return type.

        Uses the parsed AST instead of regex, so formatting (spaces, newlines)
        cannot defeat the signature contract.
        """
        actual = node.args.args
        if [arg.arg for arg in actual] != [name for name, _ in args]:
            return False
        for arg, (_, expected_type) in zip(actual, args, strict=False):
            if not isinstance(arg.annotation, ast.Name) or arg.annotation.id != expected_type:
                return False
        return isinstance(node.returns, ast.Name) and node.returns.id == returns

    def evaluate(self, response_text: str) -> EvaluationResult:
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

        expected_signatures = {
            "greet_user": ((("name", "str"),), "str"),
            "validate_name": ((("name", "str"),), "bool"),
            "format_greeting": ((("greeting", "str"), ("times", "int")), "str"),
        }
        signature_hits = 0
        signature_evidence = []
        if tree is not None:
            for name, (args, returns) in expected_signatures.items():
                matching = [
                    node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == name
                    and self._signature_matches(node, args, returns)
                ]
                if matching:
                    signature_hits += 1
                    signature_evidence.append({"kind": "signature", "name": name})
        rubric.add_criterion(
            "Typed signatures", 1.0, 1.0 * signature_hits / 3.0,
            evidence=signature_evidence,
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

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
