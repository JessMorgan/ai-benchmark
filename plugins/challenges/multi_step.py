"""Executable multi-step function-generation challenge."""
from __future__ import annotations

import ast
import re
from typing import Any

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import fenced_blocks
from plugins.challenges._execution import extract_python_source, run_python_check
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_python, stub_definitions

# The prompt mandates this exact trailing summary line. It is not Python, so
# for unfenced responses it would corrupt the raw-ast.parse fallback and the
# execution source; it is stripped before any raw-text parsing or execution.
_SUMMARY_LINE_RE = re.compile(
    r"\[SUMMARY:\s*3\s+functions,\s*3\s+code\s+blocks,\s*completed all steps\]\.\s*"
)

# Matches only Python fenced blocks (label aliases: python3/python/py3/py).
# Used to remove just the Python blocks when scanning for forbidden prose, so
# prose hidden in other fences (e.g. ```text) is still visible to the scan.
_PY_FENCE_RE = re.compile(r"```(?:python3|python|py3|py)[^\n]*\n[\s\S]*?```", re.IGNORECASE)


class MultiStepPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "multi-step"

    @property
    def version(self) -> str:
        return "1.4.0"

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
    def _module_level_defs(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Return only the module-level function definitions (top-level body).

        Nested or wrapped definitions (inside a class, another function, or a
        factory) do not expose a module-level API, so the contract and
        signature criteria must not credit them.
        """
        if not isinstance(tree, ast.Module):
            return []
        return [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    @staticmethod
    def _any_block_defines_required(blocks: list[str], expected: set[str]) -> bool:
        """Return whether any Python block defines a required function at module level."""
        for block in blocks:
            try:
                block_tree = ast.parse(block)
            except SyntaxError:
                continue
            for node in block_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in expected:
                    return True
        return False

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

    @staticmethod
    def _strip_trailing_summary(text: str) -> str:
        """Drop the prompt-mandated trailing summary line when it is the last line.

        The summary line is not Python. For an unfenced response it would make
        the raw-ast.parse fallback (and the execution source) fail to compile,
        zeroing nearly every criterion; stripping it lets the fallback retry
        parse the actual code. It is a no-op for fenced responses, where only
        the fenced blocks are parsed.
        """
        lines = text.splitlines()
        if lines and _SUMMARY_LINE_RE.fullmatch(lines[-1]):
            return "\n".join(lines[:-1]).rstrip()
        return text

    @staticmethod
    def _text_minus_python_blocks(text: str) -> str:
        """Remove only the Python fenced blocks, keeping other fences and prose.

        The forbidden-prose scan must see prose hidden in non-Python fences
        (e.g. a ```text block); stripping every fence would hide that prose.
        """
        return _PY_FENCE_RE.sub("", text)

    def evaluate(self, response_text: str) -> EvaluationResult:
        if not response_text or not response_text.strip():
            return EvaluationResult(0.0, [])
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        blocks = fenced_blocks(text, "python")
        parse_text = self._strip_trailing_summary(text)
        validation = parse_python(parse_text)
        rubric.record_validation(validation)
        tree = validation.value if validation.valid else None
        module_defs = self._module_level_defs(tree) if tree is not None else []
        definitions = {node.name for node in module_defs}

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
        for name, (args, returns) in expected_signatures.items():
            matching = [
                node for node in module_defs
                if node.name == name
                and self._signature_matches(node, args, returns)
            ]
            if matching:
                signature_hits += 1
                signature_evidence.append({"kind": "signature", "name": name})
        rubric.add_criterion(
            "Typed signatures", 1.0, 1.0 * signature_hits / 3.0,
            evidence=signature_evidence,
        )

        summary = (
            _SUMMARY_LINE_RE.fullmatch(text.splitlines()[-1]) if text.splitlines() else None
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
        # Strip only the Python blocks (not all fences) so the "followed only by
        # the exact summary" check still sees prose hidden in a non-Python fence.
        rubric.add_criterion(
            "Exact response contract", 2.0,
            2.0 if block_contract and self._text_minus_python_blocks(text).strip() == text.splitlines()[-1].strip() else 0.0,
            negative_findings=(
                [{"finding": "each required function must occupy its own Python block, followed only by the exact summary"}]
                if not block_contract else []
            ),
        )

        stubs = stub_definitions(tree, expected) if tree is not None else []
        rubric.add_criterion(
            "Non-stub implementation", 1.0,
            1.0 if not stubs and present == expected else 0.0,
            negative_findings=[{"finding": f"stub definition: {name}"} for name in stubs],
        )

        forbidden = []
        # A __main__ guard is a main block wherever it appears, including
        # inside a Python fence, so scan the whole text for it.
        if re.search(r"(?m)^\s*if\s+__name__\s*==", text):
            forbidden.append("main block")
        # Prose is only forbidden outside the Python blocks; prose hidden in a
        # non-Python fence (e.g. ```text) must still be visible to the scan.
        non_python = self._text_minus_python_blocks(text)
        if re.search(r"(?m)^\s*(?:Here|Explanation|The following|This code)\b", non_python, re.IGNORECASE):
            forbidden.append("explanatory prose")
        # The discipline point is only credited when the response actually
        # contains code: a codeless response must not earn it for free.
        has_required_code = self._any_block_defines_required(blocks, expected)
        negative_findings = [{"finding": value} for value in forbidden]
        if not has_required_code:
            negative_findings.append({"finding": "no Python block defines a required function"})
        rubric.add_criterion(
            "No forbidden prose or main block", 1.0,
            1.0 if not forbidden and has_required_code else 0.0,
            negative_findings=negative_findings,
        )

        source = extract_python_source(parse_text)
        execution = None
        if source:
            # Type-identity asserts pin the return types, so a response that
            # returns an object with a lying ``__eq__`` cannot pass the
            # behavioral contract (value-only ``==`` compares credit it). The
            # extra probes cover inputs the base asserts never exercised:
            # a second name, the 50-character boundary, symbol rejection,
            # single repetition, and negative ``times``.
            checks = """
assert type(greet_user("Ada")) is str
assert greet_user("Ada") == "Hello, Ada! Welcome."
assert greet_user("Grace") == "Hello, Grace! Welcome."
assert type(validate_name("Ada Lovelace")) is bool
assert validate_name("Ada Lovelace") is True
assert validate_name("x" * 50) is True
assert validate_name("") is False
assert validate_name("Ada123") is False
assert validate_name("Ada-Lovelace") is False
assert validate_name("x" * 51) is False
assert type(format_greeting("Hi", 3)) is str
assert format_greeting("Hi", 3) == "Hi\\nHi\\nHi"
assert format_greeting("Hi", 1) == "Hi"
assert format_greeting("Hi", 0) == ""
assert format_greeting("Hi", -1) == ""
"""
            execution = run_python_check(source, checks)
            rubric.record_execution(
                execution,
                criterion="Non-stub implementation",
                penalty=1.0,
                failure_reason="required function behavior failed its isolated API tests",
            )
            # Credit only when the harness actually ran to completion (the
            # sentinel was printed); an early exit (sys.exit / SystemExit)
            # exits 0 before the harness and must not credit it.
            if execution.harness_ok:
                rubric.credit_criterion("Non-stub implementation", 1.0, "all API tests passed")
        else:
            rubric.add_criterion("Behavioral API tests", 11.0, 0.0, negative_findings=[{"finding": "no executable Python source"}])
        if execution is not None and not execution.harness_ok:
            # The harness did not run to completion: it failed/timed out, or the
            # response exited early (sys.exit / SystemExit) before the sentinel.
            finding = (
                "harness did not run to completion (early exit before the completion sentinel)"
                if execution.status == "passed"
                else execution.error or execution.status
            )
            rubric.add_criterion("Behavioral API tests", 11.0, 0.0, negative_findings=[{"finding": finding}])
        elif execution is not None:
            rubric.add_criterion("Behavioral API tests", 11.0, 11.0, evidence=[{"kind": "execution", "status": execution.status}])

        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
