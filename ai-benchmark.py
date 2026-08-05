#!/usr/bin/env python3
"""Thin launcher for the AI Benchmark CLI.

The full implementation lives in :mod:`benchmark.cli` so the command is
importable as a package module and installable as a console script
(``ai-benchmark``) via ``[project.scripts]`` in ``pyproject.toml``.

Kept at the repository root so ``python ai-benchmark.py`` keeps working
for docs, pre-commit hooks, and shell completion examples.
"""
from benchmark.cli import main

if __name__ == "__main__":
    main()
