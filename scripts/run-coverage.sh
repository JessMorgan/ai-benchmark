#!/usr/bin/env bash
# Run the test suite with coverage reporting for the benchmark package and plugins.
# Coverage configuration lives in pyproject.toml ([tool.coverage]).
# Generates terminal, HTML, and XML reports.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run coverage run -m pytest tests/ plugins/challenges/ plugins/outputs/ -q
uv run coverage report -m
uv run coverage html
uv run coverage xml
uv run coverage report --fail-under=90
