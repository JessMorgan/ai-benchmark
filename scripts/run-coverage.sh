#!/usr/bin/env bash
# Run the test suite with coverage reporting for plugins.
# Generates terminal, HTML, and XML reports.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pytest tests/ plugins/challenges/ plugins/outputs/ \
  --cov=plugins \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-report=xml:coverage.xml \
  --cov-fail-under=90 \
  -q
