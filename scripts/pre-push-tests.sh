#!/usr/bin/env bash
# Run the full test suite against the committed state, temporarily stashing
# any unstaged or untracked changes. This lets you push one logical commit
# while unrelated work-in-progress remains in the working tree.
set -euo pipefail

cd "$(dirname "$0")/.."

STASH_CREATED=0

cleanup() {
    if [ "$STASH_CREATED" -eq 1 ]; then
        git stash pop --quiet || true
    fi
}
trap cleanup EXIT

# Only stash if the working tree differs from HEAD or there are untracked files.
# This lets tests run against the committed state instead of the dirty checkout.
if ! git diff --quiet HEAD || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git stash push --include-untracked --message "pre-push auto-stash" --quiet
    STASH_CREATED=1
fi

coverage run -m pytest tests/ plugins/challenges/ plugins/outputs/ -q
coverage report -m
coverage report --fail-under=90
