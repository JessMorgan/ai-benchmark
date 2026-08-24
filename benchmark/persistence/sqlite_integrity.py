"""Read-only integrity diagnostics for normalized benchmark databases."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntegrityIssue:
    category: str
    message: str
    details: Any = None


@dataclass
class IntegrityReport:
    """SQLite integrity result suitable for CLI and run-info output."""

    ok: bool
    issues: list[IntegrityIssue] = field(default_factory=list)
    sqlite_integrity: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sqlite_integrity": self.sqlite_integrity,
            "issues": [
                {
                    "category": issue.category,
                    "message": issue.message,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
        }


def check_integrity(connection: sqlite3.Connection) -> IntegrityReport:
    """Run SQLite, foreign-key, and normalized identity checks without writes."""
    issues: list[IntegrityIssue] = []
    row = connection.execute("PRAGMA integrity_check").fetchone()
    sqlite_integrity = str(row[0]) if row is not None else "unknown"
    if sqlite_integrity != "ok":
        issues.append(IntegrityIssue("sqlite", "PRAGMA integrity_check failed", sqlite_integrity))
    foreign_key_rows = list(connection.execute("PRAGMA foreign_key_check"))
    if foreign_key_rows:
        issues.append(IntegrityIssue(
            "foreign-key", "foreign-key violations found", [tuple(row) for row in foreign_key_rows],
        ))

    checks = (
        (
            "orphan-selection",
            "benchmark_selections reference missing attempts",
            """
            SELECT s.revision_id, s.cell_id, s.attempt_id
            FROM benchmark_selections s
            LEFT JOIN benchmark_attempts a ON a.attempt_id = s.attempt_id
            WHERE a.attempt_id IS NULL
            """,
        ),
        (
            "orphan-judge-vote",
            "current judge votes reference missing vote attempts",
            """
            SELECT c.revision_id, c.cell_id, c.judge_model, c.contract_id
            FROM current_judge_votes c
            LEFT JOIN judge_vote_attempts v ON v.vote_attempt_id = c.vote_attempt_id
            WHERE v.vote_attempt_id IS NULL
            """,
        ),
        (
            "missing-current-revision",
            "run has no current revision",
            "SELECT run_id FROM runs WHERE current_revision_id IS NULL",
        ),
    )
    for category, message, query in checks:
        rows = [tuple(row) for row in connection.execute(query)]
        if rows:
            issues.append(IntegrityIssue(category, message, rows))
    return IntegrityReport(not issues, issues, sqlite_integrity)
