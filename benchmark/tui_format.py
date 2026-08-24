"""Small, side-effect-free helpers for benchmark table rendering."""
from __future__ import annotations

MODEL_NUMBER_COLUMN_WIDTH = 5
FROZEN_VIEW_WIDTH = 35
SCORE_COLUMN_WIDTH = 9
PLUGIN_BLOCK_WIDTH = 30


def row_style(state_row: dict[str, object]) -> str | None:
    """Return the canonical style for a table row.

    Active work takes precedence over stale terminal status, so a retry or
    sibling plugin remains yellow until all work finishes.
    """
    if state_row.get("running_pids") or state_row.get("preloading"):
        return "yellow"
    status = state_row.get("status")
    if status == "completed":
        return "green"
    if status == "failed":
        return "red"
    return None
