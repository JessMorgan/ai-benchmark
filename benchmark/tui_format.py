"""Small, side-effect-free helpers for benchmark table rendering."""
from __future__ import annotations

MODEL_NUMBER_COLUMN_WIDTH = 5
FROZEN_VIEW_WIDTH = 35
SCORE_COLUMN_WIDTH = 9
PLUGIN_BLOCK_WIDTH = 30

# Fixed chrome lines around the model table / live view: title, summary,
# divider, plugin header, table divider, the ``Live:`` header, and the
# footer line(s) -- 7 enumerated, plus a 2-line safety margin that absorbs
# footer wrapping and the unconditional ``Live:`` header (which renders
# even when the live view has zero rows). Both the frame builder and the
# Textual app's resize handler allocate table/live rows from
# ``max_y - CHROME_LINES``.
_TUI_CHROME_LINES = 9

# Smallest scrollable model-table height. The table keeps this many rows
# even on short terminals so the scored rows stay reachable by scrolling;
# on very short terminals the live view shrinks first (down to zero lines)
# before the table is squeezed further.
_TUI_MIN_TABLE_ROWS = 5


def table_and_live_heights(max_y: int, num_sources: int, total_models: int) -> tuple[int, int]:
    """Allocate frame rows between the scrollable model table and the live view.

    Returns ``(visible_rows, live_height)`` where ``visible_rows`` is how
    many model rows the frame can show (the rest are reachable by scrolling)
    and ``live_height`` caps the live-status section (running models,
    preloading, judges, 429 sleepers) below it.

    Allocation policy, in priority order:

    * The model table keeps a usable minimum (``_TUI_MIN_TABLE_ROWS``) on
      short terminals; the live view shrinks (down to zero) rather than push
      the scored rows off-screen.
    * The table takes priority for the *additional* space beyond the live
      view's own minimum (``max(3, num_sources + 1)``) -- a run with many
      models grows the table as the terminal grows, matching the historical
      behaviour.
    * Once the FULL table fits beside the live minimum, leftover space
      raises the live view's cap instead of sitting as blank lines below a
      capped live section: a tall terminal with few models can display more
      live status. Note the live section still stops at its actual content,
      so the extra budget only materialises as more lines when there is
      genuinely more activity (running models, judges, 429 sleepers) than
      the old ``max(3, num_sources + 1)`` cap could show.

    The returned heights never exceed ``max_y`` and never leave blank rows
    inside the frame: content shorter than ``live_height`` simply renders
    fewer live lines.
    """
    min_live = max(3, num_sources + 1)
    min_table = min(_TUI_MIN_TABLE_ROWS, max(0, max_y - _TUI_CHROME_LINES))
    avail = max(0, max_y - _TUI_CHROME_LINES)
    if avail < min_table + min_live:
        # Short terminal: the table keeps its usable minimum and the live
        # view shrinks (possibly to zero) so the scored rows stay visible.
        return min_table, max(0, avail - min_table)
    if total_models <= avail - min_live:
        # The whole table fits beside the live minimum: show every model and
        # hand every remaining line to the live view.
        return total_models, avail - total_models
    # The table is too large to fit; it takes all space above the live
    # minimum and the live view stays at its floor.
    return avail - min_live, min_live


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
