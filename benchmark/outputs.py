"""Output generators for the AI benchmark.

This module contains the report generators (Markdown, CSV, HTML, PDF) and the
helper used to persist them to disk.
"""
import contextlib
import re

from .plugin import normalize_score


def sanitize_filename(name):
    """Sanitize a model name for use as a filename."""
    s = re.sub(r'[^\w\-\.\(\) ]', '_', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


def _numeric_plugin_scores(result, active_plugins):
    """Return normalized numeric plugin scores, excluding failures."""
    return [
        score
        for plugin in active_plugins
        for score in [result.get(f"{plugin.id}_score")]
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    ]


def _plugin_total_score(result, active_plugins):
    """Return the normalized overall mean score, or ``None`` if unscored."""
    scores = _numeric_plugin_scores(result, active_plugins)
    return normalize_score(sum(scores) / len(scores), 100) if scores else None


def _scored_plugin_count(result, active_plugins):
    """Return the number of active plugins with numeric public scores."""
    return len(_numeric_plugin_scores(result, active_plugins))


def _plugin_token_counts(result, plugin_id):
    """Return ``(thinking, content, total)`` token counts for a plugin result.

    ``{pid}_output_tokens`` is the content-only count (backward compatible);
    ``{pid}_thinking_tokens`` and ``{pid}_total_tokens`` are the split added
    for thinking models. Handles legacy state files that predate the split
    by deriving ``thinking=0`` / ``total=content`` from the content-only
    value, and non-numeric (``fail``/``-``) values by passing them through
    unchanged.
    """
    content = result.get(f"{plugin_id}_output_tokens", "-")
    thinking = result.get(f"{plugin_id}_thinking_tokens", "-")
    total = result.get(f"{plugin_id}_total_tokens", "-")
    # Legacy pre-split results: derive the split from the content-only count.
    if thinking in (None, "-", "") and isinstance(content, (int, float)):
        thinking = 0
    if total in (None, "-", "") and isinstance(content, (int, float)):
        thinking = thinking if isinstance(thinking, (int, float)) else 0
        total = int(content) + int(thinking)
    # Non-numeric content (``fail``) from a legacy state file predating the
    # split: mirror it into thinking/total so old and new fail rows render
    # identically (new fail rows carry ``fail`` in all three fields).
    if isinstance(content, str) and not isinstance(thinking, (int, float)):
        thinking = content if thinking in (None, "-", "") else thinking
    if isinstance(content, str) and not isinstance(total, (int, float)):
        total = content if total in (None, "-", "") else total
    return thinking, content, total


def _numeric_score(result, plugin_id, default=0):
    """Return a normalized numeric score for sorting, or ``default``."""
    score = result.get(f"{plugin_id}_score", default)
    if isinstance(score, (int, float)):
        return score
    return default


def _numeric_judge_score(result, plugin_id, default=0):
    """Return a numeric semantic judge score for sorting."""
    score = result.get(f"{plugin_id}_judge_score", default)
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return score
    return default


def _judge_enabled(results):
    """Return whether any result carries judge metadata."""
    return any(
        result.get("judge_model") is not None
        or result.get("judge_status") not in (None, "disabled")
        or any(key.endswith(("_judge_score", "_judge_error")) for key in result)
        for result in results
    )


def _get_output_plugin(plugin_id):
    """Get an output plugin by ID."""
    from plugins import discover_output_plugins
    for plugin in discover_output_plugins():
        if plugin.id == plugin_id:
            return plugin
    return None


def gen_markdown(results, active_plugins, output_dir=None, session_seed=None):
    """Backward-compatible wrapper that delegates to MarkdownOutputPlugin."""
    plugin = _get_output_plugin("output-markdown")
    if plugin:
        return plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
    return None


def gen_csv(results, active_plugins):
    """Backward-compatible wrapper that delegates to CSVOutputPlugin."""
    plugin = _get_output_plugin("output-csv")
    if plugin:
        return plugin.generate(results, active_plugins)
    return None


def gen_html(results, active_plugins, output_dir=None, session_seed=None):
    """Backward-compatible wrapper that delegates to HTMLOutputPlugin."""
    plugin = _get_output_plugin("output-html")
    if plugin:
        return plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
    return None


def gen_pdf(results, active_plugins, output_dir, session_seed=None):
    """Backward-compatible wrapper that delegates to PDFOutputPlugin."""
    plugin = _get_output_plugin("output-pdf")
    if plugin:
        return plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
    return None


def _save_outputs(state, output_dir, active_plugins):
    """Regenerate CSV/markdown/HTML from latest deduplicated results."""
    from plugins import discover_output_plugins

    results = state.latest_results()
    session_seed = getattr(state, "session_seed", None)

    output_plugins = discover_output_plugins()
    for plugin in output_plugins:
        with contextlib.suppress(Exception):
            plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
