"""Output generators for the AI benchmark.

This module contains the report generators (Markdown, CSV, HTML, PDF) and the
helper used to persist them to disk.
"""
import re


def sanitize_filename(name):
    """Sanitize a model name for use as a filename."""
    s = re.sub(r'[^\w\-\.\(\) ]', '_', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


def _plugin_total_score(result, active_plugins):
    """Sum numeric plugin scores, ignoring non-numeric/failed values."""
    total = 0
    for plugin in active_plugins:
        score = result.get(f"{plugin.id}_score", 0)
        if isinstance(score, (int, float)):
            total += score
    return total


def _numeric_score(result, plugin_id, default=0):
    """Return a numeric score for sorting, falling back to default for non-numeric values."""
    score = result.get(f"{plugin_id}_score", default)
    if isinstance(score, (int, float)):
        return score
    return default


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
        try:
            plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
        except Exception:
            pass
