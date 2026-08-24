"""Output generators for the AI benchmark.

This module contains the report generators (Markdown, CSV, HTML, PDF) and the
helper used to persist them to disk.
"""
from __future__ import annotations

import contextlib
import os
import re
import tempfile
from typing import Any, cast

from .plugin import BenchmarkOutputPlugin, BenchmarkTaskPlugin, normalize_score


def sanitize_filename(name: str) -> str:
    """Sanitize a model name for use as a filename."""
    s = re.sub(r'[^\w\-\.\(\) ]', '_', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


def _numeric_plugin_scores(result: dict[str, Any], active_plugins: list[BenchmarkTaskPlugin]) -> list[Any]:
    """Return normalized numeric plugin scores, excluding failures."""
    return [
        score
        for plugin in active_plugins
        for score in [result.get(f"{plugin.id}_score")]
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    ]


def _plugin_total_score(result: dict[str, Any], active_plugins: list[BenchmarkTaskPlugin]) -> int | None:
    """Return the normalized overall mean score, or ``None`` if unscored."""
    scores = _numeric_plugin_scores(result, active_plugins)
    return normalize_score(sum(scores) / len(scores), 100) if scores else None


def _scored_plugin_count(result: dict[str, Any], active_plugins: list[BenchmarkTaskPlugin]) -> int:
    """Return the number of active plugins with numeric public scores."""
    return len(_numeric_plugin_scores(result, active_plugins))


def _plugin_token_counts(result: dict[str, Any], plugin_id: str) -> tuple[Any, Any, Any]:
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


def _numeric_score(result: dict[str, Any], plugin_id: str, default: Any = 0) -> Any:
    """Return a normalized numeric score for sorting, or ``default``."""
    score = result.get(f"{plugin_id}_score", default)
    if isinstance(score, (int, float)):
        return score
    return default


def _numeric_judge_score(result: dict[str, Any], plugin_id: str, default: Any = 0) -> Any:
    """Return a numeric semantic judge score for sorting."""
    score = result.get(f"{plugin_id}_judge_score", default)
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return score
    return default


def _judge_consensus_by_contract(result: dict[str, Any], plugin_id: str) -> dict[str, Any]:
    """Return versioned consensus, with a legacy vote fallback."""
    consensus = result.get(f"{plugin_id}_judge_consensus_by_contract")
    if isinstance(consensus, dict) and consensus:
        return consensus
    return {}


def _judge_criteria(result: dict[str, Any], plugin_id: str) -> list[dict[str, Any]]:
    """Return structured judge criteria, including contract identity."""
    votes = result.get(f"{plugin_id}_judge_votes", [])
    if isinstance(votes, list):
        reports = [
            {
                "judge": vote.get("model"),
                "judge_contract_id": vote.get("judge_contract_id"),
                "criteria": vote.get("criteria", []),
            }
            for vote in votes
            if isinstance(vote, dict)
            and isinstance(vote.get("criteria"), list)
            and vote.get("criteria")
        ]
        if reports:
            return reports
    criteria = result.get(f"{plugin_id}_judge_criteria")
    return criteria if isinstance(criteria, list) else []


def _judge_enabled(results: list[dict[str, Any]]) -> bool:
    """Return whether any result carries judge metadata."""
    return any(
        result.get("judge_models")
        or result.get("judge_model") is not None
        or result.get("judge_status") not in (None, "disabled")
        or any(key.endswith(("_judge_score", "_judge_error")) for key in result)
        for result in results
    )


def _get_output_plugin(plugin_id: str) -> BenchmarkOutputPlugin | None:
    """Get an output plugin by ID."""
    from plugins import discover_output_plugins
    for plugin in discover_output_plugins():
        if plugin.id == plugin_id:
            return cast(BenchmarkOutputPlugin, plugin)
    return None


def gen_markdown(results: list[dict[str, Any]], active_plugins: list[BenchmarkTaskPlugin], output_dir: str | None = None, session_seed: int | None = None) -> str | None:
    """Backward-compatible wrapper that delegates to MarkdownOutputPlugin."""
    plugin = _get_output_plugin("output-markdown")
    if plugin:
        generated = plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
        return generated if isinstance(generated, str) else None
    return None


def gen_csv(results: list[dict[str, Any]], active_plugins: list[BenchmarkTaskPlugin]) -> str | None:
    """Backward-compatible wrapper that delegates to CSVOutputPlugin."""
    plugin = _get_output_plugin("output-csv")
    if plugin:
        generated = plugin.generate(results, active_plugins)
        return generated if isinstance(generated, str) else None
    return None


def gen_html(results: list[dict[str, Any]], active_plugins: list[BenchmarkTaskPlugin], output_dir: str | None = None, session_seed: int | None = None) -> str | None:
    """Backward-compatible wrapper that delegates to HTMLOutputPlugin."""
    plugin = _get_output_plugin("output-html")
    if plugin:
        generated = plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
        return generated if isinstance(generated, str) else None
    return None


def gen_pdf(results: list[dict[str, Any]], active_plugins: list[BenchmarkTaskPlugin], output_dir: str, session_seed: int | None = None) -> str | None:
    """Backward-compatible wrapper that delegates to PDFOutputPlugin."""
    plugin = _get_output_plugin("output-pdf")
    if plugin:
        generated = plugin.generate(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
        return generated if isinstance(generated, str) else None
    return None


_OUTPUT_FORMAT_PLUGIN_IDS = {
    "csv": "output-csv",
    "md": "output-markdown",
    "html": "output-html",
    "pdf": "output-pdf",
}


def normalize_output_formats(output_formats: list[str] | None) -> list[str]:
    """Return valid report formats once, preserving the caller's order."""
    if output_formats is None:
        return list(_OUTPUT_FORMAT_PLUGIN_IDS)
    formats = []
    for output_format in output_formats:
        if output_format not in _OUTPUT_FORMAT_PLUGIN_IDS:
            raise ValueError(f"Unsupported output format: {output_format}")
        if output_format not in formats:
            formats.append(output_format)
    return formats


def _atomic_replace_report(path: str, content: str | bytes, *, binary: bool = False) -> None:
    """Write report content through a sibling temporary file and replace atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path) or ".")
    try:
        if binary:
            if not isinstance(content, bytes):
                raise TypeError("binary reports require bytes")
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            if not isinstance(content, str):
                raise TypeError("text reports require str")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def save_outputs(results: list[dict[str, Any]], output_dir: str, active_plugins: list[BenchmarkTaskPlugin], *, output_formats: list[str] | None = None, session_seed: int | None = None) -> list[str]:
    """Generate the selected report formats from result dictionaries."""
    from plugins import discover_output_plugins

    output_plugins_list = discover_output_plugins()
    if output_formats is None:
        generated = []
        for plugin in output_plugins_list:
            with contextlib.suppress(Exception):
                path = plugin.generate(
                    results, active_plugins, output_dir=output_dir,
                    session_seed=session_seed,
                )
                if path:
                    generated.append(path)
        return generated

    selected = normalize_output_formats(output_formats)
    output_plugins = {plugin.id: plugin for plugin in output_plugins_list}
    generated = []
    for output_format in selected:
        plugin = output_plugins.get(_OUTPUT_FORMAT_PLUGIN_IDS[output_format])
        if plugin is None:
            continue
        with contextlib.suppress(Exception):
            path = plugin.generate(
                results, active_plugins, output_dir=output_dir,
                session_seed=session_seed,
            )
            if path:
                generated.append(path)
    return generated


def _save_outputs(state: Any, output_dir: str, active_plugins: list[BenchmarkTaskPlugin], output_formats: list[str] | None = None) -> list[str]:
    """Regenerate selected reports from latest deduplicated results.

    ``output_formats=None`` preserves the historical helper behavior for
    callers and tests; the CLI passes an explicit empty/selected list so a
    run can intentionally produce no reports.
    """
    return save_outputs(
        state.latest_results(), output_dir, active_plugins,
        output_formats=output_formats,
        session_seed=getattr(state, "session_seed", None),
    )
