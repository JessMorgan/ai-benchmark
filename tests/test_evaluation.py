"""Tests for offline response evaluation diagnostics."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark.evaluation import evaluate_saved_response
from plugins import discover_plugins


def test_saved_response_diagnostics_are_json_serializable():
    plugin = next(p for p in discover_plugins() if p.id == "rate-limiter")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "response.txt"
        path.write_text(
            "class TokenBucket:\n    pass\ntry:\n    raise ValueError('bad')\nexcept ValueError:\n    pass\n",
            encoding="utf-8",
        )
        result = evaluate_saved_response(plugin.id, path)

    assert result["plugin"] == "rate-limiter"
    assert result["diagnostics"]["criterion_count"] > 0
    token_bucket = next(item for item in result["rubric"] if item["name"] == "Strategy API contract")
    assert token_bucket["matched"] is True
    assert token_bucket["evidence"] == [{"kind": "class", "name": "TokenBucket"}]
    assert any(item.get("evidence") for item in result["rubric"])
    json.dumps(result)


def test_evaluation_module_cli_prints_diagnostics():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "response.txt"
        path.write_text(
            "class TokenBucket:\n    pass\ntry:\n    raise ValueError('bad')\nexcept ValueError:\n    pass\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-m", "benchmark.evaluation", "rate-limiter", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["plugin"] == "rate-limiter"
    assert payload["response_length"] > 0
