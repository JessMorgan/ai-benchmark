"""Shell completion script generation for the AI benchmark CLI.

Completions are generated from the CLI's argparse parser via ``shtab``, so
they never drift from the actual flags the way a hand-maintained list does.
The parser itself is built here (not in ``benchmark.cli``) so completion
generation and the CLI share one definition without a circular import.
"""
import argparse
from typing import Any

import shtab

DEFAULT_CONFIG_PATH = "benchmark-config.json"

# Both the installed console script (``ai-benchmark``) and the repository
# launcher (``ai-benchmark.py``) share one completion function.
COMMAND_NAMES = ("ai-benchmark", "ai-benchmark.py")

_SUPPORTED_SHELLS = ("bash", "zsh", "fish")


def build_parser(
    prog: str | None = None,
    plugin_ids: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Build the CLI argument parser (shared with ``benchmark.cli.main``).

    ``prog`` defaults to argparse's inference (so ``ai-benchmark.py --help``
    still reports the launcher name); pass an explicit ``prog`` when the
    parser is used for completion generation. ``plugin_ids``, when given, are
    attached as ``choices`` to the whitelist/blacklist options so generated
    completions can offer them.
    """
    kwargs: dict[str, Any] = {
        "description": "AI Model Benchmark — Run plugin-based benchmarks across multiple API sources.",
        "epilog": "Challenge plugins are loaded from plugins/challenges/ and report plugins from plugins/outputs/.\n\n"
                  "Examples:\n"
                  "  python ai-benchmark.py --restart\n"
                  "  python ai-benchmark.py --config my-config.json\n"
                  "  python ai-benchmark.py --out /tmp/bench-run --timeout 300\n"
                  "  python ai-benchmark.py --plugins-whitelist rate-limiter\n"
                  "  python ai-benchmark.py --dump-default-config --base-url http://localhost:11434 > config.json\n"
                  "  python ai-benchmark.py --dump-default-config > benchmark-config.json\n\n"
                  "Shell completions:\n"
                  "  eval \"$(python ai-benchmark.py --generate-shell-completion bash)\"\n"
                  "  python ai-benchmark.py --generate-shell-completion zsh > ~/.zsh/completions/_ai-benchmark.py\n"
                  "  python ai-benchmark.py --generate-shell-completion fish > ~/.config/fish/completions/ai-benchmark.py.fish",
        "formatter_class": argparse.RawDescriptionHelpFormatter,
        "add_help": False,
    }
    if prog is not None:
        kwargs["prog"] = prog
    parser = argparse.ArgumentParser(**kwargs)

    general = parser.add_argument_group('General')
    general.add_argument('-h', '--help', action='help',
                         help='Show this help message and exit')
    general.add_argument('--restart', action='store_true',
                         help='Restart the run from scratch, discarding prior results')
    general.add_argument('--scripted', action='store_true',
                         help='Non-interactive mode: never prompt for input; default to continuing runs')
    general.add_argument('--seed', type=int, default=None,
                         help='Fixed random seed for all API requests (default: random)')

    config_group = parser.add_argument_group('Benchmark configuration')
    config_group.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                              help=f'Config file path (default: {DEFAULT_CONFIG_PATH})')
    config_group.add_argument('--out', default=None,
                              help='Override output directory from config')
    config_group.add_argument('--timeout', type=int, default=None,
                              help='Override request timeout in seconds from config')
    config_group.add_argument('--max-tokens', type=int, default=None,
                              help='Override the scalar max-token budget (e.g. --max-tokens 4096)')
    config_group.add_argument('--temperature', type=float, default=None,
                              help='Default temperature for all plugins (overrides config; individual --plugin-temperature takes priority)')
    config_group.add_argument('--plugin-temperature', type=str, nargs='+', default=None,
                              help='Per-plugin temperatures as id=value (e.g. --plugin-temperature rate-limiter=0.2 moe-dense=0.7)')
    config_group.add_argument('--plugin-thread-limit', type=int, default=None,
                              help='Max threads per model for plugin execution. 0 means one thread per plugin (default: 1)')
    config_group.add_argument('--plugins-whitelist', type=str, nargs='+', default=None,
                              help='Run only these plugins (e.g. --plugins-whitelist rate-limiter moe-dense)')
    config_group.add_argument('--plugins-blacklist', type=str, nargs='+', default=None,
                              help='Run all plugins except these (e.g. --plugins-blacklist moe-dense)')
    config_group.add_argument('--no-rerun-failed', action='store_true',
                              help='Do not re-run models that failed in a previous session')

    execution_group = parser.add_argument_group('Execution')
    execution_group.add_argument('--runner', choices=['http', 'opencode', 'pi', 'both'], default='http',
                                 help='Execution runner: http (default), opencode, pi, or both (per-target OpenCode-to-HTTP pipeline)')
    execution_group.add_argument('--runners', type=str, default=None, metavar='RUNNERS',
                                 help='Explicit comma-separated runner list (for example: http,pi); --runner both remains the legacy OpenCode-to-HTTP pipeline')
    execution_group.add_argument('--no-install-opencode', action='store_true',
                                 help='Do not auto-download OpenCode into .tools/opencode/ when it is missing or too old; fail with an error instead')
    execution_group.add_argument('--no-preload', action='store_true',
                                 help='Disable per-source model pre-loading for this run')
    retry_group = execution_group.add_mutually_exclusive_group()
    retry_group.add_argument('--retry-on-429', action='store_true', default=True,
                             help='Retry HTTP 429 responses with exponential backoff for any source '
                                  'that did not set its own max_429_retries (default: enabled). Each '
                                  'rate-limited request can sleep up to (max_429_retries x max_backoff_seconds) '
                                  'before failing. Use --no-retry-on-429 if you want the legacy fail-fast '
                                  'behaviour back.')
    retry_group.add_argument('--no-retry-on-429', action='store_false',
                             help='Disable HTTP 429 retries globally. Overrides per-source max_429_retries '
                                  'only when the source did not set its own; explicit per-source values '
                                  'are preserved.')

    tools_group = parser.add_argument_group('Tools')
    tools_group.add_argument('--list-plugins', action='store_true',
                             help='List discovered challenge plugins (from plugins/challenges/) with their IDs, names, and versions, then exit')
    tools_group.add_argument('--generate-shell-completion', type=str, default=None,
                             choices=['bash', 'zsh', 'fish'],
                             help='Generate shell completion script for the specified shell and exit')
    tools_group.add_argument('--dump-default-config', action='store_true',
                             help='Print a default config file to stdout and exit')
    tools_group.add_argument('--convert-config', type=str, default=None,
                             help='Convert a YAML config to JSON or a JSON config to YAML and print to stdout')
    tools_group.add_argument('--base-url', default=None,
                             help='Base URL for model discovery via /v1/models API (used with --dump-default-config)')
    tools_group.add_argument('--api-key', default=None,
                             help='API key for model discovery (used with --dump-default-config --base-url)')
    tools_group.add_argument('--chatplayground-config', action='store_true',
                             help='Enumerate ChatPlayground.ai models from the web UI and print a ready-to-run config to stdout (uses CHATPLAYGROUND_EMAIL/PASSWORD env vars)')
    tools_group.add_argument('--schema-sentinel', action='store_true',
                             help='Run a non-scoring schema compatibility probe for every configured model and print JSON')
    tools_group.add_argument('--pi-probe', action='store_true',
                             help='Run the non-scoring Pi worker/provider compatibility probe and print JSON')
    tools_group.add_argument('--import-to-sqlite', type=str, default=None, metavar='STATE_JSON',
                             help='Import a legacy benchmark_state.json into SQLite and exit')
    tools_group.add_argument('--check-sqlite', type=str, default=None, metavar='PATH',
                             help='Run read-only SQLite integrity diagnostics on a database or run directory and print JSON')
    tools_group.add_argument('--measure-storage', action='store_true',
                             help='Measure synthetic JSON/SQLite storage size and persistence latency, then print JSON')
    tools_group.add_argument('--compare-storage', nargs=2, metavar=('JSON_STATE', 'SQLITE_PATH'),
                             help='Compare JSON and SQLite current read models without modifying either backend')
    tools_group.add_argument('--sqlite-output', type=str, default=None, metavar='PATH',
                             help='SQLite output path for --import-to-sqlite (default: run.sqlite3 beside the JSON)')
    tools_group.add_argument('--overwrite-sqlite', action='store_true',
                             help='Allow --import-to-sqlite to replace an existing SQLite output file')
    tools_group.add_argument('--import-debug-logs', action='store_true',
                             help='Also import legacy .log/.log.gz files as compressed debug logs')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--storage', choices=['json', 'sqlite'], default='sqlite',
                              help='Run storage backend (default: sqlite; use json for legacy compatibility)')
    output_group.add_argument('--storage-profile', choices=['compact', 'debug', 'portable'],
                              default='compact',
                              help='Storage policy for run artifacts (default: compact)')
    output_group.add_argument('--debug-logs', action='store_true',
                              help='Retain full request/response debug transcripts (overrides compact logging policy)')
    output_group.add_argument('--output-format', nargs='+',
                              choices=['csv', 'md', 'html', 'pdf'],
                              default=None, metavar='FORMAT',
                              help='Generate one or more reports (csv, md, html, pdf); omit to generate no reports')
    output_group.add_argument('--generate-reports', type=str, default=None, metavar='PATH',
                              help='Generate selected reports from an existing run directory or benchmark_state.json, without running models')
    output_group.add_argument('--revision', type=int, default=None, metavar='N',
                              help='Report a specific SQLite revision number or ID (default: current revision)')
    output_group.add_argument('--save-responses', action='store_true',
                              help='Save each model\'s plugin response text to <output_dir>/responses/')

    judge_group = parser.add_argument_group('Judge analysis')
    judge_group.add_argument('--judge-models', nargs='+', default=None, metavar='MODEL',
                             help='Judge benchmark responses with one or more configured models and combine their results into a confidence-weighted consensus')
    judge_group.add_argument('--build-judge-queue', type=str, default=None, metavar='STATE_FILE',
                             help='Build a ranked judge-disagreement queue JSON from a benchmark_state.json and exit')
    judge_group.add_argument('--judge-queue-output', type=str, default=None, metavar='PATH',
                             help='Output path for --build-judge-queue (default: beside STATE_FILE)')
    judge_group.add_argument('--judge-spread-threshold', type=float, default=30.0, metavar='POINTS',
                             help='Include queue cells when judge spread reaches POINTS (default: 30; use --no-judge-spread to disable)')
    judge_group.add_argument('--no-judge-spread', action='store_true',
                             help='Disable the judge-spread queue criterion')
    judge_group.add_argument('--judge-deviation-threshold', type=float, default=40.0, metavar='POINTS',
                             help='Include queue cells when consensus differs from deterministic score by POINTS (default: 40; use --no-judge-deviation to disable)')
    judge_group.add_argument('--no-judge-deviation', action='store_true',
                             help='Disable the consensus-deviation queue criterion')

    if plugin_ids:
        plugin_ids = list(plugin_ids)
        for action in parser._actions:
            if action.option_strings and action.option_strings[0] in (
                "--plugins-whitelist", "--plugins-blacklist"
            ):
                action.choices = plugin_ids
    return parser


def _register_second_command(script: str, shell: str) -> str:
    """Register the ``ai-benchmark.py`` launcher alongside ``ai-benchmark``.

    shtab's generated script registers only the primary program name; the
    launcher is a thin alias of the same command, so it points at the same
    completion function.
    """
    if shell == "bash":
        return script.rstrip("\n") + "\ncomplete -F _shtab_ai_benchmark ai-benchmark.py\n"
    if shell == "zsh":
        return script.replace(
            "#compdef ai-benchmark",
            "#compdef ai-benchmark ai-benchmark.py",
            1,
        )
    if shell == "fish":
        extra = [
            line.replace("complete -c ai-benchmark ", "complete -c ai-benchmark.py ", 1)
            for line in script.splitlines()
            if line.startswith("complete -c ai-benchmark ")
        ]
        if extra:
            return script.rstrip("\n") + "\n" + "\n".join(extra) + "\n"
    return script


def generate_shell_completion(shell: str, plugins: list[Any]) -> str:
    """Return a shell completion script for the specified shell (via shtab).

    Unsupported shells return an empty string. ``ai-benchmark.py`` is
    registered alongside ``ai-benchmark`` so both entry points complete.
    """
    if shell not in _SUPPORTED_SHELLS:
        return ""
    parser = build_parser(prog="ai-benchmark", plugin_ids=[p.id for p in plugins])
    script = shtab.complete(parser, shell=shell)
    return _register_second_command(script, shell)
