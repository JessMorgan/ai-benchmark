"""Shell completion script generation for the AI benchmark CLI."""
import shlex

# Both the installed console script (``ai-benchmark``) and the repository
# launcher (``ai-benchmark.py``) share one completion function.
COMMAND_NAMES = ("ai-benchmark", "ai-benchmark.py")


def generate_shell_completion(shell, plugins):
    """Return a shell completion script for the specified shell."""
    plugin_ids = " ".join(p.id for p in plugins)
    flags = [
        "--restart",
        "--config",
        "--out",
        "--timeout",
        "--token-levels",
        "--plugin-temperature",
        "--plugin-thread-limit",
        "--plugins-whitelist",
        "--plugins-blacklist",
        "--list-plugins",
        "--generate-shell-completion",
        "--dump-default-config",
        "--base-url",
        "--api-key",
        "--save-responses",
        "--no-preload",
        "--runner",
        "--judge-models",
    ]
    flag_list = " ".join(flags)

    if shell == "bash":
        return f"""_ai_benchmark_complete() {{
    local cur prev
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    local plugin_ids="{plugin_ids}"
    local flags="{flag_list}"

    case "$prev" in
        --plugins-whitelist|--plugins-blacklist)
            COMPREPLY=( $(compgen -W "$plugin_ids" -- "$cur") )
            return 0
            ;;
        --config)
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
        --out)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
        --runner)
            COMPREPLY=( $(compgen -W "http opencode both" -- "$cur") )
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$flags" -- "$cur") )
    fi
}}
complete -F _ai_benchmark_complete ai-benchmark ai-benchmark.py
"""
    if shell == "zsh":
        plugin_quoted = " ".join(shlex.quote(p.id) for p in plugins)
        flags_quoted = " ".join(f'"{f}"' for f in flags)
        return f"""#compdef ai-benchmark ai-benchmark.py

local plugin_ids=({plugin_quoted})
local flags=({flags_quoted})

_arguments \\
    '--restart[Restart the run from scratch, discarding prior results]' \\
    '--config[Config file path]:file:_files' \\
    '--out[Override output directory from config]:directory:_directories' \\
    '--timeout[Override request timeout in seconds from config]:timeout:' \\
    '--token-levels[Override token levels]:token levels:' \\
    '--plugin-temperature[Per-plugin temperatures]:temperature:' \\
    '--plugin-thread-limit[Max threads per model for plugin execution]:thread limit:' \\
    '--plugins-whitelist[Run only these plugins]:plugin:->plugin' \\
    '--plugins-blacklist[Run all plugins except these]:plugin:->plugin' \\
    '--list-plugins[List discovered plugins with their IDs, names, and versions]' \\
    '--generate-shell-completion[Generate shell completion script]:shell:(bash zsh fish)' \\
    '--dump-default-config[Print a default config file to stdout and exit]' \\
    '--base-url[Base URL for model discovery via /v1/models API]:url:' \\
    '--api-key[API key for model discovery]:key:' \\
    '--save-responses[Save each model\'s plugin response text to <output_dir>/responses/]' \\
    '--no-preload[Disable per-source model pre-loading for this run]' \\        '--runner[Execution runner]:runner:(http opencode both)' \\
    '--judge-models[Configured semantic judge model]:model:' \\

    '*:file:_files'

case \"$state\" in
    plugin)
        _describe -t plugin-ids 'plugin IDs' plugin_ids
        ;;
esac
"""
    if shell == "fish":
        plugin_ids_escaped = " ".join(p.id for p in plugins)
        flag_specs = [
            "-l restart -d 'Restart the run from scratch, discarding prior results'",
            "-l config -r -F -d 'Config file path'",
            "-l out -r -d 'Override output directory from config'",
            "-l timeout -r -d 'Override request timeout in seconds from config'",
            "-l token-levels -r -d 'Override token levels'",
            "-l plugin-temperature -r -d 'Per-plugin temperatures'",
            "-l plugin-thread-limit -r -d 'Max threads per model for plugin execution'",
            f"-l plugins-whitelist -x -a '{plugin_ids_escaped}' -d 'Run only these plugins'",
            f"-l plugins-blacklist -x -a '{plugin_ids_escaped}' -d 'Run all plugins except these'",
            "-l list-plugins -d 'List discovered plugins with their IDs, names, and versions'",
            "-l generate-shell-completion -x -a 'bash zsh fish' -d 'Generate shell completion script'",
            "-l dump-default-config -d 'Print a default config file to stdout and exit'",
            "-l base-url -F -d 'Base URL for model discovery via /v1/models API'",
            "-l api-key -F -d 'API key for model discovery'",
            "-l save-responses -d 'Save each model\\'s plugin response text to <output_dir>/responses/'",
            "-l no-preload -d 'Disable per-source model pre-loading for this run'",
            "-l runner -x -a 'http opencode both' -d 'Execution runner'",
            "-l judge-models -r -d 'Configured semantic judge model'",
        ]
        lines = [
            f"complete -c {cmd} {spec}"
            for cmd in COMMAND_NAMES
            for spec in flag_specs
        ]
        return "\n".join(lines) + "\n"

    return ""
