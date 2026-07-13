"""Shell completion script generation for the AI benchmark CLI."""
import shlex


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
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$flags" -- "$cur") )
    fi
}}
complete -F _ai_benchmark_complete ai-benchmark.py
"""
    if shell == "zsh":
        plugin_quoted = " ".join(shlex.quote(p.id) for p in plugins)
        flags_quoted = " ".join(f'"{f}"' for f in flags)
        return f"""#compdef ai-benchmark.py

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
    '*:file:_files'

case \"$state\" in
    plugin)
        _describe -t plugin-ids 'plugin IDs' plugin_ids
        ;;
esac
"""
    if shell == "fish":
        plugin_ids_escaped = " ".join(p.id for p in plugins)
        lines = [
            "complete -c ai-benchmark.py -l restart -d 'Restart the run from scratch, discarding prior results'",
            "complete -c ai-benchmark.py -l config -r -F -d 'Config file path'",
            "complete -c ai-benchmark.py -l out -r -d 'Override output directory from config'",
            "complete -c ai-benchmark.py -l timeout -r -d 'Override request timeout in seconds from config'",
            "complete -c ai-benchmark.py -l token-levels -r -d 'Override token levels'",
            "complete -c ai-benchmark.py -l plugin-temperature -r -d 'Per-plugin temperatures'",
            "complete -c ai-benchmark.py -l plugin-thread-limit -r -d 'Max threads per model for plugin execution'",
            f"complete -c ai-benchmark.py -l plugins-whitelist -x -a '{plugin_ids_escaped}' -d 'Run only these plugins'",
            f"complete -c ai-benchmark.py -l plugins-blacklist -x -a '{plugin_ids_escaped}' -d 'Run all plugins except these'",
            "complete -c ai-benchmark.py -l list-plugins -d 'List discovered plugins with their IDs, names, and versions'",
            "complete -c ai-benchmark.py -l generate-shell-completion -x -a 'bash zsh fish' -d 'Generate shell completion script'",
            "complete -c ai-benchmark.py -l dump-default-config -d 'Print a default config file to stdout and exit'",
            "complete -c ai-benchmark.py -l base-url -F -d 'Base URL for model discovery via /v1/models API'",
            "complete -c ai-benchmark.py -l api-key -F -d 'API key for model discovery'",
        ]
        return "\n".join(lines) + "\n"

    return ""
