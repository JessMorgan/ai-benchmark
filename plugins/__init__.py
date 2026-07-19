"""Plugin discovery and selection for the AI benchmark."""
import importlib.util
import inspect
import os
import sys

from benchmark_plugin import BenchmarkTaskPlugin, BenchmarkOutputPlugin


BASE_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CHALLENGES_DIR = os.path.join(BASE_PLUGIN_DIR, "challenges")
OUTPUTS_DIR = os.path.join(BASE_PLUGIN_DIR, "outputs")


def format_plugin_list(plugins):
    """Return a formatted table of plugin IDs, names, and versions."""
    if not plugins:
        return "No plugins discovered."
    id_width = max(len(p.id) for p in plugins)
    name_width = max(len(p.name) for p in plugins)
    lines = [f"{'ID':<{id_width}}  {'Name':<{name_width}}  Version"]
    for p in plugins:
        lines.append(f"{p.id:<{id_width}}  {p.name:<{name_width}}  {p.version}")
    lines.append("\nUse these IDs with --plugins-whitelist or --plugins-blacklist.")
    return "\n".join(lines)


def _discover_plugins_in_dir(directory, package_name, base_class):
    """Discover and instantiate plugins from a directory.

    Args:
        directory: Path to the directory to scan.
        package_name: Dotted package name to use for dynamic imports.
        base_class: Base class that discovered plugin classes must inherit from.

    Returns:
        A list of plugin instances ordered by module name.
    """
    plugins = []
    if not os.path.isdir(directory):
        return plugins

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        path = os.path.join(directory, filename)
        module_name = f"{package_name}.{filename[:-3]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        # Make the parent package importable for relative imports if needed
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is base_class:
                continue
            if not issubclass(obj, base_class):
                continue
            try:
                plugin = obj()
            except Exception as exc:
                raise RuntimeError(f"Failed to instantiate plugin {obj.__name__}") from exc
            plugins.append(plugin)

    plugins.sort(key=lambda p: p.id)
    return plugins


def discover_plugins(whitelist=None, blacklist=None):
    """Discover and instantiate challenge plugins from the plugins/challenges/ directory.

    Args:
        whitelist: Optional iterable of plugin IDs to include.
        blacklist: Optional iterable of plugin IDs to exclude.

    Returns:
        A list of BenchmarkTaskPlugin instances ordered by plugin ID.
    """
    whitelist = set(whitelist or [])
    blacklist = set(blacklist or [])
    if whitelist and blacklist:
        raise ValueError("Cannot specify both plugin whitelist and blacklist")

    plugins = _discover_plugins_in_dir(CHALLENGES_DIR, "plugins.challenges", BenchmarkTaskPlugin)

    if whitelist:
        plugins = [p for p in plugins if p.id in whitelist]
    if blacklist:
        plugins = [p for p in plugins if p.id not in blacklist]

    return plugins


def discover_output_plugins(whitelist=None, blacklist=None):
    """Discover and instantiate output plugins from the plugins/outputs/ directory.

    Args:
        whitelist: Optional iterable of plugin IDs to include.
        blacklist: Optional iterable of plugin IDs to exclude.

    Returns:
        A list of BenchmarkOutputPlugin instances ordered by plugin ID.
    """
    whitelist = set(whitelist or [])
    blacklist = set(blacklist or [])
    if whitelist and blacklist:
        raise ValueError("Cannot specify both plugin whitelist and blacklist")

    plugins = _discover_plugins_in_dir(OUTPUTS_DIR, "plugins.outputs", BenchmarkOutputPlugin)

    if whitelist:
        plugins = [p for p in plugins if p.id in whitelist]
    if blacklist:
        plugins = [p for p in plugins if p.id not in blacklist]

    return plugins
