"""Plugin discovery and selection for the AI benchmark."""
import importlib.util
import inspect
import os
import sys

from benchmark_plugin import BenchmarkTaskPlugin, BenchmarkOutputPlugin


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


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


def discover_plugins(whitelist=None, blacklist=None):
    """Discover and instantiate plugins from the plugins/ directory.

    Args:
        whitelist: Optional iterable of plugin IDs to include.
        blacklist: Optional iterable of plugin IDs to exclude.

    Returns:
        A list of BenchmarkTaskPlugin instances ordered by module name.
    """
    whitelist = set(whitelist or [])
    blacklist = set(blacklist or [])
    if whitelist and blacklist:
        raise ValueError("Cannot specify both plugin whitelist and blacklist")

    plugins = []
    for filename in sorted(os.listdir(PLUGIN_DIR)):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        path = os.path.join(PLUGIN_DIR, filename)
        module_name = f"plugins.{filename[:-3]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        # Make the parent package importable for relative imports if needed
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BenchmarkTaskPlugin:
                continue
            if not issubclass(obj, BenchmarkTaskPlugin):
                continue
            try:
                plugin = obj()
            except Exception as exc:
                raise RuntimeError(f"Failed to instantiate plugin {obj.__name__}") from exc
            plugins.append(plugin)

    # Sort by ID for stable ordering
    plugins.sort(key=lambda p: p.id)

    if whitelist:
        plugins = [p for p in plugins if p.id in whitelist]
    if blacklist:
        plugins = [p for p in plugins if p.id not in blacklist]

    return plugins


def discover_output_plugins(whitelist=None, blacklist=None):
    """Discover and instantiate output plugins from the plugins/ directory.

    Args:
        whitelist: Optional iterable of plugin IDs to include.
        blacklist: Optional iterable of plugin IDs to exclude.

    Returns:
        A list of BenchmarkOutputPlugin instances ordered by module name.
    """
    whitelist = set(whitelist or [])
    blacklist = set(blacklist or [])
    if whitelist and blacklist:
        raise ValueError("Cannot specify both plugin whitelist and blacklist")

    plugins = []
    for filename in sorted(os.listdir(PLUGIN_DIR)):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        path = os.path.join(PLUGIN_DIR, filename)
        module_name = f"plugins.{filename[:-3]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BenchmarkOutputPlugin:
                continue
            if not issubclass(obj, BenchmarkOutputPlugin):
                continue
            try:
                plugin = obj()
            except Exception as exc:
                raise RuntimeError(f"Failed to instantiate output plugin {obj.__name__}") from exc
            plugins.append(plugin)

    plugins.sort(key=lambda p: p.id)

    if whitelist:
        plugins = [p for p in plugins if p.id in whitelist]
    if blacklist:
        plugins = [p for p in plugins if p.id not in blacklist]

    return plugins
