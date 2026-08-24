"""Plugin discovery and selection for the AI benchmark."""
import importlib.util
import inspect
import os
import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Any, cast

from benchmark.plugin import BenchmarkOutputPlugin, BenchmarkTaskPlugin


class PluginDiscoveryError(RuntimeError):
    """Raised when a discovered plugin violates the plugin contract."""


def _validate_plugin(
    plugin: BenchmarkTaskPlugin | BenchmarkOutputPlugin,
    path: str,
    base_class: type[BenchmarkTaskPlugin] | type[BenchmarkOutputPlugin],
) -> None:
    """Validate required metadata before exposing a plugin instance."""
    required: tuple[str, ...] = ("id", "name")
    if base_class is BenchmarkTaskPlugin:
        required += ("version", "max_score")
    else:
        required += ("extension",)

    for attr in required:
        try:
            value = getattr(plugin, attr)
        except Exception as exc:
            raise PluginDiscoveryError(
                f"{path}: {type(plugin).__name__} could not provide {attr!r}"
            ) from exc
        if attr == "max_score":
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            )
            expected = "a positive number"
        else:
            valid = isinstance(value, str) and bool(value.strip())
            expected = "a non-empty string"
        if not valid:
            raise PluginDiscoveryError(
                f"{path}: {type(plugin).__name__}.{attr} must be {expected}"
            )

    if base_class is BenchmarkTaskPlugin:
        supports_streaming = getattr(plugin, "supports_streaming", True)
        if not isinstance(supports_streaming, bool):
            raise PluginDiscoveryError(
                f"{path}: {type(plugin).__name__}.supports_streaming must be a boolean"
            )


def _validate_unique_ids(
    plugins: Iterable[BenchmarkTaskPlugin | BenchmarkOutputPlugin], directory: str,
) -> None:
    """Reject duplicate IDs so resume/report keys cannot collide."""
    seen: dict[str, str] = {}
    for plugin in plugins:
        previous = seen.get(plugin.id)
        if previous is not None:
            raise PluginDiscoveryError(
                f"Duplicate plugin id {plugin.id!r} in {directory}: "
                f"{previous} and {type(plugin).__name__}"
            )
        seen[plugin.id] = type(plugin).__name__


def _plugin_inventory(
    plugins: Iterable[BenchmarkTaskPlugin],
) -> list[dict[str, Any]]:
    """Return a stable, serializable inventory for CLI/docs/tests."""
    return [
        {
            "id": plugin.id,
            "name": plugin.name,
            "version": plugin.version,
            "max_score": plugin.max_score,
            "supports_streaming": getattr(plugin, "supports_streaming", True),
        }
        for plugin in plugins
    ]


BASE_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CHALLENGES_DIR = os.path.join(BASE_PLUGIN_DIR, "challenges")
OUTPUTS_DIR = os.path.join(BASE_PLUGIN_DIR, "outputs")


def format_plugin_list(plugins: Iterable[BenchmarkTaskPlugin]) -> str:
    """Return a formatted table generated from validated plugin metadata."""
    inventory = _plugin_inventory(plugins)
    if not inventory:
        return "No plugins discovered."
    id_width = max(len(entry["id"]) for entry in inventory)
    name_width = max(len(entry["name"]) for entry in inventory)
    lines = [f"{'ID':<{id_width}}  {'Name':<{name_width}}  Version"]
    for entry in inventory:
        lines.append(
            f"{entry['id']:<{id_width}}  {entry['name']:<{name_width}}  "
            f"{entry['version']}"
        )
    lines.append("\nUse these IDs with --plugins-whitelist or --plugins-blacklist.")
    return "\n".join(lines)


def _discover_plugins_in_dir(
    directory: str,
    package_name: str,
    base_class: type[BenchmarkTaskPlugin] | type[BenchmarkOutputPlugin],
) -> list[Any]:
    """Discover and instantiate plugins from a directory.

    Args:
        directory: Path to the directory to scan.
        package_name: Dotted package name to use for dynamic imports.
        base_class: Base class that discovered plugin classes must inherit from.

    Returns:
        A list of plugin instances ordered by module name.
    """
    plugins: list[Any] = []
    if not os.path.isdir(directory):
        return plugins

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename.startswith(("__", "test_")):
            continue
        path = os.path.join(directory, filename)
        module_name = f"{package_name}.{filename[:-3]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PluginDiscoveryError(f"{path}: could not create import specification")
        module: ModuleType = importlib.util.module_from_spec(spec)
        # Make the parent package importable for relative imports if needed
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is base_class or obj.__module__ != module.__name__:
                continue
            if not issubclass(obj, base_class):
                continue
            try:
                plugin = obj()
            except Exception as exc:
                raise RuntimeError(f"Failed to instantiate plugin {obj.__name__}") from exc
            _validate_plugin(plugin, path, base_class)
            plugins.append(plugin)

    _validate_unique_ids(plugins, directory)
    plugins.sort(key=lambda p: p.id)
    return plugins


def discover_plugins(
    whitelist: Iterable[str] | None = None,
    blacklist: Iterable[str] | None = None,
) -> list[BenchmarkTaskPlugin]:
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

    plugins = _discover_plugins_in_dir(
        CHALLENGES_DIR,
        "plugins.challenges",
        BenchmarkTaskPlugin,
    )

    if whitelist:
        plugins = [p for p in plugins if p.id in whitelist]
    if blacklist:
        plugins = [p for p in plugins if p.id not in blacklist]

    return cast(list[BenchmarkTaskPlugin], plugins)


def plugin_inventory(plugins: Iterable[BenchmarkTaskPlugin] | None = None) -> list[dict[str, Any]]:
    """Return metadata for all discovered challenge plugins."""
    return _plugin_inventory(plugins if plugins is not None else discover_plugins())


def discover_output_plugins(
    whitelist: Iterable[str] | None = None,
    blacklist: Iterable[str] | None = None,
) -> list[BenchmarkOutputPlugin]:
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

    plugins = _discover_plugins_in_dir(
        OUTPUTS_DIR,
        "plugins.outputs",
        BenchmarkOutputPlugin,
    )

    if whitelist:
        plugins = [p for p in plugins if p.id in whitelist]
    if blacklist:
        plugins = [p for p in plugins if p.id not in blacklist]

    return cast(list[BenchmarkOutputPlugin], plugins)
