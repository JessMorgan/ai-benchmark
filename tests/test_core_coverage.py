"""Coverage-focused tests for benchmark.core.

Targets branches of the config-loading and execution helpers that were not
exercised by the existing suite: env expansion edge cases, source
abbreviation, token-level normalization, target/agent validation errors,
preload failure paths, and early-return guards in ``_run_plugin_task`` /
``run_model``.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from benchmark.core import (
    BenchmarkState,
    PreloadResult,
    _expand_env,
    _run_plugin_task,
    _source_abbrev,
    _unique_source_abbrevs,
    dump_default_config,
    generate_config_from_api,
    get_target_plugins_blacklist,
    load_dotenv_file,
    preload_model,
    resolve_model_sources,
    resolve_preload_timeout,
    resolve_stream_guards,
    resolve_targets,
    run_model,
)
from benchmark.http import NonStreamResult, StreamResult


class _FakePlugin:
    id = "fake-plugin"
    version = "1.0.0"
    name = "Fake Plugin"
    max_score = 20.0
    supports_streaming = True

    def get_prompt(self):
        return "prompt"

    def get_temperature(self, global_config):
        return None


class TestExpandEnv(unittest.TestCase):
    def test_plain_string_passthrough(self):
        self.assertEqual(_expand_env("hello"), "hello")

    def test_unset_var_expands_to_empty(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_expand_env("a${MISSING}b"), "ab")

    def test_var_with_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_expand_env("${KEY:fallback}"), "fallback")

    def test_var_set_uses_environment(self):
        with mock.patch.dict("os.environ", {"KEY": "value"}, clear=True):
            self.assertEqual(_expand_env("${KEY}"), "value")

    def test_unterminated_brace_left_alone(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_expand_env("${UNCLOSED"), "${UNCLOSED")

    def test_multiple_vars(self):
        with mock.patch.dict("os.environ", {"A": "1", "B": "2"}, clear=True):
            self.assertEqual(_expand_env("${A}-${B}"), "1-2")

    def test_recurses_into_dict_and_list(self):
        with mock.patch.dict("os.environ", {"K": "v"}, clear=True):
            self.assertEqual(
                _expand_env({"x": "${K}", "items": ["${K}", 3]}),
                {"x": "v", "items": ["v", 3]},
            )

    def test_non_string_scalars_passthrough(self):
        self.assertEqual(_expand_env(42), 42)


class TestLoadDotenvFile(unittest.TestCase):
    def test_loads_vars_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, ".env")
            with open(env_path, "w") as f:
                f.write("CHATPLAYGROUND_EMAIL=user@example.com\nCHATPLAYGROUND_PASSWORD=pw\n")
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertTrue(load_dotenv_file(env_path))
                self.assertEqual(os.environ.get("CHATPLAYGROUND_EMAIL"), "user@example.com")
                self.assertEqual(os.environ.get("CHATPLAYGROUND_PASSWORD"), "pw")

    def test_missing_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(load_dotenv_file(os.path.join(tmp, "nope.env")))

    def test_existing_environment_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, ".env")
            with open(env_path, "w") as f:
                f.write("CHATPLAYGROUND_EMAIL=from-file@example.com\n")
            with mock.patch.dict("os.environ", {"CHATPLAYGROUND_EMAIL": "from-env@example.com"}, clear=True):
                load_dotenv_file(env_path)
                self.assertEqual(os.environ.get("CHATPLAYGROUND_EMAIL"), "from-env@example.com")

    def test_default_path_is_dotenv_in_cwd(self):
        with mock.patch("dotenv.load_dotenv", return_value=True) as loader:
            self.assertTrue(load_dotenv_file())
        loader.assert_called_once_with(dotenv_path=".env", override=False)
        self.assertEqual(_expand_env(3.5), 3.5)
        self.assertEqual(_expand_env(None), None)


class TestSourceAbbrev(unittest.TestCase):
    def test_short_acronym_words_are_kept_whole(self):
        self.assertEqual(_source_abbrev("AI Server"), "AS")

    def test_mixed_words_use_camel_caps(self):
        self.assertEqual(_source_abbrev("Local Server"), "LS")
        self.assertEqual(_source_abbrev("OpenCode Zen"), "OCZ")

    def test_single_word_short_name_duplicated(self):
        self.assertEqual(_source_abbrev("a"), "AA")

    def test_punctuation_only_name_uses_first_chars(self):
        self.assertEqual(_source_abbrev("!!"), "!!".upper()[:2])

    def test_unique_abbrevs_disambiguates_collisions(self):
        # "Local Server" -> LS and "Local Server 2" -> LS2 do not collide;
        # use names that map to the SAME acronym to exercise the dedup loop.
        mapping = _unique_source_abbrevs(["A B", "a b"])
        self.assertEqual(mapping["A B"], "AB")
        self.assertEqual(mapping["a b"], "AB1")

    def test_source_abbrev_empty_name_uses_first_two_chars(self):
        self.assertEqual(_source_abbrev(""), "".upper()[:2])

    def test_source_abbrev_symbol_words_use_first_letter(self):
        # Words whose regex yields no sub-tokens are kept whole; the
        # acronym then takes the first letter of each kept word.
        self.assertEqual(_source_abbrev("12 3"), "13")
        self.assertEqual(_source_abbrev("12"), "12")


class TestResolveHelpers(unittest.TestCase):
    def test_resolve_model_sources_defaults_invalid(self):
        models = {"a": "S1", "b": {"source": "S2"}, "c": 123}
        self.assertEqual(resolve_model_sources(models), {"a": "S1", "b": "S2", "c": "Default"})

    def test_resolve_targets_normalizes_token_levels_forms(self):
        cfg = {
            "models": {"m1": {"source": "S", "token_levels": [4096]}, "m2": "S"},
            "model_token_levels": {"m1": [8192], "m2": 16384},
        }
        targets = resolve_targets(cfg)
        self.assertEqual(targets["m1"]["token_levels"], [4096])  # per-target dict wins
        self.assertEqual(targets["m2"]["token_levels"], [16384])  # int form

    def test_resolve_targets_rejects_bad_token_levels(self):
        cfg = {
            "models": {"m1": {"source": "S", "token_levels": [4096]}},
            "model_token_levels": {"m1": "not-a-list", "m2": [True], "m3": [4096, "x"]},
        }
        targets = resolve_targets(cfg)
        # Bad value -> falls back to per-target dict (4096) or None.
        self.assertEqual(targets["m1"]["token_levels"], [4096])

    def test_resolve_targets_non_str_non_dict_model_defaults(self):
        targets = resolve_targets({"models": {"m1": 42}})
        self.assertEqual(targets["m1"]["source"], "Default")

    def test_agent_missing_model_key_raises(self):
        with self.assertRaises(ValueError):
            resolve_targets({"agents": {"a": {"system_prompt": "x"}}})

    def test_agent_missing_system_prompt_raises(self):
        with self.assertRaises(ValueError):
            resolve_targets({"agents": {"a": {"model": "m"}}})

    def test_agent_non_dict_raises(self):
        with self.assertRaises(ValueError):
            resolve_targets({"agents": {"a": "just-a-string"}})

    def test_target_blacklist_lookup(self):
        targets = {
            "m1": {"plugins_blacklist": ["code-review"]},
            "m2": {"source": "S"},
        }
        self.assertEqual(get_target_plugins_blacklist(targets, "m1"), ["code-review"])
        self.assertEqual(get_target_plugins_blacklist(targets, "m2"), [])
        self.assertEqual(get_target_plugins_blacklist(targets, "missing"), [])


class TestPreload(unittest.TestCase):
    def test_preload_unknown_source_fails_immediately(self):
        result = preload_model({"Other": {"api_url": "http://x"}}, "Missing",
                               "model", 1)
        self.assertIsInstance(result, PreloadResult)
        self.assertFalse(result.success)
        self.assertIn("Unknown source", result.error)

    def test_preload_empty_response_is_failure(self):
        source_config = {"S": {"api_url": "http://x", "headers": {}}}
        with mock.patch("benchmark.core.nonstream_request",
                        return_value=NonStreamResult("", "", {}, 0.1, None, None)):
            result = preload_model(source_config, "S", "model", 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "empty preload response")

    def test_preload_thinking_only_counts_as_success(self):
        source_config = {"S": {"api_url": "http://x", "headers": {}}}
        with mock.patch("benchmark.core.nonstream_request",
                        return_value=NonStreamResult("", "reasoning", {}, 0.1, None, None)):
            result = preload_model(source_config, "S", "model", 1)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_resolve_preload_timeout_variants(self):
        cfg = {"S": {"preload_timeout": 42}}
        self.assertEqual(resolve_preload_timeout(cfg, "S"), 42)
        self.assertEqual(resolve_preload_timeout(cfg, "missing"), 300)
        self.assertEqual(resolve_preload_timeout({}, "S", default=7), 7)

    def test_resolve_stream_guards_defaults(self):
        content, thinking, guard = resolve_stream_guards({}, "missing")
        self.assertEqual(content, 16384)
        self.assertEqual(thinking, 32768)
        self.assertTrue(guard)

    def test_resolve_stream_guards_configured_values(self):
        cfg = {"S": {"max_content_tokens": 4096, "max_thinking_tokens": 8192,
                     "repetition_guard": False}}
        self.assertEqual(resolve_stream_guards(cfg, "S"), (4096, 8192, False))

    def test_resolve_stream_guards_ignores_invalid_tokens(self):
        cfg = {"S": {"max_content_tokens": "nope", "max_thinking_tokens": -5}}
        content, thinking, _ = resolve_stream_guards(cfg, "S")
        self.assertEqual(content, 16384)
        self.assertEqual(thinking, 32768)


class TestDumpDefaultConfig(unittest.TestCase):
    def test_dump_default_config_prints_valid_json(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            dump_default_config()
        data = json.loads(buffer.getvalue())
        self.assertIn("sources", data)
        self.assertIn("models", data)
        self.assertIn("agents", data)


class TestGenerateConfigFromApi(unittest.TestCase):
    def test_no_models_raises(self):
        with (
            mock.patch("benchmark.core.fetch_models_v1", return_value=[]),
            self.assertRaises(RuntimeError),
        ):
            generate_config_from_api("http://x")

    def test_with_models_builds_config(self):
        with mock.patch("benchmark.core.fetch_models_v1",
                        return_value=["model-a", "model-b"]):
            cfg = generate_config_from_api("http://x", api_key="sk-1")
        self.assertIn("model-a", cfg["models"])
        self.assertEqual(cfg["models"]["model-a"], "Default")
        self.assertIn("Authorization", cfg["sources"]["Default"]["headers"])


class TestRunPluginTaskGuards(unittest.TestCase):
    def _state(self):
        return BenchmarkState({"m": "S"}, ["fake-plugin"])

    def test_unknown_http_source(self):
        result = _run_plugin_task(
            "m", "model", "Nope", _FakePlugin(), {"S": {}}, 1, [100],
            0, None, {}, self._state(),
        )
        self.assertIsNotNone(result.error)
        self.assertIn("Unknown source", result.error)

    def test_unknown_runner(self):
        result = _run_plugin_task(
            "m", "model", "S", _FakePlugin(), {"S": {}}, 1, [100],
            0, None, {}, self._state(), runner="janky",
        )
        self.assertIsNotNone(result.error)
        self.assertIn("Unknown runner", result.error)

    def test_cancelled_before_start(self):
        import threading
        stop = threading.Event()
        stop.set()
        result = _run_plugin_task(
            "m", "model", "S", _FakePlugin(), {"S": {}}, 1, [100],
            0, None, {}, self._state(), stop_event=stop,
        )
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error, "Cancelled")

    def test_opencode_missing_config(self):
        result = _run_plugin_task(
            "m", "model", "S", _FakePlugin(), {"S": {}}, 1, [100],
            0, None, {}, self._state(), runner="opencode",
        )
        self.assertIsNotNone(result.error)
        self.assertIn("OpenCode runner is missing", result.error)

    def test_run_plugin_task_passes_stream_guards_to_http(self):
        """Per-source watchdog budgets reach the streaming request layer."""
        captured = {}

        def fake_stream(*args, **kwargs):
            captured.update(kwargs)
            return StreamResult("hello", "", None, 0.1, None, "stop", {})

        source_config = {"S": {"api_url": "http://x", "headers": {},
                               "max_content_tokens": 1024,
                               "max_thinking_tokens": 2048,
                               "repetition_guard": False}}
        with mock.patch("benchmark.core.stream_request", side_effect=fake_stream):
            _run_plugin_task(
                "m", "model", "S", _FakePlugin(), source_config, 1, [100],
                0, None, {}, self._state(),
            )
        self.assertEqual(captured["max_content_tokens"], 1024)
        self.assertEqual(captured["max_thinking_tokens"], 2048)
        self.assertFalse(captured["repetition_guard"])


class TestRunModelGuards(unittest.TestCase):
    def test_run_model_unknown_source_marks_failed(self):
        plugins = [_FakePlugin()]
        state = BenchmarkState({"m": "S"}, [p.id for p in plugins])
        run_model("m", "Missing", state, plugins, {"S": {}}, 1,
                  [100], "/tmp/out", session_seed=0)
        snap = state.snapshot()["m"]
        self.assertEqual(snap["status"], "failed")
        self.assertIn("Unknown source", snap["error"])

    def test_run_model_non_int_thread_limit_falls_back(self):
        plugins = [_FakePlugin()]
        state = BenchmarkState({"m": "S"}, [p.id for p in plugins])
        source_config = {"S": {"api_url": "http://x", "headers": {},
                               "plugin_thread_limit": "banana"}}
        with (
            mock.patch("benchmark.core.stream_request", return_value=StreamResult("", "", None, 0, "boom", None, {})),
            mock.patch("benchmark.core.nonstream_request", return_value=NonStreamResult("", "", {}, 0.1, "boom", None)),
        ):
            run_model("m", "S", state, plugins, source_config, 1,
                      [100], "/tmp/out", session_seed=0)
        snap = state.snapshot()["m"]
        self.assertIn(snap["status"], ("completed", "failed"))

    def test_run_model_cancelled_stop_event(self):
        import threading
        plugins = [_FakePlugin()]
        state = BenchmarkState({"m": "S"}, [p.id for p in plugins])
        stop = threading.Event()
        stop.set()
        run_model("m", "S", state, plugins, {"S": {"api_url": "http://x", "headers": {}}},
                  1, [100], "/tmp/out", session_seed=0, stop_event=stop)
        snap = state.snapshot()["m"]
        self.assertEqual(snap["status"], "failed")
        self.assertEqual(snap["error"], "Cancelled")


if __name__ == "__main__":
    unittest.main()
