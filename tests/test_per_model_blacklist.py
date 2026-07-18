import unittest

from benchmark_core import get_model_plugins_blacklist


class TestPerModelPluginBlacklist(unittest.TestCase):
    def test_string_model_returns_empty_blacklist(self):
        models = {"model-a": "source-1"}
        self.assertEqual(get_model_plugins_blacklist(models, "model-a"), [])

    def test_dict_model_without_blacklist_returns_empty(self):
        models = {"model-a": {"source": "source-1"}}
        self.assertEqual(get_model_plugins_blacklist(models, "model-a"), [])

    def test_dict_model_with_blacklist_returns_list(self):
        models = {"model-a": {"source": "source-1", "plugins_blacklist": ["rate-limiter"]}}
        self.assertEqual(get_model_plugins_blacklist(models, "model-a"), ["rate-limiter"])

    def test_dict_model_with_multiple_blacklisted_plugins(self):
        models = {"model-a": {"source": "source-1", "plugins_blacklist": ["rate-limiter", "moe-dense"]}}
        self.assertEqual(get_model_plugins_blacklist(models, "model-a"), ["rate-limiter", "moe-dense"])

    def test_nonexistent_model_returns_empty_blacklist(self):
        models = {"model-a": {"source": "source-1", "plugins_blacklist": ["rate-limiter"]}}
        self.assertEqual(get_model_plugins_blacklist(models, "nonexistent"), [])

    def test_empty_models_returns_empty_blacklist(self):
        models = {}
        self.assertEqual(get_model_plugins_blacklist(models, "model-a"), [])
