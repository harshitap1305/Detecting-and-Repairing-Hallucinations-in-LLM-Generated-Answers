"""
Unit tests for llm.py cache layer — no real API calls.
Run with:
    python3 -m unittest tests/test_llm.py
    OR: pytest tests/test_llm.py -v
"""
import os
import tempfile
import unittest

# Point cache to a temp file before importing the module
_tmp_db = tempfile.mktemp(suffix=".db")
os.environ["LLM_CACHE_PATH"] = _tmp_db

import src.llm as llm_module
from src.llm import (
    _cache_get,
    _cache_set,
    _make_key,
    cache_stats,
    reset_connection,
)


class TestCacheKey(unittest.TestCase):
    def test_same_inputs_same_key(self):
        k1 = _make_key("m", "hello", 0.0, 100, 42)
        k2 = _make_key("m", "hello", 0.0, 100, 42)
        self.assertEqual(k1, k2)

    def test_different_prompts(self):
        k1 = _make_key("m", "prompt A", 0.0, 100, 42)
        k2 = _make_key("m", "prompt B", 0.0, 100, 42)
        self.assertNotEqual(k1, k2)

    def test_different_temperatures(self):
        k1 = _make_key("m", "p", 0.0, 100, 42)
        k2 = _make_key("m", "p", 1.0, 100, 42)
        self.assertNotEqual(k1, k2)

    def test_different_models(self):
        k1 = _make_key("model-a", "p", 0.0, 100, 42)
        k2 = _make_key("model-b", "p", 0.0, 100, 42)
        self.assertNotEqual(k1, k2)

    def test_different_seeds(self):
        k1 = _make_key("m", "p", 0.0, 100, 1)
        k2 = _make_key("m", "p", 0.0, 100, 2)
        self.assertNotEqual(k1, k2)


class TestCacheRoundtrip(unittest.TestCase):
    def setUp(self):
        reset_connection()

    def tearDown(self):
        reset_connection()

    def test_miss_returns_none(self):
        key = _make_key("model-x", "unique prompt 999", 0.0, 100, 42)
        self.assertIsNone(_cache_get(key))

    def test_set_then_get(self):
        key = _make_key("model-x", "cache test prompt", 0.0, 100, 42)
        _cache_set(key, "model-x", "test response text")
        result = _cache_get(key)
        self.assertEqual(result, "test response text")

    def test_overwrite(self):
        key = _make_key("m", "overwrite test", 0.0, 100, 1)
        _cache_set(key, "m", "first")
        _cache_set(key, "m", "second")
        self.assertEqual(_cache_get(key), "second")

    def test_cache_stats(self):
        key = _make_key("stats-model", "stats prompt", 0.0, 100, 1)
        _cache_set(key, "stats-model", "response")
        stats = cache_stats()
        self.assertGreaterEqual(stats["total"], 1)
        self.assertIn("stats-model", stats["by_model"])


if __name__ == "__main__":
    unittest.main()
