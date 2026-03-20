"""
RagMark — Comprehensive test suite.

Covers: config loading, pass-through modes, strategy decisions,
bookmark capping, reset, empty inputs, and basic latency.
"""

import math
import os
import tempfile
import time
import unittest

# Ensure we can import from the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragmark import RagMark, Node, RagMarkConfig, load_config


# ---------------------------------------------------------------------------
# Helper — write a temp YAML and return its path
# ---------------------------------------------------------------------------

def _write_yaml(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# 1. Config Tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    """YAML loading and validation."""

    def test_load_valid_config(self):
        path = _write_yaml(
            "ragmark:\n  enabled: true\n  max_bookmarks: 3\n"
            "  similarity_threshold: 0.8\n  rag_door: post\n  debug: false\n"
        )
        cfg = load_config(path)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_bookmarks, 3)
        self.assertAlmostEqual(cfg.similarity_threshold, 0.8)
        self.assertEqual(cfg.rag_door, "post")
        self.assertFalse(cfg.debug)
        os.unlink(path)

    def test_load_defaults(self):
        path = _write_yaml("ragmark:\n  enabled: true\n")
        cfg = load_config(path)
        self.assertEqual(cfg.max_bookmarks, 5)
        self.assertAlmostEqual(cfg.similarity_threshold, 0.75)
        self.assertEqual(cfg.rag_door, "pre")
        os.unlink(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_file.yaml")

    def test_missing_ragmark_key(self):
        path = _write_yaml("other_key: 42\n")
        with self.assertRaises(ValueError):
            load_config(path)
        os.unlink(path)

    def test_invalid_rag_door(self):
        with self.assertRaises(ValueError):
            RagMarkConfig(rag_door="invalid")

    def test_invalid_threshold_high(self):
        with self.assertRaises(ValueError):
            RagMarkConfig(similarity_threshold=1.5)

    def test_invalid_threshold_low(self):
        with self.assertRaises(ValueError):
            RagMarkConfig(similarity_threshold=-0.1)

    def test_invalid_max_bookmarks(self):
        with self.assertRaises(ValueError):
            RagMarkConfig(max_bookmarks=0)


# ---------------------------------------------------------------------------
# 2. Pass-through Tests
# ---------------------------------------------------------------------------

class TestPassThrough(unittest.TestCase):
    """enabled=false and rag_door=off should be transparent."""

    def test_disabled(self):
        rm = RagMark(config=RagMarkConfig(enabled=False))
        scope = rm.get_scope([0.1, 0.2])
        self.assertEqual(scope["strategy"], "global")
        self.assertEqual(scope["node_ids"], [])

    def test_rag_door_off(self):
        rm = RagMark(config=RagMarkConfig(rag_door="off"))
        scope = rm.get_scope([0.1, 0.2])
        self.assertEqual(scope["strategy"], "global")

    def test_disabled_update_is_noop(self):
        rm = RagMark(config=RagMarkConfig(enabled=False))
        rm.update([Node("a", [1.0, 0.0])], [1.0, 0.0])
        self.assertEqual(rm.bookmark_count, 0)


# ---------------------------------------------------------------------------
# 3. Strategy Decision Tests (rag_door=pre)
# ---------------------------------------------------------------------------

class TestStrategyDecision(unittest.TestCase):
    """Local vs global switching based on similarity."""

    def _make_rm(self, threshold: float = 0.75) -> RagMark:
        return RagMark(config=RagMarkConfig(
            similarity_threshold=threshold, max_bookmarks=5
        ))

    def test_no_bookmarks_returns_global(self):
        rm = self._make_rm()
        scope = rm.get_scope([1.0, 0.0])
        self.assertEqual(scope["strategy"], "global")
        self.assertAlmostEqual(scope["confidence"], 0.0)

    def test_high_similarity_returns_local(self):
        rm = self._make_rm(threshold=0.5)
        # identical vectors → cosine similarity = 1.0
        rm.update([Node("x", [1.0, 0.0])], [1.0, 0.0])
        scope = rm.get_scope([1.0, 0.0])
        self.assertEqual(scope["strategy"], "local")
        self.assertAlmostEqual(scope["confidence"], 1.0, places=4)
        self.assertIn("x", scope["node_ids"])

    def test_low_similarity_returns_global(self):
        rm = self._make_rm(threshold=0.99)
        rm.update([Node("y", [1.0, 0.0])], [1.0, 0.0])
        # orthogonal query
        scope = rm.get_scope([0.0, 1.0])
        self.assertEqual(scope["strategy"], "global")
        self.assertEqual(scope["node_ids"], [])

    def test_confidence_equals_highest_similarity(self):
        rm = self._make_rm(threshold=0.1)
        rm.update([
            Node("a", [1.0, 0.0]),
            Node("b", [0.0, 1.0]),
        ], [1.0, 0.0])
        scope = rm.get_scope([1.0, 0.0])
        self.assertAlmostEqual(scope["confidence"], 1.0, places=4)


# ---------------------------------------------------------------------------
# 4. Post-retrieval Mode
# ---------------------------------------------------------------------------

class TestPostMode(unittest.TestCase):
    """rag_door=post — no decision, only tracking."""

    def test_post_always_global(self):
        rm = RagMark(config=RagMarkConfig(rag_door="post"))
        rm.update([Node("a", [1.0, 0.0])], [1.0, 0.0])
        scope = rm.get_scope([1.0, 0.0])
        self.assertEqual(scope["strategy"], "global")

    def test_post_still_tracks_bookmarks(self):
        rm = RagMark(config=RagMarkConfig(rag_door="post"))
        rm.update([Node("a", [1.0, 0.0])], [1.0, 0.0])
        self.assertEqual(rm.bookmark_count, 1)


# ---------------------------------------------------------------------------
# 5. Bookmark Cap (max_bookmarks)
# ---------------------------------------------------------------------------

class TestBookmarkCap(unittest.TestCase):
    def test_cap_enforced(self):
        rm = RagMark(config=RagMarkConfig(max_bookmarks=2))
        rm.update([
            Node("a", [1.0, 0.0]),
            Node("b", [0.9, 0.1]),
            Node("c", [0.0, 1.0]),
        ], [1.0, 0.0])
        self.assertEqual(rm.bookmark_count, 2)
        # "a" and "b" should survive (highest similarity to [1,0])
        self.assertIn("a", rm.bookmark_ids)
        self.assertIn("b", rm.bookmark_ids)
        self.assertNotIn("c", rm.bookmark_ids)


# ---------------------------------------------------------------------------
# 6. Reset
# ---------------------------------------------------------------------------

class TestReset(unittest.TestCase):
    def test_reset_clears_all(self):
        rm = RagMark(config=RagMarkConfig())
        rm.update([Node("a", [1.0, 0.0])], [1.0, 0.0])
        self.assertEqual(rm.bookmark_count, 1)
        rm.reset()
        self.assertEqual(rm.bookmark_count, 0)


# ---------------------------------------------------------------------------
# 7. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Empty inputs, duplicate ids, mismatched dimensions."""

    def test_empty_nodes_update(self):
        rm = RagMark(config=RagMarkConfig())
        rm.update([], [1.0, 0.0])
        self.assertEqual(rm.bookmark_count, 0)

    def test_empty_query_embedding(self):
        rm = RagMark(config=RagMarkConfig())
        scope = rm.get_scope([])
        self.assertEqual(scope["strategy"], "global")

    def test_duplicate_id_replaces(self):
        rm = RagMark(config=RagMarkConfig())
        rm.update([Node("a", [1.0, 0.0])], [1.0, 0.0])
        rm.update([Node("a", [0.0, 1.0])], [0.0, 1.0])
        self.assertEqual(rm.bookmark_count, 1)

    def test_mismatched_dimensions_no_crash(self):
        rm = RagMark(config=RagMarkConfig())
        rm.update([Node("a", [1.0, 0.0, 0.0])], [1.0, 0.0, 0.0])
        # query with different dim — should not crash, similarity → 0
        scope = rm.get_scope([1.0, 0.0])
        self.assertEqual(scope["strategy"], "global")


# ---------------------------------------------------------------------------
# 8. Latency Spot Check
# ---------------------------------------------------------------------------

class TestLatency(unittest.TestCase):
    def test_decision_under_1ms(self):
        rm = RagMark(config=RagMarkConfig(max_bookmarks=5))
        dim = 128
        emb = [float(i) / dim for i in range(dim)]
        for i in range(5):
            rm.update([Node(str(i), emb)], emb)

        start = time.perf_counter()
        rm.get_scope(emb)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 1.0, f"Decision took {elapsed_ms:.3f} ms")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
