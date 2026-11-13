"""
Unit tests for retrieve.py (Phase 5 enhancements).
Run with:
    python3 -m unittest tests/test_retrieve.py
    OR: pytest tests/test_retrieve.py -v
"""
from __future__ import annotations

import os
import importlib
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from src.schema import AnswerRecord, Claim, EvidencePassage
from src.retrieve import (
    generate_query_variants,
    _reciprocal_rank_fusion,
    rerank_cross_encoder,
    _make_retrieval_key,
    _cache_get_passages,
    _cache_set_passages,
    retrieve_bm25,
    retrieve,
    _clean_query_entity,
    ENTITY_PLANNER_VERSION,
)


def test_cache_does_not_mix_different_indexes(tmp_path, monkeypatch):
    retrieval = importlib.import_module("src.retrieve")
    monkeypatch.setenv("RETRIEVAL_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setattr(retrieval, "_retrieval_conn", None)
    first_index = tmp_path / "first.pkl"
    second_index = tmp_path / "second.pkl"
    first_index.write_bytes(b"first")
    second_index.write_bytes(b"second")
    calls = []

    def fake_retrieve(query, index_path, top_k):
        calls.append(index_path)
        title = Path(index_path).stem
        return [EvidencePassage(doc_id=title, title=title, text=title)]

    monkeypatch.setattr(retrieval, "retrieve_bm25", fake_retrieve)

    def fresh_record():
        return AnswerRecord(
            id="r", source_dataset="manual", question="Q", answer="A.",
            claims=[Claim(id="r_c1", text="same claim", sentence_id=0)],
        )

    a = retrieval.retrieve(fresh_record(), str(first_index))
    b = retrieval.retrieve(fresh_record(), str(second_index))
    again = retrieval.retrieve(fresh_record(), str(first_index))
    assert [a.claims[0].evidence[0].title, b.claims[0].evidence[0].title,
            again.claims[0].evidence[0].title] == ["first", "second", "first"]
    assert calls == [str(first_index), str(second_index)]
    retrieval._retrieval_conn.close()
    monkeypatch.setattr(retrieval, "_retrieval_conn", None)


class SimpleMockBM25:
    """Fallback BM25-like scorer for unit testing when rank_bm25 is not installed."""
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        q_set = set(query_tokens)
        scores = []
        for doc in self.corpus:
            doc_set = set(doc)
            scores.append(float(len(q_set & doc_set)))
        return np.array(scores, dtype=float)


def _build_test_index(tmp_path: str) -> str:
    try:
        from rank_bm25 import BM25Okapi
        bm25_cls = BM25Okapi
    except ImportError:
        bm25_cls = SimpleMockBM25

    passages = [
        {"doc_id": "Einstein_0", "title": "Albert Einstein",
         "text": "Title: Albert Einstein. Einstein was born in 1879 in Ulm Germany.",
         "chunk_id": 0},
        {"doc_id": "Newton_0", "title": "Isaac Newton",
         "text": "Title: Isaac Newton. Newton formulated the laws of motion in 1687.",
         "chunk_id": 0},
        {"doc_id": "Curie_0", "title": "Marie Curie",
         "text": "Title: Marie Curie. Curie was the first woman to win a Nobel Prize.",
         "chunk_id": 0},
    ]
    bm25 = bm25_cls([p["text"].lower().split() for p in passages])
    index_path = str(Path(tmp_path) / "bm25_tiny.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "passages": passages}, f)
    return index_path


class TestQueryFormulation(unittest.TestCase):
    def test_generate_query_variants_includes_raw(self):
        claim = "Albert Einstein was born in 1879 in Ulm, Germany."
        variants = generate_query_variants(claim)
        self.assertGreaterEqual(len(variants), 1)
        self.assertEqual(variants[0], claim)

    def test_generate_query_variants_extracts_entities(self):
        claim = "Albert Einstein won the Nobel Prize in 1921."
        variants = generate_query_variants(claim)
        # Should include capitalized entities & numbers: Albert Einstein Nobel Prize 1921
        self.assertTrue(any("Einstein" in v for v in variants))
        self.assertTrue(any("1921" in v for v in variants))


class TestReciprocalRankFusion(unittest.TestCase):
    def test_rrf_scoring(self):
        p1 = EvidencePassage(doc_id="doc1", title="Doc 1", text="Text 1")
        p2 = EvidencePassage(doc_id="doc2", title="Doc 2", text="Text 2")
        p3 = EvidencePassage(doc_id="doc3", title="Doc 3", text="Text 3")

        list_a = [p1, p2]  # p1 is rank 1, p2 is rank 2
        list_b = [p2, p3]  # p2 is rank 1, p3 is rank 2

        fused = _reciprocal_rank_fusion(list_a, list_b, k=60)
        self.assertEqual(len(fused), 3)
        # p2 appears in both lists (rank 2 in A, rank 1 in B), so it should have highest RRF score
        self.assertEqual(fused[0].doc_id, "doc2")
        self.assertGreater(fused[0].score, fused[1].score)


class TestCrossEncoderReranking(unittest.TestCase):
    def test_rerank_fallback_preserves_order(self):
        p1 = EvidencePassage(doc_id="p1", title="P1", text="Text 1", score=10.0)
        p2 = EvidencePassage(doc_id="p2", title="P2", text="Text 2", score=5.0)
        candidates = [p1, p2]
        # In environment without cross-encoder installed, falls back safely
        unavailable = types.ModuleType("sentence_transformers")
        unavailable.CrossEncoder = mock.Mock(side_effect=RuntimeError("model unavailable"))
        retrieval = importlib.import_module("src.retrieve")
        with mock.patch.dict(sys.modules, {"sentence_transformers": unavailable}), \
                mock.patch.object(retrieval, "_cross_encoder", None):
            reranked = rerank_cross_encoder("query", candidates, top_k=1)
        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0].doc_id, "p1")


class TestRetrievalCaching(unittest.TestCase):
    def test_cache_roundtrip(self):
        key = _make_retrieval_key("bm25", "test query", 5, False)
        passages = [
            EvidencePassage(doc_id="c1", title="Cached Doc", text="Cached text", score=1.23)
        ]
        _cache_set_passages(key, passages)
        retrieved = _cache_get_passages(key)
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].title, "Cached Doc")
        self.assertAlmostEqual(retrieved[0].score, 1.23, places=2)


class TestRetrieve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.index_path = _build_test_index(cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_retrieve_bm25_execution(self):
        record = AnswerRecord(
            id="r_test_001", source_dataset="manual", split="dev",
            question="Where was Einstein born?",
            answer="Einstein was born in Ulm.",
        )
        record.claims = [
            Claim(id="r_test_001_c1",
                  text="Albert Einstein was born in Ulm, Germany.",
                  sentence_id=0)
        ]
        result = retrieve(record, index_path=self.index_path, method="bm25", top_k=2)
        self.assertEqual(len(result.claims[0].evidence), 2)
        self.assertIn("Einstein", result.claims[0].evidence[0].title)

    def test_retrieve_with_rerank_flag(self):
        record = AnswerRecord(
            id="r_test_002", source_dataset="manual", split="dev",
            question="Q", answer="A.",
        )
        record.claims = [
            Claim(id="c1", text="Isaac Newton formulated laws in 1687.", sentence_id=0)
        ]
        retrieval = importlib.import_module("src.retrieve")
        with mock.patch.object(retrieval, "rerank_cross_encoder",
                               side_effect=lambda query, candidates, top_k, model_name:
                               candidates[:top_k]) as rerank:
            result = retrieve(record, index_path=self.index_path,
                              method="bm25+rerank", top_k=1)
        rerank.assert_called_once()
        self.assertEqual(len(result.claims[0].evidence), 1)


if __name__ == "__main__":
    unittest.main()


class TestCleanQueryEntity(unittest.TestCase):
    """Tests for the Planner v3 entity query cleaner (_clean_query_entity)."""

    def test_version_is_3(self):
        self.assertEqual(ENTITY_PLANNER_VERSION, 3)

    def test_no_change_for_clean_entity(self):
        self.assertEqual(_clean_query_entity("Alexander Graham Bell"), "Alexander Graham Bell")

    def test_deduplicates_repeated_name_runs(self):
        # FEVER link annotations sometimes produce "Name Name" repetitions.
        result = _clean_query_entity("Alexander Graham Bell Alexander Graham Bell")
        self.assertEqual(result, "Alexander Graham Bell")

    def test_deduplicates_case_insensitive(self):
        result = _clean_query_entity("Edinburgh Edinburgh")
        self.assertEqual(result, "Edinburgh")

    def test_preserves_different_words_in_order(self):
        result = _clean_query_entity("Marie Curie Nobel Prize")
        self.assertEqual(result, "Marie Curie Nobel Prize")

    def test_truncates_very_long_entity_at_word_boundary(self):
        # Construct a string that exceeds 60 chars
        long_entity = "Alexander " * 10  # 100 chars
        result = _clean_query_entity(long_entity.strip())
        self.assertLessEqual(len(result), 60)
        # Must end on a complete word (no trailing partial word)
        self.assertFalse(result.endswith(" "))

    def test_single_word_beyond_limit_is_returned_as_is(self):
        # A single word longer than 60 chars has no space to truncate at;
        # the string is truncated hard at _ENTITY_QUERY_MAX_CHARS (60) chars.
        long_word = "A" * 70
        result = _clean_query_entity(long_word)
        self.assertEqual(len(result), 60)

    def test_empty_string(self):
        self.assertEqual(_clean_query_entity(""), "")

    def test_linked_entity_query_uses_cleaned_entity(self):
        """Integration: a linked-entity query must not contain repeated name runs."""
        import importlib
        retrieval = importlib.import_module("src.retrieve")

        record = AnswerRecord(
            id="planner_v3_test", source_dataset="manual", split="dev",
            question="Who invented the telephone?",
            answer="Alexander Graham Bell invented the telephone.",
        )
        record.claims = [
            Claim(id="planner_v3_test_c1",
                  text="Alexander Graham Bell invented the telephone.",
                  sentence_id=0)
        ]
        # Passage with a repeated FEVER-style linked entity name in body.
        seed_passage = EvidencePassage(
            doc_id="Bell_0", title="Alexander Graham Bell",
            text="Title: Alexander Graham Bell.\n"
                 "[[Alexander Graham Bell|Alexander Graham Bell]] invented many things.",
            score=1.0,
        )

        queries_seen: list[str] = []

        def capturing_fetch(query, index_path, method, fetch_k,
                            faiss_index_path, passages_path, encoder_name):
            queries_seen.append(query)
            return [seed_passage]

        # Build a real (picklable) index so _index_identity() succeeds.
        with tempfile.TemporaryDirectory() as tmp:
            idx_path = _build_test_index(tmp)
            with mock.patch.object(retrieval, "_fetch_query", side_effect=capturing_fetch), \
                 mock.patch.object(retrieval, "_cache_get_passages", return_value=None), \
                 mock.patch.object(retrieval, "_cache_set_passages"):
                retrieval.retrieve(
                    record, index_path=idx_path, method="bm25",
                    query_strategy="contextual_v1", max_queries=3, top_k=1,
                )

        # At least one query was issued
        self.assertTrue(queries_seen, "No queries were issued")
        for q in queries_seen:
            self.assertNotIn(
                "Alexander Graham Bell Alexander Graham Bell", q,
                f"Repeated entity name found in query: {q!r}"
            )
