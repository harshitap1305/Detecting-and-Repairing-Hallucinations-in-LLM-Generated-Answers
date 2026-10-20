"""
Unit tests for retrieve.py
Run with:
    python3 -m unittest tests/test_retrieve.py
    OR: pytest tests/test_retrieve.py -v
"""
from __future__ import annotations

import pickle
import unittest
import numpy as np
from pathlib import Path
import tempfile

from src.schema import AnswerRecord, Claim, EvidencePassage
from src.retrieve import retrieve_bm25, retrieve


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


class TestRetrieveBm25(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.index_path = _build_test_index(cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_returns_evidence_passages(self):
        results = retrieve_bm25("Einstein born 1879", self.index_path, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, EvidencePassage) for r in results))

    def test_top_result_relevant(self):
        results = retrieve_bm25("Einstein born Ulm", self.index_path, top_k=1)
        self.assertIn("Einstein", results[0].title)

    def test_top_k_respected(self):
        results = retrieve_bm25("science", self.index_path, top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_score_computed(self):
        results = retrieve_bm25("Einstein Nobel Prize", self.index_path, top_k=3)
        self.assertTrue(any(r.score > 0 for r in results))


class TestRetrieve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.index_path = _build_test_index(cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_retrieve_populates_evidence(self):
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
        result = retrieve(record, index_path=self.index_path, top_k=2)
        self.assertEqual(len(result.claims[0].evidence), 2)

    def test_retrieve_skips_non_check_worthy(self):
        record = AnswerRecord(
            id="r_test_002", source_dataset="manual", split="dev",
            question="Q", answer="A.",
        )
        record.claims = [
            Claim(id="r_test_002_c1", text="I think this is great.",
                  sentence_id=0, check_worthy=False)
        ]
        result = retrieve(record, index_path=self.index_path, top_k=2)
        self.assertEqual(result.claims[0].evidence, [])


if __name__ == "__main__":
    unittest.main()

