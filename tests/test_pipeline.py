"""
Integration tests for the full pipeline using mock LLM and index.
Run with:
    python3 -m unittest tests/test_pipeline.py
    OR: pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from src.schema import AnswerRecord, Claim
from src.pipeline import run_pipeline, run_pipeline_batch


_DECOMPOSE_RESPONSE = json.dumps([
    {"claim": "Neil Armstrong was the first person on the moon.",
     "sentence_id": 0, "check_worthy": True},
    {"claim": "Neil Armstrong landed on the moon in 1969.",
     "sentence_id": 0, "check_worthy": True},
    {"claim": "I think space is amazing.",
     "sentence_id": 1, "check_worthy": False},
])

_VERIFY_RESPONSE = json.dumps({
    "label": "Supported",
    "rationale": "The passage confirms this.",
    "supporting_passage_ids": [1],
})

_REPAIR_RESPONSE = "Neil Armstrong was the first person to walk on the Moon on July 20, 1969."


def _mock_llm(prompt: str, **kwargs) -> str:
    prompt_lower = prompt.lower()
    if "decompose" in prompt_lower or "atomic" in prompt_lower or "check_worthy" in prompt_lower:
        return _DECOMPOSE_RESPONSE
    if "fact-checker" in prompt_lower or "evidence passages" in prompt_lower:
        return _VERIFY_RESPONSE
    return _REPAIR_RESPONSE


class SimpleMockBM25:
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
        {"doc_id": "Moon_0", "title": "Moon landing",
         "text": "Title: Moon landing. Neil Armstrong was the first person to walk on the Moon on July 20, 1969.",
         "chunk_id": 0},
        {"doc_id": "Apollo_0", "title": "Apollo 11",
         "text": "Title: Apollo 11. Apollo 11 was the first crewed mission to land on the Moon.",
         "chunk_id": 0},
    ]
    bm25 = bm25_cls([p["text"].lower().split() for p in passages])
    index_path = str(Path(tmp_path) / "bm25_pipeline_test.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "passages": passages}, f)
    return index_path


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.index_path = _build_test_index(cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def setUp(self):
        self.sample_record = AnswerRecord(
            id="test_001",
            source_dataset="manual",
            split="dev",
            question="Who was the first person on the moon?",
            answer="Neil Armstrong was the first person on the moon. I think space is amazing.",
        )

    def test_returns_answer_record(self):
        with patch("src.llm.call", side_effect=_mock_llm):
            result = run_pipeline(self.sample_record, index_path=self.index_path)
        self.assertIsInstance(result, AnswerRecord)

    def test_claims_populated(self):
        with patch("src.llm.call", side_effect=_mock_llm):
            result = run_pipeline(self.sample_record, index_path=self.index_path)
        self.assertGreater(len(result.claims), 0)

    def test_repaired_answer_not_none(self):
        with patch("src.llm.call", side_effect=_mock_llm):
            result = run_pipeline(self.sample_record, index_path=self.index_path)
        self.assertIsNotNone(result.repaired_answer)

    def test_run_id_set(self):
        with patch("src.llm.call", side_effect=_mock_llm):
            result = run_pipeline(self.sample_record, index_path=self.index_path)
        self.assertIsNotNone(result.run_id)
        self.assertGreater(len(result.run_id), 0)

    def test_reliability_scores_set(self):
        with patch("src.llm.call", side_effect=_mock_llm):
            result = run_pipeline(self.sample_record, index_path=self.index_path)
        self.assertIsNotNone(result.reliability_score_before)
        self.assertTrue(0.0 <= result.reliability_score_before <= 1.0)

    def test_evidence_retrieved(self):
        with patch("src.llm.call", side_effect=_mock_llm):
            result = run_pipeline(self.sample_record, index_path=self.index_path)
        check_worthy = [c for c in result.claims if c.check_worthy]
        for c in check_worthy:
            self.assertGreater(len(c.evidence), 0)

    def test_batch_execution(self):
        out_path = Path(self.tmp_dir.name) / "output.jsonl"
        with patch("src.llm.call", side_effect=_mock_llm):
            results = run_pipeline_batch([self.sample_record], output_path=str(out_path),
                                         index_path=self.index_path)
        self.assertEqual(len(results), 1)
        self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()

