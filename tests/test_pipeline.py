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

    def test_second_repair_uses_revised_answer_and_preserves_initial_claims(self):
        record = AnswerRecord(
            id="loop", source_dataset="manual", question="Q", answer="Original."
        )
        repaired_inputs = []

        def fake_decompose(candidate, **kwargs):
            candidate.claims = [Claim(
                id=f"{candidate.id}_c0", text=candidate.answer,
                sentence_id=0,
            )]
            return candidate

        def fake_verify(candidate, **kwargs):
            supported = candidate.answer == "Second revised."
            candidate.claims[0].label = "Supported" if supported else "Contradicted"
            candidate.reliability_score_before = float(supported)
            return candidate

        def fake_repair(candidate, **kwargs):
            repaired_inputs.append(candidate.answer)
            candidate.repaired_answer = (
                "First revised." if candidate.answer == "Original." else "Second revised."
            )
            return candidate

        with (
            patch("src.pipeline.decompose", side_effect=fake_decompose),
            patch("src.pipeline.retrieve", side_effect=lambda candidate, **kwargs: candidate),
            patch("src.pipeline.verify", side_effect=fake_verify),
            patch("src.pipeline.repair", side_effect=fake_repair),
        ):
            result = run_pipeline(record, max_repair_iterations=2)

        self.assertEqual(repaired_inputs, ["Original.", "First revised."])
        self.assertEqual(result.claims[0].id, "loop_c0")
        self.assertEqual(result.repaired_answer, "Second revised.")
        self.assertEqual(result.reliability_score_before, 0.0)
        self.assertEqual(result.reliability_score_after, 1.0)

    def test_supplied_claim_inventory_skips_initial_decomposition(self):
        record = AnswerRecord(
            id="fixed", source_dataset="manual", question="Q", answer="A fact.",
            claims=[Claim(id="human_claim_7", text="A fact.", sentence_id=0)],
        )

        def fake_verify(candidate, **kwargs):
            candidate.claims[0].label = "Supported"
            candidate.reliability_score_before = 1.0
            return candidate

        with (
            patch("src.pipeline.decompose", side_effect=AssertionError("unexpected decomposition")),
            patch("src.pipeline.retrieve", side_effect=lambda candidate, **kwargs: candidate),
            patch("src.pipeline.verify", side_effect=fake_verify),
        ):
            result = run_pipeline(record, use_existing_claims=True)

        self.assertEqual([claim.id for claim in result.claims], ["human_claim_7"])
        self.assertEqual(result.repair_iterations, 0)

    def test_remove_policy_keeps_valid_empty_answer(self):
        record = AnswerRecord(
            id="remove", source_dataset="manual", question="Q", answer="Wrong claim.",
            claims=[Claim(id="human_claim_1", text="Wrong claim.", sentence_id=0)],
        )

        def fake_verify(candidate, **kwargs):
            candidate.claims[0].label = "Contradicted"
            candidate.reliability_score_before = 0.0
            return candidate

        with (
            patch("src.pipeline.decompose", side_effect=AssertionError("unexpected decomposition")),
            patch("src.pipeline.retrieve", side_effect=lambda candidate, **kwargs: candidate),
            patch("src.pipeline.verify", side_effect=fake_verify),
            patch("src.repair.llm.call", return_value="REMOVE"),
        ):
            result = run_pipeline(
                record, use_existing_claims=True, repair_policy="rewrite+remove",
            )

        self.assertEqual(result.repaired_answer, "")
        self.assertIsNone(result.reliability_score_after)
        self.assertEqual(result.verification_diagnostics["after"]["reason"], "empty_answer")
        self.assertEqual(result.repair_iterations, 1)


if __name__ == "__main__":
    unittest.main()

