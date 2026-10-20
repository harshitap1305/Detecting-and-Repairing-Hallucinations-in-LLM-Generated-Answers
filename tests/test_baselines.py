"""
Unit tests for comparison baselines (B0 through B7).
Tests execution, record updates, and mocked outputs.
"""
import unittest
from unittest.mock import patch

from src.baselines import (
    run_b0,
    run_b1,
    run_b2,
    run_b3,
    run_b4,
    run_b5,
    run_b6,
    run_b7,
)
from src.baselines.b2_selfcheck import set_mock_selfcheck_sampler
from src.schema import AnswerRecord, Claim, EvidencePassage


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.sample_record = AnswerRecord(
            id="test_rec_01",
            source_dataset="manual",
            split="dev",
            question="Where was Albert Einstein born?",
            answer="Albert Einstein was born in 1879 in Berlin. He won the Nobel Prize in 1921.",
            claims=[
                Claim(
                    id="test_rec_01_c1",
                    text="Albert Einstein was born in 1879.",
                    sentence_id=0,
                    check_worthy=True,
                    label="Supported",
                ),
                Claim(
                    id="test_rec_01_c2",
                    text="Albert Einstein was born in Berlin.",
                    sentence_id=0,
                    check_worthy=True,
                    label="Contradicted",
                ),
                Claim(
                    id="test_rec_01_c3",
                    text="Albert Einstein won the Nobel Prize in 1921.",
                    sentence_id=1,
                    check_worthy=True,
                    label="Supported",
                ),
            ],
        )

    def test_b0_no_repair(self):
        rec = run_b0(self.sample_record)
        self.assertEqual(rec.repaired_answer, rec.answer)
        self.assertEqual(rec.repair_iterations, 0)

    @patch("src.baselines.b1_answer_level.llm_call")
    def test_b1_answer_level(self, mock_llm):
        mock_llm.return_value = '{"has_hallucination": true, "confidence": 0.85, "explanation": "Born in Ulm, not Berlin"}'
        res = run_b1(self.sample_record)
        self.assertTrue(res["has_hallucination"])
        self.assertAlmostEqual(res["confidence"], 0.85)

    def test_b2_selfcheck(self):
        # Set mock stochastic samples
        set_mock_selfcheck_sampler(
            lambda q, n: [
                "Einstein was born in Ulm in 1879.",
                "He was born in Ulm, Germany.",
                "Albert Einstein was born in 1879 in Ulm.",
            ]
        )
        try:
            with patch("src.baselines.b2_selfcheck.llm_call") as mock_eval:
                mock_eval.side_effect = [
                    '{"verdict": "Inconsistent", "score": 0.2}',  # sentence 1 (Berlin)
                    '{"verdict": "Consistent", "score": 0.9}',    # sentence 2 (Nobel Prize)
                ]
                rec = run_b2(self.sample_record, n_samples=3)
                self.assertIsNotNone(rec.repaired_answer)
                self.assertIn("Reportedly", rec.repaired_answer)
                self.assertIsNotNone(rec.reliability_score_after)
        finally:
            set_mock_selfcheck_sampler(None)

    @patch("src.baselines.b3_self_repair.llm_call")
    def test_b3_self_repair(self, mock_llm):
        mock_llm.return_value = "Albert Einstein was born in 1879 in Ulm. He won the Nobel Prize in 1921."
        rec = run_b3(self.sample_record)
        self.assertEqual(rec.repaired_answer, "Albert Einstein was born in 1879 in Ulm. He won the Nobel Prize in 1921.")
        self.assertEqual(rec.repair_iterations, 1)

    @patch("src.baselines.b4_cove.llm_call")
    def test_b4_cove(self, mock_llm):
        # Return mock outputs for: 1. Generate questions, 2. Answer Q1, 3. Revise
        mock_llm.side_effect = [
            '["Where was Albert Einstein born?"]',  # verification question
            "Albert Einstein was born in Ulm, Germany.",  # independent answer
            "Albert Einstein was born in 1879 in Ulm, Germany. He won the Nobel Prize in 1921.",  # revised
        ]
        rec = run_b4(self.sample_record)
        self.assertIn("Ulm", rec.repaired_answer)
        self.assertEqual(rec.repair_iterations, 1)

    @patch("src.baselines.b5_rarr.llm_call")
    def test_b5_rarr(self, mock_llm):
        mock_llm.return_value = "Albert Einstein was born in Ulm, Germany in 1879."
        passages = [
            EvidencePassage(
                doc_id="einstein_0",
                title="Albert Einstein",
                text="Albert Einstein was born in Ulm in the Kingdom of Wurttemberg in 1879.",
                score=0.9,
            )
        ]
        rec = run_b5(self.sample_record, passages=passages)
        self.assertEqual(rec.repaired_answer, "Albert Einstein was born in Ulm, Germany in 1879.")

    def test_b6_delete_only(self):
        rec = run_b6(self.sample_record)
        # Sentence 0 had Contradicted claim -> deleted; Sentence 1 kept
        self.assertEqual(rec.repaired_answer, "He won the Nobel Prize in 1921.")

    @patch("src.baselines.b7_factcheck_gpt.llm_call")
    def test_b7_factcheck_gpt(self, mock_llm):
        ev = [EvidencePassage(doc_id="einstein_0", title="Albert Einstein", text="Einstein was born in Ulm.", score=0.9)]
        for c in self.sample_record.claims:
            c.evidence = ev

        mock_llm.side_effect = [
            '{"stance": "Support", "reason": "1879 matches"}',       # Claim 1
            '{"stance": "Refute", "reason": "Born in Ulm"}',          # Claim 2
            '{"stance": "Support", "reason": "Nobel in 1921 matches"}', # Claim 3
            "Albert Einstein was born in Ulm in 1879. He won the Nobel Prize in 1921.", # Final revision
        ]
        rec = run_b7(self.sample_record)
        self.assertIn("Ulm", rec.repaired_answer)
        self.assertEqual(self.sample_record.claims[1].label, "Contradicted")


if __name__ == "__main__":
    unittest.main()
