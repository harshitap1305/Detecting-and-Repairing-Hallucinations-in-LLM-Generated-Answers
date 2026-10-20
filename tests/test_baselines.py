"""
Unit tests for baseline models (B0, B1, B3, B6).
Run with:
    python3 -m unittest tests/test_baselines.py
    OR: pytest tests/test_baselines.py -v
"""
import json
import unittest
from unittest.mock import patch

from src.schema import AnswerRecord, Claim
from src.baselines.b0_no_repair import run_b0
from src.baselines.b1_answer_level import run_b1
from src.baselines.b3_self_repair import run_b3
from src.baselines.b6_delete_only import run_b6


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.record = AnswerRecord(
            id="base_001",
            source_dataset="manual",
            split="dev",
            question="When was the Eiffel Tower built?",
            answer="The Eiffel Tower was built in 1889. It is located in Berlin.",
            claims=[
                Claim(id="c1", text="The Eiffel Tower was built in 1889.", sentence_id=0, label="Supported"),
                Claim(id="c2", text="The Eiffel Tower is located in Berlin.", sentence_id=1, label="Contradicted"),
            ]
        )

    def test_b0_no_repair(self):
        result = run_b0(self.record)
        self.assertEqual(result.repaired_answer, self.record.answer)
        self.assertEqual(result.repair_iterations, 0)

    def test_b1_answer_level(self):
        mock_resp = json.dumps({"has_hallucination": True, "confidence": 0.9, "explanation": "Located in Paris, not Berlin."})
        with patch("src.baselines.b1_answer_level.llm_call", return_value=mock_resp):
            out = run_b1(self.record)
        self.assertTrue(out["has_hallucination"])
        self.assertEqual(out["confidence"], 0.9)

    def test_b3_self_repair(self):
        fixed_text = "The Eiffel Tower was built in 1889. It is located in Paris."
        with patch("src.baselines.b3_self_repair.llm_call", return_value=fixed_text):
            result = run_b3(self.record)
        self.assertEqual(result.repaired_answer, fixed_text)
        self.assertEqual(result.repair_iterations, 1)

    def test_b6_delete_only(self):
        result = run_b6(self.record)
        # Sentence 1 (index 1) contained Contradicted claim, so it should be deleted.
        # Sentence 0 (index 0) was Supported, so it should be kept.
        self.assertIn("1889", result.repaired_answer)
        self.assertNotIn("Berlin", result.repaired_answer)


if __name__ == "__main__":
    unittest.main()

