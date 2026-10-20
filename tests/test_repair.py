"""
Unit tests for repair.py and guards.py (Phase 7 enhancements).
Run with:
    python3 -m unittest tests/test_repair.py
    OR: pytest tests/test_repair.py -v
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from src.schema import AnswerRecord, Claim, EvidencePassage
from src.guards import (
    check_preservation_guard,
    check_grounding_guard,
    validate_repair_guards,
    extract_named_entities_and_numbers,
)
from src.repair import repair, _apply_fallback_policy


class TestProgrammaticGuards(unittest.TestCase):
    def test_extract_entities_and_numbers(self):
        text = "Albert Einstein won the Nobel Prize in 1921 for his 1905 paper."
        entities = extract_named_entities_and_numbers(text)
        self.assertIn("Einstein", entities)
        self.assertIn("1921", entities)
        self.assertIn("1905", entities)

    def test_preservation_guard_pass(self):
        supported = [Claim(id="c1", text="Albert Einstein was born in 1879.", sentence_id=0)]
        edited = "Albert Einstein was born in 1879 in Ulm, Germany."
        ok, reason = check_preservation_guard(supported, edited)
        self.assertTrue(ok)

    def test_preservation_guard_violation(self):
        # The edit completely eliminated the fact that he was born in 1879
        supported = [Claim(id="c1", text="Albert Einstein was born in 1879.", sentence_id=0)]
        edited = "Isaac Newton was an English mathematician."
        ok, reason = check_preservation_guard(supported, edited)
        self.assertFalse(ok)
        self.assertIn("Preservation violation", reason)

    def test_grounding_guard_pass(self):
        orig = "The Eiffel Tower is in Berlin."
        # Evidence mentions Paris
        evidence = [EvidencePassage(doc_id="d1", title="Eiffel Tower", text="The Eiffel Tower is a wrought-iron tower in Paris, France.")]
        edited = "The Eiffel Tower is in Paris."
        ok, reason = check_grounding_guard(orig, edited, evidence)
        self.assertTrue(ok)

    def test_grounding_guard_violation(self):
        orig = "The Eiffel Tower was built in 1999."
        # Evidence only mentions 1889
        evidence = [EvidencePassage(doc_id="d1", title="Eiffel Tower", text="The Eiffel Tower was completed in 1889.")]
        # Edited introduces "2015" which is NOT in evidence
        edited = "The Eiffel Tower was built in 2015."
        ok, reason = check_grounding_guard(orig, edited, evidence)
        self.assertFalse(ok)
        self.assertIn("Grounding violation", reason)
        self.assertIn("2015", reason)


class TestFallbackPolicies(unittest.TestCase):
    def test_fallback_hedge(self):
        s = "The monument was completed in 1889."
        hedged = _apply_fallback_policy(s, "rewrite+hedge")
        self.assertTrue(hedged.startswith("Reportedly,"))

    def test_fallback_remove(self):
        s = "The monument was completed in 1889."
        deleted = _apply_fallback_policy(s, "rewrite+remove")
        self.assertEqual(deleted, "")

    def test_fallback_rewrite_only(self):
        s = "The monument was completed in 1889."
        kept = _apply_fallback_policy(s, "rewrite-only")
        self.assertEqual(kept, s)


class TestGuardedRepair(unittest.TestCase):
    def setUp(self):
        ev = EvidencePassage(doc_id="d1", title="Eiffel Tower", text="The Eiffel Tower is located in Paris, France.")
        self.record = AnswerRecord(
            id="rep_001",
            source_dataset="manual",
            split="dev",
            question="Where is the Eiffel Tower?",
            answer="The Eiffel Tower was built in 1889. It is located in Berlin.",
            claims=[
                Claim(id="c1", text="The Eiffel Tower was built in 1889.", sentence_id=0, label="Supported"),
                Claim(id="c2", text="The Eiffel Tower is located in Berlin.", sentence_id=1, label="Contradicted", evidence=[ev]),
            ]
        )

    def test_repair_accepts_grounded_edit(self):
        # LLM returns grounded fix
        mock_response = "It is located in Paris."
        with patch("src.llm.call", return_value=mock_response):
            result = repair(self.record, policy="rewrite+hedge")
        # Untouched sentence 0 should be byte-identical
        self.assertIn("The Eiffel Tower was built in 1889.", result.repaired_answer)
        # Sentence 1 should be corrected
        self.assertIn("Paris", result.repaired_answer)
        self.assertNotIn("Berlin", result.repaired_answer)

    def test_repair_retries_and_falls_back_on_ungrounded_hallucination(self):
        # LLM hallucinates "Rome" which is not in evidence
        mock_hallucination = "It is located in Rome."
        with patch("src.llm.call", return_value=mock_hallucination):
            result = repair(self.record, policy="rewrite+remove", max_retries=1)
        # Sentence 1 should be deleted per rewrite+remove policy after rejection
        self.assertNotIn("Rome", result.repaired_answer)
        self.assertNotIn("Berlin", result.repaired_answer)
        self.assertEqual(result.repaired_answer.strip(), "The Eiffel Tower was built in 1889.")


if __name__ == "__main__":
    unittest.main()
