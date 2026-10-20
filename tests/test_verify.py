"""
Unit tests for verify.py and error_analysis.py (Phase 6 enhancements).
Run with:
    python3 -m unittest tests/test_verify.py
    OR: pytest tests/test_verify.py -v
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.schema import AnswerRecord, Claim, EvidencePassage
from src.verify import (
    aggregate_verdict,
    set_mock_nli_scorer,
    verify_claim_nli,
    verify_claim_llm,
    verify_claim_ensemble,
    _parse_verdict,
    verify,
)
from src.error_analysis import categorize_verification_error, analyze_error_distribution


def _make_claim_with_evidence(claim_text: str = "Albert Einstein was born in 1879.") -> Claim:
    ev = EvidencePassage(
        doc_id="Test_0", title="Test",
        text="Title: Test. Albert Einstein was born in 1879.",
        score=0.9,
    )
    return Claim(
        id="v_test_c1",
        text=claim_text,
        sentence_id=0,
        evidence=[ev],
    )


class TestThresholdAggregation(unittest.TestCase):
    def test_aggregate_supported(self):
        lbl, reason = aggregate_verdict(max_entail=0.85, max_contra=0.10, tau_s=0.70, tau_c=0.50)
        self.assertEqual(lbl, "Supported")
        self.assertIn("0.850", reason)

    def test_aggregate_contradicted_priority(self):
        # Even if entailment is high, if contradiction exceeds tau_c, it is Contradicted
        lbl, reason = aggregate_verdict(max_entail=0.75, max_contra=0.65, tau_s=0.70, tau_c=0.50)
        self.assertEqual(lbl, "Contradicted")

    def test_aggregate_unverifiable(self):
        lbl, _ = aggregate_verdict(max_entail=0.45, max_contra=0.20, tau_s=0.70, tau_c=0.50)
        self.assertEqual(lbl, "Unverifiable")


class TestNliVerification(unittest.TestCase):
    def tearDown(self):
        set_mock_nli_scorer(None)

    def test_nli_supported_mock(self):
        set_mock_nli_scorer(lambda p, h: (0.90, 0.05))
        claim = _make_claim_with_evidence()
        result = verify_claim_nli(claim, tau_s=0.70, tau_c=0.50)
        self.assertEqual(result.label, "Supported")

    def test_nli_contradicted_mock(self):
        set_mock_nli_scorer(lambda p, h: (0.05, 0.85))
        claim = _make_claim_with_evidence()
        result = verify_claim_nli(claim, tau_s=0.70, tau_c=0.50)
        self.assertEqual(result.label, "Contradicted")


class TestEnsembleVerification(unittest.TestCase):
    def tearDown(self):
        set_mock_nli_scorer(None)

    def test_ensemble_consensus(self):
        set_mock_nli_scorer(lambda p, h: (0.85, 0.05))
        claim = _make_claim_with_evidence()
        mock_llm = json.dumps({"label": "Supported", "rationale": "Directly confirmed."})
        with patch("src.llm.call", return_value=mock_llm):
            result = verify_claim_ensemble(claim, tau_s=0.70, tau_c=0.50)
        self.assertEqual(result.label, "Supported")
        self.assertIn("Ensemble consensus", result.label_rationale)

    def test_ensemble_contradiction_safety_guard(self):
        # NLI flags contradiction, LLM says unverifiable -> ensemble flags Contradicted
        set_mock_nli_scorer(lambda p, h: (0.10, 0.75))
        claim = _make_claim_with_evidence()
        mock_llm = json.dumps({"label": "Unverifiable", "rationale": "Not sure."})
        with patch("src.llm.call", return_value=mock_llm):
            result = verify_claim_ensemble(claim, tau_s=0.70, tau_c=0.50)
        self.assertEqual(result.label, "Contradicted")


class TestErrorAnalysis(unittest.TestCase):
    def test_retrieval_miss(self):
        c = Claim(id="c1", text="Neil Armstrong landed on Mars in 1999.", sentence_id=0, evidence=[])
        cat, _ = categorize_verification_error(c)
        self.assertEqual(cat, "RETRIEVAL_MISS")

    def test_numeric_mismatch(self):
        ev = EvidencePassage(doc_id="d1", title="War", text="The battle took place in 1812.")
        c = Claim(id="c2", text="The battle took place in 1899.", sentence_id=0, evidence=[ev], label="Contradicted")
        cat, _ = categorize_verification_error(c)
        self.assertEqual(cat, "NUMERIC_MISMATCH")

    def test_negation_inversion(self):
        ev = EvidencePassage(doc_id="d2", title="Treaty", text="The country signed the treaty in 1945.")
        c = Claim(id="c3", text="The country never signed the treaty in 1945.", sentence_id=0, evidence=[ev], label="Contradicted")
        cat, _ = categorize_verification_error(c)
        self.assertEqual(cat, "NEGATION_INVERSION")


class TestParseVerdict(unittest.TestCase):
    def test_supported(self):
        raw = json.dumps({"label": "Supported", "rationale": "passage confirms",
                          "supporting_passage_ids": [1]})
        label, rationale, ids = _parse_verdict(raw)
        self.assertEqual(label, "Supported")
        self.assertIn("confirms", rationale)

    def test_contradicted(self):
        raw = json.dumps({"label": "Contradicted", "rationale": "passage conflicts",
                          "supporting_passage_ids": []})
        label, _, _ = _parse_verdict(raw)
        self.assertEqual(label, "Contradicted")

    def test_invalid_label_defaults_unverifiable(self):
        raw = json.dumps({"label": "UNKNOWN", "rationale": ""})
        label, _, _ = _parse_verdict(raw)
        self.assertEqual(label, "Unverifiable")

    def test_malformed_json(self):
        label, _, _ = _parse_verdict("not json at all")
        self.assertEqual(label, "Unverifiable")


class TestVerify(unittest.TestCase):
    def test_reliability_score_set(self):
        record = AnswerRecord(
            id="vr_001", source_dataset="manual", split="dev",
            question="Q", answer="A.",
        )
        ev = EvidencePassage(doc_id="X_0", title="X", text="X text.", score=0.8)
        record.claims = [
            Claim(id="c1", text="Claim one.", sentence_id=0, evidence=[ev]),
            Claim(id="c2", text="Claim two.", sentence_id=0, evidence=[ev]),
        ]
        mock_response = json.dumps({"label": "Supported", "rationale": "OK"})
        with patch("src.llm.call", return_value=mock_response):
            result = verify(record, verifier="llm")
        self.assertEqual(result.reliability_score_before, 1.0)


if __name__ == "__main__":
    unittest.main()
