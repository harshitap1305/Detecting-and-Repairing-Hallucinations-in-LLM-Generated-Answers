"""
Unit tests for verify.py
Run with:
    python3 -m unittest tests/test_verify.py
    OR: pytest tests/test_verify.py -v
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.schema import AnswerRecord, Claim, EvidencePassage
from src.verify import verify_claim_llm, _parse_verdict, verify


def _make_claim_with_evidence() -> Claim:
    ev = EvidencePassage(
        doc_id="Test_0", title="Test",
        text="Title: Test. Albert Einstein was born in 1879.",
        score=0.9,
    )
    return Claim(
        id="v_test_c1",
        text="Albert Einstein was born in 1879.",
        sentence_id=0,
        evidence=[ev],
    )


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

    def test_markdown_fenced(self):
        raw = "```json\n" + json.dumps({"label": "Supported", "rationale": "ok"}) + "\n```"
        label, _, _ = _parse_verdict(raw)
        self.assertEqual(label, "Supported")


class TestVerifyClaimLlm(unittest.TestCase):
    def test_supported_claim(self):
        claim = _make_claim_with_evidence()
        mock_response = json.dumps({"label": "Supported", "rationale": "Confirmed by passage."})
        with patch("src.llm.call", return_value=mock_response):
            result = verify_claim_llm(claim)
        self.assertEqual(result.label, "Supported")

    def test_no_evidence_is_unverifiable(self):
        claim = Claim(id="v_c2", text="Some claim.", sentence_id=0)
        result = verify_claim_llm(claim)
        self.assertEqual(result.label, "Unverifiable")

    def test_not_check_worthy_is_unverifiable(self):
        claim = Claim(id="v_c3", text="I think this.", sentence_id=0, check_worthy=False)
        result = verify_claim_llm(claim)
        self.assertEqual(result.label, "Unverifiable")


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
            result = verify(record)
        self.assertEqual(result.reliability_score_before, 1.0)

    def test_empty_claims_score_one(self):
        record = AnswerRecord(
            id="vr_002", source_dataset="manual", split="dev",
            question="Q", answer="A.",
        )
        result = verify(record)
        self.assertEqual(result.reliability_score_before, 1.0)


if __name__ == "__main__":
    unittest.main()

