"""
Unit tests for decompose.py
Run with:
    python3 -m unittest tests/test_decompose.py
    OR: pytest tests/test_decompose.py -v
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.schema import AnswerRecord
from src.decompose import _parse_claims_json, split_sentences, decompose


class TestSplitSentences(unittest.TestCase):
    def test_basic(self):
        text = "Einstein was born in 1879. He studied physics."
        sents = split_sentences(text)
        self.assertEqual(len(sents), 2)

    def test_single_sentence(self):
        sents = split_sentences("This is one sentence.")
        self.assertEqual(len(sents), 1)

    def test_empty(self):
        sents = split_sentences("")
        self.assertEqual(sents, [])


class TestParseClaimsJson(unittest.TestCase):
    def test_clean_json(self):
        raw = json.dumps([
            {"claim": "The sky is blue.", "sentence_id": 0, "check_worthy": True}
        ])
        result = _parse_claims_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "The sky is blue.")

    def test_json_with_markdown_fence(self):
        raw = "```json\n[{\"claim\": \"X.\", \"sentence_id\": 0, \"check_worthy\": true}]\n```"
        result = _parse_claims_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_json_embedded_in_text(self):
        raw = 'Here are the claims:\n[{"claim": "A.", "sentence_id": 0, "check_worthy": true}]\nDone.'
        result = _parse_claims_json(raw)
        self.assertIsNotNone(result)

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_claims_json("no json here"))
        self.assertIsNone(_parse_claims_json("{not a list}"))

    def test_empty_array(self):
        result = _parse_claims_json("[]")
        self.assertEqual(result, [])


class TestDecompose(unittest.TestCase):
    def test_decompose_populates_claims(self):
        record = AnswerRecord(
            id="test_d_001", source_dataset="manual", split="dev",
            question="When was Einstein born?",
            answer="Albert Einstein was born in 1879 in Ulm, Germany.",
        )
        mock_response = json.dumps([
            {"claim": "Albert Einstein was born in 1879.", "sentence_id": 0, "check_worthy": True},
            {"claim": "Albert Einstein was born in Ulm, Germany.", "sentence_id": 0, "check_worthy": True},
        ])
        with patch("src.llm.call", return_value=mock_response):
            result = decompose(record)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].id, "test_d_001_c1")
        self.assertTrue(result.claims[0].check_worthy)

    def test_decompose_filters_short_claims(self):
        record = AnswerRecord(
            id="test_d_002", source_dataset="manual", split="dev",
            question="Q", answer="A.",
        )
        mock_response = json.dumps([
            {"claim": "OK.", "sentence_id": 0, "check_worthy": True},
            {"claim": "Albert Einstein was born in 1879.", "sentence_id": 0, "check_worthy": True},
        ])
        with patch("src.llm.call", return_value=mock_response):
            result = decompose(record, min_claim_length=10)
        self.assertEqual(len(result.claims), 1)

    def test_decompose_handles_json_failure(self):
        record = AnswerRecord(
            id="test_d_003", source_dataset="manual", split="dev",
            question="Q", answer="A.",
        )
        with patch("src.llm.call", return_value="not valid json at all"):
            result = decompose(record, max_retries=1)
        self.assertEqual(result.claims, [])


if __name__ == "__main__":
    unittest.main()

