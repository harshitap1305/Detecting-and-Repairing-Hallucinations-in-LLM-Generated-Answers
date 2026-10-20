"""
Unit tests for decompose.py (Phase 4 enhancements).
Run with:
    python3 -m unittest tests/test_decompose.py
    OR: pytest tests/test_decompose.py -v
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.schema import AnswerRecord
from src.decompose import (
    _parse_claims_json,
    split_sentences,
    split_sentences_with_spans,
    align_claim_span,
    decompose,
)
from scripts.eval_decomposition import (
    check_is_atomic,
    check_is_decontextualized,
    check_span_valid,
)


class TestSentenceSplittingAndSpans(unittest.TestCase):
    def test_split_sentences_basic(self):
        text = "Einstein was born in 1879. He studied physics."
        sents = split_sentences(text)
        self.assertEqual(len(sents), 2)

    def test_split_sentences_with_spans_accuracy(self):
        text = "Neil Armstrong landed on the Moon. It was 1969."
        spans = split_sentences_with_spans(text)
        self.assertEqual(len(spans), 2)
        # Check first sentence offsets
        s1_text, s1_start, s1_end = spans[0]
        self.assertEqual(s1_text, "Neil Armstrong landed on the Moon.")
        self.assertEqual(text[s1_start:s1_end], s1_text)
        # Check second sentence offsets
        s2_text, s2_start, s2_end = spans[1]
        self.assertEqual(s2_text, "It was 1969.")
        self.assertEqual(text[s2_start:s2_end], s2_text)

    def test_split_sentences_empty(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences_with_spans(""), [])


class TestSpanAlignment(unittest.TestCase):
    def test_align_claim_span_exact_substring(self):
        sent = "Marie Curie was a Polish physicist and chemist."
        claim = "Marie Curie was a Polish physicist."
        span = align_claim_span(claim, sent, 0, len(sent))
        self.assertEqual(span[0], 0)
        self.assertEqual(span[1], len("Marie Curie was a Polish physicist"))
        self.assertIn("Marie Curie was a Polish physicist", sent[span[0]:span[1]])

    def test_align_claim_span_token_window(self):
        # Sentence uses pronoun "She", claim uses explicit entity "Marie Curie"
        sent = "She conducted pioneering research on radioactivity."
        claim = "Marie Curie conducted pioneering research on radioactivity."
        span = align_claim_span(claim, sent, 100, 100 + len(sent))
        # Should locate "conducted" ... "radioactivity" in the sentence
        extracted = sent[span[0] - 100 : span[1] - 100]
        self.assertIn("conducted", extracted)
        self.assertIn("radioactivity", extracted)

    def test_align_claim_span_fallback(self):
        sent = "The experiment succeeded."
        claim = "Completely unrelated statement."
        span = align_claim_span(claim, sent, 50, 75)
        # Falls back to entire sentence
        self.assertEqual(span, [50, 75])


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


class TestDecompose(unittest.TestCase):
    def test_decompose_populates_claims_with_spans(self):
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
        # Check that span was computed
        self.assertIsNotNone(result.claims[0].span)
        self.assertEqual(len(result.claims[0].span), 2)
        start, end = result.claims[0].span
        self.assertTrue(0 <= start < end <= len(record.answer))

    def test_decompose_edge_cases_handling(self):
        record = AnswerRecord(
            id="test_edge_001", source_dataset="manual", split="dev",
            question="What is the Eiffel Tower?",
            answer="As an AI, I know that the Eiffel Tower is in Paris. It is the most beautiful structure ever.",
        )
        mock_response = json.dumps([
            {"claim": "As an AI, I know this fact.", "sentence_id": 0, "check_worthy": False},
            {"claim": "The Eiffel Tower is in Paris.", "sentence_id": 0, "check_worthy": True},
            {"claim": "The Eiffel Tower is the most beautiful structure ever.", "sentence_id": 1, "check_worthy": False},
        ])
        with patch("src.llm.call", return_value=mock_response):
            result = decompose(record)
        self.assertEqual(len(result.claims), 3)
        check_worthy_flags = [c.check_worthy for c in result.claims]
        self.assertEqual(check_worthy_flags, [False, True, False])


class TestDecompositionQualityHeuristics(unittest.TestCase):
    def test_check_is_decontextualized_pass(self):
        ok, _ = check_is_decontextualized("Albert Einstein won the Nobel Prize.")
        self.assertTrue(ok)

    def test_check_is_decontextualized_fail(self):
        ok, reason = check_is_decontextualized("He won the Nobel Prize in 1921.")
        self.assertFalse(ok)
        self.assertIn("he", reason.lower())

    def test_check_is_atomic_pass(self):
        ok, _ = check_is_atomic("Marie Curie was born in Warsaw.")
        self.assertTrue(ok)

    def test_check_is_atomic_compound_fail(self):
        ok, _ = check_is_atomic("Marie Curie moved to Paris and studied physics as well as won prizes.")
        self.assertFalse(ok)

    def test_check_span_valid(self):
        from src.schema import Claim
        c = Claim(id="c1", text="Test", sentence_id=0, span=[5, 15])
        ok, _ = check_span_valid(c, "01234567890123456789")  # len 20
        self.assertTrue(ok)
        # Out of bounds
        c_bad = Claim(id="c2", text="Bad", sentence_id=0, span=[15, 25])
        ok2, _ = check_span_valid(c_bad, "Short text")
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
