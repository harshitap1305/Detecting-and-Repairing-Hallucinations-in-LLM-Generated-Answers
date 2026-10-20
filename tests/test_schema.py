"""
Unit tests for schema.py
Run with:
    python3 -m unittest tests/test_schema.py
    OR: pytest tests/test_schema.py -v
"""
import json
import unittest

from src.schema import AnswerRecord, Claim, EvidencePassage


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_evidence() -> EvidencePassage:
    return EvidencePassage(
        doc_id="Albert_Einstein_0",
        title="Albert Einstein",
        text="Einstein was born in 1879 in Ulm, Germany.",
        score=0.95,
        chunk_id=0,
    )


def _make_claim(label=None) -> Claim:
    return Claim(
        id="gold_001_c1",
        text="Albert Einstein was born in 1879.",
        sentence_id=0,
        label=label,
        evidence=[_make_evidence()],
    )


def _make_record() -> AnswerRecord:
    return AnswerRecord(
        id="gold_001",
        source_dataset="gold",
        split="dev",
        question="When was Einstein born?",
        answer="Albert Einstein was born in 1879 in Ulm, Germany.",
        claims=[_make_claim(label="Supported")],
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestEvidencePassage(unittest.TestCase):
    def test_default_score(self):
        ep = EvidencePassage(doc_id="x_0", title="X", text="hello")
        self.assertEqual(ep.score, 0.0)

    def test_fields(self):
        ep = _make_evidence()
        self.assertEqual(ep.doc_id, "Albert_Einstein_0")
        self.assertEqual(ep.chunk_id, 0)


class TestClaim(unittest.TestCase):
    def test_defaults(self):
        c = Claim(id="x", text="test fact", sentence_id=0)
        self.assertTrue(c.check_worthy)
        self.assertIsNone(c.label)
        self.assertEqual(c.evidence, [])
        self.assertIsNone(c.repaired_text)

    def test_with_label(self):
        c = _make_claim(label="Contradicted")
        self.assertEqual(c.label, "Contradicted")


class TestAnswerRecord(unittest.TestCase):
    def test_roundtrip_jsonl(self):
        rec = _make_record()
        line = rec.to_jsonl_line()
        rec2 = AnswerRecord.from_jsonl_line(line)
        self.assertEqual(rec2.id, rec.id)
        self.assertEqual(rec2.claims[0].text, rec.claims[0].text)
        self.assertEqual(rec2.claims[0].evidence[0].score, 0.95)

    def test_to_dict_is_json_serialisable(self):
        rec = _make_record()
        d = rec.to_dict()
        json.dumps(d)   # must not raise

    def test_from_dict_round_trip(self):
        rec = _make_record()
        d = rec.to_dict()
        rec2 = AnswerRecord.from_dict(d)
        self.assertEqual(rec2.source_dataset, "gold")
        self.assertEqual(len(rec2.claims), 1)

    def test_check_worthy_claims(self):
        rec = _make_record()
        rec.claims.append(Claim(id="c2", text="opinion", sentence_id=0, check_worthy=False))
        self.assertEqual(len(rec.check_worthy_claims), 1)

    def test_hallucinated_claims(self):
        rec = _make_record()
        rec.claims.append(_make_claim(label="Contradicted"))
        self.assertEqual(len(rec.hallucinated_claims), 1)

    def test_summary_string(self):
        rec = _make_record()
        s = rec.summary()
        self.assertIn("gold_001", s)
        self.assertIn("S=1", s)

    def test_empty_claims(self):
        rec = AnswerRecord(id="x", source_dataset="test", question="q",
                           answer="a", split="dev")
        self.assertEqual(rec.claims, [])
        self.assertEqual(rec.check_worthy_claims, [])

    def test_nested_evidence_preserved(self):
        rec = _make_record()
        line = rec.to_jsonl_line()
        rec2 = AnswerRecord.from_jsonl_line(line)
        self.assertEqual(rec2.claims[0].evidence[0].title, "Albert Einstein")

    def test_split_field(self):
        rec = _make_record()
        self.assertEqual(rec.split, "dev")


if __name__ == "__main__":
    unittest.main()
