"""Focused safety checks for FEVER data construction and evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.eval_verifier import bootstrap_intervals, classification_metrics, read_baseline
from scripts.prepare_fever_holdout import filter_heldout
from scripts.retrieve_fever_evidence import main as retrieve_main
from scripts.train_verifier import validate_splits
from src.schema import EvidencePassage
from src.training.verifier_data import prepare_examples, write_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestFeverPreparation(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temporary.name)
        self.claims = self.root / "train.jsonl"
        self.wiki = self.root / "wiki.jsonl"
        self.retrieval = self.root / "retrieved.jsonl"
        _write_jsonl(self.claims, [
            {"id": 1, "claim": "Alpha was founded in 1900.", "label": "SUPPORTS",
             "evidence": [[[0, 0, "Alpha", 0]]]},
            {"id": 2, "claim": "Beta was founded in 1901.", "label": "REFUTES",
             "evidence": [[[0, 0, "Beta", 0]]]},
            {"id": 3, "claim": "Gamma won an award.", "label": "NOT ENOUGH INFO",
             "evidence": []},
            {"id": 4, "claim": "Alpha is in Europe.", "label": "SUPPORTS",
             "evidence": [[[0, 0, "Alpha", 1]]]},
            {"id": 5, "claim": "Missing page says something.", "label": "REFUTES",
             "evidence": [[[0, 0, "Missing", 0]]]},
        ])
        _write_jsonl(self.wiki, [
            {"id": "Alpha", "lines": "0\tAlpha was founded in 1900.\n1\tAlpha is in Europe."},
            {"id": "Beta", "lines": "0\tBeta was founded in 1999."},
            {"id": "Gamma", "lines": "0\tGamma is a city."},
        ])
        _write_jsonl(self.retrieval, [
            {"claim_id": 1, "passages": [{"title": "Alpha", "text": "Alpha has a museum."}]},
            {"claim_id": 2, "passages": [{"title": "Beta", "text": "Beta has a bridge."}]},
            {"claim_id": 3, "passages": [
                {"title": "Gamma", "text": "Gamma is a city."},
                {"title": "Alpha", "text": "Alpha has a museum."}]},
            {"claim_id": 4, "passages": [{"title": "Alpha", "text": "Alpha has a museum."}]},
        ])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_article_disjoint_and_nei_uses_retrieved_evidence(self) -> None:
        train, dev, manifest = prepare_examples(self.claims, self.wiki, self.retrieval,
                                                seed=13, dev_fraction=0.25)
        rows = {row["claim_id"]: row for row in train + dev}
        self.assertEqual(set(rows), {"1", "2", "3", "4"})
        self.assertEqual(rows["3"]["label_name"], "Unverifiable")
        self.assertIn("Gamma is a city", rows["3"]["premise"])
        self.assertNotIn("Alpha has a museum", rows["3"]["premise"])
        self.assertIn("Alpha has a museum", rows["1"]["premise"])
        self.assertEqual(rows["2"]["label_name"], "Contradicted")
        self.assertEqual(manifest["skipped"], {"gold_evidence_missing": 1})
        train_articles = {t for row in train for t in row["article_ids"]}
        dev_articles = {t for row in dev for t in row["article_ids"]}
        self.assertFalse(train_articles & dev_articles)
        # Two claims from Alpha must be on the same side of the split.
        self.assertEqual(rows["1"] in train, rows["4"] in train)
        train2, dev2, _ = prepare_examples(self.claims, self.wiki, self.retrieval,
                                           seed=13, dev_fraction=0.25)
        self.assertEqual([r["id"] for r in train], [r["id"] for r in train2])
        self.assertEqual([r["id"] for r in dev], [r["id"] for r in dev2])

    def test_written_splits_validate_and_overlap_is_rejected(self) -> None:
        train, dev, manifest = prepare_examples(self.claims, self.wiki, self.retrieval)
        output = self.root / "prepared"
        write_dataset(output, train, dev, manifest)
        self.assertEqual(validate_splits(output / "train.jsonl", output / "dev.jsonl"),
                         (len(train), len(dev)))
        with self.assertRaises(FileExistsError):
            write_dataset(output, train, dev, manifest)
        bad = self.root / "overlap.jsonl"
        clone = {**train[0], "id": "new", "article_ids": dev[0]["article_ids"]}
        _write_jsonl(bad, [clone])
        with self.assertRaisesRegex(ValueError, "Article leakage"):
            validate_splits(output / "dev.jsonl", bad)

    def test_heldout_article_is_excluded_from_training(self) -> None:
        train, dev, manifest = prepare_examples(self.claims, self.wiki, self.retrieval,
                                                excluded_keys={"Gamma"})
        self.assertNotIn("3", {row["claim_id"] for row in train + dev})
        self.assertEqual(manifest["skipped"]["heldout_entity_or_article"], 1)
        self.assertTrue(manifest["heldout_exclusions_applied"])

    def test_retrieval_producer_uses_claim_text_and_expected_schema(self) -> None:
        index = self.root / "bm25.pkl"
        index.write_bytes(b"fixture")
        output = self.root / "produced.jsonl"
        calls = []

        def fake_retriever(query, index_path, top_k):
            calls.append((query, index_path, top_k))
            return [EvidencePassage(doc_id="Alpha_0", title="Alpha",
                                    text="Alpha passage.", score=0.8)]

        with patch("src.retrieve.retrieve_bm25", side_effect=fake_retriever):
            retrieve_main(["--fever-train", str(self.claims), "--index-path", str(index),
                           "--output", str(output), "--max-claims", "2"])
        produced = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["claim_id"] for row in produced], [1, 2])
        self.assertEqual(calls[0][0], "Alpha was founded in 1900.")
        self.assertEqual(produced[0]["passages"][0]["title"], "Alpha")
        self.assertNotIn("label", produced[0])

    def test_holdout_discards_seen_articles(self) -> None:
        train, dev, _ = prepare_examples(self.claims, self.wiki, self.retrieval)
        all_rows = train + dev
        filtered, dropped = filter_heldout(all_rows, train, dev)
        self.assertEqual(filtered, [])
        self.assertEqual(dropped, len(all_rows))


class TestVerifierMetrics(unittest.TestCase):
    def test_confusion_and_paired_bootstrap_are_deterministic(self) -> None:
        gold = [0, 1, 2, 0, 1, 2]
        trained = [0, 1, 0, 0, 2, 2]
        baseline = [0, 0, 0, 0, 0, 0]
        metrics = classification_metrics(gold, trained)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 1, 1], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        first = bootstrap_intervals(gold, {"trained": trained, "baseline": baseline},
                                    iterations=30, seed=42)
        second = bootstrap_intervals(gold, {"trained": trained, "baseline": baseline},
                                     iterations=30, seed=42)
        self.assertEqual(first, second)

    def test_baseline_requires_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp:
            path = Path(temp) / "baseline.jsonl"
            _write_jsonl(path, [{"id": "a", "prediction": "Supported"}])
            with self.assertRaisesRegex(ValueError, "Baseline IDs differ"):
                read_baseline(path, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
