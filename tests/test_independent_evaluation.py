"""Synthetic offline regressions for scoring, blinded review, and paired inference."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from src.evaluate import compute_detection_metrics, evaluate_claim_detection, evaluate_dataset
from src.experiments.artifacts import canonical
from src.experiments.assessment import (export_packet, import_ratings, validate_assessments,
                                        agreement, adjudicate, score_assessments)
from src.experiments.runner import run_benchmark
from src.experiments.scoring import load_gold, load_run, score_run
from src.experiments.statistics import clustered_interval, holm_adjust, paired_family, signed_rank
from src.schema import AnswerRecord, Claim, EvidencePassage

ROOT = Path(__file__).resolve().parents[1]


def example(rid, labels):
    text = f"Original synthetic answer {rid}."
    return AnswerRecord(id=rid, source_dataset="synthetic", split="dev", question=f"Question {rid}?",
        answer=text, gold_repaired_answer="Correct synthetic answer." if rid == "d1" else "",
        claims=[Claim(id=f"{rid}-c{i}", text=f"Synthetic claim {i} for {rid}.", sentence_id=0,
                      span=[0, len(text)], label=label,
                      evidence=[EvidencePassage(doc_id="e", title="Synthetic citation", text="Inspected fixture.")]
                      if label != "Unverifiable" else []) for i, label in enumerate(labels)])


class TestCorrectedMetrics(unittest.TestCase):
    def test_contradicted_unverifiable_confusion_is_not_three_way_success(self):
        metric = compute_detection_metrics(["Contradicted"], ["Unverifiable"])
        self.assertEqual(metric["binary"]["f1"], 1)
        self.assertEqual(metric["three_way"]["macro_f1"], 0)
        self.assertEqual(metric["three_way"]["accuracy"], 0)
        self.assertEqual(metric["three_way"]["confusion_matrix"]["rows_gold_columns_pred"][2][1], 1)
        self.assertEqual(metric["three_way"]["per_class"]["Supported"]["support"], 0)

    def test_absent_classes_use_explicit_zero_division(self):
        metric = compute_detection_metrics(["Supported"], ["Supported"])
        self.assertAlmostEqual(metric["three_way"]["macro_f1"], 1 / 3)
        for label in (None, "unknown"):
            with self.assertRaises(ValueError):
                compute_detection_metrics([label], ["Supported"])

    def test_missing_extra_duplicate_and_changed_claims_rejected(self):
        gold = [example("d1", ["Supported", "Contradicted"])]
        for change in (lambda rows: rows[0].claims.pop(),
                       lambda rows: rows[0].claims.append(copy.deepcopy(rows[0].claims[0])),
                       lambda rows: setattr(rows[0].claims[0], "text", "different fact"),
                       lambda rows: setattr(rows[0].claims[0], "label", None),
                       lambda rows: setattr(rows[0], "answer", "different answer"),
                       lambda rows: rows.append(copy.deepcopy(rows[0]))):
            rows = copy.deepcopy(gold)
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_claim_detection(rows, gold, n_bootstrap=10)
        with self.assertRaises(ValueError):
            evaluate_dataset(gold, gold + [example("extra", ["Unverifiable"])], n_bootstrap=10)

    def test_order_does_not_affect_detection_and_answers_are_bootstrap_units(self):
        gold = [example("d1", ["Supported", "Contradicted"]), example("d2", ["Unverifiable"])]
        predictions = copy.deepcopy(gold)
        predictions[0].claims[1].label = "Unverifiable"
        result = evaluate_claim_detection(predictions, gold, 30, 7)
        predictions.reverse()
        for row in predictions:
            row.claims.reverse()
        self.assertEqual(result, evaluate_claim_detection(predictions, list(reversed(gold)), 30, 7))
        self.assertEqual(result["intervals"]["three_way_macro_f1"]["unit"], "answer")
        seen_sizes = []
        def statistic(groups):
            seen_sizes.append(tuple(len(group) for group in groups))
            return sum(map(len, groups))
        clustered_interval([[1] * 10, [1]], statistic, 20, 3)
        self.assertTrue(all(size in (1, 10) for sample in seen_sizes for size in sample))
        self.assertIn((10, 10), seen_sizes)

    def test_paired_tests_holm_zero_differences_and_exact_ids(self):
        self.assertEqual(signed_rank([0., 1.], [0., 1.])["p_value"], 1)
        self.assertEqual(signed_rank([0., 1.], [0., 1.])["method"], "all_zero")
        tied = signed_rank([1.] * 8, [0.] * 8)
        self.assertEqual(tied["method"], "conditional_exact_sign_flip")
        self.assertEqual(tied["p_value"], 2 / 2**8)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        with self.assertRaises(ValueError):
            paired_family({"bad": ({"a": 1}, {"b": 0})}, 10)
        with self.assertRaises(ValueError):
            signed_rank([float("nan")], [0.])
        with patch.dict("sys.modules", {"scipy.stats": None}):
            with self.assertRaises(ImportError):
                signed_rank([1.], [1.])


class TestOfflineAssessment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [example("d1", ["Supported", "Contradicted"]), example("d2", ["Unverifiable"])]
        self.gold_path = self.root / "gold.jsonl"
        self.gold_path.write_bytes(b"".join(canonical(record.to_dict()) + b"\n" for record in self.records))
        self.gold = load_gold(self.gold_path)
        self.config = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
        self.config["splits"] = {"dev": 2, "test": 1}
        self.config["systems"] = [{"id": "pipeline", "tasks": ["repair", "claim_verification"], "retrieval": "bm25"},
                                  {"id": "b0", "tasks": ["repair"], "retrieval": "none"},
                                  {"id": "b7", "tasks": ["claim_verification"], "retrieval": "bm25"}]
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.splits = self.root / "splits.json"
        self.splits.write_text(json.dumps({"dev_ids": ["d1", "d2"], "test_ids": ["t1"]}))
        def segmenter(text):
            return SimpleNamespace(sents=[SimpleNamespace(start_char=0, end_char=len(text))])
        self.addCleanup(patch.stopall)
        patch("src.experiments.runner.require_segmenter", return_value=segmenter).start()
        patch("src.experiments.runner.preflight", return_value={}).start()
        patch("src.experiments.runner.adapters.execute", side_effect=self.adapter).start()

    def adapter(self, system, record, task, options):
        if task == "repair":
            record.repaired_answer = (record.answer if system["id"] == "b0"
                                      else self.gold[record.id].gold_repaired_answer)
        else:
            for claim in record.claims:
                actual = next(c for c in self.gold[record.id].claims if c.id == claim.id)
                claim.label = actual.label
                if system["id"] == "pipeline" and actual.label == "Contradicted":
                    claim.label = "Unverifiable"
        return {"record": record.to_dict(), "native": {}, "retrieved_evidence": []}

    def prepare_run(self, task="repair"):
        directory = self.root / task
        run_benchmark(config_path=self.config_path, inputs_path=self.gold_path, splits_path=self.splits,
                      output_dir=directory, task=task, split="dev",
                      inventory_path=self.gold_path if task == "claim_verification" else None)
        return load_run(directory, self.gold)

    def packet_and_ratings(self):
        run = self.prepare_run()
        packet = self.root / "packet"
        export_packet(run, self.gold, packet)
        key = json.loads((packet / "private/key.json").read_text())
        rows = [json.loads(line) for line in (packet / "reviewer/ratings_template.jsonl").read_text().splitlines()]
        for rating in rows:
            link = key["mapping"][rating["item_id"]]
            record = self.gold[link["record_id"]]
            empty = not run["outputs"][link["system"]][record.id]["record"]["repaired_answer"]
            for value, claim in zip(rating["claim_outcomes"], record.check_worthy_claims):
                outcome = ("deleted" if empty else "retained" if claim.label == "Supported"
                           else "corrected" if link["system"] == "pipeline" else "still_incorrect")
                value.update(outcome=outcome, rationale="Independent synthetic judgment.",
                             evidence=[{"doc_id": "e", "title": "Inspected", "text": "Source fact."}]
                             if outcome == "corrected" else [])
            rating.update(new_errors=[], completeness=0 if empty else 2)
        path = self.root / "ratings.jsonl"
        self.write_ratings(path, rows)
        return run, packet, path, rows

    def write_ratings(self, path, rows):
        path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))

    def test_offline_scores_preserve_empty_deletion_and_do_not_claim_factuality(self):
        run = self.prepare_run()
        with patch("src.llm.call", side_effect=AssertionError("No inference allowed")):
            result = score_run(run, self.gold, 20)
        metrics = result["systems"]["pipeline"]
        self.assertEqual(metrics["reference_agreement"]["exact_match"]["estimate"], 1.)
        self.assertEqual(metrics["per_record_reference"]["d2"]["rouge_l_f1"], 1.)
        self.assertEqual(metrics["independent_repair"]["status"], "unavailable")
        self.assertEqual(metrics["claim_detection"]["status"], "unavailable")

    def test_claim_run_scored_separately_with_paired_cluster_differences(self):
        result = score_run(self.prepare_run("claim_verification"), self.gold, 20)
        self.assertLess(result["systems"]["pipeline"]["detection"]["three_way"]["macro_f1"], 1)
        self.assertEqual(result["systems"]["b7"]["detection"]["three_way"]["macro_f1"], 1)
        comparison = result["comparisons"]["pooled_macro_f1_differences"]["b7_vs_pipeline"]
        self.assertGreater(comparison["estimate"], 0)
        self.assertEqual(comparison["unit"], "answer")

    def test_missing_duplicate_and_modified_output_views_are_rejected(self):
        run = self.prepare_run()
        path = self.root / "repair/outputs/b0.jsonl"
        original = path.read_bytes()
        path.write_bytes(original.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            load_run(self.root / "repair", self.gold)
        path.write_bytes(original + original.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_run(self.root / "repair", self.gold)
        row = json.loads(original.splitlines()[0])
        row["record"]["repaired_answer"] = "Tampered"
        path.write_bytes(canonical(row) + b"\n" + original.splitlines(keepends=True)[1])
        with self.assertRaisesRegex(ValueError, "committed attempt"):
            load_run(self.root / "repair", self.gold)

    def test_packet_hides_names_imports_complete_reviews_and_scores_deletion_honestly(self):
        run, packet, path, _ = self.packet_and_ratings()
        public = (packet / "reviewer/items.jsonl").read_text()
        self.assertNotIn('"system"', public)
        self.assertNotIn('"record_id"', public)
        self.assertNotIn('"pipeline"', public)
        review = validate_assessments(import_ratings(packet, path, run, self.gold, "reviewer-a"), run, self.gold)
        result = score_assessments(review, self.gold, 20)
        self.assertEqual(result["systems"]["pipeline"]["metrics"]["correction_rate"]["estimate"], .5)
        self.assertEqual(result["systems"]["pipeline"]["per_record"]["d2"]["correction_rate"], 0)
        self.assertEqual(result["paired_repair_family"]["family_size"], 2)
        with self.assertRaises(FileExistsError):
            export_packet(run, self.gold, packet)

    def test_blank_incomplete_duplicate_and_empty_success_ratings_are_rejected(self):
        run, packet, path, rows = self.packet_and_ratings()
        invalid = [rows[:-1], rows + [copy.deepcopy(rows[0])]]
        blank = copy.deepcopy(rows)
        blank[0]["claim_outcomes"][0]["outcome"] = None
        invalid.append(blank)
        missing_error_review = copy.deepcopy(rows)
        missing_error_review[0]["new_errors"] = None
        invalid.append(missing_error_review)
        empty = copy.deepcopy(rows)
        key = json.loads((packet / "private/key.json").read_text())
        for row in empty:
            link = key["mapping"][row["item_id"]]
            if link["system"] == "pipeline" and link["record_id"] == "d2":
                row["claim_outcomes"][0].update(outcome="corrected", evidence=[{"doc_id":"e","title":"T","text":"X"}])
        invalid.append(empty)
        for sample in invalid:
            self.write_ratings(path, sample)
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                import_ratings(packet, path, run, self.gold, "reviewer")

    def test_gold_or_packet_changes_invalidate_review_import(self):
        run, packet, path, rows = self.packet_and_ratings()
        changed = copy.deepcopy(self.gold)
        changed["d1"].claims[0].label = "Unverifiable"
        with self.assertRaisesRegex(ValueError, "Gold changed"):
            import_ratings(packet, path, run, changed, "reviewer")
        items = packet / "reviewer/items.jsonl"
        items.write_bytes(items.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            import_ratings(packet, path, run, self.gold, "reviewer")

    def test_disagreements_require_notes_and_distinct_adjudicator(self):
        run, packet, path, rows = self.packet_and_ratings()
        first = import_ratings(packet, path, run, self.gold, "first")
        second = copy.deepcopy(first)
        second["reviewer"] = "second"
        changed = next(row for row in second["assessments"] if row["system"] == "pipeline" and row["record_id"] == "d1")
        changed["completeness"] = 1
        self.assertEqual(len(agreement(first, second)["disagreements"]), 1)
        final = copy.deepcopy(first)
        final["reviewer"] = "adjudicator"
        with self.assertRaisesRegex(ValueError, "notes"):
            adjudicate(first, second, final)
        next(row for row in final["assessments"] if row["system"] == "pipeline" and row["record_id"] == "d1")["notes"] = "Evidence resolves completeness."
        result = adjudicate(first, second, final)
        self.assertTrue(result["adjudicated"])
        validate_assessments(result, run, self.gold)
        altered = copy.deepcopy(result)
        altered["agreement"]["outcome_agreement"] = -1
        with self.assertRaisesRegex(ValueError, "provenance/decisions"):
            validate_assessments(altered, run, self.gold)
        self.assertEqual(len(result["decisions"]), 1)
        self.assertEqual(len(result["source_review_sha256"]), 2)

    def test_cli_scoring_and_review_import_are_offline(self):
        from scripts.run_eval import main as evaluate
        from scripts.review_repairs import main as review
        run, packet, ratings, rows = self.packet_and_ratings()
        imported, metrics = self.root / "human.json", self.root / "metrics.json"
        with patch("src.llm.call", side_effect=AssertionError("No API calls")):
            self.assertEqual(review(["import", "--run-dir", str(self.root / "repair"),
                "--gold", str(self.gold_path), "--packet-dir", str(packet), "--ratings", str(ratings),
                "--reviewer", "first", "--output", str(imported)]), 0)
            self.assertEqual(evaluate(["--run-dir", str(self.root / "repair"), "--gold", str(self.gold_path),
                "--assessments", str(imported), "--output", str(metrics), "--n_bootstrap", "20"]), 0)
        data = json.loads(metrics.read_text())
        self.assertEqual(data["assessment_status"], "single_reviewer")
        self.assertEqual(data["systems"]["pipeline"]["independent_repair"]["status"], "available")
        self.assertIn("gold_sha256", data["provenance"])

    def test_answer_detection_is_scored_at_answer_level_only(self):
        self.config["systems"] = [{"id": "b1", "tasks": ["answer_detection"], "retrieval": "none"}]
        self.config_path.write_text(yaml.safe_dump(self.config))
        def adapter(system, record, task, options):
            return {"record": record.to_dict(), "native": {"answer_detection": {
                    "has_hallucination": True, "confidence": .9}}, "retrieved_evidence": []}
        with patch("src.experiments.runner.adapters.execute", side_effect=adapter):
            run = self.prepare_run("answer_detection")
        metrics = score_run(run, self.gold, 20)["systems"]["b1"]
        self.assertEqual(metrics["answer_detection"]["f1"], 1.)
        self.assertNotIn("three_way", metrics)

    def test_missing_normalized_review_and_duplicate_claims_rejected(self):
        run, packet, path, rows = self.packet_and_ratings()
        review = import_ratings(packet, path, run, self.gold, "human")
        partial = copy.deepcopy(review)
        partial["assessments"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_assessments(partial, run, self.gold)
        duplicate = copy.deepcopy(rows)
        duplicate[0]["claim_outcomes"].append(copy.deepcopy(duplicate[0]["claim_outcomes"][0]))
        self.write_ratings(path, duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            import_ratings(packet, path, run, self.gold, "human")

    def test_legacy_significance_preserves_deletion_and_rejects_partial_pairs(self):
        import contextlib
        import io
        from scripts.compute_significance import main, load_records
        system = self.root / "system.jsonl"
        baseline = self.root / "baseline.jsonl"
        output = self.root / "legacy.json"
        repaired, original = copy.deepcopy(self.records), copy.deepcopy(self.records)
        for record in repaired:
            record.repaired_answer = ""
        for record in original:
            record.repaired_answer = record.answer
        system.write_bytes(b"".join(canonical(record.to_dict()) + b"\n" for record in repaired))
        baseline.write_bytes(b"".join(canonical(record.to_dict()) + b"\n" for record in original))
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--system", str(system), "--baseline", str(baseline), "--output", str(output),
                  "--n_bootstrap", "20"])
        metrics = json.loads(output.read_text())
        self.assertEqual(metrics["rouge_l"]["system_mean"], 0.)
        self.assertEqual(metrics["rouge_l"]["baseline_mean"], 1.)
        self.assertIn("p_value_holm", metrics["rouge_l"])
        baseline.write_bytes(canonical(original[0].to_dict()) + b"\n")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--system", str(system), "--baseline", str(baseline)])
        system.write_bytes((canonical(repaired[0].to_dict()) + b"\n") * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_records(str(system))


if __name__ == "__main__":
    unittest.main()
