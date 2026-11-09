"""Offline report lifecycle, reproduction, completeness and corruption checks."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from scripts.audit_baseline import main as audit_cli
from scripts.build_evaluation_report import main as report_cli
from src.experiments.artifacts import canonical, file_hash
from src.experiments.assessment import import_ratings, adjudicate
from src.experiments.reporting import build_bundle, audit_bundle
from src.experiments.scoring import load_run
import tests.test_independent_evaluation as fixtures


class TestEvaluationBundle(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestOfflineAssessment(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def final_repair(self):
        run, packet, path, rows = self.fixture.packet_and_ratings()
        first = import_ratings(packet, path, run, self.fixture.gold, "reviewer-a")
        second, final = copy.deepcopy(first), copy.deepcopy(first)
        second["reviewer"], final["reviewer"] = "reviewer-b", "adjudicator"
        final = adjudicate(first, second, final)
        ratings = self.root / "adjudicated.json"
        ratings.write_bytes(canonical(final) + b"\n")
        bundle = self.root / "bundle"
        build_bundle(run_dir=self.root / "repair", gold_path=self.fixture.gold_path,
                     output_dir=bundle, assessments_path=ratings, n_bootstrap=20)
        return bundle, ratings

    def test_final_bundle_reproduces_after_original_run_and_review_files_are_removed(self):
        import shutil
        bundle, ratings = self.final_repair()
        report = (bundle / "report.md").read_bytes()
        metrics = (bundle / "metrics.json").read_bytes()
        shutil.rmtree(self.root / "repair")
        shutil.rmtree(self.root / "packet")
        ratings.unlink()
        with patch("src.llm.call", side_effect=AssertionError("No inference")):
            result = audit_bundle(bundle, self.fixture.gold_path)
        self.assertEqual(result["status"], "valid_final_task_bundle")
        self.assertTrue(result["recomputed"])
        self.assertEqual(report, (bundle / "report.md").read_bytes())
        self.assertEqual(metrics, (bundle / "metrics.json").read_bytes())
        self.assertIn(b"Adjudicated reviews", report)
        self.assertFalse((bundle / "gold.jsonl").exists())
        self.assertFalse(any(bundle.rglob("*.db")))
        self.assertNotIn(b"Original synthetic answer", report)

    def test_final_repair_requires_adjudicated_complete_reviews(self):
        run, packet, path, rows = self.fixture.packet_and_ratings()
        single = import_ratings(packet, path, run, self.fixture.gold, "single")
        ratings = self.root / "single.json"
        ratings.write_bytes(canonical(single))
        for supplied in (None, ratings):
            with self.subTest(supplied=supplied), self.assertRaisesRegex(ValueError, "adjudicated"):
                build_bundle(run_dir=self.root / "repair", gold_path=self.fixture.gold_path,
                             output_dir=self.root / "invalid", assessments_path=supplied, n_bootstrap=10)
        self.assertFalse((self.root / "invalid").exists())

    def test_progress_report_has_no_comparative_metrics(self):
        self.fixture.prepare_run()
        bundle = self.root / "progress"
        build_bundle(run_dir=self.root / "repair", gold_path=self.fixture.gold_path,
                     output_dir=bundle, mode="progress", n_bootstrap=10)
        self.assertEqual(audit_bundle(bundle, self.fixture.gold_path)["status"], "valid_progress_bundle")
        metrics = json.loads((bundle / "metrics.json").read_text())
        self.assertEqual(metrics["comparative_metrics"]["status"], "unavailable")
        report = (bundle / "report.md").read_text()
        self.assertIn("no final performance claims", report)
        self.assertNotIn("| Correction", report)

    def test_incomplete_execution_can_only_produce_progress_with_failure_counts(self):
        from src.experiments.runner import run_benchmark
        def flaky(system, record, task, options):
            if record.id == "d2":
                raise RuntimeError("Synthetic failed record")
            return self.fixture.adapter(system, record, task, options)
        with patch("src.experiments.runner.adapters.execute", side_effect=flaky):
            run_benchmark(config_path=self.fixture.config_path, inputs_path=self.fixture.gold_path,
                          splits_path=self.fixture.splits, output_dir=self.root / "repair", task="repair", split="dev")
        with self.assertRaises(ValueError):
            build_bundle(run_dir=self.root / "repair", gold_path=self.fixture.gold_path,
                         output_dir=self.root / "invalid", n_bootstrap=10)
        bundle = self.root / "progress"
        build_bundle(run_dir=self.root / "repair", gold_path=self.fixture.gold_path,
                     output_dir=bundle, mode="progress", n_bootstrap=10)
        self.assertEqual(audit_bundle(bundle, self.fixture.gold_path)["status"], "valid_progress_bundle")
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(manifest["failure_count"], 2)
        self.assertEqual(manifest["coverage"]["pipeline"]["missing_ids"], ["d2"])

    def test_missing_altered_extra_and_symlink_files_fail_checksums(self):
        bundle, _ = self.final_repair()
        target = bundle / "metrics.json"
        original = target.read_bytes()
        for action in (lambda: target.unlink(), lambda: target.write_bytes(original + b"\n"),
                       lambda: (bundle / "unexpected.txt").write_text("extra")):
            action()
            with self.assertRaisesRegex(ValueError, "inventory/checksum"):
                audit_bundle(bundle, self.fixture.gold_path)
            target.write_bytes(original)
            (bundle / "unexpected.txt").unlink(missing_ok=True)
        target.unlink()
        target.symlink_to(self.fixture.gold_path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            audit_bundle(bundle, self.fixture.gold_path)

    def test_modified_gold_fails_even_with_same_record_ids(self):
        bundle, _ = self.final_repair()
        self.fixture.gold_path.write_bytes(self.fixture.gold_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "gold hash"):
            audit_bundle(bundle, self.fixture.gold_path)

    def test_recomputed_metrics_reject_rehashed_fabricated_scores(self):
        bundle, _ = self.final_repair()
        metrics = json.loads((bundle / "metrics.json").read_text())
        metrics["human_repair"]["systems"]["pipeline"]["metrics"]["correction_rate"]["estimate"] = 99
        (bundle / "metrics.json").write_bytes(canonical(metrics) + b"\n")
        from src.experiments.reporting import render_report
        manifest = json.loads((bundle / "manifest.json").read_text())
        comparisons = json.loads((bundle / "comparisons.json").read_text())
        (bundle / "report.md").write_text(render_report(manifest, metrics, comparisons))
        self.rehash(bundle)
        with self.assertRaisesRegex(ValueError, "recomputation"):
            audit_bundle(bundle, self.fixture.gold_path)

    def rehash(self, bundle):
        (bundle / "checksums.json").write_bytes(canonical({"schema_version": 1, "files": {
            path.relative_to(bundle).as_posix(): file_hash(path)
            for path in bundle.rglob("*") if path.is_file() and path.name != "checksums.json"}}) + b"\n")

    def test_duplicate_outputs_rejected_even_when_checksums_are_updated(self):
        bundle, _ = self.final_repair()
        target = bundle / "execution/outputs/b0.jsonl"
        with target.open("ab") as stream:
            stream.write(target.read_bytes().splitlines(keepends=True)[0])
        self.rehash(bundle)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_bundle(bundle, self.fixture.gold_path)

    def test_detection_bundle_and_cli_audit(self):
        self.fixture.prepare_run("claim_verification")
        bundle = self.root / "detection-bundle"
        self.assertEqual(report_cli(["--run-dir", str(self.root / "claim_verification"),
            "--gold", str(self.fixture.gold_path), "--output-dir", str(bundle), "--n-bootstrap", "20"]), 0)
        self.assertEqual(audit_cli(["--bundle", str(bundle), "--gold", str(self.fixture.gold_path)]), 0)
        self.assertEqual(report_cli(["--verify", str(bundle), "--gold", str(self.fixture.gold_path)]), 0)
        report = (bundle / "report.md").read_text()
        self.assertIn("confusion matrix", report)
        self.assertIn("Secondary signed-rank", report)
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(report_cli(["--run-dir", str(self.root / "claim_verification"),
                "--gold", str(self.fixture.gold_path), "--output-dir", str(bundle)]), 1)

    def test_answer_detection_bundle_needs_no_repair_reviews(self):
        import yaml
        self.fixture.config["systems"] = [{"id": "b1", "tasks": ["answer_detection"], "retrieval": "none"}]
        self.fixture.config_path.write_text(yaml.safe_dump(self.fixture.config))
        def adapter(system, record, task, options):
            return {"record": record.to_dict(), "native": {"answer_detection": {
                    "has_hallucination": True, "confidence": .8}}, "retrieved_evidence": []}
        with patch("src.experiments.runner.adapters.execute", side_effect=adapter):
            self.fixture.prepare_run("answer_detection")
        bundle = self.root / "answer-bundle"
        build_bundle(run_dir=self.root / "answer_detection", gold_path=self.fixture.gold_path,
                     output_dir=bundle, n_bootstrap=10)
        self.assertEqual(audit_bundle(bundle, self.fixture.gold_path)["task"], "answer_detection")
        self.assertIn("## Answer-level detection", (bundle / "report.md").read_text())
        rows = [json.loads(line) for line in (bundle / "evaluations/b1.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["answer_detection"]["gold"] for row in rows))

    def test_fast_audit_still_requires_per_system_evaluation_files(self):
        bundle, _ = self.final_repair()
        self.assertFalse(audit_bundle(bundle, self.fixture.gold_path, recompute=False)["recomputed"])
        (bundle / "evaluations/b0.jsonl").unlink()
        self.rehash(bundle)
        with self.assertRaisesRegex(ValueError, "per-system evaluation"):
            audit_bundle(bundle, self.fixture.gold_path, recompute=False)

    def test_annotation_to_report_lifecycle(self):
        # All source data, gold judgments and model outputs are synthetic fixtures.
        from scripts.prepare_annotation_batch import prepare
        from src.experiments.runner import run_benchmark
        rows = [record.to_dict() for record in self.fixture.records]
        test = copy.deepcopy(rows[0])
        test.update(id="test-candidate", split="test", question="Distinct test question?")
        rows.append(test)
        source = self.root / "source.jsonl"
        source.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        config = self.fixture.config
        config["sources"] = [{"id": "synthetic", "version": "synthetic-fixture-only",
                              "path": "source.jsonl", "quotas": {"dev": 2, "test": 1}}]
        import yaml
        self.fixture.config_path.write_text(yaml.safe_dump(config))
        prepared = prepare(self.fixture.config_path, self.root / "annotation")
        inputs = prepared / "inputs/dev.jsonl"
        self.assertNotIn(b"Synthetic claim", inputs.read_bytes())
        original_adapter = self.fixture.adapter
        def interrupt(system, record, task, options):
            if system["id"] == "b0":
                raise KeyboardInterrupt()
            return original_adapter(system, record, task, options)
        kwargs = dict(config_path=self.fixture.config_path, inputs_path=inputs,
                      splits_path=prepared / "splits.json", output_dir=self.root / "repair", task="repair", split="dev")
        with patch("src.experiments.runner.adapters.execute", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_benchmark(**kwargs)
        run_benchmark(**kwargs, resume=True)
        # The separately supplied fixture gold fills the blank human annotation role.
        load_run(self.root / "repair", self.fixture.gold)
        existing = load_run(self.root / "repair", self.fixture.gold)
        with patch.object(self.fixture, "prepare_run", return_value=existing):
            bundle, _ = self.final_repair()
        self.assertEqual(audit_bundle(bundle, self.fixture.gold_path)["status"], "valid_final_task_bundle")


if __name__ == "__main__":
    unittest.main()
