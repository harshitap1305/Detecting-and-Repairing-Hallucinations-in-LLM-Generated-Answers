"""Offline contract tests for protocol validation and annotation preparation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.prepare_annotation_batch import prepare, main
from src.experiments.protocol import SYSTEMS, load_config, require_segmenter, validate_config

ROOT = Path(__file__).resolve().parents[1]


def example(record_id, split, question=None):
    return {"id": record_id, "source_dataset": "fixture", "split": split,
            "question": question or f"Question {record_id}?", "answer": "Synthetic answer.",
            "gold_repaired_answer": "LEAKED reference", "reliability_score_before": 0.9,
            "claims": [{"id": "leaked", "label": "Supported", "evidence": ["secret"]}]}


class TestEvaluationPreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/evaluation.yaml")
        self.config["splits"] = {"dev": 1, "test": 1}
        self.config["sources"] = [{"id": "fixture", "version": "synthetic-v1",
                                  "path": "source.jsonl", "quotas": {"dev": 1, "test": 1}}]
        self.records = [example("d1", "dev"), example("d2", "dev"),
                        example("t1", "test"), example("t2", "test")]

    def write_inputs(self):
        (self.root / "source.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")
        path = self.root / "evaluation.yaml"
        path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        return path

    def test_registry_rejects_inapplicable_metrics_and_retrievers(self):
        for change in (
            lambda cfg: cfg["systems"][2].update(tasks=["claim_verification"]),
            lambda cfg: cfg["systems"][6].update(retrieval="hybrid"),
            lambda cfg: cfg["systems"][7].pop("detector"),
            lambda cfg: cfg.update(typo=True),
            lambda cfg: cfg["splits"].update(dev=True),
            lambda cfg: cfg["sources"][0].update(version=""),
            lambda cfg: cfg["sources"][0]["quotas"].update(test=3),
            lambda cfg: cfg["systems"].append(copy.deepcopy(cfg["systems"][0])),
        ):
            with self.subTest(change=change):
                cfg = copy.deepcopy(self.config)
                change(cfg)
                with self.assertRaises(ValueError):
                    validate_config(cfg)
        self.assertNotIn("claim_verification", SYSTEMS["b1"].tasks)
        self.assertEqual(SYSTEMS["b6"].prerequisite, "pipeline_detector")

    def test_template_requires_actual_sources_before_preparation(self):
        with self.assertRaisesRegex(ValueError, "at least one source"):
            load_config(ROOT / "configs/evaluation.yaml", require_sources=True)

    def test_selection_reproducible_and_source_predictions_removed(self):
        config = self.write_inputs()
        first = prepare(config, self.root / "first")
        second = prepare(config, self.root / "second")
        self.assertEqual((first / "manifest.json").read_bytes(),
                         (second / "manifest.json").read_bytes())
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "awaiting_human_annotation")
        self.assertEqual(len(manifest["selected"]), 2)
        self.assertEqual(len(manifest["excluded"]), 2)
        for split in ("dev", "test"):
            safe = json.loads((first / "inputs" / f"{split}.jsonl").read_text())
            self.assertEqual(safe["split"], split)
            self.assertEqual(safe["claims"], [])
            self.assertIsNone(safe["gold_repaired_answer"])
            self.assertIsNone(safe["reliability_score_before"])
            self.assertNotIn("LEAKED", (first / "inputs" / f"{split}.jsonl").read_text())
            self.assertEqual((first / "inputs" / f"{split}.jsonl").read_bytes(),
                             (first / "annotator_a" / f"{split}.jsonl").read_bytes())
        for filename, sha in manifest["artifact_sha256"].items():
            import hashlib
            self.assertEqual(hashlib.sha256((first / filename).read_bytes()).hexdigest(), sha)
        self.records.reverse()
        self.write_inputs()
        third = prepare(config, self.root / "third")
        self.assertEqual((first / "splits.json").read_bytes(), (third / "splits.json").read_bytes())
        self.assertNotEqual(manifest["sources"][0]["sha256"],
                            json.loads((third / "manifest.json").read_text())["sources"][0]["sha256"])

    def test_prompt_groups_never_cross_splits(self):
        self.records += [example("dup-dev", "dev", "  SAME   Prompt "),
                         example("dup-test", "test", "same prompt")]
        self.records += [example("duplicate-1", "dev", "same dev question"),
                         example("duplicate-2", "dev", "Same Dev Question")]
        output = prepare(self.write_inputs(), self.root / "batch")
        manifest = json.loads((output / "manifest.json").read_text())
        excluded = {item["id"]: item["reason"] for item in manifest["excluded"]}
        self.assertEqual(excluded["dup-dev"], "cross_split_prompt")
        self.assertEqual(excluded["dup-test"], "cross_split_prompt")
        self.assertEqual(list(excluded.values()).count("duplicate_prompt"), 1)

    def test_insufficient_data_and_bad_ids_leave_no_output(self):
        for records, error in (([example("only-dev", "dev")], "quota"),
                               ([example("same", "dev"), example("same", "test")], "duplicate"),
                               ([example("bad", "train")], "declare dev or test")):
            self.records = records
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                prepare(self.write_inputs(), self.root / "batch")
            self.assertFalse((self.root / "batch").exists())

    def test_existing_batch_never_overwritten(self):
        config = self.write_inputs()
        output = prepare(config, self.root / "batch")
        before = (output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(config, output)
        self.assertEqual(before, (output / "manifest.json").read_bytes())

    def test_multiple_sources_keep_quotas_and_test_only_assignments(self):
        self.config["sources"][0]["quotas"]["test"] = 0
        self.config["sources"].append({"id": "external", "version": "fixture-only",
                                      "path": "external.jsonl", "quotas": {"dev": 0, "test": 1}})
        external = example("external-test", "test")
        external["source_dataset"] = "external"
        (self.root / "external.jsonl").write_text(json.dumps(external) + "\n", encoding="utf-8")
        config = self.write_inputs()
        first = prepare(config, self.root / "first")
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual({(row["source_dataset"], row["split"]) for row in manifest["selected"]},
                         {("fixture", "dev"), ("external", "test")})
        all_ids = {row["id"] for row in manifest["selected"] + manifest["excluded"]}
        self.assertEqual(all_ids, {row["id"] for row in self.records} | {"external-test"})
        self.config["sources"].reverse()
        self.write_inputs()
        second = prepare(config, self.root / "second")
        self.assertEqual((first / "splits.json").read_bytes(), (second / "splits.json").read_bytes())

    def test_blank_packet_cannot_pass_as_gold(self):
        from scripts.validate_gold import load_gold
        output = prepare(self.write_inputs(), self.root / "batch")
        with self.assertRaises(ValueError):
            load_gold(output / "annotator_a" / "dev.jsonl", "dev")

    def test_cli_nonzero_on_invalid_input(self):
        with patch("sys.stderr"):
            self.assertEqual(main(["--config", str(self.root / "missing.yaml"),
                                   "--output-dir", str(self.root / "out")]), 1)

    def test_segmenter_version_mismatch_has_no_fallback(self):
        with patch("importlib.metadata.version", return_value="wrong"), \
                patch.dict("sys.modules", {"spacy": unittest.mock.Mock()}):
            with self.assertRaisesRegex(ValueError, "Segmenter version"):
                require_segmenter(self.config)


if __name__ == "__main__":
    unittest.main()
