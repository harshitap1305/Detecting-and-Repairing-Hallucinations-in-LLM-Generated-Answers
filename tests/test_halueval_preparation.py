"""Source provenance and isolation tests; fixtures are not benchmark gold."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from scripts.prepare_halueval_source import prepare_source
from scripts.export_annotation_sentences import export_sentences
from scripts.prepare_annotation_batch import prepare


class TestPinnedHaluEvalPreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.git("init", "--quiet")
        (self.repo / "data").mkdir()
        rows = [{"question": f"Synthetic question {i}?", "right_answer": "Reference answer.",
                 "hallucinated_answer": "Generated answer.", "knowledge": "Source context."} for i in range(100)]
        rows.append(dict(rows[0]))
        self.source = self.repo / "data/qa_data.json"
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (self.repo / "README.md").write_text("Synthetic fixture only.")
        (self.repo / "LICENSE").write_text("Fixture license text.")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.revision = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    def test_pinned_source_is_deterministic_mixed_and_unlabeled(self):
        first = prepare_source(self.repo, self.revision, self.root / "first")
        second = prepare_source(self.repo, self.revision, self.root / "second")
        self.assertEqual((first / "candidates.jsonl").read_bytes(), (second / "candidates.jsonl").read_bytes())
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual(manifest["raw_rows"], 101)
        self.assertEqual(manifest["unique_prompts"], 100)
        self.assertEqual(len(manifest["exclusions"]), 1)
        self.assertTrue(all(count > 0 for count in manifest["origin_counts"].values()))
        for line in (first / "candidates.jsonl").read_text().splitlines():
            row = json.loads(line)
            self.assertIsNone(row["gold_repaired_answer"])
            self.assertEqual(row["claims"], [])
            self.assertNotIn("knowledge", row)
            self.assertNotIn("answer_field", row)
        with self.assertRaises(FileExistsError):
            prepare_source(self.repo, self.revision, first)

    def test_mutated_source_and_wrong_revision_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "HEAD"):
            prepare_source(self.repo, "0" * 40, self.root / "out")
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Git object"):
            prepare_source(self.repo, self.revision, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_sentence_offsets_match_frozen_inputs_and_reject_modified_candidates(self):
        source = prepare_source(self.repo, self.revision, self.root / "processed")
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/evaluation.yaml").read_text())
        config["splits"] = {"dev": 2, "test": 2}
        config["sources"] = [{"id": "halueval", "version": self.revision, "path": str(source / "candidates.jsonl"),
                               "quotas": {"dev": 2, "test": 2}}]
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        batch = prepare(path, self.root / "batch")
        def nlp(text):
            return SimpleNamespace(sents=[SimpleNamespace(text=text, start_char=0, end_char=len(text))])
        with patch("scripts.export_annotation_sentences.require_segmenter", return_value=nlp), \
                patch("scripts.export_annotation_sentences.version", return_value="fixture"):
            output = export_sentences(batch, self.root / "sentences")
            for line in (output / "dev.jsonl").read_text().splitlines():
                row = json.loads(line)
                for sentence in row["sentences"]:
                    start, end = sentence["span"]
                    self.assertEqual(row["answer"][start:end], sentence["text"])
            candidate = batch / "inputs/dev.jsonl"
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                export_sentences(batch, self.root / "changed")


if __name__ == "__main__":
    unittest.main()
