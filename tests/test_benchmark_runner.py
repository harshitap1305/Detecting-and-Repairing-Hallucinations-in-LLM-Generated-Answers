"""Offline execution, isolation, integrity, and resume regressions."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from src.experiments.artifacts import IDENTITY, canonical, file_hash, fingerprint, load_inputs, read_inventory
from src.experiments.runner import run_benchmark
from src.experiments.runtime import execution
from src import llm

ROOT = Path(__file__).resolve().parents[1]


def segmenter(text):
    return SimpleNamespace(sents=[SimpleNamespace(start_char=0, end_char=len(text))])


class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cfg = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
        cfg["splits"] = {"dev": 2, "test": 1}
        cfg["systems"] = [{"id": "b0", "tasks": ["repair"], "retrieval": "none"}]
        self.config = cfg
        self.config_path = self.root / "config.yaml"
        self.write_config()
        self.inputs = self.root / "inputs.jsonl"
        self.rows = [{"id": name, "source_dataset": "synthetic", "split": "dev",
                      "question": name + "?", "answer": "Synthetic answer.",
                      "gold_repaired_answer": "secret", "claims": [{"secret": "labels"}]}
                     for name in ("first", "second")]
        self.inputs.write_bytes(b"".join(canonical(row) + b"\n" for row in self.rows))
        self.splits = self.root / "splits.json"
        self.splits.write_text(json.dumps({"dev_ids": ["first", "second"], "test_ids": ["test"]}))
        self.kwargs = dict(config_path=self.config_path, inputs_path=self.inputs,
                           splits_path=self.splits, output_dir=self.root / "run", task="repair", split="dev")
        self.addCleanup(patch.stopall)
        patch("src.experiments.runner.require_segmenter", return_value=segmenter).start()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config))

    def test_b0_runs_offline_and_complete_resume_skips_every_record(self):
        manifest = run_benchmark(**self.kwargs)
        self.assertEqual(manifest["status"], "execution_complete")
        self.assertFalse(manifest["benchmark_complete"])
        paths = list((self.root / "run" / "attempts").glob("*/*.json"))
        self.assertEqual(len(paths), 2)
        outputs = (self.root / "run" / "outputs" / "b0.jsonl").read_bytes()
        self.assertNotIn(b"secret", outputs)
        self.assertIsNone(json.loads(outputs.splitlines()[0])["record"]["gold_repaired_answer"])
        with patch("src.experiments.runner.adapters.execute", side_effect=AssertionError("must skip")):
            resumed = run_benchmark(**self.kwargs, resume=True)
        self.assertEqual(resumed["successful_outputs"], 2)
        self.assertEqual(outputs, (self.root / "run" / "outputs" / "b0.jsonl").read_bytes())
        with self.assertRaises(FileExistsError):
            run_benchmark(**self.kwargs)

    def test_retrieval_limits_are_frozen_in_identity_and_reject_changed_resume(self):
        manifest = run_benchmark(**self.kwargs)
        options = manifest["identity"]["options"]
        self.assertEqual(options["query_strategy"], "contextual_v1")
        self.assertEqual(options["retrieval_max_queries"], 3)
        self.assertEqual(options["retrieval_max_passages"], 20)
        self.assertEqual(options["tracking_max_claims"], 64)
        with self.assertRaisesRegex(ValueError, "Resume identity changed"):
            run_benchmark(**self.kwargs, resume=True, options={"retrieval_max_queries": 4})

    def test_modified_generating_source_snapshot_prevents_resume(self):
        run_benchmark(**self.kwargs)
        source = self.root / "run/source.zip"
        self.assertTrue(source.is_file())
        source.write_bytes(b"modified source snapshot")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            run_benchmark(**self.kwargs, resume=True)

    def test_failure_remains_visible_and_retry_does_not_repeat_success(self):
        from src.experiments.adapters import execute
        calls = []
        def flaky(system, record, task, options):
            calls.append(record.id)
            if record.id == "second":
                raise RuntimeError("synthetic failure")
            return execute(system, record, task, options)
        with patch("src.experiments.runner.adapters.execute", side_effect=flaky):
            manifest = run_benchmark(**self.kwargs)
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(manifest["successful_outputs"], 1)
        self.assertIn("synthetic failure", (self.root / "run" / "errors.jsonl").read_text())
        with patch("src.experiments.runner.adapters.execute", wraps=execute) as adapter:
            resumed = run_benchmark(**self.kwargs, resume=True)
        self.assertEqual(adapter.call_count, 1)
        self.assertEqual(resumed["status"], "execution_complete")
        self.assertEqual(len(list((self.root / "run" / "attempts").glob("*/*.json"))), 3)
        self.assertIn("synthetic failure", (self.root / "run" / "errors.jsonl").read_text())

    def test_interrupt_exports_progress_and_resumes(self):
        from src.experiments.adapters import execute
        def interrupt(system, record, task, options):
            if record.id == "second":
                raise KeyboardInterrupt()
            return execute(system, record, task, options)
        with patch("src.experiments.runner.adapters.execute", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_benchmark(**self.kwargs)
        self.assertEqual(json.loads((self.root / "run" / "manifest.json").read_text())["status"], "incomplete")
        self.assertEqual(run_benchmark(**self.kwargs, resume=True)["status"], "execution_complete")

    def test_changed_input_code_and_options_cannot_resume(self):
        run_benchmark(**self.kwargs)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            run_benchmark(**self.kwargs, resume=True, options={"model": "changed"})
        with patch("src.experiments.runner.code_identity", return_value={"commit": "changed"}):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                run_benchmark(**self.kwargs, resume=True)
        self.inputs.write_bytes(self.inputs.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            run_benchmark(**self.kwargs, resume=True)

    def test_modified_trace_and_deleted_success_prevent_resume(self):
        run_benchmark(**self.kwargs)
        trace = next((self.root / "run" / "traces").glob("*.jsonl"))
        trace.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Trace checksum"):
            run_benchmark(**self.kwargs, resume=True)
        trace.write_text("")
        next((self.root / "run" / "attempts").glob("*/*.json")).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            run_benchmark(**self.kwargs, resume=True)

    def test_fixed_inventory_only_passes_unlabeled_claims_and_never_repairs(self):
        self.config["systems"] = [{"id": "pipeline", "tasks": ["claim_verification"], "retrieval": "bm25"}]
        self.write_config()
        inventory = self.root / "inventory.jsonl"
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["claims"] = [{"id": row["id"] + "_c", "text": "Synthetic claim.", "sentence_id": 0,
                              "span": [0, 5], "check_worthy": True, "label": "Contradicted",
                              "label_rationale": "secret", "evidence": [{"secret": True}]}]
        inventory.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        def adapter(system, record, task, options):
            self.assertIsNone(record.gold_repaired_answer)
            self.assertIsNone(record.claims[0].label)
            self.assertEqual(record.claims[0].evidence, [])
            self.assertEqual(record.claims[0].label_rationale, "")
            record.claims[0].label = "Supported"
            return {"record": record.to_dict(), "native": {}, "retrieved_evidence": []}
        args = {**self.kwargs, "task": "claim_verification", "inventory_path": inventory}
        with patch("src.experiments.runner.preflight", return_value={}), \
                patch("src.experiments.runner.adapters.execute", side_effect=adapter):
            manifest = run_benchmark(**args)
        self.assertEqual(manifest["status"], "execution_complete")

    def test_missing_or_duplicate_inputs_rejected_before_directory_creation(self):
        self.inputs.write_bytes(canonical(self.rows[0]) + b"\n")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            run_benchmark(**self.kwargs)
        self.assertFalse((self.root / "run").exists())
        self.inputs.write_bytes((canonical(self.rows[0]) + b"\n") * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_benchmark(**self.kwargs)

    def test_attempt_payload_corruption_is_detected(self):
        run_benchmark(**self.kwargs)
        path = next((self.root / "run" / "attempts").glob("*/*.json"))
        wrapper = json.loads(path.read_text())
        wrapper["result"]["record"]["repaired_answer"] = "Changed without checksum"
        path.write_text(json.dumps(wrapper))
        with self.assertRaisesRegex(ValueError, "Attempt checksum"):
            run_benchmark(**self.kwargs, resume=True)

    def test_concurrent_writer_is_refused(self):
        from src.experiments.runner import run_lock
        directory = self.root / "locked"
        directory.mkdir()
        with run_lock(directory):
            with self.assertRaisesRegex(ValueError, "Another process"):
                with run_lock(directory):
                    self.fail("Second lock must not succeed")

    def test_missing_credentials_fail_before_creating_a_run(self):
        self.config["systems"] = [{"id": "b1", "tasks": ["answer_detection"], "retrieval": "none"}]
        self.write_config()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "credential"):
                run_benchmark(**{**self.kwargs, "task": "answer_detection"})
        self.assertFalse((self.root / "run").exists())


class TestBenchmarkCacheAndStrictness(unittest.TestCase):
    def test_cache_scoped_to_provider_endpoint_and_path(self):
        self.assertNotEqual(llm._make_key("m", "p", 0, 1, 42, "groq"),
                            llm._make_key("m", "p", 0, 1, 42, "openai"))
        self.assertNotEqual(llm._make_key("m", "p", 0, 1, 42, "openai", "https://a"),
                            llm._make_key("m", "p", 0, 1, 42, "openai", "https://b"))
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"LLM_CACHE_PATH": str(Path(root) / "a.db")}):
                llm.reset_connection()
                llm._cache_set("key", "m", "answer")
                with patch.dict(os.environ, {"LLM_CACHE_PATH": str(Path(root) / "b.db")}):
                    self.assertIsNone(llm._cache_get("key"))
                self.assertEqual(llm._cache_get("key"), "answer")
                llm.reset_connection()

    def test_actual_calls_are_recorded_including_cache_hits_and_seed(self):
        calls = []
        with patch.object(llm, "_call_groq", return_value="response") as backend, \
                execution(record=calls.append, seed=100):
            self.assertEqual(llm.call("prompt", use_cache=False), "response")
        self.assertEqual(backend.call_args.args[-1], 100)
        self.assertEqual(calls[0]["response"], "response")
        with patch.object(llm, "_cache_get", return_value="cached"), execution(record=calls.append):
            llm.call("prompt")
        self.assertTrue(calls[-1]["cache_hit"])

    def test_failed_requests_are_recorded_and_retries_bypass_cache(self):
        calls = []
        with patch.object(llm, "_call_groq", side_effect=RuntimeError("offline")), \
                execution(record=calls.append):
            with self.assertRaises(RuntimeError):
                llm.call("failed request", use_cache=False)
        self.assertEqual(calls[0]["prompt"], "failed request")
        self.assertEqual(calls[0]["error"]["type"], "RuntimeError")
        with patch.object(llm, "_cache_get", side_effect=AssertionError("must bypass")), \
                patch.object(llm, "_call_groq", return_value="fresh"), execution(refresh=True):
            self.assertEqual(llm.call("retry request"), "fresh")

    def test_strict_parsers_surface_failures_and_legacy_mode_still_works(self):
        from src.baselines.b1_answer_level import run_b1
        from src.baselines.b7_factcheck_gpt import classify_claim_stance
        from src.verify import _parse_verdict
        from src.schema import AnswerRecord, EvidencePassage
        record = AnswerRecord(id="r", source_dataset="fixture", question="Q?", answer="A.")
        with patch("src.baselines.b1_answer_level.llm_call", return_value="invalid"):
            self.assertFalse(run_b1(record)["has_hallucination"])
            with execution(), self.assertRaises(ValueError):
                run_b1(record)
        with execution(), self.assertRaises(ValueError):
            _parse_verdict('{"unknown": "label"}')
        with patch("src.baselines.b7_factcheck_gpt.llm_call", return_value="{}"):
            with execution(), self.assertRaises(ValueError):
                classify_claim_stance("fact", [EvidencePassage(doc_id="d", title="T", text="Fact.")])


class TestNativeAdapters(unittest.TestCase):
    def test_all_native_adapters_preserve_input_and_separate_scores(self):
        from contextlib import ExitStack
        from src.experiments.adapters import execute
        from src.experiments.runner import DEFAULTS
        from src.experiments.protocol import SYSTEMS
        from src.schema import AnswerRecord, EvidencePassage
        evidence = EvidencePassage(doc_id="e", title="Synthetic", text="Synthetic answer.")
        options = {**DEFAULTS, "index_path": "unused"}
        def llm_response(prompt, **kwargs):
            if "NOW DECOMPOSE" in prompt:
                return '[{"claim":"Synthetic factual claim.","sentence_id":0,"check_worthy":true}]'
            if "objective fact-checking judge" in prompt:
                return '{"label":"Supported","rationale":"fixture","supporting_passage_ids":[1]}'
            if "Does this answer contain" in prompt:
                return '{"has_hallucination":false,"confidence":0.8,"explanation":"fixture"}'
            if "Context samples generated" in prompt:
                return '{"verdict":"Consistent","score":0.8}'
            if "generate 2 to 4" in prompt:
                return '["A synthetic question?"]'
            if "Classify the evidence stance" in prompt:
                return '{"stance":"Support","reason":"fixture"}'
            return "Synthetic revised answer."
        def retrieve(record, **kwargs):
            for claim in record.claims:
                claim.evidence = [evidence]
            return record
        with ExitStack() as stack:
            for module in ("src.llm", "src.baselines.b1_answer_level", "src.baselines.b2_selfcheck",
                           "src.baselines.b3_self_repair", "src.baselines.b4_cove",
                           "src.baselines.b5_rarr", "src.baselines.b7_factcheck_gpt"):
                stack.enter_context(patch(module + (".call" if module == "src.llm" else ".llm_call"),
                                          side_effect=llm_response))
            stack.enter_context(patch("src.decompose._get_nlp", return_value=None))
            stack.enter_context(patch("src.pipeline.retrieve", side_effect=retrieve))
            stack.enter_context(patch("src.retrieve.retrieve_bm25", return_value=[evidence]))
            stack.enter_context(patch("src.baselines.b7_factcheck_gpt.retrieve_bm25", return_value=[evidence]))
            for name in SYSTEMS:
                record = AnswerRecord(id="r", source_dataset="synthetic", question="Q?", answer="Synthetic answer.")
                task = "answer_detection" if name == "b1" else "repair"
                method = "bm25" if name in ("pipeline", "b5", "b6", "b7") else "none"
                with self.subTest(system=name), execution():
                    payload = execute({"id": name, "retrieval": method}, record, task, options)
                    self.assertEqual(payload["record"]["answer"], "Synthetic answer.")
                    self.assertIsNone(payload["record"]["gold_repaired_answer"])
                    if name == "b1":
                        self.assertIn("answer_detection", payload["native"])
                        self.assertIsNone(payload["record"]["repaired_answer"])
                    else:
                        self.assertIsInstance(payload["record"]["repaired_answer"], str)
                    if name == "b2":
                        self.assertIsNone(payload["record"]["reliability_score_after"])
                        self.assertAlmostEqual(payload["native"]["sentence_consistency"][0]["score"], 0.8)
                    if name == "b5":
                        self.assertEqual(payload["retrieved_evidence"][0]["doc_id"], "e")

    def test_verification_adapters_disable_repair(self):
        from src.experiments.adapters import execute
        from src.experiments.runner import DEFAULTS
        from src.schema import AnswerRecord
        options = {**DEFAULTS, "index_path": "unused"}
        record = AnswerRecord(id="r", source_dataset="fixture", question="Q?", answer="A.")
        with patch("src.pipeline.run_pipeline", return_value=record) as pipeline:
            execute({"id": "pipeline", "retrieval": "bm25"}, record, "claim_verification", options)
            self.assertEqual(pipeline.call_args.kwargs["max_repair_iterations"], 0)
            self.assertTrue(pipeline.call_args.kwargs["use_existing_claims"])
        with patch("src.baselines.run_b7", return_value=record) as baseline:
            execute({"id": "b7", "retrieval": "bm25"}, record, "claim_verification", options)
            self.assertFalse(baseline.call_args.kwargs["revise"])

    def test_retrieval_errors_are_not_converted_to_empty_evidence(self):
        from src.baselines import run_b5, run_b7
        from src.schema import AnswerRecord, Claim
        record = AnswerRecord(id="r", source_dataset="fixture", question="Q?", answer="A.",
                              claims=[Claim(id="c", text="A fact.", sentence_id=0)])
        for target, function in (("src.baselines.b5_rarr.retrieve_bm25", run_b5),
                                 ("src.baselines.b7_factcheck_gpt.retrieve_bm25", run_b7)):
            with self.subTest(target=target), patch(target, side_effect=OSError("broken index")):
                with execution(), self.assertRaises(OSError):
                    function(record, index_path="broken")


if __name__ == "__main__":
    unittest.main()
