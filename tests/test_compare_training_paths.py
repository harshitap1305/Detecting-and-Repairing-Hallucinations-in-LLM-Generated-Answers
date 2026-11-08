"""The paired comparison must reject incomplete data and partial pipeline runs."""

import json
import sqlite3
from dataclasses import replace

import pytest

from scripts import compare_training_paths as comparison
from scripts.build_fever_fts import build_index
from src.schema import AnswerRecord, Claim


def record(record_id, split, *, gold=False):
    return AnswerRecord(
        id=record_id, source_dataset="gold", split=split,
        question=f"Question for {record_id}?", answer=f"Answer for {record_id}.",
        claims=[Claim(id=f"human-annotated:{record_id}", text="A factual claim",
                      sentence_id=0, label="Supported")] if gold else [],
        gold_repaired_answer=f"Answer for {record_id}." if gold else None,
    )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.to_jsonl_line() + "\n" for item in records),
                    encoding="utf-8")


def fixture_config(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    gold_dev = tmp_path / "gold_dev.jsonl"
    gold_test = tmp_path / "gold_test.jsonl"
    write_jsonl(inputs, [record("test-1", "test")])
    write_jsonl(gold_dev, [record("dev-1", "dev", gold=True)])
    write_jsonl(gold_test, [record("test-1", "test", gold=True)])
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"dev_ids": ["dev-1"], "test_ids": ["test-1"]}),
                      encoding="utf-8")
    for name in ("verifier", "repair"):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"dummy model weights")
    index = tmp_path / "index.pkl"
    index.write_bytes(b"dummy index")
    return comparison.Config(
        inputs=inputs, gold_dev=gold_dev, gold_test=gold_test, splits=splits,
        verifier_checkpoint=tmp_path / "verifier",
        repair_checkpoint=tmp_path / "repair", index=index,
        output_dir=tmp_path / "comparison", repo_root=tmp_path, n_bootstrap=10,
    )


def mock_pipeline_result(source):
    assert len(source.claims) == 1
    assert source.claims[0].id == f"human-annotated:{source.id}"
    assert source.claims[0].label is None
    assert source.claims[0].evidence == []
    source.claims[0].label = "Supported"
    source.repaired_answer = source.answer
    source.reliability_score_before = 1.0
    source.reliability_score_after = 1.0
    source.run_id = "mock-run"
    return source


def test_comparison_writes_only_complete_paired_outputs(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    modes = []

    def fake_pipeline(source, **kwargs):
        assert kwargs["use_existing_claims"] is True
        modes.append((kwargs["verifier"], kwargs["repair_backend"]))
        return mock_pipeline_result(source)

    monkeypatch.setattr(comparison, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(comparison, "seed_everything", lambda seed: None)
    monkeypatch.setattr(comparison, "git_info", lambda root: {
        "commit": "test-commit", "working_tree_dirty": False
    })
    monkeypatch.setattr(comparison, "package_versions", lambda: {})
    monkeypatch.setattr(comparison, "evaluate_dataset", lambda records, **kwargs: {
        "n_records": len(records), "detection_vs_gold": {"accuracy": 1.0}
    })

    output_dir = comparison.run_comparison(config)
    assert modes == list(comparison.VARIANTS.values())
    assert {path.name for path in output_dir.iterdir()} == {
        *(f"{name}.jsonl" for name in comparison.VARIANTS),
        "metrics.json", "manifest.json",
    }
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))[
        "test_record_ids"
    ] == ["test-1"]
    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["prompted_baseline"]["repair_vs_gold"]["exact_match"]["mean"] == 1.0
    assert report["trained_both"]["paired_delta_vs_prompted"]["rouge_l_f1"]["mean"] == 0.0


def test_duplicate_held_out_id_rejected_before_any_run(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    write_jsonl(config.inputs, [record("test-1", "test"), record("test-1", "test")])
    monkeypatch.setattr(comparison, "run_pipeline", lambda *args, **kwargs: pytest.fail(
        "pipeline must not run with duplicate IDs"
    ))
    with pytest.raises(ValueError, match="Duplicate record ID"):
        comparison.run_comparison(config)
    assert not config.output_dir.exists()


def test_pipeline_failure_leaves_no_results_or_metrics(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    metrics_calls = []

    def incomplete_pipeline(source, **kwargs):
        result = mock_pipeline_result(source)
        if kwargs["verifier"] == "trained":
            result.repaired_answer = None
        return result

    monkeypatch.setattr(comparison, "run_pipeline", incomplete_pipeline)
    monkeypatch.setattr(comparison, "seed_everything", lambda seed: None)
    monkeypatch.setattr(comparison, "git_info", lambda root: {
        "commit": "test-commit", "working_tree_dirty": False
    })
    monkeypatch.setattr(comparison, "evaluate_dataset", lambda *args, **kwargs: metrics_calls.append(1))
    with pytest.raises(RuntimeError, match="trained_verifier failed"):
        comparison.run_comparison(config)
    assert not metrics_calls
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".ac-compare-*"))


def test_gold_claim_sentence_id_must_refer_to_answer(tmp_path):
    config = fixture_config(tmp_path)
    bad = record("test-1", "test", gold=True)
    bad.claims[0].sentence_id = 2
    write_jsonl(config.gold_test, [bad])
    with pytest.raises(ValueError, match="sentence_id outside answer"):
        comparison.validate_inputs(config)


def test_empty_repair_is_valid_for_delete_policy(tmp_path):
    config = fixture_config(tmp_path)
    source = record("test-1", "test")
    gold = record("test-1", "test", gold=True)
    gold.gold_repaired_answer = ""
    comparison.validate_gold([gold], config.gold_test)
    result = mock_pipeline_result(
        comparison.AnswerRecord(
            id=source.id, source_dataset=source.source_dataset,
            split=source.split, question=source.question, answer=source.answer,
            claims=[Claim(id=gold.claims[0].id, text=gold.claims[0].text,
                          sentence_id=gold.claims[0].sentence_id)],
        )
    )
    result.repaired_answer = ""
    comparison.validate_output(result, source, gold)
    metrics = comparison.reference_repair_metrics([result], {source.id: gold}, 10, 42)
    assert metrics["exact_match"]["mean"] == 1.0


def test_fts5_comparison_accepts_a_valid_shared_index(tmp_path):
    config = fixture_config(tmp_path)
    index = tmp_path / "index.sqlite"
    build_index(iter([{"doc_id": "doc_0", "title": "Doc", "text": "A fact",
                       "chunk_id": 0}]), index, source="fixture")
    comparison.validate_inputs(replace(config, index=index, retriever_method="fts5"))
    with pytest.raises(sqlite3.DatabaseError):
        comparison.validate_inputs(replace(config, retriever_method="fts5"))
