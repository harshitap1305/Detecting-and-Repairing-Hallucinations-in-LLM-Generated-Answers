"""Benchmark conversion must keep the hallucinated response as the input."""

import json
import sys
import types

import pytest

from scripts.download_data import load_halueval, save_splits


def _load(monkeypatch, tmp_path, rows):
    calls = []

    def fake_load(name, config, **kwargs):
        calls.append((name, config, kwargs["split"]))
        return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load))
    records = load_halueval(tmp_path, seed=42)
    assert calls == [("pminervini/HaluEval", "qa_samples", "data")]
    saved = [json.loads(line) for line in
             (tmp_path / "halueval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved == [record.to_dict() for record in records]
    return records


def test_qa_samples_maps_hallucination_to_input_and_groups_prompts(monkeypatch, tmp_path):
    rows = [
        {"question": "Where is Lyon?", "answer": "Lyon is in France.",
         "hallucination": "Lyon is in Germany."},
        {"question": " WHERE   is lyon? ", "answer": "Lyon is in France.",
         "hallucination": "Lyon is in Spain."},
    ]
    records = _load(monkeypatch, tmp_path, rows)
    assert {record.answer for record in records} == {
        "Lyon is in Germany.", "Lyon is in Spain."
    }
    assert {record.gold_repaired_answer for record in records} == {"Lyon is in France."}
    assert len({record.split for record in records}) == 1


def test_qa_schema_maps_explicit_answer_names(monkeypatch, tmp_path):
    rows = [{"question": "Who wrote Hamlet?", "right_answer": "Shakespeare.",
             "hallucinated_answer": "Dickens."}]
    records = _load(monkeypatch, tmp_path, rows)
    assert records[0].answer == "Dickens."
    assert records[0].gold_repaired_answer == "Shakespeare."


@pytest.mark.parametrize("rows,match", [
    ([{"question": "Q", "answer": "right"}], "Unsupported HaluEval"),
    ([{"question": "Q", "answer": "right", "hallucination": ""}], "Incomplete HaluEval"),
    ([], "dataset is empty"),
])
def test_bad_qa_rows_fail_closed(monkeypatch, tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        _load(monkeypatch, tmp_path, rows)
    assert not (tmp_path / "halueval.jsonl").exists()


def test_split_manifest_records_actual_seed(monkeypatch, tmp_path):
    records = _load(monkeypatch, tmp_path, [
        {"question": "A question?", "answer": "right", "hallucination": "wrong"}
    ])
    manifest = tmp_path / "splits.json"
    save_splits(records, manifest, seed=17)
    assert json.loads(manifest.read_text(encoding="utf-8"))["seed"] == 17
