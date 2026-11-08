"""Human gold files must be complete before they drive a reported comparison."""

import json

import pytest

from scripts.validate_gold import build_holdout, load_gold, main, validate_gold_set


@pytest.fixture(autouse=True)
def deterministic_sentence_splitter(monkeypatch):
    # Avoid a spaCy model download during structural validation tests.
    monkeypatch.setattr("src.decompose._get_nlp", lambda: None)


def record(record_id, split, question, *, label="Supported", title="Paris"):
    answer = f"{record_id} is in France."
    return {
        "id": record_id, "source_dataset": "gold", "split": split,
        "question": question, "answer": answer,
        "gold_repaired_answer": answer,
        "claims": [{
            "id": f"{record_id}_c1", "text": answer, "sentence_id": 0,
            "span": [0, len(answer)], "check_worthy": True, "label": label,
            "evidence": ([{"doc_id": f"{title}_0", "title": title,
                           "text": f"{title} is in France."}] if title else []),
        }],
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_paths(tmp_path):
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    splits_path = tmp_path / "splits.json"
    dev = record("dev-1", "dev", "Where is the dev city?")
    test = record("test-1", "test", "Where is the test city?", title="Lyon")
    write_jsonl(dev_path, [dev])
    write_jsonl(test_path, [test])
    splits_path.write_text(json.dumps({"dev_ids": ["dev-1"], "test_ids": ["test-1"]}),
                           encoding="utf-8")
    return dev_path, test_path, splits_path, dev, test


def test_valid_gold_and_test_only_holdout(tmp_path):
    dev_path, test_path, splits_path, _, _ = fixture_paths(tmp_path)
    dev, test, report = validate_gold_set(dev_path, test_path, splits_path,
                                          expected_dev=1, expected_test=1)
    assert len(dev) == len(test) == 1
    assert report["check_worthy_labels"] == {"Supported": 2}
    holdout = build_holdout(test, {}, report["gold_test_sha256"])
    assert holdout["article_ids"] == ["Lyon"]
    assert holdout["prompt_ids"] == ["test-1"]
    assert "dev-1" not in holdout["prompt_ids"]


def test_cli_writes_holdout_after_validation(tmp_path, capsys):
    dev_path, test_path, splits_path, _, _ = fixture_paths(tmp_path)
    output = tmp_path / "holdout.json"
    status = main(["--gold-dev", str(dev_path), "--gold-test", str(test_path),
                   "--splits", str(splits_path), "--holdout-out", str(output)])
    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["article_ids"] == ["Lyon"]
    assert "valid_structure_only" in capsys.readouterr().out
    assert main(["--gold-dev", str(dev_path), "--gold-test", str(test_path),
                 "--splits", str(splits_path), "--holdout-out", str(output)]) == 1


@pytest.mark.parametrize("change,match", [
    (lambda row: row["claims"][0].update(label=None), "needs a verdict"),
    (lambda row: row["claims"][0].update(evidence=[]), "needs cited evidence"),
    (lambda row: row["claims"][0].update(span=[0, 999]), "span must lie"),
    (lambda row: row["claims"][0].update(sentence_id=2), "sentence_id"),
    (lambda row: row.update(gold_repaired_answer=""), "remove supported claims"),
])
def test_rejects_incomplete_gold_claim(tmp_path, change, match):
    path = tmp_path / "dev.jsonl"
    row = record("dev-1", "dev", "Where is the city?")
    change(row)
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match=match):
        load_gold(path, "dev")


def test_requires_exact_split_ids(tmp_path):
    dev_path, test_path, splits_path, _, _ = fixture_paths(tmp_path)
    splits_path.write_text(json.dumps({"dev_ids": ["dev-1", "dev-2"],
                                       "test_ids": ["test-1"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        validate_gold_set(dev_path, test_path, splits_path)


def test_rejects_prompt_overlap_across_splits(tmp_path):
    dev_path, test_path, splits_path, dev, test = fixture_paths(tmp_path)
    test["question"] = "  WHERE is THE dev city?  "
    write_jsonl(test_path, [test])
    with pytest.raises(ValueError, match="share a question"):
        validate_gold_set(dev_path, test_path, splits_path)


def test_holdout_needs_article_or_explicit_entity_for_every_test_record(tmp_path):
    dev_path, test_path, splits_path, _, test = fixture_paths(tmp_path)
    test["claims"][0]["label"] = "Unverifiable"
    test["claims"][0]["evidence"] = []
    write_jsonl(test_path, [test])
    _, test_rows, report = validate_gold_set(dev_path, test_path, splits_path)
    with pytest.raises(ValueError, match="no cited article or mapped entity"):
        build_holdout(test_rows, {}, report["gold_test_sha256"])
    holdout = build_holdout(test_rows, {"test-1": ["Lyon"]},
                            report["gold_test_sha256"])
    assert holdout["entities"] == ["Lyon"]


def test_holdout_checks_each_check_worthy_claim(tmp_path):
    dev_path, test_path, splits_path, _, test = fixture_paths(tmp_path)
    test["claims"].append({
        "id": "test-1_c2", "text": "An unresolved second fact.",
        "sentence_id": 0, "check_worthy": True,
        "label": "Unverifiable", "evidence": [],
    })
    write_jsonl(test_path, [test])
    _, test_rows, report = validate_gold_set(dev_path, test_path, splits_path)
    with pytest.raises(ValueError, match="test-1/test-1_c2"):
        build_holdout(test_rows, {}, report["gold_test_sha256"])
