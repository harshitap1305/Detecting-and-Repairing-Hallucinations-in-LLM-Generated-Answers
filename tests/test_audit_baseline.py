"""The evidence audit must reject missing or mismatched experiment artifacts."""

import json

from scripts.audit_baseline import audit


def write_jsonl(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": record_id}) + "\n" for record_id in ids),
        encoding="utf-8",
    )


def complete_fixture(root):
    (root / "data").mkdir()
    (root / "data/splits.json").write_text(
        json.dumps({"dev_ids": ["dev-1"], "test_ids": ["test-1"]}),
        encoding="utf-8",
    )
    write_jsonl(root / "data/gold/dev.jsonl", ["dev-1"])
    write_jsonl(root / "data/gold/test.jsonl", ["test-1"])
    write_jsonl(root / "results/pipeline_output.jsonl", ["test-1"])
    for i in range(8):
        write_jsonl(root / f"results/baseline_b{i}.jsonl", ["test-1"])


def test_missing_artifacts_fail_closed(tmp_path):
    lines, baseline_present, checkpoint_present = audit(tmp_path)
    assert not baseline_present
    assert not checkpoint_present
    assert any("splits.json: missing" in line for line in lines)


def test_matching_artifacts_satisfy_minimum_check(tmp_path):
    complete_fixture(tmp_path)
    (tmp_path / "data/splits.json").write_text(
        json.dumps({"dev_ids": ["dev-1", "dev-2"], "test_ids": ["test-1", "test-2"]}),
        encoding="utf-8",
    )
    lines, baseline_present, checkpoint_present = audit(tmp_path)
    assert baseline_present
    assert not checkpoint_present
    assert any("baseline_b7.jsonl: 1 records" in line for line in lines)


def test_mismatched_output_ids_fail_closed(tmp_path):
    complete_fixture(tmp_path)
    write_jsonl(tmp_path / "results/baseline_b5.jsonl", ["other-test"])
    lines, baseline_present, _ = audit(tmp_path)
    assert not baseline_present
    assert any("baseline_b5.jsonl: 1 records; IDs differ" in line for line in lines)
