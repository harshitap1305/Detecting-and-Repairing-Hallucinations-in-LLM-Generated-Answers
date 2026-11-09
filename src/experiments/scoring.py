"""Offline validation and scoring of complete runner artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from src.schema import AnswerRecord
from src.evaluate import (compute_rouge_l, evaluate_claim_detection, unique_records,
                          compute_detection_metrics, align_claim_records)
from src.experiments.artifacts import IDENTITY, canonical, file_hash, fingerprint, validate_result
from src.experiments.statistics import clustered_interval


def jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            canonical(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def load_gold(path):
    records = [AnswerRecord.from_dict(row) for row in jsonl(path)]
    result = unique_records(records)
    seen = set()
    for record in records:
        if any(not isinstance(getattr(record, key), str) or not getattr(record, key).strip() for key in IDENTITY):
            raise ValueError("Invalid gold input identity")
        if record.split not in ("dev", "test") or not isinstance(record.gold_repaired_answer, str):
            raise ValueError("Gold requires declared split and human reference repair")
        if not record.claims or not record.check_worthy_claims:
            raise ValueError("Gold must contain reviewed check-worthy claims")
        if not record.gold_repaired_answer.strip() and any(claim.label == "Supported" for claim in record.check_worthy_claims):
            raise ValueError("Empty reference would remove supported gold claims")
        for claim in record.claims:
            if (not isinstance(claim.id, str) or not claim.id.strip() or claim.id in seen
                    or not isinstance(claim.text, str) or not claim.text.strip()
                    or type(claim.check_worthy) is not bool or type(claim.sentence_id) is not int
                    or claim.sentence_id < 0):
                raise ValueError("Malformed/duplicate gold claim")
            seen.add(claim.id)
            if claim.span is not None and (not isinstance(claim.span, list) or len(claim.span) != 2
                    or any(type(offset) is not int for offset in claim.span)
                    or not 0 <= claim.span[0] < claim.span[1] <= len(record.answer)):
                raise ValueError("Gold span outside original answer")
            if claim.check_worthy and claim.label not in ("Supported", "Contradicted", "Unverifiable"):
                raise ValueError("Unlabeled gold check-worthy claim")
            if claim.check_worthy and claim.label in ("Supported", "Contradicted") and not claim.evidence:
                raise ValueError("Decisive gold verdict needs human-inspected evidence")
            for passage in claim.evidence:
                if any(not isinstance(getattr(passage, key), str) or not getattr(passage, key).strip()
                       for key in ("doc_id", "title", "text")):
                    raise ValueError("Malformed gold evidence")
    return result


def load_run(directory: Path, gold: dict) -> dict:
    """Verify saved outputs against immutable attempts without loading any model."""
    from src.experiments.runner import _read_attempts
    from src.experiments.protocol import validate_config
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if manifest.get("schema_version") != 1 or identity.get("schema_version") != 1:
        raise ValueError("Unsupported run schema")
    validate_config(identity["config"])
    if manifest.get("run_id") != fingerprint(identity) or manifest.get("status") != "execution_complete":
        raise ValueError("Evaluation requires a complete, consistent execution manifest")
    task = identity["task"]
    if task not in ("repair", "claim_verification", "answer_detection"):
        raise ValueError("Unknown run task")
    ids = identity["input_ids"]
    if len(set(ids)) != len(ids) or set(ids) != set(gold):
        raise ValueError("Run and gold IDs must match exactly")
    if any(record.split != identity["split"] for record in gold.values()):
        raise ValueError("Run and gold split mismatch")
    systems = {system["id"]: system for system in identity["config"]["systems"] if task in system["tasks"]}
    expected = len(gold) * len(systems)
    if not systems or manifest.get("successful_outputs") != expected or manifest.get("expected_outputs") != expected:
        raise ValueError("Incomplete declared system coverage")
    inventory = {key: record.claims for key, record in gold.items()} if task == "claim_verification" else {}
    attempts, _ = _read_attempts(directory, gold, systems, task, inventory)
    if len(attempts) != expected:
        raise ValueError("Missing committed successful attempts")
    outputs = {}
    declared = {f"{system}.jsonl" for system in systems}
    if {path.name for path in (directory / "outputs").glob("*.jsonl")} != declared:
        raise ValueError("Missing or extra system output files")
    for system in systems:
        rows = jsonl(directory / "outputs" / f"{system}.jsonl")
        mapped = {}
        for row in rows:
            record_id = row["input"]["id"]
            if record_id not in gold or record_id in mapped or row["status"] != "success":
                raise ValueError("Duplicate, unknown, or failed output record")
            validate_result(row, gold[record_id], system, task, inventory.get(record_id))
            if canonical(row) != canonical(attempts[(system, record_id)]):
                raise ValueError("Aggregate output differs from committed attempt")
            mapped[record_id] = row
        if set(mapped) != set(gold):
            raise ValueError("Output IDs must exactly match gold")
        outputs[system] = mapped
    return {"manifest": manifest, "outputs": outputs,
            "manifest_sha256": file_hash(directory / "manifest.json")}


def score_run(run, gold, n_bootstrap=1000, seed=42):
    task = run["manifest"]["identity"]["task"]
    ids = sorted(gold)
    results = {}
    for system, outputs in run["outputs"].items():
        if task == "claim_verification":
            results[system] = {"detection": evaluate_claim_detection(
                [AnswerRecord.from_dict(outputs[key]["record"]) for key in ids],
                [gold[key] for key in ids], n_bootstrap, seed)}
        elif task == "answer_detection":
            pairs = [("Contradicted" if outputs[key]["native"]["answer_detection"]["has_hallucination"]
                      else "Supported", "Contradicted" if gold[key].hallucinated_claims else "Supported")
                     for key in ids]
            def metric(sample):
                return compute_detection_metrics([p for p, g in sample], [g for p, g in sample])["binary"]
            results[system] = {"answer_detection": metric(pairs),
                               "coverage": {"records": len(ids), "fraction": 1.0},
                               "f1_interval": clustered_interval(pairs, lambda sample: metric(sample)["f1"],
                                                                 n_bootstrap, seed)}
        else:
            per_record = {}
            for key in ids:
                answer = outputs[key]["record"]["repaired_answer"]
                reference = gold[key].gold_repaired_answer
                per_record[key] = {"exact_match": float(answer.strip() == reference.strip()),
                                   "rouge_l_f1": compute_rouge_l(answer, reference)["f1"]}
            results[system] = {
                "reference_agreement": {name: clustered_interval([per_record[key][name] for key in ids],
                                          lambda sample: sum(sample) / len(sample), n_bootstrap, seed)
                                        for name in ("exact_match", "rouge_l_f1")},
                "per_record_reference": per_record,
                "independent_repair": {"status": "unavailable", "reason": "Human assessments not supplied"},
                "claim_detection": {"status": "unavailable", "reason": "End-to-end inventories are not aligned"},
                "diagnostics": {key: outputs[key]["native"] for key in ids},
            }
    comparisons = {}
    if task == "claim_verification" and "pipeline" in run["outputs"]:
        from src.experiments.statistics import paired_family
        baseline = [AnswerRecord.from_dict(run["outputs"]["pipeline"][key]["record"]) for key in ids]
        gold_records = [gold[key] for key in ids]
        secondary = {}
        def macro(groups):
            pairs = [pair for group in groups for pair in group]
            return compute_detection_metrics([p for p, g in pairs], [g for p, g in pairs])["three_way"]["macro_f1"]
        baseline_groups = align_claim_records(baseline, gold_records)
        for system in run["outputs"]:
            if system == "pipeline":
                continue
            groups = align_claim_records([AnswerRecord.from_dict(run["outputs"][system][key]["record"])
                                          for key in ids], gold_records)
            comparisons[system + "_vs_pipeline"] = clustered_interval(list(range(len(ids))),
                lambda sample: macro([groups[i] for i in sample]) - macro([baseline_groups[i] for i in sample]),
                n_bootstrap, seed)
            secondary[system + "_vs_pipeline"] = (
                {rid: macro([groups[i]]) for i, rid in enumerate(ids)},
                {rid: macro([baseline_groups[i]]) for i, rid in enumerate(ids)})
        comparisons = {"pooled_macro_f1_differences": comparisons,
                       "secondary_per_answer_macro_f1_tests": paired_family(secondary, n_bootstrap, seed)}
    return {"comparisons": comparisons, "schema_version": 1, "run_id": run["manifest"]["run_id"], "task": task,
            "systems": results, "n_records": len(ids), "record_ids": ids,
            "interpretation": "Reference similarity and native diagnostics are not independent factual judgments"}
