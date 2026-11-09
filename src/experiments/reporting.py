"""Deterministic task reports and portable, externally-gold-bound result bundles."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from src.experiments.artifacts import canonical, file_hash, fingerprint
from src.experiments.assessment import RUBRIC_VERSION, validate_assessments
from src.experiments.scoring import load_gold, load_run, evaluate_run

SCHEMA = 1
REQUIRED = {"manifest.json", "protocol.json", "inputs_manifest.json", "metrics.json",
            "comparisons.json", "report.md", "errors.jsonl", "checksums.json"}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def _summary(gold):
    return {"records": len(gold), "source_counts": dict(sorted(Counter(record.source_dataset for record in gold.values()).items())),
            "label_counts": dict(sorted(Counter(claim.label for record in gold.values()
                                                for claim in record.check_worthy_claims).items())),
            "claim_count": sum(len(record.check_worthy_claims) for record in gold.values())}


def _products(run, gold, assessments, mode, n_bootstrap, seed):
    if mode not in ("final", "progress") or type(n_bootstrap) is not int or n_bootstrap < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid report mode or resampling settings")
    complete = run["manifest"]["status"] == "execution_complete"
    task = run["manifest"]["identity"]["task"]
    if mode == "final" and not complete:
        raise ValueError("Final reports require complete execution")
    if mode == "final" and task == "repair":
        if assessments is None or assessments.get("adjudicated") is not True:
            raise ValueError("Final repair reports require adjudicated independent human assessments")
    if assessments is not None:
        if task != "repair" or not complete:
            raise ValueError("Assessments require a complete repair run")
        validate_assessments(assessments, run, gold)
    coverage = {}
    for system, rows in sorted(run["outputs"].items()):
        coverage[system] = {"expected": len(gold), "successful": len(rows),
                            "missing_ids": sorted(set(gold) - set(rows))}
    if mode == "progress":
        # Never calculate partial comparisons or promote a reference score to a
        # factual evaluation. All progress reports show coverage only.
        metrics = {"schema_version": SCHEMA, "run_id": run["manifest"]["run_id"], "task": task,
                   "status": "progress", "coverage": coverage,
                   "comparative_metrics": {"status": "unavailable", "reason": "Progress report; no final performance claims"}}
        comparisons = {"status": "unavailable", "reason": "Progress report"}
    else:
        metrics = evaluate_run(run, gold, assessments, n_bootstrap, seed)
        comparisons = {"detection": metrics.get("comparisons", {}),
                       "repair": metrics.get("human_repair", {}).get("paired_repair_family", {})}
    evaluations = {}
    for system, rows in sorted(run["outputs"].items()):
        human_rows = {row["record_id"]: row for row in assessments["assessments"] if row["system"] == system} if assessments else {}
        evaluations[system] = [{"record_id": rid, "output_sha256": fingerprint(row),
                                "human_assessment": human_rows.get(rid),
                                "reference_agreement": metrics.get("systems", {}).get(system, {}).get("per_record_reference", {}).get(rid)}
                               for rid, row in sorted(rows.items())]
        for item in evaluations[system]:
            rid = item["record_id"]
            if task == "claim_verification":
                predicted = {claim["id"]: claim for claim in rows[rid]["record"]["claims"]}
                item["claim_verdicts"] = [{"claim_id": claim.id, "prediction": predicted[claim.id]["label"],
                                            "gold": claim.label} for claim in sorted(gold[rid].check_worthy_claims, key=lambda c: c.id)]
            elif task == "answer_detection":
                item["answer_detection"] = {"prediction": rows[rid]["native"]["answer_detection"]["has_hallucination"],
                                            "gold": bool(gold[rid].hallucinated_claims)}
    return metrics, comparisons, evaluations, coverage


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _ci(metric):
    if not metric or metric.get("estimate") is None:
        return "N/A"
    value = f"{metric['estimate']:.4f}"
    if metric.get("ci_lower") is not None:
        value += f" [{metric['ci_lower']:.4f}, {metric['ci_upper']:.4f}]"
    return value


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows] + [""]


def render_report(manifest, metrics, comparisons):
    scope = manifest["scope"]
    progress = manifest["mode"] == "progress"
    lines = ["# " + ("Progress report — no final performance claims" if progress else "Generated task evaluation report"), "",
             f"Task: **{_cell(scope['task'])}**. Split: **{_cell(scope['split'])}**. "
             f"Gold records in this split: **{scope['gold_summary']['records']}**.", "",
             f"Execution run: `{scope['run_id']}`.", "",
             "This report covers only the declared systems for this task and split. "
             "It does not certify completion of the full project benchmark.", "",
             "## Data and coverage", ""]
    lines += _table(["Source", "Answers"], sorted(scope["gold_summary"]["source_counts"].items()))
    lines += _table(["Gold verdict", "Check-worthy claims"], sorted(scope["gold_summary"]["label_counts"].items()))
    lines += _table(["System", "Successful outputs", "Expected outputs", "Missing outputs"],
                    [(system, item["successful"], item["expected"], len(item["missing_ids"]))
                     for system, item in sorted(manifest["coverage"].items())])
    lines += [f"Recorded failed attempts: **{manifest['failure_count']}**. "
              "Failures can precede successful retries; they are retained in errors.jsonl.", ""]
    if not progress:
        task, systems = scope["task"], metrics["systems"]
        if task == "claim_verification":
            lines += ["## Fixed-inventory detection", "", "Human claim inventories are oracle inputs. "
                      "Intervals resample answers and retain all their claims; macro-F1 includes all three classes.", ""]
            lines += _table(["System", "Three-way macro-F1 [95% CI]", "Binary F1 [95% CI]", "Three-way accuracy"],
                [(name, _ci(value["detection"]["intervals"]["three_way_macro_f1"]),
                  _ci(value["detection"]["intervals"]["binary_f1"]), f"{value['detection']['three_way']['accuracy']:.4f}")
                 for name, value in sorted(systems.items())])
            lines += _table(["System", "Class", "Precision", "Recall", "F1", "Support"],
                [(name, label, f"{values['precision']:.4f}", f"{values['recall']:.4f}",
                  f"{values['f1']:.4f}", values["support"])
                 for name, value in sorted(systems.items())
                 for label, values in sorted(value["detection"]["three_way"]["per_class"].items())])
            for name, value in sorted(systems.items()):
                matrix = value["detection"]["three_way"]["confusion_matrix"]
                lines += [f"### {_cell(name)} confusion matrix", "", "Rows are gold; columns are predicted.", ""]
                lines += _table(["Gold / predicted"] + matrix["labels"],
                                [(label, *row) for label, row in zip(matrix["labels"], matrix["rows_gold_columns_pred"])])
            detection = comparisons["detection"]
            lines += ["## Paired detection comparisons", "", "Primary effect: difference in pooled claim macro-F1.", ""]
            deltas = detection.get("pooled_macro_f1_differences", {})
            lines += _table(["Comparison", "Macro-F1 difference [95% CI]"], [(name, _ci(value)) for name, value in sorted(deltas.items())]) if deltas else ["Unavailable: no configured comparator against the prompted pipeline.", ""]
            family = detection.get("secondary_per_answer_macro_f1_tests", {})
            lines += ["Secondary signed-rank tests target per-answer macro-F1, a different estimand.", ""]
            lines += _family_table(family)
        elif task == "answer_detection":
            lines += ["## Answer-level detection", "", "Positive gold: any original check-worthy Contradicted or Unverifiable claim.", ""]
            lines += _table(["System", "Precision", "Recall", "F1 [95% CI]", "Accuracy"],
                [(name, f"{value['answer_detection']['precision']:.4f}", f"{value['answer_detection']['recall']:.4f}",
                  _ci(value["f1_interval"]), f"{value['answer_detection']['accuracy']:.4f}") for name, value in sorted(systems.items())])
        else:
            lines += ["## Independent human repair outcomes", "", "Adjudicated reviews; rates are means over gold-eligible answers. "
                      "Deletion and hedging do not count as correction. Zero-eligible answers are inapplicable, not perfect.", ""]
            human = metrics["human_repair"]
            agreement = manifest["review_agreement"]
            kappa = f"{agreement['cohens_kappa']:.4f}" if agreement["cohens_kappa"] is not None else "N/A (constant marginals)"
            lines += [f"Independent outcome agreement: {agreement['outcome_agreement']:.4f}; "
                      f"Cohen's kappa: {kappa}; reviewed claim ratings: {agreement['n_claim_ratings']}; "
                      f"records requiring adjudication: {len(agreement['disagreements'])}.", ""]
            lines += _table(["System", "Correction [95% CI]", "Eligible answers", "Preservation [95% CI]", "Eligible answers", "New errors [95% CI]", "Completeness [95% CI]"],
                [(name, _ci(value["metrics"]["correction_rate"]), value["metrics"]["correction_rate"]["eligible_answers"],
                  _ci(value["metrics"]["preservation_rate"]), value["metrics"]["preservation_rate"]["eligible_answers"],
                  _ci(value["metrics"]["new_error_count"]), _ci(value["metrics"]["completeness"])) for name, value in sorted(human["systems"].items())])
            outcomes = ("retained", "corrected", "deleted", "hedged", "still_incorrect")
            lines += _table(["System"] + list(outcomes),
                [(name, *(value["outcome_counts"].get(outcome, 0) for outcome in outcomes))
                 for name, value in sorted(human["systems"].items())])
            lines += ["Completeness uses 0=no useful answer, 1=partial, 2=complete.", "",
                      "## Paired repair comparisons", "", "B0 correction/preservation comparisons share one Holm family.", ""]
            lines += _family_table(comparisons["repair"])
            lines += ["## Reference editing agreement", "", "Exact match and ROUGE-L compare with human reference wording; "
                      "they do not establish factual correctness. Empty deletions remain empty.", ""]
            lines += _table(["System", "Exact match [95% CI]", "ROUGE-L [95% CI]"],
                [(name, _ci(value["reference_agreement"]["exact_match"]), _ci(value["reference_agreement"]["rouge_l_f1"]))
                 for name, value in sorted(systems.items())])
    lines += ["## Provenance and interpretation", "",
              f"Corpus snapshot: {_cell(scope['corpus']['snapshot'])}. "
              f"Generator: {_cell(scope['model']['provider'])} / {_cell(scope['model']['model'])}.", "",
              f"Bootstrap: {manifest['scoring']['n_bootstrap']} resamples, seed {manifest['scoring']['seed']}, "
              "95% percentile intervals, answer as the sampling unit.", "",
              "Self-verification SCR, consistency scores, and token overlap are diagnostics, not independent factual judgments. "
              "Checksums establish artifact integrity, not annotation correctness or reviewer independence.", "",
              "Only the selected split is measured here; configured dev/test targets are not combined into a test count. "
              "Unpinned upstream/hosted model behavior may change on new inference runs.", "",
              "This bundle contains private source text, evidence, and traces. Share the aggregate report only after reviewing dataset terms. "
              "The gold annotations remain external; use the exact hashed gold file for offline recomputation.", ""]
    return "\n".join(lines)


def _family_table(family):
    values = family.get("comparisons", {})
    if not values:
        return ["Unavailable or inapplicable: no eligible declared paired comparisons.", ""]
    return _table(["Comparison", "Mean paired difference [95% CI]", "Pairs", "Raw p", "Holm p", "Method"],
        [(name, _ci(value["paired_difference"]), value["n_pairs"], f"{value['p_value']:.6g}",
          f"{value['p_value_holm']:.6g}", value["method"]) for name, value in sorted(values.items())])


def _files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Bundle artifacts must not be symlinks")
        if path.is_file() and path.relative_to(directory).as_posix() != "checksums.json":
            result[path.relative_to(directory).as_posix()] = file_hash(path)
    return result


def build_bundle(*, run_dir: Path, gold_path: Path, output_dir: Path, assessments_path: Path | None = None,
                 mode="final", n_bootstrap=1000, seed=42):
    from src.experiments.runner import run_lock, code_identity, environment, ROOT
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    if output_dir == run_dir or output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir):
        raise ValueError("Bundle must be separate from the source execution directory")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    gold = load_gold(gold_path)
    assessments = _read(assessments_path) if assessments_path else None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".report-", dir=output_dir.parent))
    try:
        with run_lock(run_dir):
            run = load_run(run_dir, gold, allow_partial=mode == "progress")
            metrics, comparisons, evaluations, coverage = _products(run, gold, assessments, mode, n_bootstrap, seed)
            snapshot = staging / "execution"
            snapshot.mkdir()
            for name in ("manifest.json",):
                shutil.copyfile(run_dir / name, snapshot / name)
            for name in ("attempts", "traces", "outputs"):
                # Reject links rather than following a source artifact outside the run.
                if any(path.is_symlink() for path in (run_dir / name).rglob("*")) or (run_dir / name).is_symlink():
                    raise ValueError("Execution snapshot must not contain symlinks")
                shutil.copytree(run_dir / name, snapshot / name)
            # Derive retained errors from authoritative attempts, not a potentially stale view.
            errors = [wrapper["result"] for path in sorted((snapshot / "attempts").glob("*/*.json"))
                      for wrapper in [_read(path)] if wrapper["result"]["status"] == "error"]
            (staging / "errors.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in errors))
        identity = run["manifest"]["identity"]
        protocol = {"schema_version": SCHEMA, "config": identity["config"], "rubric_version": RUBRIC_VERSION,
                    "reporting_protocol_markdown": (ROOT / "docs/evaluation_protocol.md").read_text(encoding="utf-8")}
        inputs = {"schema_version": SCHEMA, "gold_sha256": file_hash(gold_path),
                  "gold_content_sha256": fingerprint({key: record.to_dict() for key, record in sorted(gold.items())}),
                  "gold_access": "external; supply --gold to audit/recompute", "record_ids": sorted(gold),
                  "generation_input_sha256": identity["input_sha256"], "generation_splits_sha256": identity["splits_sha256"],
                  "generation_inventory_sha256": identity["inventory_sha256"]}
        manifest = {"bundle_schema_version": SCHEMA, "mode": mode,
                    "task_evaluation_complete": mode == "final", "coverage": coverage, "failure_count": len(errors),
                    "scope": {"run_id": run["manifest"]["run_id"], "task": identity["task"], "split": identity["split"],
                              "gold_summary": _summary(gold), "corpus": identity["config"]["corpus"],
                              "model": {key: identity["options"][key] for key in ("provider", "model", "model_revision")}},
                    "scoring": {"n_bootstrap": n_bootstrap, "seed": seed},
                    "generator": {"code": code_identity(), "environment": environment()},
                    "assessment_file": "evaluations/assessments.json" if assessments else None,
                    "review_agreement": assessments.get("agreement") if assessments else None,
                    "inputs_manifest_sha256": fingerprint(inputs), "protocol_sha256": fingerprint(protocol)}
        for name, value in (("manifest.json", manifest), ("protocol.json", protocol),
                            ("inputs_manifest.json", inputs), ("metrics.json", metrics), ("comparisons.json", comparisons)):
            _write(staging / name, value)
        if assessments:
            _write(staging / "evaluations/assessments.json", assessments)
        for system, rows in evaluations.items():
            target = staging / "evaluations" / f"{system}.jsonl"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        (staging / "report.md").write_text(render_report(manifest, metrics, comparisons), encoding="utf-8")
        _write(staging / "checksums.json", {"schema_version": SCHEMA, "files": _files(staging)})
        # Re-read the copied snapshot before publishing the bundle directory.
        audit_bundle(staging, gold_path, recompute=False)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir


def audit_bundle(directory: Path, gold_path: Path, *, recompute=True):
    directory = directory.resolve()
    checksum_path = directory / "checksums.json"
    if checksum_path.is_symlink():
        raise ValueError("Bundle checksum manifest must not be a symlink")
    checksums = _read(checksum_path)
    if checksums.get("schema_version") != SCHEMA or not isinstance(checksums.get("files"), dict):
        raise ValueError("Unsupported checksum manifest")
    expected = checksums["files"]
    actual = _files(directory)
    if set(expected) != set(actual) or any(expected[name] != actual[name] for name in expected):
        raise ValueError("Bundle file inventory/checksum mismatch")
    if not REQUIRED <= set(actual) | {"checksums.json"}:
        raise ValueError("Missing required bundle artifacts")
    manifest, protocol, inputs = (_read(directory / name) for name in
                                  ("manifest.json", "protocol.json", "inputs_manifest.json"))
    if manifest.get("bundle_schema_version") != SCHEMA or manifest.get("mode") not in ("final", "progress"):
        raise ValueError("Unsupported bundle manifest")
    if manifest.get("task_evaluation_complete") != (manifest["mode"] == "final"):
        raise ValueError("Inconsistent bundle completion state")
    if manifest["inputs_manifest_sha256"] != fingerprint(inputs) or manifest["protocol_sha256"] != fingerprint(protocol):
        raise ValueError("Bundle protocol/input manifest identity mismatch")
    if inputs["gold_sha256"] != file_hash(gold_path):
        raise ValueError("External gold hash mismatch")
    gold = load_gold(gold_path)
    if inputs["gold_content_sha256"] != fingerprint({key: record.to_dict() for key, record in sorted(gold.items())}) or inputs["record_ids"] != sorted(gold):
        raise ValueError("External gold content/ID mismatch")
    run = load_run(directory / "execution", gold, allow_partial=manifest["mode"] == "progress")
    identity = run["manifest"]["identity"]
    if (protocol["config"] != identity["config"] or protocol["rubric_version"] != RUBRIC_VERSION
            or manifest["scope"]["run_id"] != run["manifest"]["run_id"]
            or manifest["scope"]["task"] != identity["task"] or manifest["scope"]["split"] != identity["split"]
            or manifest["scope"]["gold_summary"] != _summary(gold)):
        raise ValueError("Bundle scope differs from execution/gold")
    for key, source in (("generation_input_sha256", "input_sha256"), ("generation_splits_sha256", "splits_sha256"),
                        ("generation_inventory_sha256", "inventory_sha256")):
        if inputs[key] != identity[source]:
            raise ValueError("Generation input identity mismatch")
    if manifest["scope"]["corpus"] != identity["config"]["corpus"] or manifest["scope"]["model"] != {
            key: identity["options"][key] for key in ("provider", "model", "model_revision")}:
        raise ValueError("Report model/corpus scope mismatch")
    expected_evaluations = {f"{system}.jsonl" for system in run["outputs"]}
    if {path.name for path in (directory / "evaluations").glob("*.jsonl")} != expected_evaluations:
        raise ValueError("Missing or extra per-system evaluation files")
    assessment_file = manifest.get("assessment_file")
    if assessment_file not in (None, "evaluations/assessments.json"):
        raise ValueError("Invalid assessment file reference")
    assessments = _read(directory / assessment_file) if assessment_file else None
    # Structural validation always enforces finality gates. Recompute optionally
    # skips costly resampling, not completeness or assessment validation.
    if manifest["mode"] == "final" and identity["task"] == "repair" and (
            assessments is None or assessments.get("adjudicated") is not True):
        raise ValueError("Final repair bundle requires adjudicated assessments")
    if assessments is not None:
        validate_assessments(assessments, run, gold)
    if manifest.get("review_agreement") != (assessments.get("agreement") if assessments else None):
        raise ValueError("Review agreement metadata differs from source assessments")
    stored_metrics, stored_comparisons = (_read(directory / name) for name in ("metrics.json", "comparisons.json"))
    if (directory / "report.md").read_text(encoding="utf-8") != render_report(manifest, stored_metrics, stored_comparisons):
        raise ValueError("Report tables differ from saved metrics")
    if recompute:
        metrics, comparisons, evaluations, coverage = _products(run, gold, assessments, manifest["mode"], **manifest["scoring"])
        if canonical(metrics) != canonical(stored_metrics) or canonical(comparisons) != canonical(stored_comparisons):
            raise ValueError("Offline metric recomputation differs from saved values")
        for system, rows in evaluations.items():
            expected_bytes = b"".join(canonical(row) + b"\n" for row in rows)
            if (directory / "evaluations" / f"{system}.jsonl").read_bytes() != expected_bytes:
                raise ValueError("Saved per-record evaluations differ from recomputation")
    else:
        coverage = {system: {"expected": len(gold), "successful": len(rows), "missing_ids": sorted(set(gold) - set(rows))}
                    for system, rows in sorted(run["outputs"].items())}
    if coverage != manifest["coverage"]:
        raise ValueError("Bundle coverage differs from committed outputs")
    errors = [wrapper["result"] for path in sorted((directory / "execution/attempts").glob("*/*.json"))
              for wrapper in [_read(path)] if wrapper["result"]["status"] == "error"]
    if (directory / "errors.jsonl").read_bytes() != b"".join(canonical(row) + b"\n" for row in errors) or len(errors) != manifest["failure_count"]:
        raise ValueError("Bundle error history mismatch")
    return {"status": "valid_final_task_bundle" if manifest["mode"] == "final" else "valid_progress_bundle",
            "recomputed": recompute, "task": identity["task"], "split": identity["split"],
            "systems": sorted(run["outputs"]), "records": len(gold)}
