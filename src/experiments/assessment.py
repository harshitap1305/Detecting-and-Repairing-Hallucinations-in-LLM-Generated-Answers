"""Blinded human repair review packets and strict independent-rating import."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import random
import secrets
import shutil
import tempfile

from src.experiments.artifacts import canonical, file_hash, fingerprint
from src.experiments.scoring import jsonl
from src.experiments.statistics import clustered_interval, paired_family

RUBRIC_VERSION = "repair-review-v1"
OUTCOMES = {"retained", "corrected", "deleted", "hedged", "still_incorrect"}


def export_packet(run, gold, output_dir: Path, seed=42):
    if run["manifest"]["identity"]["task"] != "repair":
        raise ValueError("Human repair packets require a repair run")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    public, mapping = [], {}
    # Random opaque tokens, not reversible deterministic hashes of system IDs.
    for system, rows in sorted(run["outputs"].items()):
        for record_id in sorted(gold):
            source = gold[record_id]
            token = secrets.token_hex(16)
            item = {"item_id": token, "question": source.question, "original_answer": source.answer,
                    "candidate_answer": rows[record_id]["record"]["repaired_answer"],
                    "claims": [{"claim_id": f"claim-{i + 1}", "text": claim.text,
                                "original_label": claim.label,
                                "evidence": [passage.__dict__ for passage in claim.evidence]}
                               for i, claim in enumerate(source.check_worthy_claims)]}
            public.append(item)
            mapping[token] = {"system": system, "record_id": record_id,
                              "output_sha256": fingerprint(rows[record_id]),
                              "claim_ids": {f"claim-{i + 1}": claim.id for i, claim in
                                            enumerate(source.check_worthy_claims)}}
    random.Random(seed).shuffle(public)
    template = [{"item_id": item["item_id"], "claim_outcomes": [
                    {"claim_id": claim["claim_id"], "outcome": None, "rationale": "", "evidence": []}
                    for claim in item["claims"]], "new_errors": None,
                 "completeness": None, "notes": ""} for item in public]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".assessment-", dir=output_dir.parent))
    try:
        (staging / "reviewer").mkdir()
        (staging / "private").mkdir()
        for filename, rows in (("items.jsonl", public), ("ratings_template.jsonl", template)):
            (staging / "reviewer" / filename).write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        instructions = (
            "Human repair review v1\nReview all original claims against the candidate and supplied evidence.\n"
            "Supported claims: retained, deleted, hedged, or still_incorrect (corrupted).\n"
            "Contradicted/Unverifiable claims: corrected, deleted, hedged, or still_incorrect.\n"
            "Correction requires an evidence-grounded fact; hedging is not correction.\n"
            "For each judgment supply a rationale; corrected claims need inspected evidence\n"
            "with doc_id, title and text. List every new factual error with text/rationale/evidence.\n"
            "Set new_errors to [] only after checking for new errors.\n"
            "Completeness: 0=no useful answer, 1=partial answer, 2=complete answer to the question.\n"
            "An empty candidate must have all outcomes deleted and completeness 0.\n"
            "Use separate template copies for two independent reviewers. Preserve raw submissions.\n"
            "Do not share the private mapping with reviewers. System identity is hidden, but\n"
            "writing style may still reveal a method. Gold labels are context, not candidate verdicts.\n"
        )
        (staging / "reviewer" / "instructions.txt").write_text(instructions, encoding="utf-8")
        key = {"schema_version": 1, "rubric_version": RUBRIC_VERSION,
               "run_id": run["manifest"]["run_id"], "mapping": mapping,
               "gold_sha256": fingerprint({cid: record.to_dict() for cid, record in sorted(gold.items())}),
               "items_sha256": file_hash(staging / "reviewer" / "items.jsonl"),
               "shuffle_seed": seed}
        (staging / "private" / "key.json").write_bytes(canonical(key) + b"\n")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir


def _evidence(passages, required=False):
    if not isinstance(passages, list) or (required and not passages):
        raise ValueError("Inspected evidence required")
    for passage in passages:
        if not isinstance(passage, dict) or any(not isinstance(passage.get(key), str)
                or not passage[key].strip() for key in ("doc_id", "title", "text")):
            raise ValueError("Malformed inspected citation")


def import_ratings(packet_dir: Path, ratings_path: Path, run, gold, reviewer: str):
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Reviewer identity is required")
    key = json.loads((packet_dir / "private" / "key.json").read_text(encoding="utf-8"))
    if key.get("rubric_version") != RUBRIC_VERSION or key.get("run_id") != run["manifest"]["run_id"]:
        raise ValueError("Wrong assessment rubric or run")
    if key.get("gold_sha256") != fingerprint({cid: record.to_dict() for cid, record in sorted(gold.items())}):
        raise ValueError("Gold changed since assessment export")
    items_path = packet_dir / "reviewer" / "items.jsonl"
    if key["items_sha256"] != file_hash(items_path):
        raise ValueError("Review items were modified")
    items = {row["item_id"]: row for row in jsonl(items_path)}
    if set(items) != set(key["mapping"]):
        raise ValueError("Review item/key mismatch")
    ratings = jsonl(ratings_path)
    seen, normalized = set(), []
    for rating in ratings:
        token = rating.get("item_id")
        if token not in items or token in seen:
            raise ValueError("Duplicate or unknown assessment item")
        seen.add(token)
        link = key["mapping"][token]
        source = gold[link["record_id"]]
        output = run["outputs"][link["system"]][source.id]
        if link["output_sha256"] != fingerprint(output):
            raise ValueError("Assessment is bound to different generated output")
        original = {claim.id: claim for claim in source.check_worthy_claims}
        expected_claims = {f"claim-{i + 1}": claim.id for i, claim in enumerate(source.check_worthy_claims)}
        if link["claim_ids"] != expected_claims:
            raise ValueError("Gold inventory changed since review export")
        outcomes = rating.get("claim_outcomes")
        if not isinstance(outcomes, list):
            raise ValueError("Claim outcomes must be a list")
        assessed, found = {}, set()
        for judgment in outcomes:
            if not isinstance(judgment, dict):
                raise ValueError("Malformed claim assessment")
            cid = judgment.get("claim_id")
            if cid not in expected_claims or cid in found:
                raise ValueError("Duplicate or unknown reviewed claim")
            found.add(cid)
            claim = original[expected_claims[cid]]
            outcome = judgment.get("outcome")
            allowed = OUTCOMES - ({"corrected"} if claim.label == "Supported" else {"retained"})
            if outcome not in allowed or not isinstance(judgment.get("rationale"), str) or not judgment["rationale"].strip():
                raise ValueError("Invalid or unfinished outcome/rationale")
            _evidence(judgment.get("evidence"), required=outcome == "corrected")
            assessed[claim.id] = copy.deepcopy(judgment)
        if found != set(expected_claims):
            raise ValueError("Missing claim assessments")
        if not isinstance(rating.get("notes", ""), str):
            raise ValueError("Review notes must be text")
        completeness, errors = rating.get("completeness"), rating.get("new_errors")
        if type(completeness) is not int or completeness not in (0, 1, 2) or not isinstance(errors, list):
            raise ValueError("Review must include completeness and explicit new-error list")
        for error in errors:
            if not isinstance(error, dict) or any(not isinstance(error.get(field), str)
                    or not error[field].strip() for field in ("text", "rationale")):
                raise ValueError("Malformed new factual error")
            _evidence(error.get("evidence"))
        if not output["record"]["repaired_answer"].strip() and (
                completeness != 0 or errors or any(value["outcome"] != "deleted" for value in assessed.values())):
            raise ValueError("Empty output cannot correct/retain claims or provide a complete answer")
        normalized.append({"system": link["system"], "record_id": source.id,
                           "outcomes": {cid: value["outcome"] for cid, value in assessed.items()},
                           "judgments": assessed, "new_errors": errors, "completeness": completeness,
                           "notes": rating.get("notes", ""), "output_sha256": fingerprint(output)})
    if seen != set(items):
        raise ValueError("Incomplete assessment coverage")
    return {"schema_version": 1, "rubric_version": RUBRIC_VERSION,
            "run_id": key["run_id"], "reviewer": reviewer, "reviewer_kind": "human",
            "ratings_sha256": file_hash(ratings_path), "key_sha256": file_hash(packet_dir / "private" / "key.json"),
            "gold_sha256": fingerprint({cid: record.to_dict() for cid, record in sorted(gold.items())}),
            "assessments": sorted(normalized, key=lambda row: (row["system"], row["record_id"]))}


def validate_assessments(data, run, gold):
    if (data.get("schema_version") != 1 or data.get("rubric_version") != RUBRIC_VERSION
            or data.get("run_id") != run["manifest"]["run_id"] or data.get("reviewer_kind") != "human"
            or not isinstance(data.get("reviewer"), str) or not data["reviewer"].strip()
            or data.get("gold_sha256") != fingerprint({cid: record.to_dict() for cid, record in sorted(gold.items())})):
        raise ValueError("Assessment identity/provenance mismatch")
    expected = {(system, rid) for system in run["outputs"] for rid in gold}
    seen = set()
    for row in data["assessments"]:
        key = (row["system"], row["record_id"])
        if key not in expected or key in seen:
            raise ValueError("Duplicate or extra normalized assessment")
        seen.add(key)
        record = gold[key[1]]
        output = run["outputs"][key[0]][key[1]]
        if row.get("output_sha256") != fingerprint(output):
            raise ValueError("Assessment output fingerprint mismatch")
        claims = {claim.id: claim for claim in record.check_worthy_claims}
        if set(row["outcomes"]) != set(claims) or set(row["judgments"]) != set(claims):
            raise ValueError("Incomplete normalized claim coverage")
        for cid, outcome in row["outcomes"].items():
            allowed = OUTCOMES - ({"corrected"} if claims[cid].label == "Supported" else {"retained"})
            judgment = row["judgments"][cid]
            if (outcome not in allowed or judgment.get("outcome") != outcome
                    or not isinstance(judgment.get("rationale"), str) or not judgment["rationale"].strip()):
                raise ValueError("Invalid normalized outcome")
            _evidence(judgment.get("evidence"), required=outcome == "corrected")
        if type(row["completeness"]) is not int or row["completeness"] not in (0, 1, 2):
            raise ValueError("Invalid completeness")
        if not isinstance(row["new_errors"], list):
            raise ValueError("Missing new-error review")
        for error in row["new_errors"]:
            if not isinstance(error, dict) or any(not isinstance(error.get(field), str)
                    or not error[field].strip() for field in ("text", "rationale")):
                raise ValueError("Invalid new error")
            _evidence(error.get("evidence"))
        if not output["record"]["repaired_answer"].strip() and (row["completeness"] != 0
                or row["new_errors"] or any(value != "deleted" for value in row["outcomes"].values())):
            raise ValueError("Empty answer cannot count as a successful repair")
    if seen != expected:
        raise ValueError("Incomplete human assessment coverage")
    if "adjudicated" in data:
        if data["adjudicated"] is not True:
            raise ValueError("Invalid adjudication state")
        sources = data.get("source_reviews")
        if not isinstance(sources, list) or len(sources) != 2 or any("adjudicated" in source for source in sources):
            raise ValueError("Adjudication needs both original independent reviews")
        for source in sources:
            validate_assessments(source, run, gold)
        recomputed = adjudicate(sources[0], sources[1], data)
        for field in ("agreement", "decisions", "source_review_sha256"):
            if data.get(field) != recomputed[field]:
                raise ValueError("Adjudication provenance/decisions mismatch")
    return data


def adjudicate(first, second, final):
    if "adjudicated" in first or "adjudicated" in second:
        raise ValueError("Supply original independent reviews for adjudication")
    report = agreement(first, second)
    if (final["reviewer"] in (first["reviewer"], second["reviewer"])
            or final["run_id"] != first["run_id"] or final["gold_sha256"] != first["gold_sha256"]):
        raise ValueError("Adjudication requires a distinct reviewer on the same run/gold")
    maps = [{(row["system"], row["record_id"]): row for row in data["assessments"]}
            for data in (first, second, final)]
    if maps[0].keys() != maps[2].keys():
        raise ValueError("Incomplete adjudication coverage")
    fields = ("outcomes", "new_errors", "completeness")
    decisions = []
    for key in sorted(maps[0]):
        a, b, chosen = (mapping[key] for mapping in maps)
        if any(a[field] != b[field] or a[field] != chosen[field] for field in fields):
            if not isinstance(chosen.get("notes"), str) or not chosen["notes"].strip():
                raise ValueError("Disagreements and changed decisions require adjudication notes")
            decisions.append({"system": key[0], "record_id": key[1], "rationale": chosen["notes"],
                              "first": {field: a[field] for field in fields},
                              "second": {field: b[field] for field in fields},
                              "chosen": {field: chosen[field] for field in fields}})
    result = copy.deepcopy(final)
    result.update(adjudicated=True, agreement=report, decisions=decisions,
                  source_review_sha256=[fingerprint(first), fingerprint(second)],
                  source_reviews=[copy.deepcopy(first), copy.deepcopy(second)])
    return result


def agreement(first, second):
    if first["reviewer"] == second["reviewer"]:
        raise ValueError("Independent reviewer identities must differ")
    if first["run_id"] != second["run_id"] or first["gold_sha256"] != second["gold_sha256"]:
        raise ValueError("Reviewers assessed different runs/gold")
    index = lambda data: {(row["system"], row["record_id"]): row for row in data["assessments"]}
    a, b = index(first), index(second)
    if a.keys() != b.keys():
        raise ValueError("Reviewer coverage mismatch")
    pairs, disagreements = [], []
    for key in sorted(a):
        for claim in a[key]["outcomes"]:
            x, y = a[key]["outcomes"][claim], b[key]["outcomes"][claim]
            pairs.append((x, y))
        if (a[key]["outcomes"] != b[key]["outcomes"] or a[key]["new_errors"] != b[key]["new_errors"]
                or a[key]["completeness"] != b[key]["completeness"]):
            disagreements.append({"system": key[0], "record_id": key[1]})
    observed = sum(x == y for x, y in pairs) / len(pairs)
    left, right = Counter(x for x, y in pairs), Counter(y for x, y in pairs)
    expected = sum(left[label] * right[label] for label in OUTCOMES) / len(pairs)**2
    return {"n_claim_ratings": len(pairs), "outcome_agreement": observed,
            "cohens_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "kappa_note": "undefined for constant marginals" if expected == 1 else None,
            "reviewer_label_counts": [dict(left), dict(right)],
            "disagreements": disagreements}


def score_assessments(data, gold, n_bootstrap=1000, seed=42):
    result = {}
    systems = sorted({row["system"] for row in data["assessments"]})
    for system in systems:
        records = {}
        totals = Counter()
        for row in data["assessments"]:
            if row["system"] != system:
                continue
            record = gold[row["record_id"]]
            bad = [claim.id for claim in record.check_worthy_claims if claim.label != "Supported"]
            good = [claim.id for claim in record.check_worthy_claims if claim.label == "Supported"]
            outcomes = row["outcomes"]
            totals.update(outcomes.values())
            records[record.id] = {"correction_rate": sum(outcomes[cid] == "corrected" for cid in bad) / len(bad) if bad else None,
                                  "preservation_rate": sum(outcomes[cid] == "retained" for cid in good) / len(good) if good else None,
                                  "new_error_count": len(row["new_errors"]), "completeness": row["completeness"]}
        metrics = {}
        for metric in ("correction_rate", "preservation_rate", "new_error_count", "completeness"):
            eligible = [record[metric] for record in records.values() if record[metric] is not None]
            metrics[metric] = (clustered_interval(eligible, lambda sample: sum(sample) / len(sample), n_bootstrap, seed)
                               if eligible else {"estimate": None, "reason": "No eligible original claims"})
            metrics[metric]["eligible_answers"] = len(eligible)
        result[system] = {"metrics": metrics, "outcome_counts": dict(totals), "per_record": records}
    comparisons = {"status": "unavailable", "reason": "B0 is absent from this run"}
    if "b0" in result:
        pairs = {}
        for system in systems:
            if system == "b0":
                continue
            for metric in ("correction_rate", "preservation_rate"):
                values = {rid: value[metric] for rid, value in result[system]["per_record"].items() if value[metric] is not None}
                baseline = {rid: value[metric] for rid, value in result["b0"]["per_record"].items() if value[metric] is not None}
                if values:
                    pairs[f"{system}_vs_b0/{metric}"] = (values, baseline)
        comparisons = paired_family(pairs, n_bootstrap, seed)
    return {"systems": result, "paired_repair_family": comparisons,
            "estimand": "mean per-answer rate over gold-eligible answers; not pooled claim ratio",
            "reviewer": data["reviewer"], "reviewer_kind": "human"}
