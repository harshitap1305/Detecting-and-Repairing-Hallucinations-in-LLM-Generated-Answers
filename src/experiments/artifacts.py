"""Shared input isolation, JSON persistence, and output contract validation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from src.schema import AnswerRecord, Claim

LABELS = {"Supported", "Contradicted", "Unverifiable"}
IDENTITY = ("id", "source_dataset", "split", "question", "answer")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value) -> None:
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_inputs(path: Path, split: str) -> list[AnswerRecord]:
    records = []
    seen = set()
    prompts = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str)
                                            or not raw[key].strip() for key in IDENTITY):
            raise ValueError(f"{path}:{number}: missing input identity/text")
        if raw["split"] != split or raw["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate ID or wrong split")
        prompt = " ".join(raw["question"].casefold().split())
        if prompt in prompts:
            raise ValueError(f"{path}:{number}: duplicate prompt")
        seen.add(raw["id"])
        prompts.add(prompt)
        # Ignore gold and arbitrary source fields, even if the input is a gold file.
        records.append(AnswerRecord(**{key: raw[key] for key in IDENTITY}))
    if not records:
        raise ValueError("Inputs must be nonempty")
    return records


def read_inventory(path: Path, inputs: list[AnswerRecord], segmenter) -> dict[str, list[Claim]]:
    """Match original text exactly and copy ONLY unlabeled claim inventory fields."""
    sources = {record.id: record for record in inputs}
    result = {}
    all_claim_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or raw.get("id") not in sources or raw["id"] in result:
            raise ValueError("Inventory has extra/duplicate record IDs")
        source = sources[raw["id"]]
        if any(raw.get(key) != getattr(source, key) for key in IDENTITY):
            raise ValueError("Inventory input identity/text mismatch")
        spans = [(sent.start_char, sent.end_char) for sent in segmenter(source.answer).sents]
        claims = raw.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("Fixed-inventory mode requires nonempty human claims")
        cleaned = []
        for claim in claims:
            if (not isinstance(claim, dict) or not isinstance(claim.get("id"), str)
                    or not claim["id"].strip() or claim["id"] in all_claim_ids
                    or not isinstance(claim.get("text"), str) or not claim["text"].strip()
                    or type(claim.get("sentence_id")) is not int
                    or not 0 <= claim["sentence_id"] < len(spans)
                    or type(claim.get("check_worthy")) is not bool):
                raise ValueError("Invalid or duplicate inventory claim")
            span = claim.get("span")
            start, end = spans[claim["sentence_id"]]
            if span is not None and (not isinstance(span, list) or len(span) != 2
                    or any(type(offset) is not int for offset in span)
                    or not start <= span[0] < span[1] <= end):
                raise ValueError("Inventory claim span outside sentence")
            all_claim_ids.add(claim["id"])
            cleaned.append(Claim(**{key: claim[key] for key in
                                    ("id", "text", "sentence_id", "check_worthy")}, span=span))
        if not any(claim.check_worthy for claim in cleaned):
            raise ValueError("Inventory has no check-worthy claims")
        result[source.id] = cleaned
    if set(result) != set(sources):
        raise ValueError("Inventory IDs must exactly match input IDs")
    return result


def validate_result(result: dict, source: AnswerRecord, system: str, task: str,
                    inventory=None) -> None:
    if (not isinstance(result, dict) or result.get("schema_version") != 1
            or result.get("system") != system or result.get("task") != task
            or result.get("input") != {key: getattr(source, key) for key in IDENTITY}
            or result.get("status") not in ("success", "error")):
        raise ValueError("Output envelope identity/status mismatch")
    elapsed = result.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid output timing")
    if result["status"] == "error":
        if not isinstance(result.get("error"), dict) or not result["error"].get("type"):
            raise ValueError("Missing execution error")
        return
    record = AnswerRecord.from_dict(result["record"])
    if any(getattr(record, key) != getattr(source, key) for key in IDENTITY):
        raise ValueError("System mutated original input")
    if record.gold_repaired_answer is not None:
        raise ValueError("Gold reference leaked into system output")
    if task == "repair" and not isinstance(record.repaired_answer, str):
        raise ValueError("Missing generated answer (empty deletion is valid)")
    if task == "claim_verification":
        fields = lambda c: (c.id, c.text, c.sentence_id, c.span, c.check_worthy)
        if inventory is None or [fields(c) for c in record.claims] != [fields(c) for c in inventory]:
            raise ValueError("System changed fixed claim inventory")
        if any(claim.label not in LABELS for claim in record.check_worthy_claims):
            raise ValueError("Missing/invalid claim verdict")
    if task == "answer_detection":
        verdict = result.get("native", {}).get("answer_detection")
        if (not isinstance(verdict, dict) or type(verdict.get("has_hallucination")) is not bool
                or type(verdict.get("confidence")) not in (int, float)
                or not 0 <= verdict["confidence"] <= 1):
            raise ValueError("Invalid answer-level detection result")
    canonical(result)  # reject NaN/Infinity and unserializable fields before persistence
