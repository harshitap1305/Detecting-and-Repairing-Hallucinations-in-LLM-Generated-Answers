"""Validate human gold annotations before model training or paired comparison.

The optional holdout manifest contains only test-set keys. Wikipedia article
titles come from human-cited evidence; entities for records without cited
articles must be supplied explicitly with --entity-map. This command checks
structure and split leakage, not whether a human verdict is factually correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_baseline import read_splits
from src.decompose import split_sentences_with_spans
from src.schema import AnswerRecord


LABELS = {"Supported", "Contradicted", "Unverifiable"}
REQUIRED = {"id", "source_dataset", "split", "question", "answer",
            "claims", "gold_repaired_answer"}


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gold(path: Path, split: str) -> list[AnswerRecord]:
    """Read JSONL and reject ambiguous or partially annotated records."""
    if not path.is_file():
        raise FileNotFoundError(f"Gold {split} file is missing: {path}")
    records: list[AnswerRecord] = []
    seen_records: set[str] = set()
    seen_claims: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_no}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict) or not REQUIRED <= raw.keys():
                raise ValueError(f"{location}: expected an object with {sorted(REQUIRED)}")
            if raw["split"] != split:
                raise ValueError(f"{location}: expected split={split}")
            for name in ("id", "source_dataset", "question", "answer"):
                if not _nonempty_text(raw[name]):
                    raise ValueError(f"{location}: {name} must be nonempty text")
            if not isinstance(raw["gold_repaired_answer"], str):
                raise ValueError(f"{location}: gold_repaired_answer must be text (empty is allowed)")
            if not isinstance(raw["claims"], list) or not raw["claims"]:
                raise ValueError(f"{location}: claims must be a nonempty list")
            if raw["id"] in seen_records:
                raise ValueError(f"{location}: duplicate record id {raw['id']}")
            seen_records.add(raw["id"])

            spans = split_sentences_with_spans(raw["answer"])
            check_worthy = 0
            supported = 0
            for index, claim in enumerate(raw["claims"]):
                item = f"{location} claim[{index}]"
                if not isinstance(claim, dict):
                    raise ValueError(f"{item}: expected object")
                if not _nonempty_text(claim.get("id")) or not _nonempty_text(claim.get("text")):
                    raise ValueError(f"{item}: id and text must be nonempty")
                if claim["id"] in seen_claims:
                    raise ValueError(f"{item}: duplicate claim id {claim['id']}")
                seen_claims.add(claim["id"])
                worthy = claim.get("check_worthy", True)
                if type(worthy) is not bool:
                    raise ValueError(f"{item}: check_worthy must be boolean")
                sentence_id = claim.get("sentence_id")
                if type(sentence_id) is not int or not 0 <= sentence_id < len(spans):
                    raise ValueError(f"{item}: sentence_id is outside the answer")
                if claim.get("span") is not None:
                    span = claim["span"]
                    start, end = spans[sentence_id][1:]
                    if (not isinstance(span, list) or len(span) != 2
                            or any(type(offset) is not int for offset in span)
                            or not start <= span[0] < span[1] <= end):
                        raise ValueError(f"{item}: span must lie within its sentence")
                label = claim.get("label")
                if worthy:
                    check_worthy += 1
                    if label not in LABELS:
                        raise ValueError(f"{item}: check-worthy claim needs a verdict label")
                    supported += label == "Supported"
                elif label is not None and label not in LABELS:
                    raise ValueError(f"{item}: invalid optional verdict label")

                evidence = claim.get("evidence", [])
                if not isinstance(evidence, list):
                    raise ValueError(f"{item}: evidence must be a list")
                if worthy and label in {"Supported", "Contradicted"} and not evidence:
                    raise ValueError(f"{item}: {label} needs cited evidence")
                for evidence_index, passage in enumerate(evidence):
                    if (not isinstance(passage, dict)
                            or any(not _nonempty_text(passage.get(key))
                                   for key in ("doc_id", "title", "text"))):
                        raise ValueError(
                            f"{item} evidence[{evidence_index}]: doc_id, title and text are required"
                        )
            if not check_worthy:
                raise ValueError(f"{location}: at least one check-worthy claim is required")
            if supported and not raw["gold_repaired_answer"].strip():
                raise ValueError(f"{location}: an empty repair would remove supported claims")
            try:
                records.append(AnswerRecord.from_dict(raw))
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"{location}: schema error: {exc}") from exc
    if not records:
        raise ValueError(f"Gold {split} file is empty: {path}")
    return records


def validate_gold_set(dev_path: Path, test_path: Path, splits_path: Path,
                      *, expected_dev: int | None = None,
                      expected_test: int | None = None) -> tuple[list[AnswerRecord], list[AnswerRecord], dict]:
    """Validate exact split membership, unique claims and prompt separation."""
    splits, note = read_splits(splits_path)
    if splits is None:
        raise ValueError(f"Invalid split file {splits_path}: {note}")
    dev = load_gold(dev_path, "dev")
    test = load_gold(test_path, "test")
    dev_ids = {record.id for record in dev}
    test_ids = {record.id for record in test}
    if dev_ids != splits["dev"] or test_ids != splits["test"]:
        raise ValueError(
            "Gold IDs must exactly match data/splits.json: "
            f"dev missing={len(splits['dev'] - dev_ids)}, extra={len(dev_ids - splits['dev'])}; "
            f"test missing={len(splits['test'] - test_ids)}, extra={len(test_ids - splits['test'])}"
        )
    if dev_ids & test_ids:
        raise ValueError("Dev and test records share IDs")
    dev_claims = {claim.id for record in dev for claim in record.claims}
    test_claims = {claim.id for record in test for claim in record.claims}
    if dev_claims & test_claims:
        raise ValueError("Dev and test records share claim IDs")
    normalize = lambda value: " ".join(value.casefold().split())
    dev_prompts = {normalize(record.question) for record in dev}
    test_prompts = {normalize(record.question) for record in test}
    if dev_prompts & test_prompts:
        raise ValueError("Dev and test records share a question/prompt")
    for split, rows, expected in (("dev", dev, expected_dev), ("test", test, expected_test)):
        if expected is not None and len(rows) != expected:
            raise ValueError(f"Expected {expected} {split} records, found {len(rows)}")
    labels = Counter(claim.label for record in dev + test
                     for claim in record.check_worthy_claims)
    report = {
        "status": "valid_structure_only",
        "dev_records": len(dev), "test_records": len(test),
        "check_worthy_labels": dict(sorted(labels.items())),
        "gold_dev_sha256": _sha256(dev_path),
        "gold_test_sha256": _sha256(test_path),
        "splits_sha256": _sha256(splits_path),
    }
    return dev, test, report


def read_entity_map(path: Path | None, test_ids: set[str]) -> dict[str, list[str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Entity map is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid entity map JSON: {exc.msg}") from exc
    if not isinstance(raw, dict) or not set(raw) <= test_ids:
        raise ValueError("Entity map must be an object keyed only by test record IDs")
    result = {}
    for record_id, values in raw.items():
        if not isinstance(values, list) or any(not _nonempty_text(v) for v in values):
            raise ValueError(f"Entity map {record_id} must be a list of nonempty strings")
        result[record_id] = [value.strip() for value in values]
    return result


def build_holdout(test: list[AnswerRecord], entity_map: dict[str, list[str]],
                  test_sha256: str) -> dict:
    """Derive test-only training exclusions from citations and explicit entities."""
    articles: set[str] = set()
    entities: set[str] = set()
    uncovered: list[str] = []
    for record in test:
        record_entities = set(entity_map.get(record.id, []))
        for claim in record.check_worthy_claims:
            titles = {passage.title.strip() for passage in claim.evidence
                      if passage.title.strip()}
            if not titles and not record_entities:
                uncovered.append(f"{record.id}/{claim.id}")
            articles.update(titles)
        entities.update(record_entities)
    if uncovered:
        raise ValueError(
            "Cannot build a leakage holdout: no cited article or mapped entity for "
            + ", ".join(uncovered[:5]) + ("..." if len(uncovered) > 5 else "")
        )
    return {
        "article_ids": sorted(articles, key=str.casefold),
        "entities": sorted(entities, key=str.casefold),
        "prompt_ids": sorted(record.id for record in test),
        "prompts": sorted({record.question for record in test}, key=str.casefold),
        "source": "human gold test citations and explicit entity map",
        "test_records_covered": len(test),
        "gold_test_sha256": test_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dev", type=Path, default=Path("data/gold/dev.jsonl"))
    parser.add_argument("--gold-test", type=Path, default=Path("data/gold/test.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--expected-dev", type=int)
    parser.add_argument("--expected-test", type=int)
    parser.add_argument("--holdout-out", type=Path,
                        help="Write test-only article/entity/prompt exclusions after validation")
    parser.add_argument("--entity-map", type=Path,
                        help="JSON mapping test record IDs to extra entity names")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if ((args.expected_dev is not None and args.expected_dev < 1)
            or (args.expected_test is not None and args.expected_test < 1)):
        parser.error("Expected split counts must be positive")
    if args.entity_map and not args.holdout_out:
        parser.error("--entity-map requires --holdout-out")
    try:
        _, test, report = validate_gold_set(
            args.gold_dev, args.gold_test, args.splits,
            expected_dev=args.expected_dev, expected_test=args.expected_test,
        )
        if args.holdout_out:
            if args.holdout_out.exists() and not args.overwrite:
                raise FileExistsError(f"Holdout output already exists: {args.holdout_out}")
            entity_map = read_entity_map(args.entity_map, {record.id for record in test})
            manifest = build_holdout(test, entity_map, report["gold_test_sha256"])
            args.holdout_out.parent.mkdir(parents=True, exist_ok=True)
            args.holdout_out.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            report["holdout_file"] = str(args.holdout_out)
            report["holdout_articles"] = len(manifest["article_ids"])
            report["holdout_entities"] = len(manifest["entities"])
    except (OSError, ValueError) as exc:
        print(f"Gold validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
