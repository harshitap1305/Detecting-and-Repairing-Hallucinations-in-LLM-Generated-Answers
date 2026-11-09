"""Deterministically prepare blank annotation packets; never generate gold labels.

python scripts/prepare_annotation_batch.py --config configs/evaluation.yaml \
    --output-dir data/annotation_batches/v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.protocol import load_config
from src.schema import AnswerRecord


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def prepare(config_path: Path, output_dir: Path) -> Path:
    config_path = config_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    config = load_config(config_path, require_sources=True)
    rows = []
    sources = []
    seen_ids = set()
    # Canonical order means source-list order and JSONL order cannot change selection.
    for source in sorted(config["sources"], key=lambda item: item["id"]):
        path = (config_path.parent / source["path"]).resolve()
        data = path.read_bytes()
        source_rows = 0
        for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            where = f"{path}:{line_no}"
            if not isinstance(raw, dict):
                raise ValueError(f"{where}: expected record object")
            for field in ("id", "source_dataset", "question", "answer"):
                if not isinstance(raw.get(field), str) or not raw[field].strip():
                    raise ValueError(f"{where}: missing nonempty {field}")
            if raw["id"] in seen_ids:
                raise ValueError(f"{where}: duplicate record ID {raw['id']}")
            seen_ids.add(raw["id"])
            if raw["source_dataset"] != source["id"]:
                raise ValueError(f"{where}: source_dataset differs from configured source")
            if raw.get("split") not in ("dev", "test"):
                raise ValueError(f"{where}: source must declare dev or test split")
            # Construct by allowlist: never deserialize/forward source labels, claims,
            # evidence, reference repairs, reliability scores, or arbitrary metadata.
            safe = AnswerRecord(**{key: raw[key] for key in
                                   ("id", "source_dataset", "question", "answer", "split")})
            rows.append(safe)
            source_rows += 1
        sources.append({"id": source["id"], "path": str(path), "version": source["version"],
                        "sha256": digest(data), "records": source_rows})

    def rank(record: AnswerRecord) -> str:
        return digest(json.dumps([config["seed"], record.source_dataset, record.id],
                                 ensure_ascii=False).encode())

    groups = {}
    for record in rows:
        groups.setdefault(normalized_prompt(record.question), []).append(record)
    eligible = []
    excluded = []
    for prompt, group in sorted(groups.items()):
        if len({record.split for record in group}) > 1:
            # Exclude the entire group rather than reassign frozen source splits.
            excluded.extend({"id": record.id, "reason": "cross_split_prompt"} for record in group)
        else:
            ordered = sorted(group, key=lambda record: (rank(record), record.id))
            eligible.append(ordered[0])
            excluded.extend({"id": record.id, "reason": "duplicate_prompt",
                             "representative_id": ordered[0].id} for record in ordered[1:])
    selected = {"dev": [], "test": []}
    for source in sorted(config["sources"], key=lambda item: item["id"]):
        for split in selected:
            pool = sorted((record for record in eligible
                           if record.source_dataset == source["id"] and record.split == split),
                          key=lambda record: (rank(record), record.id))
            quota = source["quotas"][split]
            if len(pool) < quota:
                raise ValueError(f"{source['id']}/{split}: quota {quota}, only {len(pool)} unique eligible prompts")
            selected[split].extend(pool[:quota])
            excluded.extend({"id": record.id, "reason": "outside_quota"} for record in pool[quota:])
    for split in selected:
        selected[split].sort(key=lambda record: (rank(record), record.id))
    splits = {"seed": config["seed"], **{f"{split}_ids": [record.id for record in records]
                                        for split, records in selected.items()},
              "note": "Annotation candidates only; original source splits preserved."}
    manifest = {
        "schema_version": 1, "status": "awaiting_human_annotation",
        "selection_algorithm": "sha256-seed-source-id-v1",
        "prompt_grouping": "casefold-whitespace-v1",
        "config_sha256": digest(config_path.read_bytes()), "config": config,
        "sources": sources, "split_ids": splits,
        "selected": [{"id": record.id, "source_dataset": record.source_dataset,
                      "original_split": record.split, "split": split}
                     for split, records in selected.items() for record in records],
        "excluded": sorted(excluded, key=lambda item: item["id"]),
        "annotation_instructions": (
            "Review shared inventory first; then issue identical inventory to two independent "
            "annotators. Blank packets are not gold. Never copy source predictions as labels."
        ),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".annotation-", dir=output_dir.parent))
    try:
        (staging / "splits.json").write_text(_json(splits), encoding="utf-8")
        for split, records in selected.items():
            content = "".join(record.to_jsonl_line() + "\n" for record in records)
            for folder in ("inputs", "inventory", "annotator_a", "annotator_b"):
                target = staging / folder / f"{split}.jsonl"
                target.parent.mkdir(exist_ok=True)
                target.write_text(content, encoding="utf-8")
        # Input packet fields intentionally share the canonical AnswerRecord schema;
        # claims/evidence are empty and every gold/prediction field is null/default.
        manifest["artifact_sha256"] = {
            str(path.relative_to(staging)): digest(path.read_bytes())
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        (staging / "manifest.json").write_text(_json(manifest), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Output already exists: {output_dir}")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = prepare(args.config, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Annotation preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Blank annotation batch saved to {output}; human review and labels required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
