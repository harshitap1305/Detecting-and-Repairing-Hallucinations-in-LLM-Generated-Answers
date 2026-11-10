"""Convert a pinned local HaluEval checkout into unlabeled annotation candidates.

One variant per normalized question; balanced upstream origins within each split.
Upstream designations are provenance, never human factual labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import AnswerRecord
from src.experiments.artifacts import canonical, file_hash

CONVERSION = "halueval-prompt-mix-v1"
UPSTREAM = "https://github.com/RUCAIBox/HaluEval"


def _git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True,
                            env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"})
    if result.returncode:
        raise ValueError(f"Local Git provenance check failed: {result.stderr.strip()}")
    return result.stdout.strip()


def prepare_source(repo: Path, revision: str, output_dir: Path, seed=42):
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Supply the full lowercase upstream commit SHA")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if _git(repo, "rev-parse", "HEAD") != revision:
        raise ValueError("Checkout HEAD differs from the pinned revision")
    for name in ("data/qa_data.json", "README.md", "LICENSE"):
        expected = _git(repo, "rev-parse", f"{revision}:{name}")
        if _git(repo, "hash-object", str((repo / name).resolve())) != expected:
            raise ValueError(f"Source differs from pinned Git object: {name}")
    source = repo / "data/qa_data.json"
    groups, exclusions = {}, []
    count = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("question", "right_answer", "hallucinated_answer", "knowledge")):
            raise ValueError(f"Invalid HaluEval QA row {line_number}")
        count += 1
        question = " ".join(row["question"].casefold().split())
        if question in groups:
            exclusions.append({"source_line": line_number, "reason": "duplicate_prompt",
                               "representative_line": groups[question][0]})
        else:
            groups[question] = (line_number, row)
    if not groups:
        raise ValueError("Source is empty")
    pools = {"dev": [], "test": []}
    for question, (line_number, row) in groups.items():
        # Match the existing loader's prompt-disjoint, approximately 20% test rule.
        bucket = hashlib.sha256(f"{seed}:{question}".encode()).digest()[0]
        split = "test" if bucket < 51 else "dev"
        rank = hashlib.sha256(f"{seed}:variant:{question}".encode()).hexdigest()
        pools[split].append((rank, line_number, row))
    records, lineage = [], []
    for split, pool in pools.items():
        for index, (_, line_number, row) in enumerate(sorted(pool)):
            field = "right_answer" if index % 2 == 0 else "hallucinated_answer"
            record = AnswerRecord(id=f"halueval_qa_{line_number:05d}", source_dataset="halueval",
                                  split=split, question=row["question"], answer=row[field])
            records.append(record)
            lineage.append({"record_id": record.id, "source_line": line_number, "split": split,
                            "answer_field": field})
    records.sort(key=lambda record: record.id)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".halueval-", dir=output_dir.parent))
    try:
        (staging / "candidates.jsonl").write_bytes(b"".join(canonical(record.to_dict()) + b"\n" for record in records))
        raw = staging / "raw"
        raw.mkdir()
        for name in ("data/qa_data.json", "README.md", "LICENSE"):
            shutil.copyfile(repo / name, raw / Path(name).name)
        manifest = {"schema_version": 1, "status": "unlabeled_candidates", "upstream": UPSTREAM,
                    "revision": revision, "conversion": CONVERSION, "seed": seed,
                    "raw_rows": count, "unique_prompts": len(groups), "lineage": lineage,
                    "exclusions": exclusions, "split_counts": dict(Counter(record.split for record in records)),
                    "origin_counts": dict(Counter(item["answer_field"] for item in lineage)),
                    "note": "Origin fields are upstream provenance, not reviewed gold labels; assessors must verify all facts.",
                    "files_sha256": {path.relative_to(staging).as_posix(): file_hash(path)
                                     for path in sorted(staging.rglob("*")) if path.is_file()}}
        (staging / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        path = prepare_source(args.source_repo, args.revision, args.output_dir, args.seed)
    except (OSError, ValueError) as exc:
        print(f"Source preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Unlabeled candidates and upstream provenance saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
