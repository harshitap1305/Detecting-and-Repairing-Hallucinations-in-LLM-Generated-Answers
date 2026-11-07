"""Prepare an article-disjoint FEVER official-dev subset for final evaluation.

Use official ``shared_task_dev.jsonl`` and retrieval produced for those claims
by ``retrieve_fever_evidence.py``. Rows sharing an article or explicit entity
with either the training or checkpoint-selection split are dropped and counted.
This is a leakage-controlled subset, not the complete official benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verifier_data import iter_jsonl, normalize_title, prepare_examples


def _keys(rows):
    articles = {normalize_title(title) for row in rows for title in row["article_ids"]}
    entities = {normalize_title(row["entity_key"]) for row in rows if row.get("entity_key")}
    return articles, entities


def filter_heldout(rows: list[dict], train_rows: list[dict], dev_rows: list[dict]
                   ) -> tuple[list[dict], int]:
    seen_articles, seen_entities = _keys(train_rows + dev_rows)
    accepted = []
    dropped = 0
    for row in rows:
        articles, entities = _keys([row])
        if articles & seen_articles or entities & seen_entities:
            dropped += 1
        else:
            accepted.append(row)
    return accepted, dropped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-dev", type=Path, required=True)
    parser.add_argument("--wiki-pages", type=Path, required=True)
    parser.add_argument("--retrieved", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True,
                        help="Internal checkpoint-selection dev split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-claims", type=int)
    parser.add_argument("--max-passages", type=int, default=5)
    parser.add_argument("--max-noise-passages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir / "eval.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"Holdout output already exists in {args.output_dir}")
    # The preparation function makes two disjoint halves internally; joining
    # them recovers all usable official-dev rows without training on either.
    part_a, part_b, raw_manifest = prepare_examples(
        args.fever_dev, args.wiki_pages, args.retrieved,
        max_claims=args.max_claims, max_passages=args.max_passages,
        max_noise_passages=args.max_noise_passages, seed=args.seed,
    )
    prepared = part_a + part_b
    train_rows = list(iter_jsonl(args.train_file))
    dev_rows = list(iter_jsonl(args.dev_file))
    heldout, dropped = filter_heldout(prepared, train_rows, dev_rows)
    if not heldout:
        raise ValueError("No article-disjoint FEVER dev examples remain")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in heldout:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"source": "FEVER official dev, filtered to article/entity-disjoint subset",
                "seed": args.seed, "raw_usable": len(prepared),
                "overlap_dropped": dropped, "heldout_count": len(heldout),
                "labels": dict(Counter(row["label_name"] for row in heldout)),
                "source_skips": raw_manifest["skipped"],
                "inputs": {"fever_dev": str(args.fever_dev), "wiki_pages": str(args.wiki_pages),
                           "retrieved": str(args.retrieved), "train": str(args.train_file),
                           "internal_dev": str(args.dev_file)}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "heldout_count": len(heldout),
                      "overlap_dropped": dropped, "labels": manifest["labels"]}, indent=2))


if __name__ == "__main__":
    main()
