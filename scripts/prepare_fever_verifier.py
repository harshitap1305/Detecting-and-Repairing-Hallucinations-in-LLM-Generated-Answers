"""Create article-disjoint FEVER train/dev JSONL for the trained verifier.

Example (run from the repository root)::

    python -m scripts.prepare_fever_verifier \
      --fever-train data/raw/fever/train.jsonl \
      --wiki-pages data/raw/fever/wiki-pages.zip \
      --retrieved data/raw/fever/retrieved_train.jsonl \
      --output-dir data/processed/verifier

``retrieved_train.jsonl`` must contain one object per FEVER claim ID with
``claim_id`` and a ranked ``passages`` list of ``{title, text, score?}``.
It must come from the project's retriever, run without looking at labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verifier_data import prepare_examples, write_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-train", type=Path, required=True,
                        help="Official FEVER train.jsonl; never pass an evaluation/test split")
    parser.add_argument("--wiki-pages", type=Path, required=True,
                        help="FEVER wiki-pages directory, JSONL file, or original zip")
    parser.add_argument("--retrieved", type=Path, required=True,
                        help="Ranked, label-independent retrieval JSONL for FEVER train claims")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-claims", type=int,
                        help="Use only the first N claims for a smoke run")
    parser.add_argument("--max-passages", type=int, default=5)
    parser.add_argument("--max-noise-passages", type=int, default=2)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-keys", type=Path,
                        help="Newline-delimited held-out test entities/article titles to exclude")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    excluded_keys = None
    if args.exclude_keys:
        excluded_keys = {line.strip() for line in args.exclude_keys.read_text(encoding="utf-8").splitlines()
                         if line.strip() and not line.lstrip().startswith("#")}
        if not excluded_keys:
            parser.error("--exclude-keys file is empty")
    train, dev, manifest = prepare_examples(
        args.fever_train, args.wiki_pages, args.retrieved,
        max_claims=args.max_claims, max_passages=args.max_passages,
        max_noise_passages=args.max_noise_passages,
        dev_fraction=args.dev_fraction, seed=args.seed, excluded_keys=excluded_keys,
    )
    write_dataset(args.output_dir, train, dev, manifest, overwrite=args.overwrite)
    print(json.dumps({"output_dir": str(args.output_dir), "train": manifest["train"],
                      "dev": manifest["dev"], "skipped": manifest["skipped"],
                      "article_overlap": 0}, indent=2))
    if not excluded_keys:
        print("WARNING: no held-out entity/article exclusion list was provided; "
              "treat this as a pilot, not a final leakage-controlled experiment.")


if __name__ == "__main__":
    main()
