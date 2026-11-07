"""Run the project's retriever on raw FEVER claims for Path A preparation.

This creates the label-independent retrieval JSONL expected by
``prepare_fever_verifier.py``. Build a Wikipedia index with
``scripts/build_index.py`` first. Use ``--max-claims`` for a bounded pilot.
All claim labels are ignored while retrieving.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verifier_data import iter_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-train", type=Path, required=True)
    parser.add_argument("--index-path", help="BM25 index pickle")
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--faiss-index-path")
    parser.add_argument("--passages-path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-claims", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k <= 0 or (args.max_claims is not None and args.max_claims <= 0):
        parser.error("top-k and max-claims must be positive")
    if args.method in ("dense", "hybrid") and (not args.faiss_index_path or not args.passages_path):
        parser.error("dense/hybrid requires --faiss-index-path and --passages-path")
    if args.method in ("bm25", "hybrid") and not args.index_path:
        parser.error("bm25/hybrid requires --index-path")
    required_paths = [args.fever_train]
    if args.method in ("bm25", "hybrid"):
        required_paths.append(Path(args.index_path))
    if args.method in ("dense", "hybrid"):
        required_paths.extend((Path(args.faiss_index_path), Path(args.passages_path)))
    for path in required_paths:
        if not path.is_file():
            parser.error(f"Required input file not found: {path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)

    from src.retrieve import retrieve_bm25, retrieve_dense, retrieve_hybrid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for row in iter_jsonl(args.fever_train):
            claim_id, claim = row.get("id"), row.get("claim")
            if claim_id is None or not isinstance(claim, str) or not claim.strip():
                raise ValueError(f"Invalid FEVER claim id/text: {claim_id!r}")
            if args.method == "bm25":
                passages = retrieve_bm25(claim, args.index_path, top_k=args.top_k)
            elif args.method == "dense":
                passages = retrieve_dense(claim, args.faiss_index_path, args.passages_path,
                                          top_k=args.top_k)
            else:
                passages = retrieve_hybrid(claim, args.index_path, args.faiss_index_path,
                                           args.passages_path, top_k=args.top_k)
            stream.write(json.dumps({"claim_id": claim_id,
                                     "passages": [asdict(p) for p in passages]},
                                    ensure_ascii=False) + "\n")
            count += 1
            if args.max_claims is not None and count >= args.max_claims:
                break
    print(json.dumps({"claims": count, "method": args.method, "top_k": args.top_k,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
