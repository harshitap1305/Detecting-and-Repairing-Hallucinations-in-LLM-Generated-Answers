"""Run the project's retriever on raw FEVER claims for Path A preparation.

This creates the label-independent retrieval JSONL expected by
``prepare_fever_verifier.py``. Build a Wikipedia index with
``scripts/build_index.py`` for the in-memory BM25 path, or
``scripts/build_fever_fts.py`` for a streaming full-corpus SQLite FTS5 path.
Use ``--max-claims`` for a bounded pilot. All claim labels are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verifier_data import iter_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _read_existing_ids(path: Path, top_k: int) -> list[str]:
    if path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                raise ValueError(f"Retrieval output ends in a partial row: {path}")
    seen: set[str] = set()
    ids: list[str] = []
    for row in iter_jsonl(path):
        claim_id = row.get("claim_id")
        key = str(claim_id) if claim_id is not None else ""
        passages = row.get("passages")
        if (not key or key in seen or not isinstance(passages, list)
                or len(passages) > top_k
                or any(not isinstance(p, dict) or not isinstance(p.get("title"), str)
                       or not isinstance(p.get("text"), str) for p in passages)):
            raise ValueError(f"Invalid existing retrieval row for claim {key!r}")
        seen.add(key)
        ids.append(key)
    return ids


def _run_fts(args: argparse.Namespace) -> dict:
    from scripts.build_fever_fts import FtsRetriever

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    output = args.output
    meta_path = _manifest_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FtsRetriever(Path(args.index_path)) as retriever:
        manifest = {
            "method": "fts5",
            "claims_sha256": _sha256(args.fever_train),
            "index_source_sha256": retriever.info["source_sha256"],
            "index_passage_count": int(retriever.info["passage_count"]),
            "top_k": args.top_k,
            "max_claims": args.max_claims,
            "query_version": 1,
        }
        existing_ids: list[str] = []
        if args.resume and output.exists():
            if not meta_path.is_file():
                raise ValueError(f"Cannot resume without retrieval manifest: {meta_path}")
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in manifest.items()):
                raise ValueError("Resume input/index/settings differ from retrieval manifest")
            existing_ids = _read_existing_ids(output, args.top_k)
        else:
            if (output.exists() or meta_path.exists()) and not args.overwrite:
                raise FileExistsError(f"Retrieval output or manifest already exists: {output}")
            _write_manifest(meta_path, {**manifest, "complete": False, "retrieved_rows": 0})
            output.write_text("", encoding="utf-8")

        seen: set[str] = set()
        count = 0
        with output.open("a", encoding="utf-8") as stream:
            for row in iter_jsonl(args.fever_train):
                if args.max_claims is not None and count >= args.max_claims:
                    break
                claim_id, claim = row.get("id"), row.get("claim")
                if claim_id is None or not isinstance(claim, str) or not claim.strip():
                    raise ValueError(f"Invalid FEVER claim id/text: {claim_id!r}")
                key = str(claim_id)
                if key in seen:
                    raise ValueError(f"Duplicate FEVER claim id: {key}")
                seen.add(key)
                if count < len(existing_ids):
                    if existing_ids[count] != key:
                        raise ValueError(
                            f"Existing retrieval prefix differs at row {count + 1}"
                        )
                else:
                    passages = retriever.retrieve(claim, top_k=args.top_k)
                    stream.write(json.dumps({
                        "claim_id": claim_id,
                        "passages": [asdict(p) for p in passages],
                    }, ensure_ascii=False) + "\n")
                    stream.flush()
                count += 1
        if len(existing_ids) > count:
            raise ValueError("Existing retrieval has more rows than requested claims")
        _write_manifest(meta_path, {**manifest, "complete": True,
                                    "retrieved_rows": count})
        return {"claims": count, "resumed": len(existing_ids), "method": "fts5",
                "top_k": args.top_k, "index_passages": manifest["index_passage_count"],
                "output": str(output), "manifest": str(meta_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-train", type=Path, required=True)
    parser.add_argument("--index-path", help="BM25 pickle or SQLite FTS5 index")
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid", "fts5"),
                        default="bm25")
    parser.add_argument("--faiss-index-path")
    parser.add_argument("--passages-path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-claims", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a validated FTS5 output prefix")
    args = parser.parse_args(argv)
    if args.top_k <= 0 or (args.max_claims is not None and args.max_claims <= 0):
        parser.error("top-k and max-claims must be positive")
    if args.method in ("dense", "hybrid") and (not args.faiss_index_path or not args.passages_path):
        parser.error("dense/hybrid requires --faiss-index-path and --passages-path")
    if args.method in ("bm25", "hybrid", "fts5") and not args.index_path:
        parser.error("bm25/hybrid/fts5 requires --index-path")
    if args.resume and args.method != "fts5":
        parser.error("--resume is currently supported for --method fts5 only")
    required_paths = [args.fever_train]
    if args.method in ("bm25", "hybrid", "fts5"):
        required_paths.append(Path(args.index_path))
    if args.method in ("dense", "hybrid"):
        required_paths.extend((Path(args.faiss_index_path), Path(args.passages_path)))
    for path in required_paths:
        if not path.is_file():
            parser.error(f"Required input file not found: {path}")
    if args.method == "fts5":
        print(json.dumps(_run_fts(args), indent=2))
        return
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
