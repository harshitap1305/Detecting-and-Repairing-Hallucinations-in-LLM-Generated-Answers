"""Build Path C examples in two auditable stages.

1. ``candidates`` streams FEVER wiki sentences (or curated source JSONL) and
   writes deterministic corruptions. A nonempty holdout manifest is mandatory.
2. ``filter`` requires either explicit verifier verdicts or a working NLI model.
   Only verified rows become train/dev/test examples; each split is by article.

Verdict JSONL format: {"id": "...", "original_label": "Supported",
"candidate_label": "Contradicted"}. The candidate file is suitable for an
external verifier or human audit before running the filter stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.repair_data import (
    filter_verified_candidates, generate_candidates, iter_source_jsonl,
    iter_wiki_sentences, load_holdouts, write_jsonl,
)
from src.training.repair_training import assert_disjoint_splits


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def _verdict_lookup(path: Path):
    verdicts = {}
    for row in _read_jsonl(path):
        row_id = str(row.get("id", ""))
        if not row_id or row_id in verdicts:
            raise ValueError(f"Missing or duplicate verdict id: {row_id!r}")
        verdicts[row_id] = (row.get("original_label"), row.get("candidate_label"))

    def check(candidate):
        if candidate["id"] not in verdicts:
            raise ValueError(f"No verifier verdict for candidate {candidate['id']}")
        return verdicts[candidate["id"]]

    return check


def _nli_verifier(model_name: str, verdicts_out: Path | None = None):
    from src.schema import Claim, EvidencePassage
    from src.verify import _get_nli_model, verify_claim_nli

    if _get_nli_model(model_name) is None:
        raise RuntimeError(f"Could not load NLI model {model_name}; refusing heuristic labels")
    original_cache = {}
    previous = {}
    if verdicts_out and verdicts_out.exists():
        for row in _read_jsonl(verdicts_out):
            row_id = str(row.get("id", ""))
            if not row_id or row_id in previous:
                raise ValueError(f"Missing or duplicate cached verdict id: {row_id!r}")
            previous[row_id] = row
    if verdicts_out:
        verdicts_out.parent.mkdir(parents=True, exist_ok=True)

    def check(candidate):
        fingerprint = hashlib.sha256(json.dumps({
            key: candidate[key] for key in ("original_sentence", "sentence", "evidence")
        }, sort_keys=True).encode("utf-8")).hexdigest()
        if candidate["id"] in previous:
            row = previous[candidate["id"]]
            if row.get("candidate_sha256") != fingerprint or row.get("model_name") != model_name:
                raise ValueError(f"Stale cached verdict for candidate {candidate['id']}")
            return row["original_label"], row["candidate_label"]
        evidence = [EvidencePassage(
            doc_id=f"{candidate['id']}:e{i}", title=e["title"], text=e["text"]
        ) for i, e in enumerate(candidate["evidence"])]
        key = (candidate["source_id"], model_name)
        if key not in original_cache:
            original = Claim(id=candidate["id"] + ":original", text=candidate["original_sentence"],
                             sentence_id=0, evidence=evidence)
            original_cache[key] = verify_claim_nli(original, model_name=model_name).label
        changed = Claim(id=candidate["id"], text=candidate["sentence"],
                        sentence_id=0, evidence=evidence)
        label = verify_claim_nli(changed, model_name=model_name).label
        verdict = (original_cache[key], label)
        if verdicts_out:
            with verdicts_out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": candidate["id"],
                                         "candidate_sha256": fingerprint,
                                         "model_name": model_name,
                                         "original_label": verdict[0],
                                         "candidate_label": verdict[1]}, sort_keys=True) + "\n")
        return verdict

    return check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("candidates", help="generate unverified examples")
    source = gen.add_mutually_exclusive_group(required=True)
    source.add_argument("--wiki-pages-dir", type=Path)
    source.add_argument("--sources", type=Path, help="JSONL with article_id, sentence, evidence")
    gen.add_argument("--holdout-manifest", type=Path, required=True,
                     help="JSON lists: article_ids, entities, prompt_ids, prompts")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--max-sources", type=int, default=None)

    filt = sub.add_parser("filter", help="filter by verifier verdicts into train/dev/test")
    filt.add_argument("--candidates", type=Path, required=True)
    filt.add_argument("--output-dir", type=Path, required=True)
    source = filt.add_mutually_exclusive_group(required=True)
    source.add_argument("--verdicts", type=Path, help="audited candidate verdict JSONL")
    source.add_argument("--nli-model", help="run the existing NLI verifier; downloads model")
    filt.add_argument("--verdicts-out", type=Path,
                      help="append/resume NLI verdicts for long Colab runs")
    filt.add_argument("--unchanged-fraction", type=float, default=0.2)
    filt.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.mode == "candidates":
        if args.max_sources is not None and args.max_sources < 1:
            parser.error("--max-sources must be positive")
        holdouts = load_holdouts(args.holdout_manifest)
        if args.wiki_pages_dir and not (holdouts["article_ids"] or holdouts["entities"]):
            parser.error(
                "--wiki-pages-dir requires article_ids or entities in the holdout manifest; "
                "wiki sources do not carry prompt IDs or prompts"
            )
        sources = (iter_wiki_sentences(args.wiki_pages_dir) if args.wiki_pages_dir
                   else iter_source_jsonl(args.sources))
        n = write_jsonl(generate_candidates(sources, holdouts, args.seed, args.max_sources), args.output)
        print(f"Wrote {n} unverified candidate rows to {args.output}")
        return 0

    if args.verdicts and args.verdicts_out:
        parser.error("--verdicts-out applies only with --nli-model")
    verifier = (_verdict_lookup(args.verdicts) if args.verdicts
                else _nli_verifier(args.nli_model, args.verdicts_out))
    examples, counts = filter_verified_candidates(
        _read_jsonl(args.candidates), verifier, args.unchanged_fraction, args.seed
    )
    assert_disjoint_splits(*(examples.get(split, []) for split in ("train", "dev", "test")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        write_jsonl(examples.get(split, []), args.output_dir / f"repair_{split}.jsonl")
    manifest = {
        "seed": args.seed, "split_unit": "normalized Wikipedia article ID",
        "verification": str(args.verdicts) if args.verdicts else args.nli_model,
        "nli_verdicts_file": str(args.verdicts_out) if args.verdicts_out else None,
        "unchanged_fraction_requested": args.unchanged_fraction,
        "counts": dict(counts),
    }
    (args.output_dir / "repair_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
