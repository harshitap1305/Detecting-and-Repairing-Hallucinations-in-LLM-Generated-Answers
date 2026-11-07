"""Evaluate a trained repair editor on held-out examples and optional baseline.

Baseline JSONL rows must have ``id`` and ``prediction`` for exactly the same
examples. Paired bootstrap intervals are by source article to avoid treating
multiple sentences from one article as independent observations.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.repair_training import (
    evaluate_predictions, file_sha256, load_examples, predict_rows, token_f1,
)


def _load_baseline(path: Path, rows: list[dict]) -> list[str]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item.get("id")
            if row_id in found:
                raise ValueError(f"Duplicate baseline id: {row_id}")
            found[row_id] = item.get("prediction", "")
    requested = {r["id"] for r in rows}
    if set(found) != requested:
        raise ValueError(f"Baseline IDs differ: missing={len(requested - set(found))}, extra={len(set(found) - requested)}")
    return [str(found[r["id"]]) for r in rows]


def _paired_ci(rows: list[dict], current: list[str], baseline: list[str],
               n_bootstrap: int, seed: int) -> dict:
    by_article = defaultdict(list)
    for index, row in enumerate(rows):
        by_article[row["article_id"]].append(index)
    groups = list(by_article.values())
    rng = random.Random(seed)
    samples = {"exact_match": [], "token_f1": []}
    for _ in range(n_bootstrap):
        indices = [i for _ in groups for i in groups[rng.randrange(len(groups))]]
        exact = sum((current[i].strip() == rows[i]["target"].strip()) -
                    (baseline[i].strip() == rows[i]["target"].strip()) for i in indices) / len(indices)
        f1 = sum(token_f1(rows[i]["target"], current[i]) -
                 token_f1(rows[i]["target"], baseline[i]) for i in indices) / len(indices)
        samples["exact_match"].append(exact)
        samples["token_f1"].append(f1)
    return {name: [sorted(values)[int(0.025 * (n_bootstrap - 1))],
                   sorted(values)[int(0.975 * (n_bootstrap - 1))]]
            for name, values in samples.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", default="rewrite+hedge")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_dir():
        parser.error(f"Missing checkpoint directory: {args.checkpoint}")
    if args.n_bootstrap < 100 or args.batch_size < 1:
        parser.error("n-bootstrap must be >=100 and batch-size must be positive")
    if args.metrics_out == args.predictions_out:
        parser.error("metrics and predictions output paths must differ")
    rows = load_examples(args.data, args.split)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    predictions, seconds = predict_rows(
        model, tokenizer, rows, device, args.batch_size, args.max_source_length,
        args.max_new_tokens, args.policy)
    metrics = evaluate_predictions(rows, predictions)
    metrics.update({"split": args.split, "checkpoint": str(args.checkpoint.resolve()),
                    "data_sha256": file_sha256(args.data), "device": device,
                    "generation_seconds": seconds,
                    "seconds_per_example": seconds / len(rows)})
    if args.baseline_predictions:
        baseline = _load_baseline(args.baseline_predictions, rows)
        base_metrics = evaluate_predictions(rows, baseline)
        metrics["baseline"] = base_metrics
        metrics["delta_vs_baseline"] = {
            "exact_match": metrics["exact_match"] - base_metrics["exact_match"],
            "token_f1": metrics["token_f1"] - base_metrics["token_f1"],
            "paired_bootstrap_95_ci": _paired_ci(rows, predictions, baseline,
                                                   args.n_bootstrap, args.seed),
            "resampling_unit": "article_id",
        }
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(rows, predictions):
            handle.write(json.dumps({"id": row["id"], "article_id": row["article_id"],
                                     "prediction": prediction, "target": row["target"]},
                                    ensure_ascii=False) + "\n")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
