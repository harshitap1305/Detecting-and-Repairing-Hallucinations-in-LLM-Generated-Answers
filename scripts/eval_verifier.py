"""Evaluate a trained verifier on held-out evidence/claim JSONL.

Accepts the same row schema as ``prepare_fever_verifier.py``. Use a true
held-out file (FEVER official dev or gold-test) for final reported numbers;
the generated internal dev split is for checkpoint selection. Baseline files
may be added as ``--baseline nli=path/to/predictions.jsonl`` and must contain
one ``{"id": ..., "prediction": "Supported|Contradicted|Unverifiable"}``
object per evaluation sample. All systems are scored on identical IDs and
paired bootstrap samples. No result is invented when a baseline is absent.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verifier_data import LABELS, iter_jsonl
from src.training.verifier_model import predict_probabilities


def read_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No evaluation rows in {path}")
    ids: set[str] = set()
    for row in rows:
        sample_id = row.get("id")
        label = row.get("label")
        if not isinstance(sample_id, str) or sample_id in ids:
            raise ValueError(f"Invalid or duplicate evaluation id: {sample_id!r}")
        ids.add(sample_id)
        if (not isinstance(row.get("premise"), str) or not row["premise"].strip()
                or not isinstance(row.get("claim"), str) or not row["claim"].strip()):
            raise ValueError(f"Empty premise/claim in {sample_id}")
        if type(label) is not int or not 0 <= label < len(LABELS):
            raise ValueError(f"Invalid gold label in {sample_id}: {label!r}")
    return rows


def _binary_f1(gold: list[int], predicted: list[int]) -> float:
    true_positive = sum(g != 0 and p != 0 for g, p in zip(gold, predicted))
    false_positive = sum(g == 0 and p != 0 for g, p in zip(gold, predicted))
    false_negative = sum(g != 0 and p == 0 for g, p in zip(gold, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def classification_metrics(gold: list[int], predicted: list[int],
                           probabilities: list[list[float]] | None = None) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Gold and prediction lists must have equal nonzero length")
    confusion = [[0] * 3 for _ in range(3)]
    for actual, guess in zip(gold, predicted):
        if actual not in range(3) or guess not in range(3):
            raise ValueError("Labels must be indices 0, 1 or 2")
        confusion[actual][guess] += 1
    by_class: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for i, label in enumerate(LABELS):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(3) if j != i)
        fn = sum(confusion[i][j] for j in range(3) if j != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        by_class[label] = {"support": sum(confusion[i]), "precision": precision,
                           "recall": recall, "f1": f1}
        f1s.append(f1)
    result: dict[str, Any] = {
        "count": len(gold),
        "accuracy": sum(confusion[i][i] for i in range(3)) / len(gold),
        "macro_f1": sum(f1s) / 3,
        "hallucinated_f1": _binary_f1(gold, predicted),
        "label_order": list(LABELS),
        "confusion_matrix": confusion,
        "per_class": by_class,
    }
    if probabilities is not None:
        if len(probabilities) != len(gold):
            raise ValueError("Probability row count does not match gold")
        bins = [dict(n=0, correct=0, confidence=0.0) for _ in range(10)]
        for actual, guess, scores in zip(gold, predicted, probabilities):
            if len(scores) != 3 or any(not 0 <= score <= 1 for score in scores):
                raise ValueError("Each probability row must contain three values in [0, 1]")
            confidence = max(scores)
            bucket = min(9, int(confidence * 10))
            bins[bucket]["n"] += 1
            bins[bucket]["correct"] += int(actual == guess)
            bins[bucket]["confidence"] += confidence
        result["ece_10_bins"] = sum(
            abs(part["correct"] / part["n"] - part["confidence"] / part["n"])
            * part["n"] / len(gold) for part in bins if part["n"]
        )
        result["calibration_bins"] = [
            {"count": part["n"],
             "accuracy": part["correct"] / part["n"] if part["n"] else None,
             "mean_confidence": part["confidence"] / part["n"] if part["n"] else None}
            for part in bins
        ]
    return result


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    left = int(position)
    right = min(len(ordered) - 1, left + 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def bootstrap_intervals(gold: list[int], systems: dict[str, list[int]],
                        *, iterations: int, seed: int) -> dict[str, Any]:
    """Paired percentile CIs, resampling the same rows for every system."""
    if iterations <= 0:
        return {}
    n = len(gold)
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {name: [] for name in systems}
    deltas: dict[str, list[float]] = {name: [] for name in systems if name != "trained"}
    for _ in range(iterations):
        picks = [rng.randrange(n) for _ in range(n)]
        sampled_gold = [gold[i] for i in picks]
        scores = {}
        for name, predictions in systems.items():
            sampled_pred = [predictions[i] for i in picks]
            scores[name] = classification_metrics(sampled_gold, sampled_pred)["macro_f1"]
            distributions[name].append(scores[name])
        for name in deltas:
            deltas[name].append(scores["trained"] - scores[name])
    return {"iterations": iterations, "seed": seed,
            "macro_f1_ci95": {name: [_percentile(values, 0.025), _percentile(values, 0.975)]
                              for name, values in distributions.items()},
            "trained_minus_baseline_macro_f1_ci95": {
                name: [_percentile(values, 0.025), _percentile(values, 0.975)]
                for name, values in deltas.items()}}


def read_baseline(path: Path, expected_ids: set[str]) -> dict[str, int]:
    predictions: dict[str, int] = {}
    for row in iter_jsonl(path):
        sample_id = row.get("id")
        label = row.get("prediction")
        if not isinstance(sample_id, str) or sample_id in predictions:
            raise ValueError(f"Invalid or duplicate baseline id: {sample_id!r}")
        if label not in LABELS:
            raise ValueError(f"Invalid baseline prediction for {sample_id}: {label!r}")
        predictions[sample_id] = LABELS.index(label)
    if set(predictions) != expected_ids:
        missing = expected_ids - set(predictions)
        extra = set(predictions) - expected_ids
        raise ValueError(f"Baseline IDs differ: {len(missing)} missing, {len(extra)} extra")
    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", action="append", default=[], metavar="NAME=JSONL")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.bootstrap < 0:
        parser.error("batch-size must be positive and bootstrap nonnegative")
    rows = read_eval_rows(args.eval_file)
    baseline_paths: dict[str, Path] = {}
    for item in args.baseline:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name == "trained" or name in baseline_paths:
            parser.error(f"Invalid baseline specification: {item!r}")
        baseline_paths[name] = Path(path)
    output_metrics = args.output_dir / "metrics.json"
    output_predictions = args.output_dir / "predictions.jsonl"
    if not args.overwrite and (output_metrics.exists() or output_predictions.exists()):
        raise FileExistsError(f"Evaluation output exists in {args.output_dir}")

    pairs = [(row["premise"], row["claim"]) for row in rows]
    probabilities = predict_probabilities(pairs, args.checkpoint, batch_size=args.batch_size)
    gold = [row["label"] for row in rows]
    trained = [max(range(3), key=scores.__getitem__) for scores in probabilities]
    systems = {"trained": trained}
    expected_ids = {row["id"] for row in rows}
    for name, path in baseline_paths.items():
        predictions = read_baseline(path, expected_ids)
        systems[name] = [predictions[row["id"]] for row in rows]
    report = {"eval_file": str(args.eval_file), "checkpoint": args.checkpoint,
              "systems": {name: classification_metrics(gold, prediction,
                                                        probabilities if name == "trained" else None)
                          for name, prediction in systems.items()},
              "bootstrap": bootstrap_intervals(gold, systems, iterations=args.bootstrap,
                                               seed=args.seed)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8") as stream:
        for row, label, scores in zip(rows, trained, probabilities):
            stream.write(json.dumps({"id": row["id"], "gold": LABELS[row["label"]],
                                     "prediction": LABELS[label], "probabilities": scores}) + "\n")
    output_metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_file": str(output_metrics),
                      "systems": {name: {key: value for key, value in metrics.items()
                                         if key in ("accuracy", "macro_f1", "hallucinated_f1")}
                                  for name, metrics in report["systems"].items()}}, indent=2))


if __name__ == "__main__":
    main()
