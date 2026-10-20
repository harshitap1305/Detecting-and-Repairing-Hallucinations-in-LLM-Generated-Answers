"""
run_eval.py — Compute detection and repair metrics over pipeline output.

Usage:
    python scripts/run_eval.py \
        --results results/pipeline_output.jsonl \
        --gold data/gold/test.jsonl

Phase 8: full metrics with bootstrap CIs. Phase 3: reliability score delta and summary statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import AnswerRecord


def load_jsonl(path: str) -> list[AnswerRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AnswerRecord.from_jsonl_line(line))
    return records


def compute_reliability_delta(records: list[AnswerRecord]) -> dict:
    """Compute mean reliability score before and after repair."""
    before = [
        r.reliability_score_before
        for r in records
        if r.reliability_score_before is not None
    ]
    after = [
        r.reliability_score_after
        for r in records
        if r.reliability_score_after is not None
    ]
    if not before:
        return {"error": "No reliability scores found."}
    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after) if after else 0.0
    return {
        "n_evaluated": len(before),
        "mean_reliability_before": mean_before,
        "mean_reliability_after": mean_after,
        "mean_delta": mean_after - mean_before,
    }


def compute_claim_stats(records: list[AnswerRecord]) -> dict:
    """Compute summary statistics for claim verdicts."""
    total_claims = 0
    check_worthy = 0
    label_dist = {"Supported": 0, "Contradicted": 0, "Unverifiable": 0, "Unlabeled": 0}

    for rec in records:
        for c in rec.claims:
            total_claims += 1
            if c.check_worthy:
                check_worthy += 1
                lbl = c.label if c.label in label_dist else "Unlabeled"
                label_dist[lbl] += 1

    return {
        "total_claims": total_claims,
        "check_worthy_claims": check_worthy,
        "label_distribution": label_dist,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline results")
    parser.add_argument("--results", required=True, help="Path to pipeline output JSONL")
    parser.add_argument("--gold", default=None, help="Path to gold dataset JSONL (optional)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file {results_path} does not exist.")
        sys.exit(1)

    records = load_jsonl(str(results_path))
    print(f"\n=======================================================")
    print(f" Evaluation Report: {len(records)} records analyzed")
    print(f"=======================================================")

    claim_stats = compute_claim_stats(records)
    print("\nClaim Statistics:")
    print(f"  Total claims:        {claim_stats['total_claims']}")
    print(f"  Check-worthy claims: {claim_stats['check_worthy_claims']}")
    print(f"  Verdicts:")
    for lbl, cnt in claim_stats["label_distribution"].items():
        pct = (cnt / claim_stats["check_worthy_claims"] * 100) if claim_stats["check_worthy_claims"] > 0 else 0
        print(f"    - {lbl:15s}: {cnt:5d} ({pct:5.1f}%)")

    rel = compute_reliability_delta(records)
    print("\nReliability Score Progression:")
    if "error" in rel:
        print(f"  {rel['error']}")
    else:
        print(f"  Answers evaluated:   {rel['n_evaluated']}")
        print(f"  Mean score BEFORE:   {rel['mean_reliability_before']:.4f}")
        print(f"  Mean score AFTER:    {rel['mean_reliability_after']:.4f}")
        print(f"  Mean delta:          {rel['mean_delta']:+.4f}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()

