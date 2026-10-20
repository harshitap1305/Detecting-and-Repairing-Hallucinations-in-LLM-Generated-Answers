"""
run_eval.py — Compute detection and repair metrics over pipeline output.

Usage:
    python scripts/run_eval.py \
        --results results/pipeline_output.jsonl \
        --gold data/gold/test.jsonl

Phase 8: full scientific metrics with bootstrap CIs, SCR progression, preservation, and lexical drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate_dataset
from src.schema import AnswerRecord


def load_jsonl(path: str) -> list[AnswerRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AnswerRecord.from_jsonl_line(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline results")
    parser.add_argument("--results", required=True, help="Path to pipeline output JSONL")
    parser.add_argument("--gold", default=None, help="Path to gold dataset JSONL (optional)")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file {results_path} does not exist.")
        sys.exit(1)

    records = load_jsonl(str(results_path))
    gold_records = load_jsonl(args.gold) if args.gold and Path(args.gold).exists() else None

    print(f"\n=======================================================")
    print(f" Scientific Evaluation Report: {len(records)} records analyzed")
    if gold_records:
        print(f" Gold Reference: {len(gold_records)} gold records")
    print(f"=======================================================")

    metrics = evaluate_dataset(records, gold_records=gold_records, n_bootstrap=args.n_bootstrap)

    print("\nClaim Statistics:")
    print(f"  Total claims:        {metrics['total_claims']}")
    print(f"  Check-worthy claims: {metrics['check_worthy_claims']}")
    print(f"  Verdicts:")
    for lbl, cnt in metrics["label_distribution"].items():
        pct = (cnt / metrics["check_worthy_claims"] * 100) if metrics["check_worthy_claims"] > 0 else 0
        print(f"    - {lbl:15s}: {cnt:5d} ({pct:5.1f}%)")

    scr = metrics["scr"]
    print("\nSupported Claim Ratio (SCR / FActScore) Progression [95% CI]:")
    print(f"  SCR BEFORE: {scr['before']['mean']:.4f} [{scr['before']['ci_lower']:.4f}, {scr['before']['ci_upper']:.4f}]")
    print(f"  SCR AFTER:  {scr['after']['mean']:.4f} [{scr['after']['ci_lower']:.4f}, {scr['after']['ci_upper']:.4f}]")
    print(f"  SCR DELTA:  {scr['delta']['mean']:+.4f} [{scr['delta']['ci_lower']:+.4f}, {scr['delta']['ci_upper']:+.4f}]")

    pres = metrics["preservation_rate"]
    oe = metrics["over_edit_rate"]
    print("\nPreservation & Localization Guard Performance [95% CI]:")
    print(f"  Preservation Rate: {pres['mean'] * 100:.2f}% [{pres['ci_lower'] * 100:.2f}%, {pres['ci_upper'] * 100:.2f}%]")
    print(f"  Over-edit Rate:    {oe['mean'] * 100:.2f}% [{oe['ci_lower'] * 100:.2f}%, {oe['ci_upper'] * 100:.2f}%]")

    lex = metrics["lexical_similarity"]
    print("\nLexical & Semantic Drift [95% CI]:")
    print(f"  ROUGE-L F1:              {lex['rouge_l_f1']['mean']:.4f} [{lex['rouge_l_f1']['ci_lower']:.4f}, {lex['rouge_l_f1']['ci_upper']:.4f}]")
    print(f"  Levenshtein Similarity:  {lex['levenshtein_similarity']['mean']:.4f} [{lex['levenshtein_similarity']['ci_lower']:.4f}, {lex['levenshtein_similarity']['ci_upper']:.4f}]")
    print(f"  Word Retention:          {lex['word_retention']['mean']:.4f} [{lex['word_retention']['ci_lower']:.4f}, {lex['word_retention']['ci_upper']:.4f}]")

    if metrics.get("detection_vs_gold"):
        det = metrics["detection_vs_gold"]
        print("\nClaim-level Detection vs. Gold:")
        print(f"  Precision: {det['precision']:.4f} | Recall: {det['recall']:.4f} | F1: {det['f1']:.4f} | Accuracy: {det['accuracy']:.4f}")

    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
