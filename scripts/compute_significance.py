"""
compute_significance.py — Run paired statistical significance testing and bootstrap CIs.

Usage:
    python scripts/compute_significance.py \
        --system results/pipeline_output.jsonl \
        --baseline results/baseline_b5.jsonl \
        --output results/significance_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import (
    bootstrap_confidence_interval,
    compute_preservation_rate,
    compute_rouge_l,
    compute_scr,
    paired_wilcoxon_test,
)
from src.schema import AnswerRecord


def load_records(path: str) -> Dict[str, AnswerRecord]:
    """Load JSONL into a dict keyed by record.id."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = AnswerRecord.from_jsonl_line(line)
                if rec.id in records:
                    raise ValueError(f"Duplicate record ID: {rec.id}")
                records[rec.id] = rec
    return records


def format_p_value(p: float) -> str:
    """Format p-value with scientific notation and significance stars."""
    if p < 0.001:
        stars = "*** (p < 0.001)"
    elif p < 0.01:
        stars = "** (p < 0.01)"
    elif p < 0.05:
        stars = "* (p < 0.05)"
    else:
        stars = "n.s. (not significant)"
    return f"{p:.4e} {stars}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired statistical tests between two systems")
    parser.add_argument("--system", required=True, help="Path to system output JSONL")
    parser.add_argument("--baseline", required=True, help="Path to baseline output JSONL")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    sys_records = load_records(args.system)
    base_records = load_records(args.baseline)

    if sys_records.keys() != base_records.keys():
        parser.error("System and baseline IDs must match exactly; partial intersections are invalid")
    for record_id in sys_records:
        if any(getattr(sys_records[record_id], field) != getattr(base_records[record_id], field)
               for field in ("source_dataset", "split", "question", "answer")):
            parser.error(f"Paired input mismatch: {record_id}")
    common_ids = sorted(sys_records)
    if not common_ids:
        print(f"Error: No overlapping record IDs found between {args.system} and {args.baseline}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" Statistical Significance Test: {len(common_ids)} Paired Records")
    print(f" System:   {args.system}")
    print(f" Baseline: {args.baseline}")
    print(f"=======================================================")

    metrics = {}

    # 1. SCR (Supported Claim Ratio)
    sys_scrs = [compute_scr(sys_records[i], use_after=True) for i in common_ids]
    base_scrs = [compute_scr(base_records[i], use_after=True) for i in common_ids]
    if any(value is None for value in sys_scrs + base_scrs):
        metrics["scr"] = {"status": "unavailable", "reason": "Undefined SCR in paired records",
                          "undefined_record_ids": [rid for rid, s, b in zip(common_ids, sys_scrs, base_scrs)
                                                   if s is None or b is None]}
    else:
        scr_wilcoxon = paired_wilcoxon_test(sys_scrs, base_scrs)
        scr_diff_mean, scr_diff_low, scr_diff_high = bootstrap_confidence_interval(
            [s - b for s, b in zip(sys_scrs, base_scrs)], n_bootstrap=args.n_bootstrap
        )
        metrics["scr"] = {
            "system_mean": sum(sys_scrs) / len(sys_scrs),
            "baseline_mean": sum(base_scrs) / len(base_scrs),
            "mean_difference": scr_diff_mean,
            "ci_95": [scr_diff_low, scr_diff_high],
            "wilcoxon_stat": scr_wilcoxon["statistic"],
            "p_value": scr_wilcoxon["p_value"],
            "significant": scr_wilcoxon["is_significant_p05"],
        }

    # 2. Preservation Rate
    sys_pres = [compute_preservation_rate(sys_records[i]) for i in common_ids]
    base_pres = [compute_preservation_rate(base_records[i]) for i in common_ids]
    pres_wilcoxon = paired_wilcoxon_test(sys_pres, base_pres)
    pres_diff_mean, pres_diff_low, pres_diff_high = bootstrap_confidence_interval(
        [s - b for s, b in zip(sys_pres, base_pres)], n_bootstrap=args.n_bootstrap
    )
    metrics["preservation"] = {
        "system_mean": sum(sys_pres) / len(sys_pres),
        "baseline_mean": sum(base_pres) / len(base_pres),
        "mean_difference": pres_diff_mean,
        "ci_95": [pres_diff_low, pres_diff_high],
        "wilcoxon_stat": pres_wilcoxon["statistic"],
        "p_value": pres_wilcoxon["p_value"],
        "significant": pres_wilcoxon["is_significant_p05"],
    }

    # 3. ROUGE-L F1 (Lexical similarity to original answer)
    sys_rouge = [compute_rouge_l(sys_records[i].repaired_answer if sys_records[i].repaired_answer is not None else sys_records[i].answer, sys_records[i].answer)["f1"] for i in common_ids]
    base_rouge = [compute_rouge_l(base_records[i].repaired_answer if base_records[i].repaired_answer is not None else base_records[i].answer, base_records[i].answer)["f1"] for i in common_ids]
    rouge_wilcoxon = paired_wilcoxon_test(sys_rouge, base_rouge)
    rouge_diff_mean, rouge_diff_low, rouge_diff_high = bootstrap_confidence_interval(
        [s - b for s, b in zip(sys_rouge, base_rouge)], n_bootstrap=args.n_bootstrap
    )
    metrics["rouge_l"] = {
        "system_mean": sum(sys_rouge) / len(sys_rouge),
        "baseline_mean": sum(base_rouge) / len(base_rouge),
        "mean_difference": rouge_diff_mean,
        "ci_95": [rouge_diff_low, rouge_diff_high],
        "wilcoxon_stat": rouge_wilcoxon["statistic"],
        "p_value": rouge_wilcoxon["p_value"],
        "significant": rouge_wilcoxon["is_significant_p05"],
    }

    from src.experiments.statistics import holm_adjust
    defined = [m for m in metrics.values() if m.get("status") != "unavailable"]
    for metric, adjusted in zip(defined, holm_adjust([m["p_value"] for m in defined])):
        metric["p_value_holm"] = adjusted
        metric["significant_holm_p05"] = adjusted < 0.05
    print("Legacy diagnostics only; not independent factual repair measurements.")
    # Display results table
    print(f"\n{'Metric':<18} | {'System':<8} | {'Baseline':<8} | {'Diff [95% CI]':<22} | {'Holm-adjusted p-value'}")
    print("-" * 80)
    for name, m in metrics.items():
        if m.get("status") == "unavailable":
            print(f"{name}: N/A — {m['reason']}")
            continue
        ci_str = f"{m['mean_difference']:+.3f} [{m['ci_95'][0]:+.3f}, {m['ci_95'][1]:+.3f}]"
        print(f"{name:<18} | {m['system_mean']:<8.4f} | {m['baseline_mean']:<8.4f} | {ci_str:<22} | {format_p_value(m['p_value_holm'])}")
    print("=" * 80 + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()

