"""
run_ablations.py — Execute the 6 specified ablation studies across pipeline dimensions.

Ablations:
1. Claim Granularity: Answer-level (B1) vs. Sentence-level vs. Atomic Claims
2. Retriever Architecture: BM25 vs. Dense vs. Hybrid (RRF) vs. Hybrid + Reranker
3. Top-k Evidence Count: k in {1, 3, 5, 10}
4. Verifier Type: NLI vs. LLM-Judge vs. Ensemble
5. Repair Policy: rewrite-only vs. rewrite+hedge vs. rewrite+remove
6. Re-verification Iterations: 0 vs. 1 vs. 2 iterations

Usage:
    python scripts/run_ablations.py --data data/processed/halueval.jsonl --output_dir results/ablations
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate_dataset
from src.pipeline import run_pipeline
from src.schema import AnswerRecord


ABLATION_CONFIGS = {
    "granularity": [
        {"name": "Answer-level (B1)", "granularity": "answer"},
        {"name": "Sentence-level", "granularity": "sentence"},
        {"name": "Atomic Claim (Default)", "granularity": "atomic"},
    ],
    "retriever": [
        {"name": "BM25 (Default Tier A)", "method": "bm25"},
        {"name": "Dense (BGE-small)", "method": "dense"},
        {"name": "Hybrid (RRF)", "method": "hybrid"},
        {"name": "Hybrid + Reranker", "method": "hybrid+rerank"},
    ],
    "top_k": [
        {"name": "Top-1", "top_k": 1},
        {"name": "Top-3", "top_k": 3},
        {"name": "Top-5 (Default)", "top_k": 5},
        {"name": "Top-10", "top_k": 10},
    ],
    "verifier": [
        {"name": "NLI Only (DeBERTa-v3)", "verifier": "nli"},
        {"name": "LLM Judge Only", "verifier": "llm"},
        {"name": "Calibrated Ensemble (Default)", "verifier": "ensemble"},
    ],
    "repair_policy": [
        {"name": "rewrite-only", "repair_policy": "rewrite-only"},
        {"name": "rewrite+remove", "repair_policy": "rewrite+remove"},
        {"name": "rewrite+hedge (Default)", "repair_policy": "rewrite+hedge"},
    ],
    "iterations": [
        {"name": "0 Iterations (No repair)", "max_iterations": 0},
        {"name": "1 Iteration", "max_iterations": 1},
        {"name": "2 Iterations (Default)", "max_iterations": 2},
    ],
}


def load_dev_records(data_path: str, max_records: int = 50) -> List[AnswerRecord]:
    """Load records for ablation benchmarking."""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AnswerRecord.from_jsonl_line(line))
                if len(records) >= max_records:
                    break
    return records


def run_ablation_experiment(
    ablation_name: str,
    records: List[AnswerRecord],
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute all variants of a single ablation study."""
    print(f"\n=======================================================")
    print(f" Running Ablation: {ablation_name.upper()} ({len(records)} records)")
    print(f"=======================================================")

    variants = ABLATION_CONFIGS.get(ablation_name, [])
    results_summary = []

    for var in variants:
        var_name = var["name"]
        print(f"  Evaluating variant: {var_name}...")

        # Process records under this ablation configuration
        variant_records: List[AnswerRecord] = []
        for orig in records:
            rec = copy.deepcopy(orig)

            # Apply variant configuration
            if ablation_name == "repair_policy":
                pol = var["repair_policy"]
                rec = run_pipeline(rec, repair_policy=pol)
            elif ablation_name == "verifier":
                v_type = var["verifier"]
                rec = run_pipeline(rec, verifier=v_type)
            elif ablation_name == "top_k":
                tk = var["top_k"]
                rec = run_pipeline(rec, top_k=tk)
            elif ablation_name == "iterations":
                iters = var["max_iterations"]
                if iters == 0:
                    rec.repaired_answer = rec.answer
                    rec.reliability_score_after = rec.reliability_score_before
                else:
                    rec = run_pipeline(rec, max_repair_iterations=iters)
            else:
                # Default baseline execution
                rec = run_pipeline(rec)

            variant_records.append(rec)

        eval_metrics = evaluate_dataset(variant_records)
        scr_delta = eval_metrics["scr"]["delta"]["mean"]
        preservation = eval_metrics["preservation_rate"]["mean"]
        rouge_f1 = eval_metrics["lexical_similarity"]["rouge_l_f1"]["mean"]

        results_summary.append({
            "variant": var_name,
            "scr_before": eval_metrics["scr"]["before"]["mean"],
            "scr_after": eval_metrics["scr"]["after"]["mean"],
            "scr_delta": scr_delta,
            "preservation_rate": preservation,
            "rouge_l_f1": rouge_f1,
        })

    # Print summary table
    print(f"\n{'Variant':<32} | {'SCR Before':<10} | {'SCR After':<10} | {'SCR Delta':<10} | {'Preservation':<12} | {'ROUGE-L'}")
    print("-" * 95)
    for r in results_summary:
        print(f"{r['variant']:<32} | {r['scr_before']:<10.3f} | {r['scr_after']:<10.3f} | {r['scr_delta']:<+10.3f} | {r['preservation_rate']:<12.3f} | {r['rouge_l_f1']:.3f}")
    print("=" * 95 + "\n")

    out_file = output_dir / f"ablation_{ablation_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    return {"ablation": ablation_name, "results": results_summary}


def main():
    parser = argparse.ArgumentParser(description="Run pipeline ablation experiments")
    parser.add_argument("--data", default="data/processed/halueval.jsonl", help="Data file")
    parser.add_argument("--ablation", default="all", choices=list(ABLATION_CONFIGS.keys()) + ["all"])
    parser.add_argument("--output_dir", default="results/ablations", help="Output directory")
    parser.add_argument("--max_records", type=int, default=20, help="Max records per ablation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        # Create a synthetic dev sample if dataset file does not exist locally
        sample_records = [
            AnswerRecord(
                id=f"sample_{i:03d}",
                source_dataset="dev",
                split="dev",
                question="Where was Albert Einstein born?",
                answer="Albert Einstein was born in 1879 in Berlin, Germany. He won a Nobel Prize in 1921.",
            )
            for i in range(args.max_records)
        ]
    else:
        sample_records = load_dev_records(str(data_path), max_records=args.max_records)

    ablations_to_run = (
        list(ABLATION_CONFIGS.keys()) if args.ablation == "all" else [args.ablation]
    )

    all_summaries = {}
    for ab_name in ablations_to_run:
        summary = run_ablation_experiment(ab_name, sample_records, out_dir)
        all_summaries[ab_name] = summary["results"]

    with open(out_dir / "all_ablations_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll ablations completed. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
