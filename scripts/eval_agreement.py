"""
eval_agreement.py — Calculate Inter-Annotator Agreement (Cohen's Kappa & Confusion Matrix).

Phase 7 Deliverable:
  - Computes Cohen's kappa (kappa) between two independent annotators for 3-way claim labels:
      Supported, Contradicted, Unverifiable
  - Generates 3x3 label confusion matrix and raw percentage agreement
  - Used to validate human gold benchmark annotation quality (Target: kappa >= 0.70)

Usage:
    python scripts/eval_agreement.py --benchmark_suite
    python scripts/eval_agreement.py --annotator_a data/gold/rater_a.jsonl --annotator_b data/gold/rater_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


# Standard synthetic 30-claim dual annotation calibration set
SAMPLE_ANNOTATIONS_A = [
    "Supported", "Supported", "Supported", "Contradicted", "Supported",
    "Unverifiable", "Contradicted", "Supported", "Contradicted", "Contradicted",
    "Supported", "Supported", "Unverifiable", "Supported", "Contradicted",
    "Supported", "Supported", "Unverifiable", "Supported", "Contradicted",
    "Supported", "Contradicted", "Supported", "Supported", "Unverifiable",
    "Supported", "Supported", "Contradicted", "Supported", "Unverifiable"
]

SAMPLE_ANNOTATIONS_B = [
    "Supported", "Supported", "Supported", "Contradicted", "Supported",
    "Unverifiable", "Contradicted", "Supported", "Contradicted", "Contradicted",
    "Supported", "Supported", "Supported",   # Slight disagreement: Unverifiable vs Supported
    "Supported", "Contradicted",
    "Supported", "Supported", "Unverifiable", "Supported", "Contradicted",
    "Supported", "Contradicted", "Supported", "Supported", "Unverifiable",
    "Supported", "Supported", "Contradicted", "Supported", "Unverifiable"
]


def compute_cohens_kappa(rater_a: List[str], rater_b: List[str]) -> Tuple[float, float, Dict]:
    """
    Compute Cohen's kappa for two raters with categorical labels.
    Returns: (kappa, raw_agreement, confusion_matrix)
    """
    assert len(rater_a) == len(rater_b), "Rater arrays must be of equal length"
    n = len(rater_a)
    classes = sorted(list(set(rater_a) | set(rater_b)))

    # Compute confusion matrix
    cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for a, b in zip(rater_a, rater_b):
        cm[a][b] += 1

    # Raw agreement (Po)
    agreements = sum(cm[c][c] for c in classes)
    p_o = agreements / n

    # Chance agreement (Pe)
    p_e = 0.0
    for c in classes:
        p_a = sum(cm[c].values()) / n
        p_b = sum(cm[row][c] for row in classes) / n
        p_e += p_a * p_b

    # Kappa
    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return float(kappa), float(p_o), cm


def print_agreement_report(kappa: float, p_o: float, cm: Dict):
    print("\n" + "=" * 65)
    print(" INTER-ANNOTATOR AGREEMENT BENCHMARK (Phase 7 - Cohen's Kappa)")
    print("=" * 65)
    print(f"Total Evaluated Claims:     {sum(sum(r.values()) for r in cm.values())}")
    print(f"Raw Percentage Agreement:   {p_o * 100:.2f}%")
    print(f"Cohen's Kappa (kappa):      {kappa:.4f}  (Target: >= 0.70)")

    interpretation = (
        "Almost Perfect Agreement" if kappa >= 0.81 else
        "Substantial Agreement" if kappa >= 0.61 else
        "Moderate Agreement" if kappa >= 0.41 else
        "Fair to Poor Agreement"
    )
    print(f"Agreement Level:            {interpretation}")
    print("-" * 65)
    print("3x3 Confusion Matrix (Rater A rows vs. Rater B columns):")
    classes = sorted(cm.keys())
    header = f"{'':15s} | " + " | ".join(f"{c:12s}" for c in classes)
    print(header)
    print("-" * len(header))
    for r in classes:
        row_str = f"{r:15s} | " + " | ".join(f"{cm[r][c]:12d}" for c in classes)
        print(row_str)
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compute Inter-Annotator Agreement")
    parser.add_argument("--annotator_a", default=None, help="Path to Rater A JSONL")
    parser.add_argument("--annotator_b", default=None, help="Path to Rater B JSONL")
    parser.add_argument("--benchmark_suite", action="store_true", default=True)
    args = parser.parse_args()

    if args.annotator_a and args.annotator_b:
        def load_labels(path: str) -> List[str]:
            labels = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        labels.append(d.get("label", "Unverifiable"))
            return labels
        rater_a = load_labels(args.annotator_a)
        rater_b = load_labels(args.annotator_b)
    else:
        rater_a = SAMPLE_ANNOTATIONS_A
        rater_b = SAMPLE_ANNOTATIONS_B

    kappa, p_o, cm = compute_cohens_kappa(rater_a, rater_b)
    print_agreement_report(kappa, p_o, cm)


if __name__ == "__main__":
    main()

