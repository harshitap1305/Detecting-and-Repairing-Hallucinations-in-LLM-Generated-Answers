#!/usr/bin/env python3
"""
Comprehensive Result Visualization Suite
Detecting and Repairing Hallucinations in LLM-Generated Answers

Generates the four retained report charts (original asset numbers preserved):
1. Master Baseline Comparison (SCR before/after + Preservation vs Over-edit)
3. Retrieval Architecture Performance & Latency Trade-off
4. Re-Verification Loop Convergence & Error Reduction Dynamics
6. Audited HaluEval Gold Benchmark & Baseline Distribution
"""

import os
from pathlib import Path

# Ensure writable matplotlib cache
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "figure.dpi": 300,
})

# Values reproduce the team-supplied chart summaries. See
# docs/report/figure_provenance.md for their source history.
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_master_baseline_comparison():
    """Figure 1: Master Baseline Comparison (SCR + Preservation Rate)"""
    print("Generating Figure 1: Master Baseline Comparison...")
    
    baselines = [
        "B0: No Repair",
        "B1: Answer-Level",
        "B2: SelfCheckGPT",
        "B3: Self-Repair",
        "B4: CoVe",
        "B5: RARR (Rewrite)",
        "B6: Delete-Only",
        "B7: Factcheck-GPT",
        "Ours (Guarded)",
    ]
    
    # SCR After with 95% CIs
    scr_after = [0.542, 0.542, 0.684, 0.651, 0.732, 0.825, 0.884, 0.826, 0.912]
    ci_lower = [0.495, 0.495, 0.638, 0.602, 0.686, 0.781, 0.849, 0.783, 0.881]
    ci_upper = [0.589, 0.589, 0.730, 0.701, 0.776, 0.867, 0.918, 0.868, 0.942]
    errors = [
        [s - l for s, l in zip(scr_after, ci_lower)],
        [u - s for s, u in zip(scr_after, ci_upper)],
    ]
    
    preservation = [100.0, 100.0, 84.5, 81.3, 79.8, 61.2, 78.1, 74.6, 94.8]
    over_edit = [0.0, 0.0, 15.5, 18.7, 20.2, 38.8, 21.9, 25.4, 5.2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    # Color coding: Gray for baselines, distinctive Blue/Teal for Ours
    colors = ["#7f8c8d"] * 8 + ["#2980b9"]
    
    y_pos = np.arange(len(baselines))
    
    # Left: Supported Claim Ratio (SCR)
    bars1 = ax1.barh(y_pos, scr_after, xerr=errors, align="center", color=colors, alpha=0.85, capsize=4, edgecolor="black", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(baselines, fontweight="medium")
    ax1.invert_yaxis()
    ax1.set_xlabel("Supported Claim Ratio (SCR / FActScore)")
    ax1.set_title("(A) Factual Reliability After Repair (95% CI)", pad=12, fontweight="bold")
    ax1.set_xlim(0.4, 1.0)
    ax1.axvline(0.542, color="#c0392b", linestyle="--", linewidth=1.2, label="Original Baseline (B0: 0.542)")
    ax1.legend(loc="lower right")
    
    # Annotate values
    for bar, val in zip(bars1, scr_after):
        ax1.text(val + 0.015, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", ha="left", fontsize=9, fontweight="bold" if val > 0.9 else "normal")

    # Right: Preservation vs Over-edit trade-off
    w = 0.38
    b_pres = ax2.barh(y_pos - w/2, preservation, w, label="Preservation Rate (%)", color="#27ae60", alpha=0.85, edgecolor="black", linewidth=0.8)
    b_over = ax2.barh(y_pos + w/2, over_edit, w, label="Over-Edit Rate (%)", color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("(B) Content Preservation vs. Over-Editing", pad=12, fontweight="bold")
    ax2.set_xlim(0, 110)
    ax2.legend(loc="upper right")
    
    # Annotate preservation
    for bar, val in zip(b_pres, preservation):
        ax2.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig1_master_baseline_comparison.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_retrieval_curves():
    """Figure 3: Retrieval Recall Curves & Latency Trade-off"""
    print("Generating Figure 3: Retrieval Recall & Latency...")
    
    k_vals = [1, 3, 5, 10]
    
    models = {
        "BM25 Baseline": [0.745, 0.820, 0.865, 0.890],
        "BM25 + Entity Keywords": [0.780, 0.855, 0.890, 0.915],
        "Dense (BGE-Small-en)": [0.760, 0.840, 0.885, 0.920],
        "Hybrid RRF (Proposed Default)": [0.835, 0.900, 0.935, 0.955],
        "Hybrid + Cross-Encoder Reranker": [0.870, 0.925, 0.950, 0.965],
    }
    
    latencies = {
        "BM25 Baseline": 4.5,
        "BM25 + Keywords": 4.8,
        "Dense (BGE-Small)": 18.0,
        "Hybrid RRF": 22.0,
        "Hybrid + Reranker": 48.0,
    }
    recall_at_5 = {
        "BM25 Baseline": 86.5,
        "BM25 + Keywords": 89.0,
        "Dense (BGE-Small)": 88.5,
        "Hybrid RRF": 93.5,
        "Hybrid + Reranker": 95.0,
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    styles = {
        "BM25 Baseline": ("#7f8c8d", "o", "--"),
        "BM25 + Entity Keywords": ("#95a5a6", "s", "-."),
        "Dense (BGE-Small-en)": ("#e67e22", "^", ":"),
        "Hybrid RRF (Proposed Default)": ("#2980b9", "D", "-"),
        "Hybrid + Cross-Encoder Reranker": ("#8e44ad", "P", "-"),
    }
    
    # Left: Recall Curves
    for name, recalls in models.items():
        color, marker, ls = styles[name]
        lw = 2.5 if "Proposed" in name else 1.8
        ax1.plot(k_vals, [r * 100 for r in recalls], label=name, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=7)
        
    ax1.set_xlabel("Top-$k$ Retrieved Evidence Passages")
    ax1.set_ylabel("Passage Retrieval Recall (%)")
    ax1.set_title("(A) Retrieval Recall Across Evidence Budgets", pad=12, fontweight="bold")
    ax1.set_xticks(k_vals)
    ax1.set_ylim(70, 100)
    ax1.legend(loc="lower right", framealpha=0.9)
    
    # Right: Latency vs Recall@5 Pareto plot
    colors_pareto = ["#7f8c8d", "#95a5a6", "#e67e22", "#2980b9", "#8e44ad"]
    for i, (name, lat) in enumerate(latencies.items()):
        rec = recall_at_5[name]
        ax2.scatter(lat, rec, color=colors_pareto[i], s=140, edgecolor="black", zorder=4)
        offset = (8, -2) if "Keywords" not in name else (8, -12)
        if "Hybrid RRF" in name:
            offset = (-15, 12)
        ax2.annotate(name, (lat, rec), xytext=offset, textcoords="offset points", fontsize=9.5, fontweight="bold" if "Hybrid RRF" in name else "normal")
        
    ax2.set_xlabel("Mean Latency per Query (ms)")
    ax2.set_ylabel("Recall@5 (%)")
    ax2.set_title("(B) Retrieval Accuracy vs. Latency Trade-Off", pad=12, fontweight="bold")
    ax2.set_xlim(0, 55)
    ax2.set_ylim(84, 98)
    
    # Draw Pareto sweet-spot shaded zone
    ax2.axvspan(15, 25, color="#2980b9", alpha=0.1, label="Optimal Efficiency Zone")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig3_retrieval_recall_latency.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_reverification_convergence():
    """Figure 4: Iterative Re-Verification Convergence & Contradiction Reduction"""
    print("Generating Figure 4: Re-Verification Convergence...")
    
    iterations = [0, 1, 2, 3]
    convergence_pct = [0.0, 88.6, 97.3, 97.8]
    residual_contra_pct = [45.8, 6.5, 2.7, 2.2]
    
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    
    color_conv = "#27ae60"
    color_contra = "#e74c3c"
    
    # Left axis: Convergence
    ax1.set_xlabel("Re-Verification Iteration Count", labelpad=10)
    ax1.set_ylabel("Cumulative Fully-Converged Answers (%)", color=color_conv, labelpad=10, fontweight="bold")
    line1 = ax1.plot(iterations, convergence_pct, color=color_conv, marker="o", linewidth=2.5, markersize=8, label="Cumulative Convergence (%)")
    ax1.tick_params(axis="y", labelcolor=color_conv)
    ax1.set_ylim(-5, 110)
    ax1.set_xticks(iterations)
    
    # Annotate points
    for x, y in zip(iterations, convergence_pct):
        ax1.annotate(f"{y:.1f}%", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", color=color_conv, fontweight="bold")
        
    # Right axis: Residual Contradictions
    ax2 = ax1.twinx()
    ax2.set_ylabel("Residual Contradiction Rate (%)", color=color_contra, labelpad=10, fontweight="bold")
    line2 = ax2.plot(iterations, residual_contra_pct, color=color_contra, marker="s", linestyle="--", linewidth=2.2, markersize=8, label="Residual Contradiction Rate (%)")
    ax2.tick_params(axis="y", labelcolor=color_contra)
    ax2.set_ylim(-2, 50)
    ax2.grid(False)
    
    for x, y in zip(iterations, residual_contra_pct):
        ax2.annotate(f"{y:.1f}%", (x, y), xytext=(0, -14), textcoords="offset points", ha="center", color=color_contra, fontweight="bold")
        
    # Early-stopping cutoff boundary line
    ax1.axvline(2, color="#2c3e50", linestyle=":", linewidth=1.8, label="Default Early-Stopping Cutoff (Max Iter=2)")
    
    # Combined legend
    lines = line1 + line2 + [plt.Line2D([0], [0], color="#2c3e50", linestyle=":", linewidth=1.8)]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.95)
    
    plt.title("Iterative Re-Verification: Convergence vs. Error Elimination", pad=14, fontweight="bold")
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig4_reverification_convergence.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_audited_benchmark_distribution():
    """Figure 6: Audited HaluEval Gold Benchmark & Baseline Outcomes"""
    print("Generating Figure 6: Audited Benchmark Gold Claims & Baseline Distribution...")
    
    # Dev: 50 answers, 59 claims (28 Supported, 31 Contradicted)
    # Test: 100 answers, 110 claims (53 Supported, 57 Contradicted)
    splits = ["Dev Split (50 Answers)", "Test Split (100 Answers)"]
    supported = [28, 53]
    contradicted = [31, 57]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Left: Claim Distribution in Benchmark
    x = np.arange(len(splits))
    width = 0.35
    b1 = ax1.bar(x - width/2, supported, width, label="Supported Claims", color="#27ae60", alpha=0.85, edgecolor="black")
    b2 = ax1.bar(x + width/2, contradicted, width, label="Contradicted Claims (Hallucinations)", color="#e74c3c", alpha=0.85, edgecolor="black")
    
    ax1.set_ylabel("Number of Check-Worthy Claims")
    ax1.set_title("(A) Frozen Gold Benchmark Ground Truth Claims", pad=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontweight="bold")
    ax1.set_ylim(0, 70)
    ax1.legend(loc="upper left")
    
    # Annotate counts
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, str(int(bar.get_height())), ha="center", fontweight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, str(int(bar.get_height())), ha="center", fontweight="bold")

    # Right: Audited Baseline B0 Review Outcomes
    # Test Split: 53 Retained (100% preservation), 57 Still Incorrect, 0 New Errors, Cohen's kappa = 1.0
    outcomes = ["Preserved Valid\n(100.0%)", "Residual Hallucination\n(Unedited B0)", "Newly Introduced\nErrors (0.0%)"]
    counts = [53, 57, 0]
    colors_pie = ["#27ae60", "#e74c3c", "#95a5a6"]
    
    bars = ax2.bar(outcomes, counts, color=colors_pie, alpha=0.85, edgecolor="black", width=0.55)
    ax2.set_ylabel("Audited Claims in Test Set (Total: 110)")
    ax2.set_title("(B) Blinded Human Review Outcomes on Control (B0)", pad=12, fontweight="bold")
    ax2.set_ylim(0, 75)
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{int(h)} claims", ha="center", fontweight="bold")
        
    ax2.text(
        0.95, 0.85,
        "Dual Human Reviewers:\nCohen's $\\kappa = 1.0000$\nAgreement = 100%\nAdjudication Conflicts = 0",
        transform=ax2.transAxes,
        ha="right",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#bdc3c7", lw=1.2),
    )

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig6_audited_b0_control_distribution.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("=" * 60)
    print("Generating All Project Figures & Graphs...")
    print(f"Saving figures to directory: {OUTPUT_DIR.resolve()}")
    print("=" * 60)
    
    plot_master_baseline_comparison()
    plot_retrieval_curves()
    plot_reverification_convergence()
    plot_audited_benchmark_distribution()
    
    print("=" * 60)
    print("All 4 retained figures successfully generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

