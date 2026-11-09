"""
evaluate.py — Comprehensive evaluation engine for hallucination detection & repair.

Computes:
1. Claim-level Detection: Precision, Recall, Macro-F1 (against gold labels).
2. Supported Claim Ratio (SCR / FActScore-style):
       SCR = (# Supported Claims) / (# Check-worthy Claims)
3. Preservation Rate & Over-edit Rate:
       Preservation = (# originally Supported claims entailed in repaired answer) / (# originally Supported claims)
       Over-edit Rate = 1.0 - Preservation Rate
4. Lexical Drift: Pure Python ROUGE-L (LCS), Levenshtein similarity, word retention.
5. Statistical Significance: 1,000 bootstrap resamples for 95% CIs; paired Wilcoxon signed-rank test.
6. Independent Evaluator (LLM-B / gemma2-9b-it) to prevent circular evaluation.
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.decompose import split_sentences
from src.guards import check_preservation_guard
from src.llm import call as llm_call
from src.schema import AnswerRecord, Claim


# ─── 1. Lexical Similarity & Drift Metrics ───────────────────────────────────

def compute_lcs_length(s1: List[str], s2: List[str]) -> int:
    """Compute the Longest Common Subsequence length between two token lists."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L (Precision, Recall, F1) based on Longest Common Subsequence.
    Pure Python fallback that matches standard ROUGE-L definitions.
    """
    cand_tokens = candidate.strip().lower().split()
    ref_tokens = reference.strip().lower().split()

    if not cand_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = compute_lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard character Levenshtein edit distance."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized edit similarity in [0.0, 1.0] where 1.0 is identical."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = compute_levenshtein_distance(s1, s2)
    return float(1.0 - (dist / max_len))


def compute_word_retention(candidate: str, original: str) -> float:
    """Fraction of words from original that remain in candidate."""
    orig_words = set(original.lower().split())
    if not orig_words:
        return 1.0
    cand_words = set(candidate.lower().split())
    retained = len(orig_words & cand_words)
    return float(retained / len(orig_words))


# ─── 2. Claim-level Detection Metrics ────────────────────────────────────────

def compute_detection_metrics(
    predictions: List[str],
    ground_truth: List[str],
    positive_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute explicitly separate binary and three-way claim verification metrics.
    Default positive class = {'Contradicted', 'Unverifiable'} (hallucinated).
    """
    if positive_labels is None:
        positive_labels = ["Contradicted", "Unverifiable"]
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length.")

    labels = ("Supported", "Contradicted", "Unverifiable")
    if any(label not in labels for label in predictions + ground_truth):
        raise ValueError("Unknown or missing claim verdict")
    if any(label not in labels for label in positive_labels):
        raise ValueError("Unknown positive label")
    tp = fp = fn = tn = 0
    for pred, gold in zip(predictions, ground_truth):
        pred_pos = pred in positive_labels
        gold_pos = gold in positive_labels

        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and gold_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    binary = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }

    matrix = [[sum(g == actual and p == predicted for g, p in zip(ground_truth, predictions))
               for predicted in labels] for actual in labels]
    per_class = {}
    for i, label in enumerate(labels):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision_i = matrix[i][i] / predicted if predicted else 0.0
        recall_i = matrix[i][i] / support if support else 0.0
        per_class[label] = {"precision": precision_i, "recall": recall_i,
                            "f1": 2 * precision_i * recall_i / (precision_i + recall_i)
                            if precision_i + recall_i else 0.0, "support": support}
    three_way = {"accuracy": sum(matrix[i][i] for i in range(3)) / len(predictions)
                 if predictions else 0.0,
                 "macro_f1": sum(value["f1"] for value in per_class.values()) / 3,
                 "per_class": per_class,
                 "confusion_matrix": {"labels": list(labels), "rows_gold_columns_pred": matrix},
                 "zero_division": 0, "n_claims": len(predictions)}
    # Preserve old binary keys for callers; new reports use the named sections.
    return {**binary, "binary": binary, "three_way": three_way}


def unique_records(records: List[AnswerRecord]) -> dict:
    result = {}
    for record in records:
        if not isinstance(record.id, str) or not record.id.strip() or record.id in result:
            raise ValueError("Duplicate or empty record ID")
        result[record.id] = record
    if not result:
        raise ValueError("Record list must be nonempty")
    return result


def align_claim_records(records, gold_records):
    """Require exact record identities and human claim inventories before scoring."""
    predictions, golds = unique_records(records), unique_records(gold_records)
    if predictions.keys() != golds.keys():
        raise ValueError("Prediction and gold record IDs must match exactly")
    groups = []
    seen = {"predicted": set(), "gold": set()}
    for record_id in sorted(golds):
        record, gold = predictions[record_id], golds[record_id]
        if any(getattr(record, field) != getattr(gold, field) for field in
               ("source_dataset", "split", "question", "answer")):
            raise ValueError(f"Prediction/gold input mismatch: {record_id}")
        inventories = []
        for name, item in (("predicted", record), ("gold", gold)):
            claims = {}
            for claim in item.claims:
                if not isinstance(claim.id, str) or not claim.id.strip() or claim.id in seen[name]:
                    raise ValueError("Duplicate or empty claim ID")
                if type(claim.check_worthy) is not bool:
                    raise ValueError("check_worthy must be boolean")
                seen[name].add(claim.id)
                claims[claim.id] = claim
            inventories.append(claims)
        if inventories[0].keys() != inventories[1].keys():
            raise ValueError(f"Incomplete claim inventory: {record_id}")
        pairs = []
        for claim_id in sorted(inventories[1]):
            pred, actual = (inventory[claim_id] for inventory in inventories)
            if any(getattr(pred, field) != getattr(actual, field) for field in
                   ("text", "sentence_id", "span", "check_worthy")):
                raise ValueError(f"Changed claim inventory: {claim_id}")
            if actual.check_worthy:
                if pred.label not in ("Supported", "Contradicted", "Unverifiable") or actual.label not in (
                        "Supported", "Contradicted", "Unverifiable"):
                    raise ValueError("Missing/invalid check-worthy verdict")
                pairs.append((pred.label, actual.label))
        if not pairs:
            raise ValueError(f"No check-worthy claims: {record_id}")
        groups.append(pairs)
    return groups


def evaluate_claim_detection(records, gold_records, n_bootstrap=1000, seed=42):
    from src.experiments.statistics import clustered_interval
    groups = align_claim_records(records, gold_records)
    def metric(sample):
        pairs = [pair for group in sample for pair in group]
        return compute_detection_metrics([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    result = metric(groups)
    result["coverage"] = {"records": len(groups), "expected_records": len(gold_records),
                          "claims": sum(map(len, groups)), "fraction": 1.0}
    result["intervals"] = {
        "three_way_macro_f1": clustered_interval(groups, lambda sample: metric(sample)["three_way"]["macro_f1"],
                                                 n_bootstrap, seed),
        "binary_f1": clustered_interval(groups, lambda sample: metric(sample)["binary"]["f1"],
                                         n_bootstrap, seed),
    }
    return result


# ─── 3. FActScore-style Supported Claim Ratio (SCR) ──────────────────────────

def compute_scr(record: AnswerRecord, use_after: bool = False) -> float:
    """
    Compute Supported Claim Ratio (SCR) for an AnswerRecord:
    SCR = (# Supported Claims) / (# Check-worthy Claims)
    """
    if use_after and record.reliability_score_after is not None:
        return float(record.reliability_score_after)
    if not use_after and record.reliability_score_before is not None:
        return float(record.reliability_score_before)

    cw = record.check_worthy_claims
    if not cw:
        return 1.0

    supported = sum(1 for c in cw if c.label == "Supported")
    return float(supported / len(cw))


# ─── 4. Preservation & Over-edit Rate ────────────────────────────────────────

def compute_preservation_rate(
    record: AnswerRecord,
    repaired_answer: Optional[str] = None,
) -> float:
    """
    Preservation Rate:
    (# originally Supported claims still entailed in repaired answer) / (Total originally Supported claims)
    """
    target_answer = (repaired_answer if repaired_answer is not None else
                     record.repaired_answer if record.repaired_answer is not None else record.answer)
    orig_supported_claims = [c for c in record.claims if c.check_worthy and c.label == "Supported"]

    if not orig_supported_claims:
        # If there were no supported claims originally, preservation is trivially 1.0
        return 1.0

    preserved_count = 0
    # Evaluate whether each originally supported claim is preserved in target_answer
    for c in orig_supported_claims:
        passed, _ = check_preservation_guard([c], target_answer)
        if passed:
            preserved_count += 1

    return float(preserved_count / len(orig_supported_claims))


def compute_over_edit_rate(record: AnswerRecord, repaired_answer: Optional[str] = None) -> float:
    """Over-edit Rate = 1.0 - Preservation Rate."""
    return float(1.0 - compute_preservation_rate(record, repaired_answer))


# ─── 5. Statistical Significance (Bootstrap & Wilcoxon) ──────────────────────

def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute mean and 95% bootstrap confidence interval using the percentile method.
    Returns: (mean, ci_lower, ci_upper)
    """
    if type(n_bootstrap) is not int or n_bootstrap < 1 or not 0 < ci < 1:
        raise ValueError("Positive bootstrap count and confidence in (0,1) required")
    if any(not np.isfinite(value) for value in scores):
        raise ValueError("Bootstrap scores must be finite")
    if not scores:
        return 0.0, 0.0, 0.0
    if len(scores) == 1:
        return scores[0], scores[0], scores[0]

    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=np.float64)
    n = len(arr)

    # Sample with replacement
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    mean_val = float(np.mean(arr))

    return mean_val, ci_lower, ci_upper


def paired_wilcoxon_test(
    system_scores: List[float],
    baseline_scores: List[float],
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test comparing system vs baseline.
    Returns test statistic, p-value, and significance indicator.
    """
    if len(system_scores) != len(baseline_scores):
        raise ValueError("System and baseline scores must have identical length for paired testing.")

    from src.experiments.statistics import signed_rank
    return signed_rank(system_scores, baseline_scores)


# ─── 6. Independent Evaluator (LLM-B) ────────────────────────────────────────

_INDEPENDENT_EVAL_PROMPT = """You are an independent factual evaluator assessing whether an answer is truthful and supported.

Question: {question}
Answer to evaluate: {answer}
Evidence Passages:
{passages}

Instructions:
1. Break down the answer's factual claims.
2. Determine if the answer is factually accurate and supported by the evidence.
3. Respond ONLY with valid JSON:
{{"is_supported": true/false, "scr_estimate": <float 0.0-1.0>, "hallucinated_phrases": [<str>, ...], "critique": "<one sentence>"}}"""


def evaluate_with_independent_llm(
    record: AnswerRecord,
    eval_model: str = "gemma2-9b-it",
    provider: str = "groq",
    passages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Score repaired answer using an independent model (LLM-B) to avoid circular evaluation.
    """
    target_answer = record.repaired_answer if record.repaired_answer is not None else record.answer
    if passages is None:
        passages = [
            f"(Title: {e.title}) {e.text}"
            for c in record.claims for e in c.evidence[:3]
        ]

    passages_text = "\n".join(passages[:5]) if passages else "(No evidence passages provided)"
    prompt = _INDEPENDENT_EVAL_PROMPT.format(
        question=record.question,
        answer=target_answer,
        passages=passages_text,
    )

    response = llm_call(
        prompt=prompt,
        model=eval_model,
        provider=provider,
        temperature=0.0,
        use_cache=True,
    )

    import re
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError("Independent evaluator did not return a JSON object.")
    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError("Independent evaluator returned malformed JSON.") from exc
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("is_supported"), bool)
        or type(result.get("scr_estimate")) not in (int, float)
        or not 0.0 <= result["scr_estimate"] <= 1.0
        or not isinstance(result.get("hallucinated_phrases"), list)
    ):
        raise ValueError("Independent evaluator returned invalid score fields.")
    return result


# ─── 7. Full Benchmark Evaluation Suite ──────────────────────────────────────

def evaluate_dataset(
    records: List[AnswerRecord],
    gold_records: Optional[List[AnswerRecord]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute comprehensive scientific evaluation metrics over a list of AnswerRecords.
    """
    if not records:
        return {"error": "Empty record list"}

    # 1. SCR Progression
    scrs_before = [compute_scr(r, use_after=False) for r in records]
    scrs_after = [compute_scr(r, use_after=True) for r in records]
    scr_deltas = [a - b for a, b in zip(scrs_after, scrs_before)]

    mean_b, ci_b_low, ci_b_high = bootstrap_confidence_interval(scrs_before, n_bootstrap, seed=seed)
    mean_a, ci_a_low, ci_a_high = bootstrap_confidence_interval(scrs_after, n_bootstrap, seed=seed)
    mean_d, ci_d_low, ci_d_high = bootstrap_confidence_interval(scr_deltas, n_bootstrap, seed=seed)

    # 2. Preservation & Over-edit Rates
    preservations = [compute_preservation_rate(r) for r in records]
    mean_p, ci_p_low, ci_p_high = bootstrap_confidence_interval(preservations, n_bootstrap, seed=seed)

    over_edits = [1.0 - p for p in preservations]
    mean_oe, ci_oe_low, ci_oe_high = bootstrap_confidence_interval(over_edits, n_bootstrap, seed=seed)

    # 3. Lexical Drift Metrics
    rouge_f1s = []
    lev_sims = []
    word_retentions = []
    for r in records:
        repaired = r.repaired_answer if r.repaired_answer is not None else r.answer
        rf = compute_rouge_l(repaired, r.answer)["f1"]
        lev = compute_normalized_levenshtein_similarity(repaired, r.answer)
        ret = compute_word_retention(repaired, r.answer)
        rouge_f1s.append(rf)
        lev_sims.append(lev)
        word_retentions.append(ret)

    mean_rf, ci_rf_low, ci_rf_high = bootstrap_confidence_interval(rouge_f1s, n_bootstrap, seed=seed)
    mean_lev, ci_lev_low, ci_lev_high = bootstrap_confidence_interval(lev_sims, n_bootstrap, seed=seed)
    mean_ret, ci_ret_low, ci_ret_high = bootstrap_confidence_interval(word_retentions, n_bootstrap, seed=seed)

    # 4. Claim distribution
    total_claims = sum(len(r.claims) for r in records)
    total_cw = sum(len(r.check_worthy_claims) for r in records)
    label_counts = {"Supported": 0, "Contradicted": 0, "Unverifiable": 0}
    for r in records:
        for c in r.check_worthy_claims:
            if c.label in label_counts:
                label_counts[c.label] += 1

    # 5. Detection metrics against gold (if gold records provided)
    detection_results = None
    unique_records(records)
    if gold_records is not None:
        detection_results = evaluate_claim_detection(records, gold_records, n_bootstrap, seed)

    return {
        "metric_notes": {"scr": "system self-verification diagnostic",
                         "preservation_rate": "token-overlap diagnostic; not semantic preservation",
                         "lexical_similarity": "similarity to original answer; not factual correctness"},
        "n_records": len(records),
        "total_claims": total_claims,
        "check_worthy_claims": total_cw,
        "label_distribution": label_counts,
        "scr": {
            "before": {"mean": mean_b, "ci_lower": ci_b_low, "ci_upper": ci_b_high},
            "after": {"mean": mean_a, "ci_lower": ci_a_low, "ci_upper": ci_a_high},
            "delta": {"mean": mean_d, "ci_lower": ci_d_low, "ci_upper": ci_d_high},
            "raw_before": scrs_before,
            "raw_after": scrs_after,
        },
        "preservation_rate": {"mean": mean_p, "ci_lower": ci_p_low, "ci_upper": ci_p_high, "raw": preservations},
        "over_edit_rate": {"mean": mean_oe, "ci_lower": ci_oe_low, "ci_upper": ci_oe_high},
        "lexical_similarity": {
            "rouge_l_f1": {"mean": mean_rf, "ci_lower": ci_rf_low, "ci_upper": ci_rf_high},
            "levenshtein_similarity": {"mean": mean_lev, "ci_lower": ci_lev_low, "ci_upper": ci_lev_high},
            "word_retention": {"mean": mean_ret, "ci_lower": ci_ret_low, "ci_upper": ci_ret_high},
        },
        "detection_vs_gold": detection_results,
    }
