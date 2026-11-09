"""Answer-cluster bootstrap and explicit paired tests; no approximate fallback."""
from __future__ import annotations

import math
import numpy as np


def clustered_interval(groups, statistic, n_bootstrap=1000, seed=42):
    if not groups or type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("Bootstrap needs nonempty answer groups and positive resample count")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        value = statistic(sample)
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("Non-finite bootstrap statistic")
            values.append(value)
    point = statistic(groups)
    return {"estimate": point, "ci_lower": float(np.percentile(values, 2.5)) if values else None,
            "ci_upper": float(np.percentile(values, 97.5)) if values else None,
            "n_bootstrap": n_bootstrap, "valid_resamples": len(values), "seed": seed,
            "unit": "answer", "confidence": 0.95}


def signed_rank(system, baseline):
    # Require SciPy even for degenerate input, so dependency failures are explicit.
    from scipy.stats import wilcoxon, rankdata
    if len(system) != len(baseline) or not system:
        raise ValueError("Paired scores must have equal nonzero length")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in list(system) + list(baseline)):
        raise ValueError("Paired scores must be finite numbers")
    diffs = np.round(np.asarray(system, dtype=float) - np.asarray(baseline, dtype=float), 12)
    nonzero = diffs[diffs != 0]
    if not len(nonzero):
        statistic, p_value, method = 0.0, 1.0, "all_zero"
    else:
        # Exact Wilcoxon is valid only without zero differences or tied magnitudes.
        exact = len(nonzero) == len(diffs) and len(set(abs(nonzero))) == len(nonzero) and len(nonzero) <= 50
        if not exact and len(nonzero) <= 50:
            # Conditional null distribution over independent sign flips of the
            # observed nonzero ranks. Doubled average ranks are integers, even
            # with ties; dynamic programming counts all 2**n sign assignments.
            ranks = np.rint(rankdata(abs(nonzero)) * 2).astype(int)
            total = int(sum(ranks))
            positive = int(sum(ranks[nonzero > 0]))
            threshold = min(positive, total - positive)
            counts = {0: 1}
            for rank in ranks:
                following = dict(counts)
                for value, count in counts.items():
                    following[value + int(rank)] = following.get(value + int(rank), 0) + count
                counts = following
            p_value = min(1.0, 2 * sum(count for value, count in counts.items() if value <= threshold)
                          / 2**len(nonzero))
            statistic, method = threshold / 2, "conditional_exact_sign_flip"
        else:
            method = "exact" if exact else "approx"
            result = wilcoxon(diffs, zero_method="wilcox", correction=True,
                              alternative="two-sided", method=method)
            statistic, p_value = float(result.statistic), float(result.pvalue)
            if not math.isfinite(p_value):
                raise ValueError("Wilcoxon returned a non-finite result")
    return {"statistic": statistic, "p_value": p_value, "is_significant_p05": p_value < 0.05,
            "mean_difference": float(np.mean(diffs)), "n_pairs": len(diffs),
            "n_nonzero": len(nonzero), "method": method, "zero_method": "wilcox",
            "alternative": "two-sided", "difference_round_decimals": 12}


def holm_adjust(p_values):
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("Invalid p-value")
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, previous = [0.0] * len(p_values), 0.0
    for position, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - position) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def paired_family(comparisons, n_bootstrap=1000, seed=42):
    """Named predeclared comparisons, each containing exactly paired ID->score maps."""
    results = {}
    for name, pair in comparisons.items():
        system, baseline = pair
        if system.keys() != baseline.keys() or not system:
            raise ValueError("Paired comparisons require exactly matching IDs")
        ids = sorted(system)
        values = [system[key] for key in ids]
        controls = [baseline[key] for key in ids]
        test = signed_rank(values, controls)
        interval = clustered_interval([a - b for a, b in zip(values, controls)],
                                      lambda sample: float(np.mean(sample)), n_bootstrap, seed)
        results[name] = {**test, "paired_difference": interval, "record_ids": ids}
    adjusted = holm_adjust([value["p_value"] for value in results.values()])
    for result, p in zip(results.values(), adjusted):
        result.update(p_value_holm=p, significant_holm_p05=p < 0.05)
    return {"status": "available" if results else "inapplicable",
            "correction": "Holm", "family_size": len(results), "comparisons": results}
