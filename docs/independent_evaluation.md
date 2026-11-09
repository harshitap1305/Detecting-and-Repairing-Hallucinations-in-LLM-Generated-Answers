# Offline scoring and independent repair assessment

Increment 3 implements corrected detection scoring, blinded human review, and
paired comparisons. All commands consume saved outputs; none invokes an LLM,
retriever, or training job. Final report/bundle generation is increment 4.
Actual gold annotations and independent reviews remain human work.

## Score a complete runner directory

```bash
python3 scripts/run_eval.py \
  --run-dir results/evaluation/dev-claims-v1 \
  --gold data/gold/dev.jsonl \
  --output results/evaluation/dev-claims-v1/detection_metrics.json \
  --n_bootstrap 1000 --seed 42
```

This also works for `answer_detection` and `repair` runs. Output paths must be
new. The scorer validates the execution manifest, all declared system/record
coverage, immutable attempt checksums and trace hashes, and output JSONL views
against their committed attempts. Gold and output IDs must match exactly.
Neither partial intersections nor failed/missing records are silently scored.
Gold is structurally checked, including labels and decisive citations; factual
correctness and sentence-span annotation quality require the earlier human
validation workflow. Scoring does not require the original API credentials,
retrieval index, segmenter, or model packages. NumPy and SciPy are required.

New scoring code can evaluate an older complete execution run without resuming
inference. The metrics save gold/manifest/assessment hashes, scoring code identity,
package versions, and resampling settings separately from generation provenance.

For fixed-inventory claim verification, claim IDs, text, sentence IDs, offsets,
and check-worthiness must agree exactly, independent of row ordering. Report
three-way macro-F1 across Supported/Contradicted/Unverifiable, per-class
precision/recall/F1/support, a confusion matrix (gold rows, prediction columns),
and three-way accuracy. Absent-class F1 is explicitly zero; the macro average
always includes all three classes. Binary hallucination detection is a separately
named section combining Contradicted and Unverifiable. Thus confusing these two
labels can succeed on binary detection while failing three-way verification.
All accepted records and claims have coverage 1; incomplete coverage is an error.

B1 is scored at answer level, with gold positive when any check-worthy original
claim is Contradicted or Unverifiable. No three-way B1 claim result is invented.
Repair runs have no claim detection score because end-to-end inventories are
not automatically aligned. They receive reference exact match and whitespace
word-token ROUGE-L, explicitly labeled editing agreement. Empty repaired strings
remain empty, including when the human reference is empty. Native self-scores
remain diagnostics. Independent factual outcomes are unavailable until ratings
are supplied; a reference match is not a factual correctness judgment.

## Export review packets

```bash
python3 scripts/review_repairs.py export \
  --run-dir results/evaluation/dev-repair-v1 \
  --gold data/gold/dev.jsonl \
  --packet-dir data/review_batches/dev-v1
```

The command creates `reviewer/items.jsonl`, `reviewer/ratings_template.jsonl`,
`reviewer/instructions.txt`, and `private/key.json`. Give reviewers only the
`reviewer/` directory. Tokens are random opaque IDs, and items are shuffled;
system names, record IDs, and system-specific metadata are absent. Question,
original answer, candidate, and human original-claim labels/citations remain
visible. Text style can still reveal a method. Exporting again makes a new set
of opaque tokens; it cannot overwrite an existing packet. The private key binds
items to exact output fingerprints, the run, and the complete gold content.
Preserve it for imports and later adjudication.

Each reviewer independently copies the ratings template and completes:

- Every original check-worthy claim: an outcome and nonempty rationale.
  Supported claims allow `retained`, `deleted`, `hedged`, or `still_incorrect`
  (corrupted/lost truth). Contradicted/Unverifiable claims allow `corrected`,
  `deleted`, `hedged`, or `still_incorrect`. Corrections require inspected
  citation objects with `doc_id`, `title`, and `text`.
- `new_errors`: an explicit list of newly introduced factual errors, each with
  text, rationale, and an evidence list; use `[]` only after checking for errors.
- `completeness`: integer 0 (no useful answer), 1 (partial), or 2 (complete).
  Use `notes` for explanations and later adjudication decisions.

An empty candidate must have every outcome `deleted`, completeness 0, and no
new errors. Hedging and deletion never count as correction. Existing correct
facts are preserved only when reviewed as `retained`. Structural checks cannot
establish that a review is correct or that reviewers worked independently.

## Import two independent reviews and adjudicate

```bash
python3 scripts/review_repairs.py import \
  --run-dir results/evaluation/dev-repair-v1 --gold data/gold/dev.jsonl \
  --packet-dir data/review_batches/dev-v1 \
  --ratings data/review_batches/dev-v1/ratings_a.jsonl \
  --reviewer annotator-a --output data/review_batches/dev-v1/imported_a.json
```

Repeat for annotator B with a distinct reviewer identity and separate raw file.
The importer requires every item/claim exactly once, validates outcome/label
compatibility, citations and completeness, and rejects modified packets, changed
gold, or changed generated outputs. Blank templates and partial reviews fail.
Raw submissions remain separate; imported records retain ratings/key hashes,
judgments, citations, notes, and output fingerprints. Reviewer kind is explicitly
human. Optional model-judge assistance is not implemented in this increment.

A third reviewer can complete another copy of the same template, then import it
with a distinct adjudicator identity and `imported_final.json` as output:

```bash
python3 scripts/review_repairs.py adjudicate \
  --run-dir results/evaluation/dev-repair-v1 --gold data/gold/dev.jsonl \
  --first data/review_batches/dev-v1/imported_a.json \
  --second data/review_batches/dev-v1/imported_b.json \
  --final data/review_batches/dev-v1/imported_final.json \
  --output data/review_batches/dev-v1/adjudicated.json
```

The adjudicated result stores outcome agreement, marginal counts, Cohen's kappa,
disagreement locations, both imported source reviews and their fingerprints,
and a decision log. Saved adjudication metadata is recomputed during validation. Constant
marginals make kappa undefined (JSON null), not automatically perfect. Every
original disagreement and every changed final decision needs a nonempty `notes`
rationale. All three reviews must cover the same run/gold. The decision log
retains the two reviewers' choices and the adjudicated choice. Single-reviewer
scoring is supported but labeled `single_reviewer`, not adjudicated.

## Score independent repair outcomes

```bash
python3 scripts/run_eval.py \
  --run-dir results/evaluation/dev-repair-v1 --gold data/gold/dev.jsonl \
  --assessments data/review_batches/dev-v1/adjudicated.json \
  --output results/evaluation/dev-repair-v1/human_metrics.json
```

Primary repair estimands are the mean **per-answer** correction rate over answers
with originally incorrect claims, and mean per-answer preservation rate over
answers with originally Supported claims. Eligibility depends only on gold, so
it is the same across systems. Answers with zero relevant claims are inapplicable
for that rate, not scored as perfect. Report eligible counts, per-record rates,
claim-outcome totals, mean new-error count, and completeness. These are not pooled
claim ratios. Missing reviews are rejected, rather than excluded from denominators.

All confidence intervals use percentile bootstrap with replacement at the answer
level (default 1,000 resamples, 95%, seed 42). Claim macro-F1 is recomputed from
all claims of each resampled answer; claims are not sampled independently.
Paired comparisons resample the same answer IDs for both systems.

Repair comparisons against B0 include correction and preservation in one Holm
multiplicity family. The signed-rank test uses per-answer rates and exact paired
eligible IDs. If B0 is absent, that family is unavailable; if a metric has no
eligible answers, its comparison is inapplicable. Detection reports primary
paired bootstrap differences in pooled three-way macro-F1 against the prompted
pipeline. Holm-adjusted signed-rank tests on **per-answer** macro-F1 are secondary
and explicitly named: their estimand differs from pooled claim macro-F1.

SciPy is required for paired tests, with no silent fallback. All-zero differences
return p=1. Differences are rounded to 12 decimals before ranking. For at most
50 nonzero differences, untied data use SciPy's exact Wilcoxon method, while
ties/zero differences use an exact conditional sign-flip distribution of average
ranks. Larger tied/zero samples use the declared SciPy normal approximation,
Wilcox zero handling, and continuity correction. Method and nonzero-pair counts
are stored beside each result. Holm-adjusted and raw p-values are both saved.
Significance is not assumed, and tiny benchmarks can have very wide intervals.

The older `--results` mode of `run_eval.py` remains a diagnostic interface.
`compute_significance.py` likewise compares legacy diagnostics: it now rejects
partial pairs and duplicate IDs, preserves empty repairs, and displays adjusted
p-values. Use runner scoring plus human ratings for independent repair claims.
