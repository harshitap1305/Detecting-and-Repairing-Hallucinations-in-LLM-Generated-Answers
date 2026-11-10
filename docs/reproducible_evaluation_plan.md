# Implementation plan: reproducible evaluation package

Status: implementation increments 1–4 are implemented: preparation, resumable
execution, corrected scoring and independent-review tooling, report generation,
and offline bundle auditing with CI lifecycle coverage. Increment 5 remains:
actual human annotation/review, a live development pilot, and final benchmark
execution. No measured result is claimed by the synthetic tests. See
[the protocol](evaluation_protocol.md), [benchmark execution](benchmark_runner.md),
[offline evaluation](independent_evaluation.md), and
[result bundles](evaluation_bundles.md).

Step 5 has started: a pinned HaluEval cohort (50 dev / 100 test), blank human
packets, sentence references, and a five-record B0 operational smoke run are
prepared. Full pytest verification passed (219 tests plus 35 subtests). The live
pipeline pilot and final benchmark remain pending live resources and the full
human gold/reviews. The first five pilot records have now been finalized as
human-reviewed AI-assisted annotations with citations (user-confirmed acceptance);
corpus alignment, a real retrieval index, and an API credential remain outstanding. See the
[actual experiment handoff](halueval_experiment_handoff.md).


## Objective and boundaries

Make every reported result traceable to frozen inputs, a declared protocol,
complete per-record outputs, and an offline recomputation command. Deliver a
working package even before real annotations are available; mark benchmark
completion separately and require real human annotations and actual runs.

Reuse `validate_gold.py`, `audit_baseline.py`, the baseline modules,
`compare_training_paths.py`, and existing statistical helpers where appropriate.
Do not retrain models or redesign repair guards in this contribution. Changes
to metrics, error handling, and provenance are included where needed to make
the evaluation valid. Existing pipeline scores remain diagnostic measurements.

Default benchmark proposal: 50 development answers and 100 test answers,
matching the annotation workflow. Counts remain configurable and reports derive
them from validated files. Never describe the combined 150 as 150 test answers.

## Phase 1 — Freeze the evaluation protocol

Deliver `docs/evaluation_protocol.md` and `configs/evaluation.yaml`.

- Declare dataset versions, sampling seed, source quotas, inclusion rules,
  duplicate/prompt grouping, corpus snapshot, and split sizes before sampling.
  Never select examples based on observed system successes or failures.
- Define tasks separately: answer-level detection, fixed-inventory claim
  verification, and end-to-end answer repair. Human claim inventories are an
  oracle input for the fixed-inventory task; identify that condition explicitly.
- Define a system capability registry. B1 returns answer-level detection;
  B2 has consistency scores; B6 requires detector predictions. Record these
  distinctions instead of manufacturing comparable three-way predictions.
- Specify how B6 obtains predictions without gold labels, which retrieval
  settings each system supports, and whether evidence is shared or independently
  retrieved. Preserve each baseline's intended method and disclose differences.
- Define primary metrics, comparison families, evaluator rubric, test-use policy,
  and deterministic sentence segmentation. Record tokenizer/model versions and
  fail when a required segmenter is absent instead of silently changing it.
- Use development records to select settings. Finalize the configuration before
  running test evaluation; preserve subsequent reruns as separate recorded runs.

Acceptance: the protocol identifies the inputs and applicable metrics for every
system, and configuration validation rejects unsupported combinations.

## Phase 2 — Prepare annotation inputs and freeze the gold set

Add `scripts/prepare_annotation_batch.py`; extend the current annotation guide
and gold validator rather than creating a competing record schema.

- Sample deterministically from supplied processed datasets; save original
  dataset IDs, versions, source hashes, sampling decisions, and exclusions.
- Export separate blank annotation packets and prediction-safe system inputs.
  Optional model-proposed claims must be identified as drafts for human review.
- Establish a shared, human-reviewed claim inventory before two independent
  verdict annotations, so agreement compares the same claims. Record inventory
  changes and obtain fresh labels when the inventory changes.
- Preserve raw annotator files, citation snapshots, disagreements, adjudication
  decisions, and final human reference repairs. Report agreement, coverage, and
  class distribution; treat the existing kappa target as a quality target rather
  than evidence that a valid file is factually correct.
- Clarify annotation treatment of uncertainty, deletion, and supported facts.
  A hedge does not constitute a factual correction. Keep these outcomes distinct
  during evaluation and avoid adding unsupported facts to reference repairs.
- Validate exact split membership, IDs, spans, label types, decisive citations,
  and prompt separation; freeze gold and split hashes in a versioned manifest.
- Generate existing training holdout exclusions when needed for later A/C runs.
  Gold evidence is available to evaluators, never to system generation unless
  the run is explicitly a gold-evidence diagnostic.

Acceptance: repeated preparation with identical sources/configuration selects
the same records; incomplete annotations cannot become a frozen benchmark.
Real annotation and adjudication are a human dependency, not an automated output.

## Phase 3 — Add a common experiment runner and artifact contract

Add `src/experiments/` for configuration, adapters, validation, and persistence,
with `scripts/run_benchmark.py` as the entry point.

- Adapt the prompted pipeline and B0–B7 to a versioned output envelope carrying
  record identity, generated text, native predictions/scores, evidence references,
  execution status, timing, and errors. Preserve `AnswerRecord` interoperability.
- Build fresh system inputs for every run. Strip human verdicts, references, and
  citations; supply only unlabeled human claims in fixed-inventory mode.
- Retain native scores under explicit names. For example, B2 consistency is not
  evidence-based SCR. Missing task outputs are unavailable, never guessed.
- Preflight inputs, configuration, index files, credentials, and model paths.
  Surface parse failures and retrieval exceptions as execution errors; valid
  zero-hit retrieval remains a measured outcome. Remove swallowed-error behavior
  from benchmark execution paths, including B1, B5, and B7 as needed.
- Checkpoint completed record outputs and preserve failures. Resume only when
  source hashes, code, configuration, and model/evaluator identities match.
  Never overwrite a completed run or drop failed IDs from the denominator.
- Include provider/endpoint identity in LLM cache keys; record cache policy and
  preserve actual responses where permitted. Remote API reruns need not be
  bit-for-bit deterministic; offline scoring must be reproducible.
- Write manifests with code revision/dirty state, resolved configuration,
  input and index hashes, model/checkpoint identities, package versions, seeds,
  segmenter identity, hardware, timestamps, and measured runtime. Capture token
  usage when exposed; label cost estimates separately from measured usage.

Acceptance: interruption/resume preserves exactly one successful output per
system and record; identity changes prevent resumption; no gold labels reach
generation. An incomplete run remains visibly incomplete.

## Phase 4 — Implement comparable, validated evaluation

Extend `src/evaluate.py` and `scripts/run_eval.py`; share validation with the
benchmark runner and the A/C comparison.

- Reject duplicate, missing, extra, and mismatched record IDs before comparisons.
  Fixed-inventory evaluation must also validate claim text, IDs, check-worthiness,
  and complete prediction coverage. Independently decomposed claims need a
  separately specified alignment protocol; ordinal IDs alone are insufficient.
- Report binary detection and three-way macro-F1 separately, with per-class
  precision/recall/F1, confusion matrices, supports, and coverage. Report B1 at
  answer level; mark inapplicable claim metrics unavailable with a reason.
- Compare repairs with human reference answers using exact match and ROUGE-L,
  correctly retaining empty-string deletions. Treat text similarity as editing
  agreement, not a factual correctness judgment.
- Score every system's repairs using the same independent assessment protocol.
  Export anonymized, shuffled human-review packets; collect corrected,
  retained, deleted, hedged, and still-incorrect original claim outcomes, plus
  newly introduced factual errors and answer completeness. Preserve raw ratings
  and adjudication. An independent model judge may assist, but its scores remain
  separate from human judgments and must store evidence, model, and rubric.
- Keep self-verification SCR and token-overlap preservation in a diagnostic
  section. Empty/deleted answers do not count as successful corrections. Missing
  independent judgments are unavailable, never replaced by self-scores.
- Compute paired differences on exact ID sets. Bootstrap at answer level,
  including all claims of a sampled answer when estimating claim metrics.
  State interval settings and seeds. If significance tests are reported, require
  SciPy, handle ties/degenerate pairs explicitly, and apply Holm correction within
  declared comparison families. Do not silently substitute an approximate test.

Acceptance: targeted regressions cover contradicted/unverifiable confusion,
missing claims, partial system outputs, empty repairs, zero-support classes,
failed evaluators, shuffled IDs, and paired bootstrap reproducibility.

## Phase 5 — Generate and verify the result bundle

Add `scripts/build_evaluation_report.py` and extend `audit_baseline.py` with a
versioned bundle mode while retaining its legacy artifact check.

Proposed bundle:

```text
results/evaluation/<run_id>/
  protocol.json
  manifest.json
  inputs_manifest.json
  outputs/<system>.jsonl
  evaluations/<system>.jsonl
  errors.jsonl
  metrics.json
  comparisons.json
  report.md
  checksums.json
```

- Generate tables and confidence intervals directly from saved outputs and
  assessments. Include split counts, class mix, method limitations, completion
  status, and applicable metrics. Preserve full numeric precision in JSON.
- Recompute metrics/report offline without credentials, inference, or training.
  Hash required artifacts and check that outputs match the manifest and gold
  identity; a checksum establishes integrity, not factual correctness.
- Require complete declared outputs and assessments for the final report.
  Allow explicitly labeled partial progress reports, with failure counts and
  unavailable metrics, but no final comparison claims from incomplete runs.
- Update the README and existing reports to separate historical unverified
  tables from newly generated measured results. Never replace old numbers with
  fabricated fixture scores. Store private gold and restricted source material
  outside public bundles; document access and how to reproduce with those files.

Acceptance: deleting an output, modifying a hashed input, or supplying duplicate
IDs invalidates the bundle. Recomputing from the same valid bundle reproduces
metrics and tables without network access.

## Phase 6 — Verify, pilot, and execute

1. Add small, clearly synthetic fixtures and deterministic mocked adapters for
   CI. Install an explicit CPU test dependency set. Exercise annotation export,
   execution, interruption/resume, scoring, report generation, and audit together.
2. Run a small development-only live pilot using supplied data, index, and
   credentials. Check failure handling, latency, baseline semantics, assessor
   agreement, and scoring sensitivity before committing API budget to test runs.
3. Freeze settings and execute the final human-annotated test benchmark. Complete
   blinded repair assessment, then generate and validate the final bundle.
4. Optionally integrate trained A/C variants through the same contract after
   valid checkpoints are available. They are not prerequisites for this package.

Acceptance: CI passes offline; the development pilot has a complete audited
bundle; the final benchmark is complete only after real outputs and assessments
exist. If external inputs are absent, report software readiness and the exact
remaining data/run requirements separately.

## Suggested implementation increments

1. Protocol/configuration, capability registry, and annotation preparation.
2. Shared artifact validation, adapters, runner, and safe resume/cache behavior.
3. Correct detection metrics, independent assessment import/export, and paired
   statistics.
4. Offline report generation, bundle audit, CI, and updated documentation.
5. Human annotation, development pilot, and final benchmark execution.

Each increment should be reviewable with local fixtures. Start with B0 plus the
prompted pipeline to validate the complete artifact lifecycle, then add B1–B7
before declaring the full baseline comparison implemented.
