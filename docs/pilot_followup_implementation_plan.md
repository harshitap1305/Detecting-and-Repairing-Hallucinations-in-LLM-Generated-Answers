# Implementation plan: fixes identified by the reviewed development pilot

Status: phases 1–2 implemented with offline regression coverage; phases 3–6
remain planned. A new live development run has not yet been performed.

## Evidence and scope

The audited repair pilot contains five dev answers, seven original gold claims
(six Supported, one Unverifiable), and ten outputs across B0 and the pipeline.
The human-adjudicated report is stored privately at
`results/evaluation/colab_reviewed_pilot/pilot_final_v1/report.md`.
Both systems corrected zero of one eligible answers. Preservation was 1.00 for
B0 and 0.75 for the pipeline over four eligible answers. These small counts do
not establish general system performance or statistical superiority.

The strongest directly observed software issue is that three repaired answers
received self-verification SCR 1.0 after their final decomposition marked the
remaining content as not check-worthy. Other observed weaknesses are loss of
question context in short-answer decomposition, evidence gaps followed by
unhelpful editing, and unresolved unsupported attributions. Retrieval strategy
and repair behavior are improvement hypotheses to test, not proven fixes.

Preserve the existing gold, raw reviews, original run, final bundle and audit.
Use new configuration identities and output directories for new execution.
Production inference must never read gold labels, reference repairs, review
outcomes or known expected answers. This document plans software work and a dev
rerun; it does not authorize treating the pilot as a held-out test benchmark.

## 1. Make undefined self-verification explicit

Priority: first implementation commit. This is a correctness fix independent of
whether subsequent decomposition and retrieval changes improve the answers.

Files: `src/verify.py`, `src/pipeline.py`, `src/schema.py`,
`src/experiments/adapters.py`, affected scoring/reporting consumers and tests.
Search all score formatting, arithmetic, threshold comparisons and JSON output
consumers before changing behavior, including baselines and interactive tools.

- Replace the zero-check-worthy-claim convention of SCR 1.0 with a null score
  and a reason such as `no_check_worthy_claims`.
- Distinguish an intentionally empty candidate (`empty_answer`) from a nonempty
  answer for which decomposition produced no check-worthy claims. Neither is a
  successful factual correction. Purely nonfactual text also has no SCR.
- Store supported, contradicted, unverifiable and total check-worthy counts
  alongside the score, with an explicit diagnostic semantics version. Preserve
  original before-verification counts and final candidate counts separately.
- Do not announce all claims supported or convergence on zero claims. Record
  explicit stop reasons such as no factual claims, unchanged answer, or budget
  exhausted. A numeric 1.0 requires at least one verified check-worthy claim.
- Update logging to render null as N/A; handle null without crashes or implicit
  conversion to zero. Keep SCR a diagnostic, separate from human correction,
  preservation and completeness outcomes.
- Read existing artifacts using their recorded historical semantics. Do not
  rewrite old manifests or silently recalculate historical SCR with new rules.

Acceptance and regression tests:

- Empty answers and nonempty/no-check-worthy answers produce null, never 1.0.
- One Supported claim yields 1.0; one Unverifiable claim yields 0.0; mixed cases
  preserve the supported/total calculation.
- Mocked full-pipeline runs reproduce the three pilot empty-denominator paths
  without a false success state, including zero repair iterations.
- Native diagnostic output round-trips null/counts/reasons, and a representative
  old bundle still audits under its original recorded metric values.

## 2. Preserve propositions and modality during decomposition

Depends on step 1. Files: `src/decompose.py`, `tests/test_decompose.py`,
`tests/test_pipeline.py`; update the decomposition documentation.

- Use question context to expand short answer fragments into checkable
  propositions. Preserve the relationship being answered, rather than asserting
  only that an answer entity belongs to a broad category.
- Add generic few-shot examples for entity-only, frequency-only and definition
  answers, and ambiguous compound questions. Do not insert pilot IDs, gold
  answers or case-specific lookup rules into runtime prompts.
- Revise the instruction that categorically excludes hedges. A hedged factual
  assertion still contains factual content. Preserve its uncertainty/attribution
  in structured metadata or the claim representation; do not silently strengthen
  a tentative assertion into a certain one. Verify the factual payload separately
  where necessary and avoid inventing an attribution source for “Reportedly”.
- Keep genuine opinions, instructions, questions and boilerplate non-check-worthy.
- Add bounded structural validation/retry for nonempty factual-looking fragments
  that return no usable proposition. A failed extraction must be surfaced as
  unresolved coverage, not filled with a fact borrowed from the question.
- Preserve sentence IDs and original answer spans: expanded contextual claims
  need not occur verbatim in the answer. Separate interpretation from adding
  missing information that the answer never supplied.

Acceptance: synthetic fixtures retain the full answer relationship and modality;
ambiguous questions do not cause invented answer assertions; subjective text
remains non-check-worthy. Include pilot regressions as private dev fixtures and
use unrelated synthetic examples in public tests. Live quality is assessed only
in the later rerun, not guaranteed by mocked prompt tests.

## 3. Track original claims through repair and re-verification

Depends on steps 1–2. Files: `src/pipeline.py`, `src/schema.py`,
`src/repair.py`, `src/guards.py`, adapter diagnostics and pipeline/repair tests.

- Keep the original predicted claim inventory immutable throughout iteration.
  Separately store each candidate's decomposition and verification results.
- Record original-to-candidate claim links and explicit dispositions: retained,
  modified, hedged, removed or unresolved. These are model diagnostics, not human
  outcome ratings. Many-to-many matches and uncertain matches must be allowed;
  ID equality or exact string matching is not a semantic alignment algorithm.
- Retain original factual targets when qualifiers are inserted. Missing matches
  mean unresolved coverage or removal, never automatic correction or support.
- Inspect candidate-only propositions for introduced factual content, including
  unsupported scope restrictions, dates, names and frequency qualifiers.
- Require at least one verified factual target plus adequate target coverage for
  a factual-success stop state. A no-op or conservative fallback may end the loop
  but must not be reported as successful correction.
- Store iteration diagnostics and bounded stop reasons so a claim that disappears
  from decomposition can be traced without overwriting original evidence.

Acceptance: tests cover hedge-only edits, deletion, entity substitution, supported
plus unsupported facts in one sentence, new unsupported facts, and failed/ambiguous
alignment. An unchanged or hedged unresolved attribution never counts as corrected;
a supported fact disappearing from the candidate remains visible diagnostically.

## 4. Improve evidence acquisition before editing

Depends on step 2; step 3 provides coverage diagnostics. Files: `src/retrieve.py`,
`src/verify.py`, applicable options/config validation and retrieval tests.

- First compare actual pipeline queries/evidence with the separate gold-query
  retrieval diagnostic. Keep these distinct: relevant gold-query retrieval does
  not prove actual pipeline coverage or a match to the modern gold excerpt.
- Add a bounded question-conditioned query fallback when a claim lacks required
  entities or initial evidence is insufficient. Keep claim verification targets
  fixed; the question may supply context, not verified truth.
- Support bounded linked-entity retrieval where answering requires identifying
  an entity and then verifying an attribute. Trace each query, passage and link.
- Deduplicate and rank evidence deterministically; record query strategy, caps
  and corpus/index identity in config, cache keys and execution provenance.
- Treat missing evidence as Unverifiable. Do not infer Contradicted from absence.
  Set limits on queries, passages and model calls so this remains a reproducible
  change rather than unconstrained search.

Acceptance: fixed-corpus synthetic retrieval cases cover short fragments,
entity-to-attribute links, missing evidence, unrelated high-overlap passages,
deduplication and cache invalidation when strategy or index changes. A gold
citation unavailable in FEVER remains an explicit corpus-coverage limitation.
Do not require rebuilding or downloading the unchanged FEVER index.

## 5. Make repair conservative when evidence is insufficient

Depends on steps 3–4. Files: `src/repair.py`, `src/guards.py`,
`src/pipeline.py`, configuration and repair/guard tests.

- Separate verified replacement, unresolved claim and optional hedge/fallback in
  the output diagnostics. Keep the existing rewrite policies explicit; changing
  default behavior requires a versioned configuration, not a hidden adjustment.
- Require evidence for a replacement relation, not merely for the presence of
  its entity names. Reject arbitrary qualifications that do not follow from
  evidence, such as a new geographic scope or intermittent publication schedule.
- Preserve previously Supported content; leave the sentence unchanged when a
  proposed edit has no grounded improvement and abstention is the chosen policy.
  An unchanged unresolved answer is still unresolved.
- Keep factual minimal repair distinct from question completion. Missing answers
  to compound questions and creation-year omissions remain visible in completeness
  assessment; do not silently expand this experiment into answer generation.

Acceptance: evidence-backed corrections are accepted; plausible but ungrounded
rewrites are rejected; hedge-only/no-op edits remain unresolved; supported facts
survive mixed-sentence edits; exhausted guard retries have explicit fallback and
stop reasons. Validate across both prompted and trained repair entry points.

## 6. Freeze settings and run the next development pilot

Depends on steps 1–5. Files: `notebooks/05_run_reviewed_pilot_colab.ipynb`,
`src/llm.py`, experiment provenance/configuration, evaluation documentation.

- Make the selected model and GPT-OSS reasoning/token settings explicit. Include
  effective reasoning settings in request traces and cache identity; the current
  cache records token budget but not the added reasoning-effort option.
- Record completion finish reasons and token usage where the SDK supplies them.
  Retain clear empty-answer errors; never substitute reasoning for final content.
- Save the exact generating source state (including dirty edits), dependency
  versions and settings before inference. Save results persistently or export
  them before the Colab runtime is discarded. Avoid full environment/index setup
  when the current runtime and verified index are reusable.
- Run the same five-answer dev pilot under a fresh run directory, initially B0
  plus the pipeline. Preserve the old bundle as the baseline; do not resume it
  after changing source or options. No API key needs to be placed in the repo.
- Validate all ten outputs, compare stage-level failure patterns and actual
  evidence coverage, and regenerate blinded review packets for new candidates.
  Old human ratings bind to old outputs and cannot be reused for changed outputs.
- Obtain independent human repair ratings and adjudication, then generate and
  audit a new final bundle. Report outcomes even if correction does not improve.

Acceptance: offline tests and notebook validation pass; new execution has all
outputs; no empty denominator is presented as perfect factuality; generated claim
coverage and stop reasons are inspectable; independent human evaluation and an
audited bundle support any claim of improvement. Five dev answers remain an
operational check. Expand annotated dev coverage, freeze settings, and execute
the reserved held-out test benchmark only after this stage succeeds operationally.

## Suggested commits and validation order

1. `fix(verification): report undefined SCR and explicit stop reasons`
2. `fix(decomposition): preserve short-answer context and hedged facts`
3. `feat(pipeline): track claim coverage across repair iterations`
4. `feat(retrieval): add bounded contextual evidence fallback`
5. `fix(repair): require grounded improvements and preserve supported facts`
6. `feat(evaluation): capture model settings and reproducible pilot source`

For each change, run focused behavioral regression tests. Then run the full
offline suite, notebook schema/Python validation and `git diff --check` once the
combined implementation is ready. Audit the existing final pilot bundle to catch
accidental changes to historical evaluation semantics. No live API calls are
needed for steps 1–5; the real Colab run and new human assessments belong to
step 6. Implement step 1 first, review its null-score propagation, then proceed.
