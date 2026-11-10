# Implementation plan: defects exposed by the v3 pilot

Status (11 November 2026): phases 1–4 and the phase 5 execution workflow
were implemented. v4 completed 7/10 outputs; the v4b recovery run completed 10/10
and now has fresh human review, adjudication and an audited final bundle.
Model extraction checks still accept some inadequate claims, and entity queries
can contain noisy flattened metadata. These acceptance goals are not fully met.
The later proposed entity-lead cleanup was reverted and is not part of the
handoff. See [current results and open work](project_handoff.md).

## Scope and baseline

The v3 execution completed all ten outputs and passed offline progress-bundle
and source-snapshot audit. The pipeline preserved all five original answers,
avoiding earlier harmful edits, but did not demonstrate a factual correction.
V3 has no independent human repair assessments or final adjudicated report.
Preserve it as a development baseline; leave its review explicitly pending.

The private diagnostic report is at
`results/evaluation/colab_followup_v3/analysis/report.md`. It identifies three
observed defects: formatting headers selected as linked entities, missing
answer relationships in decomposition, and extra assertions inferred from an
unanswered question clause. These require fixes before the next human review.

Keep existing gold labels, references, splits, completed runs and reviews intact.
Do not hard-code pilot entities/answers, retrieve from human gold, or tune using
reserved test data. The following changes use generic rules and synthetic public
fixtures; private pilot traces are regression evidence, not production input.

## 1. Extract entity leads from content, not index formatting

Files: `src/retrieve.py`, `scripts/build_fever_fts.py` (inspect formatting; change
only if needed), `tests/test_contextual_retrieval.py`, FTS integration tests.

- Recognize the index's actual title/body format. Exclude header labels such as
  `Title:` as lead candidates while retaining meaningful title text as context.
  Do not globally blacklist a word wherever it occurs in genuine article text.
- Identify candidate mentions within content regions. Retain offsets into the
  original stored passage; normalization must not invalidate source spans.
- Prioritize explicit linked targets and relation-bearing body mentions over
  generic sentence-start capitalization. Include apostrophes, hyphens, Unicode
  names, FEVER title encodings and repeated mentions in the design.
- Deduplicate normalized aliases before consuming query slots. Do not schedule
  an entity query if it adds no useful lead beyond the prior query. If no valid
  lead exists, leave that slot unused rather than inserting a formatting token.
- Preserve the existing fixed-target, one-hop and query/passage limits. Record
  lead selection/rejection reasons and the exact originating document/span.
- Version the planner/cache strategy so old cached query behavior is not confused
  with the revised planner. Reuse the existing FEVER index if its text can be
  handled at query time; a corpus rebuild is not the default solution.

Acceptance:

- Realistic `Title: ...` fixture passages never generate the header label as an
  entity query, while their useful linked mentions remain discoverable.
- Extracted spans slice the original passage exactly, including after normalized
  title handling. Empty/header-only passages produce no invented lead.
- A synthetic person-to-publication-to-frequency case retrieves the attribute
  passage within the same three-query budget, including the verifier-triggered
  continuation path. Missing evidence stays unresolved.
- Repeated aliases and duplicate hits cannot exhaust slots with equivalent
  queries; cached/fresh behavior remains deterministic and index-scoped.

Suggested commit: `fix(retrieval): ignore index headers when selecting entity leads`

## 2. Define an answer-to-question attribution contract

Files: `src/decompose.py`, `src/schema.py` only if new canonical fields are
necessary, decomposition tests and benchmark documentation.

- Represent which question clause/relationship each extracted proposition
  answers, which original answer span supplies the value, and what context was
  borrowed only to resolve a referent. Keep metadata separate from factual labels.
- Distinguish answer-bearing relationships from descriptive premises already in
  the question. A claim about an event's date/location cannot stand in for the
  relationship identifying that event as the answer to a person's death question.
- Require explicit attribution for every extracted claim. Reusing an answer span
  is necessary in many cases but is not sufficient: the same one-word span must
  not automatically be assigned to every clause of a compound question.
- Allow a fragment to answer one clause, several clearly shared clauses, or an
  ambiguous clause without guessing which absent fact must be true. For an
  ambiguous binding, retain explicit unresolved coverage rather than inventing
  an entity, relation or an extra answer.
- Keep uncertainty, negation, attribution and original span mapping. Keep genuine
  nonfactual text non-check-worthy. Question context and retrieved evidence are
  not authorization to add information that was absent from the answer.
- Make new metadata optional for historical deserialization; preserve old
  record/gold hashes when it is absent. Define the new semantics version before
  implementing checks so consumers have a stable contract.

Acceptance:

- Synthetic entity/date/frequency fragments map to the requested relationship,
  with valid answer spans. Compound questions with partial answers mark other
  clauses unanswered instead of filling them in.
- Shared-clause answers, ambiguous pronouns and explicit multi-clause answers
  work without forcing one universal interpretation.
- Old records and the existing final pilot bundle retain their fingerprints and
  remain auditable. No gold reference or evidence passage enters attribution.

Suggested commit: `feat(decomposition): record answer relationship attribution`

## 3. Enforce relation coverage with bounded extraction checks

Depends on step 2. Files: `src/decompose.py`, decomposition tests and private
pilot regression fixtures.

- Replace the current longer-than-fragment check as the sole quality safeguard.
  Check that extracted claims preserve the answer-bearing relation and its
  relevant argument roles, not just a plausible related sentence.
- Validate the attribution structure and span/clause references deterministically.
  Add bounded semantic checking for missing relationships and extra assertions
  using question, answer and proposed claims only. Keep this separate from
  evidence retrieval and truth verification.
- Do not treat the generator's own attribution labels as proof. Use a separate
  check with explicit missing-relation/unsupported-assertion findings and require
  concrete references to the proposed inventory. This model check is fallible;
  it is not an independent human evaluation.
- Feed actionable failures into the existing bounded retry. Cap extraction
  attempts and extra checking calls explicitly, include them in run identity,
  and record acceptance, retry and exhausted-budget reasons in diagnostics.
- Exhaustion remains unresolved coverage. Strict runs must surface the failure
  rather than substituting a guessed proposition or dropping a denominator.
  Do not add an unbounded retry loop until a desirable answer appears.

Acceptance:

- Reject an event date/location statement when the supplied answer expresses an
  identifying death relationship. Accept an unrelated synthetic case that
  preserves the relationship with a paraphrase rather than identical verbs.
- Reject a newly asserted institution/service relationship when the fragment
  answers only a definition clause, even if that extra proposition might be true.
- Accept genuinely explicit answers to both clauses. Preserve hedges and
  negation. Malformed, self-contradictory and incomplete check responses fail
  closed with bounded calls and useful diagnostics.
- Tests verify retry behavior and failure propagation, not merely prompt wording
  or mocked successful JSON. A live pilot is still needed to assess reliability.

Suggested commit: `fix(decomposition): reject lost relations and inferred answers`

## 4. Carry extraction coverage into pipeline success states

Depends on steps 2–3. Files: `src/pipeline.py`, `src/claim_tracking.py`,
`src/experiments/adapters.py`, diagnostic summaries and pipeline tests.

- Preserve extraction-attribution diagnostics with each original/candidate
  snapshot. Original-to-candidate tracking cannot independently establish that
  the original predicted inventory covered the answer correctly.
- Prevent `all_targets_verified` when required answer relationships are missing
  or extraction attribution is unresolved. Keep current-inventory SCR distinct
  from extraction coverage and from independently reviewed factual performance.
- Preserve original predicted claims and the existing removed/hedged/unresolved
  dispositions. Missing original facts cannot be silently manufactured by a
  tracking or retrieval stage.
- Expose separate reasons for extraction failure, evidence gaps, rejected edits
  and successful verification of a complete predicted inventory. Avoid collapsing
  these into a single success percentage.

Acceptance:

- An entirely Supported but incorrectly extracted inventory cannot produce a
  successful end-to-end diagnostic state.
- Valid extraction plus supported targets can pass; valid extraction plus
  missing evidence remains unresolved. No-op, empty-answer and optional
  hedge/delete behavior remains explicit and does not become factual correction.
- Native adapters and round-trip serialization preserve these distinctions.

Suggested commit: `fix(pipeline): gate verification success on extraction coverage`

## 5. Validate and execute a fresh development pilot

Depends on steps 1–4. Files: versioned runtime profile, notebook/handoff,
`tests/`, diagnostic inspection and packaging scripts as needed.

- Run focused regressions, then the full offline suite, notebook schema/Python
  validation, diff checks and audit of the earlier human-adjudicated final bundle.
  Confirm the v3 progress bundle remains valid under its historical semantics.
- Create a new versioned planner/extraction configuration and fresh run name,
  such as `pilot_followup_v4`. Package the exact code update, preserving the
  working Colab environment and corpus index. Keep source/package snapshots,
  effective model settings, request counts and query budgets.
- Run the same five dev answers and both systems first. Compare per-record
  decomposition relations, attribution decisions, selected entity leads,
  retrieved evidence, repair attempts and output edits with v3. Lexical reference
  agreement remains a diagnostic, not the acceptance criterion for factuality.
- If extraction remains unresolved and execution is incomplete, report the
  failure and preserve its attempts. Do not exclude failed records or publish a
  final comparison on the successful subset.
- Inspect complete new outputs before asking reviewers to work. Then export a
  fresh blinded packet, collect independent human ratings and adjudication, and
  generate/audit the final bundle. Do not transfer v3 or earlier human ratings.
- The existing pilot has only one correction-eligible answer. After operational
  defects are fixed, prepare a separately versioned, annotated dev extension
  containing actual evidence-backed correctable errors and no-op cases. Do not
  silently replace the five originals or claim general efficacy from this pilot.
  Any added data needs its own annotation provenance. Keep the reserved test
  split untouched until settings are frozen.

Acceptance:

- New traces contain no header-token lead and show correct answer relationship
  coverage or an explicit unresolved extraction status for every case.
- Grounded corrections, conservative abstentions and newly introduced errors are
  reported separately. No minimum accuracy or forced edit count is stipulated.
- A final quality claim requires a complete execution, new adjudicated human
  assessments and a successful offline audit. A run that only abstains does not
  demonstrate evidence-backed correction.

Suggested commit: `chore(evaluation): prepare and validate the v4 development pilot`

## Execution order and boundaries

Implement step 1 first, then define and test the extraction contract before
adding its semantic checker and pipeline gate. Finish offline validation before
spending further API budget. Human review comes after the corrected fresh run.

This plan does not authorize a corpus rebuild, new trained models, changes to
accepted gold or reserved test examples, or relabeling model drafts as human
reviews. No live model calls are needed to implement the fixes; the later Colab
execution and human-review steps are operational dependencies.
