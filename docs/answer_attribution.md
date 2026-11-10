# Answer attribution contract, version 1

This contract records a model's proposed mapping from answer content to question
relationships. It is stored at `record.decomposition_diagnostics.attribution`;
no new field is required on canonical claims or historical gold records.
Decomposition diagnostics now use semantics version 5; the nested attribution
contract has its own `semantics_version: 1`.

## Question clauses and source coordinates

`question_clauses` contains stable `q1`, `q2`, ... identifiers, exact question text
and half-open original character spans. The current `surface_v1` segmentation
splits on question marks, semicolons, newlines and conjunctions introducing a new
WH question. It is a structural partition, not a complete syntactic or semantic
parse. Ambiguous or unsplit questions must not be treated as fully understood.

The decomposition prompt includes these clause IDs, original answer character
length, and numbered sentence offsets. Attribution character coordinates always
refer to the original strings, not cleaned or expanded claim text.

## Per-claim proposal

Each generated claim can include an `attribution` object with exactly:

- `status`: `bound`, `ambiguous`, `answer_only`, or `nonfactual`.
- `answer_spans`: a nonempty list of `[start, end]` spans supplying the claim's
  answer content. Each must lie inside the claim's mapped original sentence.
- `bindings`: a list of question-relationship proposals as described below.
- `rationale`: nonempty text explaining the attribution.

A binding has `clause_id`, `answer_span`, `answer_text`, `relation`, and
`borrowed_context`. `answer_text` must equal the original answer slice, and its
span must lie within a declared `answer_spans` span. `relation` names the
answer-bearing relationship, such as birthplace or publication frequency,
rather than a descriptive premise appearing only in the question.

Each borrowed context entry contains `source` (`question` or `answer`), `span`,
`text`, and `role` (`referent` or `relation_context`). Text must exactly match
its original source span. This supports question referents and coreference to
previous answer sentences. Context is not evidence that a proposition is true
and cannot authorize supplying a missing answer to another question clause.

`bound` requires at least one valid binding. `ambiguous` may leave bindings empty
instead of inventing a referent or relationship. `answer_only` is an explicit
answer fact without a question-clause binding. `nonfactual` covers opinion or
boilerplate and must agree with `check_worthy: false`; other statuses require
check-worthy content. Answer-only/nonfactual proposals have no bindings.
The same span can be proposed for multiple atomic claims or clauses; this is
necessary for genuinely shared answers, but is not automatically justified.

## Coverage and validation boundaries

Normalized per-claim entries reference generated claim IDs. Per-clause coverage
is `proposed_answered`, `ambiguous`, `unanswered`, or `unknown`. These labels
summarize the proposal, not independently assessed question completeness.
Unanswered means no binding was proposed by a completely annotated inventory.
Missing legacy annotations produce unknown coverage, not inferred success.

Provided malformed metadata enters the existing bounded retry with explicit
feedback. Retry exhaustion is unresolved coverage and fails strict execution.
Absent metadata is accepted as a legacy/unannotated response with attribution
status `not_provided` (or `partial` for a mixed inventory). It is never upgraded
to a validated relationship. Empty answers have explicit `empty_answer` status.

`validation: structural_only_not_semantic_or_factual` is deliberate. Correct
spans, a plausible relation string, or the generator's own `bound` label do not
establish that the answer actually expresses that fact. A compound-question
fragment can have valid coordinates while still being bound to the wrong clause.
The separate semantic extraction checker and pipeline gate from phases 3 and 4
of `docs/v3_followup_implementation_plan.md` are now implemented. They do not
change the structural contract's meaning; their results are separate diagnostics.
Direct `decompose()` calls retain `extraction_check="structural"` for compatibility.
Pipeline/runner defaults explicitly select `semantic_v1` and require full proposed
attribution for the semantic check. Missing or partial metadata cannot pass it.

Historical records without this metadata retain their previous serialization
and content hashes. Existing adjudicated and progress bundles remain historical
artifacts. A new run requires a fresh output directory and, after the remaining
fixes, new human assessments for changed outputs.


## Semantic extraction audit and pipeline gating

The audit sees the original question/answer, canonical proposed claim text and
attribution only. It receives no retrieved evidence, verdict labels, reference
repairs, reviewer ratings or gold. Its prompt independently tests answer-bearing
relationships, argument roles, missing factual content, unsupported extra
assertions, negation and uncertainty. It permits legitimate unanswered question
clauses and semantic paraphrases. This remains a fallible model diagnostic,
not independent human review or proof that an answer is factually true.

`semantic_check` records per-claim `expressed`, `nonfactual`, `wrong_relation`,
`unsupported_assertion` or `ambiguous` findings, plus per-clause `represented`,
`unanswered`, `missing_relation` or `ambiguous` findings. Missing-content findings
reference original answer spans. Every claim/clause must be covered exactly once;
references, enum values, rationales, missing-content spans and cross-check
consistency are validated. Empty blanket approvals and contradictory findings
fail closed. Each accepted result includes a fingerprint binding it to the exact
question, answer, canonical inventory and proposed attribution.

Decomposition retries use actionable findings with explicit attempt/check/inventory
caps. Attempt history and consumed budgets remain in diagnostics. Semantic retry
exhaustion is unresolved coverage; strict execution fails and retains the normal
benchmark error/trace history. Structured `event: extraction` trace entries are
emitted before strict failure, preserving attempt findings and budgets even when
no candidate output is committed. There is no retry-until-success loop. A check of a
changed inventory requires a new fingerprint; a cached/stale result cannot approve
an altered claim. Structural mode never supplies a semantic pass implicitly.

Each pipeline snapshot preserves extraction proposals and check findings. Original
and candidate extraction coverage are reported separately from their factual
verification counts. `target_verification_success` describes the original
predicted-target tracking condition alone. `factual_success` and the
`all_targets_verified` stop state additionally require passed, input-bound semantic
extraction checks on both original and final candidate. A wholly Supported but
incorrectly extracted inventory can still have numeric SCR 1.0 while its pipeline
state is `unresolved_extraction_coverage`.

In semantic mode, unresolved generated extraction abstains from repair. Supplied
fixed inventories skip original extraction and remain explicitly unassessed for
this diagnostic; they can still be verified/repaired for fixed-inventory tasks.
Generated replacements are checked normally. This does not certify end-to-end
extraction for an externally supplied original inventory. Empty answers, evidence
gaps, no-op edits, target alignment failures and extraction failures remain distinct.


## Exact-quote recovery and audit formatting retries

`attribution_anchor_mode=unique_quote_v1` can correct a borrowed-context numeric
span only when the unchanged quoted string occurs exactly once in its declared
original question/answer. It does not fuzzy-match, change source text, relabel a
relationship or choose among duplicate occurrences. Strict structural validation
runs afterwards. Attempt diagnostics retain `span_repairs` with old/new positions;
raw model responses remain in request traces. `strict` mode disables recovery.

Audit prompt version 2 clarifies that an expressed `answer_only` fact has no
answered-clause IDs; a question's absent answer belongs in `clause_checks` as
unanswered, never in `missing_content` with an empty or zero-length answer span.
One malformed-format/consistency retry can reuse the same extraction, within the
existing logical check cap. Its prior failure is retained. Semantic rejections
still return to generation, and contradictions cannot be automatically approved.
