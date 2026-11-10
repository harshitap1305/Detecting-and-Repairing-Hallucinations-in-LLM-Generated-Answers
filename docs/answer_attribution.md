# Answer attribution contract, version 1

This contract records a model's proposed mapping from answer content to question
relationships. It is stored at `record.decomposition_diagnostics.attribution`;
no new field is required on canonical claims or historical gold records.
Decomposition diagnostics now use semantics version 3; the nested attribution
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
The separate semantic extraction checker and pipeline gate are phases 3 and 4
of `docs/v3_followup_implementation_plan.md`; they are not implemented by this
contract. No new evidence queries or semantic-checker model calls are added here.

Historical records without this metadata retain their previous serialization
and content hashes. Existing adjudicated and progress bundles remain historical
artifacts. A new run requires a fresh output directory and, after the remaining
fixes, new human assessments for changed outputs.
