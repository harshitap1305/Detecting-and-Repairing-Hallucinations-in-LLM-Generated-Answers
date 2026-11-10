# Execute and resume a benchmark

Increment 2 implements system execution and artifact validation. The separate
[increment 3 scoring tools](independent_evaluation.md) handle metrics and human
review; [increment 4](evaluation_bundles.md) packages reports and audited bundles. `execution_complete` means all
requested system/record outputs exist, not that a scientific benchmark is complete.

## Inputs and preflight

Use a configured `configs/evaluation.yaml`, processed input JSONL for one split,
and a split file with populated, disjoint dev/test IDs. Input IDs must exactly
match the selected split, and both split counts must match configuration. For a
small development pilot, create a separate configuration and split file with
smaller explicit counts; the runner never silently truncates a final benchmark.

The runner validates native capabilities and selects only configured systems
supporting `--task`. Each invocation executes one task:

- `repair`: question/answer only; native decomposition and retrieval. Includes
  pipeline, B0, B2–B7. B6 uses pipeline-predicted flags with zero repair iterations,
  then deletes flagged sentences. No gold detector labels are used.
- `answer_detection`: B1's native answer-level output, with typed boolean and
  bounded numeric confidence. No fabricated claim verdicts or repair text.
- `claim_verification`: pipeline and B7 use an explicitly supplied human inventory
  (`--inventory`). Record text/IDs must match inputs exactly, and claims must have
  unique IDs, valid sentence mappings, boolean check-worthiness, and valid spans.
  Gold labels, rationales, citations, and references are removed before execution.
  Repair generation is disabled, and output inventories must remain unchanged.

Source input fields are allowlisted into fresh `AnswerRecord` objects. Even if
an input or inventory file contains gold, its labels/evidence/reference repairs
are not forwarded to generation. This is input isolation, not certification of
annotation quality: use the separate gold validation workflow before a final run.

Install the configured spaCy model at the exact declared package version.
Sentence-dependent execution is pinned to that loaded model with no fallback.
The runner also requires the selected provider's SDK and environment credential,
except for B0-only execution. It checks retrieval asset presence and hashes
contents before running. Configuration corpus/index paths are relative to the
configuration file; CLI FAISS/passages paths are relative to the working directory.

## Commands

```bash
python3 scripts/run_benchmark.py \
  --config configs/evaluation.yaml \
  --inputs data/gold/dev.jsonl \
  --splits data/splits.json \
  --task repair --split dev \
  --provider groq --model llama-3.3-70b-versatile \
  --output-dir results/evaluation/dev-repair-v1
```

For fixed-inventory verification, use a separate output directory and replace
the task with `--task claim_verification --inventory data/gold/dev.jsonl`.
For answer-level detection use `--task answer_detection` without `--inventory`.

Runtime options include `--top-k`, `--verifier` (`llm`, `nli`, `ensemble`),
`--repair-policy`, `--max-repair-iterations`, `--n-samples`, and
`--consistency-threshold`. Dense/hybrid retrieval additionally requires
`--faiss-index-path` and `--passages-path`. Trained A/C checkpoints remain on
the existing comparison route until their later adapter integration.

A provider model ID can itself specify a dated/versioned release. Hosted APIs
have no separate revision argument here; `--model-revision` remains
`hosted-unpinned`. Local Transformers loading requires either a local model
directory (all files are hashed) or `--model-revision` set to a full 40-character
upstream commit. That revision is passed to the local loader and included in
its in-memory and disk cache identity. Auxiliary NLI/dense/reranker model IDs
are recorded with an explicitly unpinned upstream revision policy. Stored
outputs support offline reproduction; new remote/pretrained inference is not
claimed to be bit-for-bit reproducible.

Resume by repeating the identical command with `--resume`. Existing directories
are refused without that flag. A nonzero exit status signals incomplete outputs
or failed preflight/integrity checks. Model, endpoint, settings, inputs, split,
inventory, retrieval contents, source code, package versions, or recorded hardware
changes prevent resumption. Fix execution conditions without changing experiment
identity, or use a new directory for a changed experiment.

## Persistence contract

```text
<run>/
  manifest.json
  attempts/<system-record-hash>/<attempt-number>.json
  traces/<unique-id>.jsonl
  outputs/<system>.jsonl
  errors.jsonl
  llm_cache.db
  retrieval_cache.db
  .lock
```

The manifest records resolved options, configuration/input/index hashes, Git
commit and source digest, dirty-worktree flag, package/Python versions, machine
information, model identities, seed, timestamps, and explicit cache policy.
GPU names, memory, and CUDA runtime are recorded when PyTorch is available;
otherwise GPU details are explicitly unavailable. Token usage/cost are also
explicitly unavailable from the legacy backends rather than estimated as
measured usage.

Each successful attempt has the versioned envelope: original input identity,
system/task, canonical `AnswerRecord`, native scores, additional retrieved
evidence where applicable, status, timing, seed, and trace path/hash. B2 retains
per-sentence and mean consistency under native names and clears its overloaded
factual-reliability field. Pipeline SCR stays a named self-verification diagnostic.
B5 whole-answer passages are saved separately; claim-based systems save evidence
inside their claims. Raw LLM requests/responses, cache hits, and backend errors are
flushed to trace files as calls occur. Trace files contain source text and should
be handled with the same access restrictions as inputs. The default run directory
is ignored by Git.

Attempt files are written atomically and checksummed. Successful cells are never
run again during resume. Errors are retained as numbered failed attempts, and
retrying adds another attempt with the same per-system/record seed. Failed-cell
retries bypass the LLM cache; B2 stochastic sampling always bypasses it. Caches
are scoped to the run, and LLM keys include provider and endpoint. In-memory
retrieval stores are isolated for each invocation so rebuilt indexes cannot reuse
stale passages from a preceding run in the same process.

Per-record attempts are authoritative. Output JSONL and errors JSONL are derived
views rebuilt on normal completion or a handled interruption. An abrupt process
kill can leave stale views or an orphan trace; resume recovers committed attempts
and retries work without a committed result. It never interprets a partial trace
as a successful record. Trace/attempt corruption, missing previously completed
outputs, duplicate successes, changed input identity, and changed experiment
identity abort resume. The output directory has an advisory POSIX lock; this
runner supports one writer on Linux/macOS, not Windows or a distributed filesystem
without compatible locking semantics.

Strict execution context makes malformed decomposition/verifier/B1/B2/B4/B7
responses and retrieval/reranker exceptions visible failures. Legitimate zero-hit
retrieval remains valid and can produce Unverifiable. Existing interactive callers
retain their legacy fallback behavior outside this context.

## Verification

Offline tests cover all adapters using synthetic model/retrieval responses, input
isolation, failed attempts, interruption/resume, output integrity, cache scoping,
and strict parsing. Run:

```bash
python3 -m unittest tests.test_benchmark_runner tests.test_evaluation_preparation -v
```

Live provider/model execution still requires real datasets, declared retrieval
assets, installed segmenter/provider dependencies, and credentials. No live scores
are supplied by this implementation.

## Offline scoring and human review

Increment 3 adds [offline evaluation and independent repair review](independent_evaluation.md).
Use `scripts/run_eval.py --run-dir ... --gold ... --output ...` for complete saved
runs and `scripts/review_repairs.py` for blinded human packets, rating imports,
and adjudication. This does not change execution completion into a factual
performance claim; independent ratings and the final report remain separate.

## Self-verification and decomposition diagnostics (semantics version 2)

New pipeline records include `verification_diagnostics` with separate `before`
and `after` scores, check-worthy/supported/contradicted/unverifiable/unlabeled
counts, undefined-score reasons, and an explicit `stop_reason`. No check-worthy
claims means `null`, never perfect support. `empty_answer` distinguishes deletion
from a nonempty candidate with `no_check_worthy_claims`. Unchanged answers and
exhausted budgets are stop conditions, not evidence of a correction. A stop
reason `all_current_claims_supported` in older runs described only the current
predicted inventory; new runs additionally gate success on tracked target coverage.
Neither diagnostic establishes independently reviewed factuality or completeness.
The runner's native SCR diagnostic includes these details. UI/logging displays
undefined values as N/A. Legacy aggregate diagnostics expose defined/paired
coverage and do not substitute original-claim scores for undefined final scores.
The legacy significance CLI makes SCR unavailable when any paired record lacks a
score rather than silently dropping pairs.

Decomposition uses question context to expand short fragments while retaining
answer spans and uncertainty/attribution in claim text. Questions do not supply
missing factual answers. A bounded retry handles malformed inventories, copied
short fragments and detectable loss of factual modality. `decomposition_diagnostics`
records attempts and coverage status. These surface heuristics are not semantic
validation: ambiguous/unrecognized cases still require empirical evaluation.
Exhaustion produces explicit unresolved coverage in interactive mode and fails
strict benchmark execution. Opinions and boilerplate remain non-check-worthy.

Historical records without these optional fields serialize as before, including
their original hashes and stored numeric SCR values. Existing bundles are not
rewritten or relabeled. New code/prompt settings require a new run directory and
new repair assessments. Original-to-candidate semantic tracking and retrieval or
repair-policy changes remain later phases of the follow-up plan.

## Claim tracking and contextual retrieval (follow-up phases 3–4)

The pipeline keeps the original predicted claims immutable during repair and
records candidate snapshots under `record.claim_tracking`. Each stage contains
the candidate answer/claims, decomposition and verification diagnostics, evidence
query traces, and original-to-candidate links. These are **model diagnostics**,
not substitutes for human outcome ratings. An unchanged answer/inventory links
directly; changed candidates use at most one model alignment request per stage.
The request contains question, original/candidate text and predicted claims,
never gold reference repairs or human repair ratings. Many-to-many links and
`retained`, `modified`, `hedged`, `removed`, `unresolved` dispositions are supported.
Malformed/incomplete alignment or an exceeded claim budget remains unresolved.
Empty candidates are removed targets; absent decomposition is not proof of
removal. Candidate-only propositions retain their verification labels/evidence.

`all_targets_verified` requires nonempty original and candidate factual targets,
full grounded target coverage, preservation of originally Supported targets, and
Supported verdicts for all candidate factual claims. A hedge, unmatched target,
unsupported new claim or ambiguous alignment prevents that state even when the
current-inventory SCR is 1.0. A stop with `unresolved_target_coverage` is diagnostic
abstention, not a successful repair. Semantic alignment and verification are
fallible, so this state still does not establish independent factual correctness
or question completeness. No-op and empty-answer stop reasons stay explicit.

Pipeline and B6 runner options now include:

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--query-strategy` | `contextual_v1` | `claim_only` or bounded contextual fallback |
| `--retrieval-max-queries` | `3` | Logical queries per target per grounding stage, including its retry; allowed 1–8 |
| `--retrieval-max-passages` | `20` | Total returned hits before deduplication per target/stage; between top-k and 200 |
| `--tracking-max-claims` | `64` | Combined original/candidate check-worthy inventory cap for model alignment; allowed 1–512 |

These options are validated before inference and frozen in run identity; changing
one invalidates resume. B6 uses contextual retrieval for its detector; its deletion output has no final
verification score or candidate tracking because it is not reverified. Its
detector-only tracking is labeled separately. Other native baselines retain their
own retrieval algorithms. The standalone `retrieve()` API defaults to `claim_only` for existing
callers; the updated pipeline explicitly selects `contextual_v1`.

Contextual search starts with the fixed claim. Short claims, missing surface
terms or too few hits can trigger a question-conditioned query. Unverifiable
verdicts can also trigger continuation with the **remaining** budget, followed
by at most one extra verification per expanded target/stage, before editing.
Passages are evidence candidates, not verified premises. Proper-name/wiki-link
mentions from initial/context passages can seed bounded single-hop entity queries.
Each lead records its source document and exact text span. No recursive crawl,
new model-generated search facts, or live web search is used. Missing evidence
remains Unverifiable; absence is not treated as contradiction.

Returned hits are deduplicated by document/chunk and fused deterministically by
rank, with a stable tie break; optional reranking uses the original claim target.
Caps count logical queries and their returned hits, not internal index scans or
the native hybrid backend's component searches. Retrieval planning makes no
extra LLM calls. For `N` original claims and repair budget `I`, at most `I+1`
grounding stages are needed; each stage's targets have their own explicit caps.
Each distinct changed, verified candidate has at most one alignment call (an
unchanged inventory or unavailable/oversized inventory needs none).

Retrieval diagnostics are stored on records and emitted as `event: retrieval`
entries in benchmark traces. Cache keys include query text, question, strategy,
caps, method, asset versions and encoder/reranker identities. In-memory lexical
and dense index stores also use file identities. Query logs retain both fresh
and cached hits. These surface/query heuristics are not evidence-recall or
factuality metrics. Original gold-query retrieval inspection remains a separate
diagnostic and never enters inference.

New runs need fresh output directories and new human assessments; existing pilot
artifacts and audit semantics are preserved. The unchanged FEVER index can be
reused. Example additional options for the next development run:

```text
--query-strategy contextual_v1 --retrieval-max-queries 3 \
--retrieval-max-passages 20 --tracking-max-claims 64
```

## Entity planner version 2 and proposed answer attribution

The contextual workflow now records `entity_planner_version: 2` in retrieval
settings and cache scope. It recognizes the corpus builder's exact `Title: ...`
prefix, selects entity leads from the remaining body, and retains offsets into
the original stored text. Header labels are excluded by source region, not by a
global word blacklist. Explicit links and body names are prioritized by target
terms and relation context; apostrophes, Unicode, hyphens and FEVER title encoding
are normalized for query/alias deduplication. `entity` remains the exact source
slice; `query_entity` is its normalized search form. Lead diagnostics explain
header/known/context/duplicate rejections and whether a lead consumed a query.
No usable lead means no fabricated query. Query/passage caps remain unchanged,
and the existing index can be reused without rebuilding it.

Decomposition also records the [answer attribution contract](answer_attribution.md).
It names question relationships, original answer spans, borrowed referent/context
spans and ambiguous or unanswered clauses. Structural validation does not establish
semantic correctness. Missing old annotations remain explicit unknown coverage;
the separate bounded semantic checker and success-state gate are now implemented.


## Semantic extraction settings

Pipeline/B6 runtime options include the following, frozen into run identity:

| CLI option | Default | Bounds |
| --- | --- | --- |
| `--extraction-check` | `semantic_v1` | `semantic_v1` or compatibility `structural` |
| `--extraction-max-attempts` | `2` | 1–4 generation attempts per decomposition |
| `--extraction-max-checks` | `2` | 1–4 logical semantic-check calls per decomposition |
| `--extraction-max-claims` | `64` | 1–128 combined proposed claims and question clauses |

Each accepted generation has at most one separate audit call; missing/ambiguous
attribution and inventory-cap overflow require none. Cache hits still consume a
logical check slot. Generation stops when attempts or check calls are exhausted;
provider failures are surfaced normally. Checks request 2048 completion tokens,
subject to the recorded effective model floor (4096 by default for GPT-OSS).
A zero-call structural precondition failure is not a successful semantic check.

Native SCR details now include before/after extraction coverage. The saved claim
tracking snapshots and `inspect_pilot_run.py` preserve extraction findings and
separate target-only verification success. These diagnostics cannot substitute
for a new independent human assessment. Both historical final and v3 progress
bundles continue to audit under their recorded semantics. Use a fresh run directory
after the phase 5 live-pilot handoff; do not resume an old run with these changes.
