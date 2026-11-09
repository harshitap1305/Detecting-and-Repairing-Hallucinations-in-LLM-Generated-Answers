# Reproducible evaluation protocol (version 1)

Status: preparation infrastructure is implemented. Human gold, system execution,
independent assessment, and final report generation are subsequent milestones.
The current numerical reports remain unverified.

## Configuration and sampling

Use `configs/evaluation.yaml` as a template. Configuration schema version 1 is
validated by `src.experiments.protocol.load_config`. Unknown keys, duplicate
systems, unsupported tasks/retrievers, invalid quotas, and missing source
versions are errors. The checked-in template deliberately has no source files:
fill these in before preparing a batch. Versions are user-supplied provenance;
the preparer additionally hashes actual source bytes.

Default target: 50 development answers and 100 test answers. Add source entries
with `id` matching `source_dataset`, a processed JSONL `path`, an exact upstream
revision and conversion revision in `version`, and explicit `dev`/`test` quotas.
Paths resolve relative to the configuration file. Quotas must sum exactly to
the target counts. Record the retrieval corpus snapshot and index location;
preparation does not require an index or credentials. Index content hashes and
actual runtime versions belong to the later experiment manifest.

Inputs must have globally unique IDs, nonempty questions/answers, and declared
`dev` or `test` assignments. Original source split assignments are preserved;
external test-only data must have a zero development quota. Dataset access and
license choices remain the data curator's responsibility.

Before any predictions are examined, group prompts using Unicode case-folding
and collapsed whitespace. Exclude all members of groups spanning source dev/test
assignments. For remaining duplicate prompts retain the lowest SHA-256 rank of
`[seed, source_dataset, id]`; exclude the rest even if answers differ. This
conservative policy spans all sources. It can reduce a source's available quota;
insufficient quota is an error, never an implicit refill from a different source.
Within each source and original split, select the lowest ranks to satisfy its
quota. Source-list and JSONL row order do not affect selected IDs. Byte hashes
still change when a source file changes order. Near-duplicate paraphrases and
entity overlap require human review; exact prompt grouping does not detect them.

## Tasks and system capabilities

Capabilities describe this repository's implementations, which are adaptations
rather than claims of faithful reproduction of published systems.

| System | Native tasks | Retrieval | Native score / prerequisite |
|---|---|---|---|
| Pipeline | Claim verification; repair | BM25, FTS5, dense, hybrid, hybrid+rerank | Self-verification SCR |
| B0 | Unchanged-answer repair control | None | No detector |
| B1 | Answer-level detection | None | Answer confidence; no claim verdicts |
| B2 | Consistency-based hedging | None | Sentence consistency; not factual SCR |
| B3 | Self-repair | None | No external evidence |
| B4 | CoVe-style repair | None | Internal verification questions |
| B5 | RARR-style repair | BM25 | Whole question/answer retrieval |
| B6 | Delete-only repair | Same methods as pipeline detector | Requires `pipeline_detector` |
| B7 | Claim verification; repair | BM25 | Per-claim retrieved evidence |

`SYSTEMS` in `src/experiments/protocol.py` is the machine-readable registry.
Configuration validation does not execute adapters or certify a completed run.

Run answer detection separately: derive gold answer hallucination as any
check-worthy Contradicted or Unverifiable claim. B1 is applicable here; do not
invent an answer detector for systems that do not provide one.

For fixed-inventory claim verification, give applicable systems the same
human-reviewed claim text, IDs, spans, and check-worthiness, with verdicts,
rationales, citations, and reference repairs removed. Report three-way macro-F1,
per-class scores, confusion matrix, supports, and coverage, and separately
binary hallucination F1. This is an oracle claim-inventory experiment and does
not measure automatic decomposition accuracy.

For end-to-end repair, supply question and original answer, allowing native
system decomposition and retrieval. Independently assess repaired text from
all systems using the same rubric. Claim detection scores from this task need
a separately validated alignment procedure; similarly numbered claims are not
necessarily the same fact.

B6 must obtain sentence flags from the pipeline's predicted detector output,
using the declared corpus/verifier/settings and no gold labels or gold evidence.
It is a shared-detector ablation. B5 retrieves using the whole answer; B7 and
the pipeline retrieve by claim. Keep the corpus snapshot and top-k fixed where
applicable, but record query formulation and retrieval differences. Evidence is
independently retrieved; gold-evidence experiments must be labeled diagnostics.

## Annotation handoff

Run after filling actual sources and quotas:

```bash
python3 scripts/prepare_annotation_batch.py \
  --config configs/evaluation.yaml \
  --output-dir data/annotation_batches/v1
```

The command writes fresh canonical `AnswerRecord` JSONL files under `inputs/`,
`inventory/`, `annotator_a/`, and `annotator_b/`, plus `splits.json` and
`manifest.json`. These packets start identically blank: no claims, labels,
citations, repairs, confidence scores, or source predictions are forwarded.
The manifest records source versions/hashes, configuration/hash, selected IDs,
original splits, all excluded IDs and reasons, and output hashes. Existing
output directories are refused. Preparation neither changes `data/splits.json`
nor creates final gold. It makes no model calls and creates no benchmark scores.

1. Curators edit `inventory/` into a shared atomic claim inventory, checking
   completeness and spans. Do not infer labels from the source's annotations.
2. Freeze and hash that inventory. Reissue identical copies to the two annotator
   directories before independent labeling; the initially blank packets cannot
   establish agreement. Preserve the frozen inventory and raw submissions.
3. Annotators label against inspected citation snapshots, then propose minimal
   reference repairs. Keep Supported, Contradicted, and Unverifiable distinct;
   uncertainty wording does not make an unsupported proposition correct.
4. Calculate agreement on the same inventory, including coverage and label
   counts; retain disagreements and an adjudication log. The existing kappa
   target of 0.70 is a review target, not an automatic correctness certificate.
   Reissue changed inventory items for annotation rather than comparing stale IDs.
5. Curators create adjudicated `data/gold/dev.jsonl`, `data/gold/test.jsonl`, and
   the corresponding split file; run `scripts/validate_gold.py` as described in
   [the gold workflow](gold_annotation_workflow.md). Freeze final hashes only
   after structural validation and human review. Keep private artifacts private.

Sentence spans use the declared spaCy model and exact model package version.
`require_segmenter(config)` checks the installed package version, loads it, and
refuses missing sentence components or missing packages; it never downloads a
model or falls back. Preparation does not need sentence segmentation because
claims are blank. Inventory editing and future benchmark adapters must call
this check and record the spaCy runtime version. The current legacy pipeline
and gold-validator CLI are not yet wired to this configuration: curators must
use the same installed model for validation. Integration and automatic version
recording are part of the runner increment.

## Metrics, independent review, and test discipline

The following is the contract for subsequent evaluation implementation, not a
claim that the current metrics module implements these checks:

- Primary claim metric: three-way macro-F1 on complete fixed inventories.
  Bootstrap answers, keeping their claims together; report 95% percentile
  intervals with 1,000 resamples and seed 42.
- Primary repair outcomes: fractions of originally incorrect factual claims
  corrected, and originally supported facts retained, judged independently.
  Record original claims as corrected, retained, deleted, hedged, or still
  incorrect. Count newly introduced factual errors and assess completeness.
- Human-reference exact match and ROUGE-L measure editing agreement. They do
  not establish factual correctness. Preserve empty-string deletions in scoring.
- Pipeline SCR, consistency, and token overlap are separately named diagnostics.
  Empty output or hedging is never automatically a successful correction.
- Blind human reviewers to system identity and shuffle system outputs. Resolve
  disagreements, preserve raw assessments and citations, and report agreement.
  Model-judge assistance is separate and records model, evidence, and rubric.
- Compare every repair system with B0 on the two primary repair outcomes. Treat
  these comparisons as one multiplicity family. Claim-verification comparisons
  against the prompted pipeline are a separate family. If reporting p-values,
  use paired tests with explicit tie/degenerate handling, SciPy, and Holm
  correction. Require exact paired IDs and report missing assessments as missing.

Tune settings on development data only. Freeze the protocol/configuration before
final test execution; preserve reruns and deviations rather than replacing a
favorable result. Software readiness, annotation completeness, successful
execution, and independently evaluated benchmark completion are separate states.
