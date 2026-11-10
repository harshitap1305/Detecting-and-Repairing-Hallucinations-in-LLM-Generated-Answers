# Step 5: real-data preparation and human handoff

Status (11 November 2026): the full benchmark remains incomplete. The first
five development records have reviewed gold/citations; a real FEVER index and API
access were used in Colab. Live v4b execution, human repair assessment,
adjudication and final-bundle auditing are complete for those five records.
The remaining 50/100 cohort annotations and full benchmark are pending.
See the [current handoff](project_handoff.md). The preparation history and original
operational instructions below are retained for reference.

## Work completed in this checkout

- Downloaded the official HaluEval QA source at commit
  [`b7253db3cdaa0ab2c382f92b26b390109174f77e`](https://github.com/RUCAIBox/HaluEval/tree/b7253db3cdaa0ab2c382f92b26b390109174f77e).
- Verified local source files against the pinned Git objects and retained raw
  data, README, and license snapshots under ignored processed-data storage.
- Converted 10,000 unique questions to one unlabeled answer candidate per prompt,
  preserving prompt-disjoint source splits (7,999 dev / 2,001 test).
- Mixed upstream `right_answer` and `hallucinated_answer` origins approximately
  equally within each source split before annotation sampling. The existing
  download route uses only hallucinated variants; this separate, explicitly
  versioned route provides candidate negative controls for answer detection.
  Upstream origins are provenance, **not human factuality labels**.
- Prepared 50 dev and 100 test candidates with seed 42. The selected dev set has
  24 upstream reference / 26 generated-hallucination origins; test has 48 / 52.
  Actual human class distributions remain unknown. Generation inputs and blank
  annotation packets contain no verdicts, citations, or reference repairs.
- Exported human-readable sentence IDs and exact character offsets using
  spaCy 3.7.5 and `en_core_web_sm` 3.7.1, matching the declared segmenter.
- Completed a **B0-only operational smoke run** on five actual dev candidates:
  5/5 outputs persisted. B0 leaves text unchanged and makes no model calls.
  This is not a live pipeline pilot or an independently assessed result.
- Ran the complete pytest suite in an isolated environment: **219 tests and
  35 subtests passed**. Dependency checks passed. A full environment freeze is
  saved privately with the preparation artifacts.

Raw QA SHA-256:
`89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`.

This initial cohort is HaluEval-only. Its eventual results cannot establish
cross-dataset generalization to FELM, WikiBio, or Factcheck-Bench. The evidence
corpus remains the declared FEVER snapshot; source context is not secretly
supplied as oracle evidence to the pipeline.

## Files ready for human work

| Artifact | Location |
|---|---|
| Pinned raw source, candidates, provenance | `data/processed/halueval_v1/` |
| Full 50/100 blank annotation packets | `data/annotation_batches/halueval_v1/` |
| Full sentence references | `data/annotation_batches/halueval_v1_sentence_reference/` |
| First-five-dev handoff | `data/annotation_batches/halueval_v1_operational_pilot/` |
| Main cohort configuration | `configs/evaluation_halueval_v1.yaml` |
| Future live pilot configuration | `configs/evaluation_halueval_pilot.yaml` |
| B0 execution smoke artifacts | `results/evaluation/halueval_b0_smoke_v1/` |
| Machine-readable readiness | `results/evaluation/halueval_step5_readiness.json` |
| Exact preparation environment | `results/evaluation/halueval_preparation_environment.freeze.txt` |

Data, reference text, and raw outputs are ignored by Git. `data/splits.json` and
final gold files have not been populated from these unreviewed candidates.
The 100 test IDs are selected, but test outputs must not guide model selection.

## Start with the five development records

Open the pilot handoff's `README.txt`, `sentence_reference.txt`, and
`inventory/dev.jsonl`. Its records are:

```text
halueval_qa_03835
halueval_qa_08844
halueval_qa_04351
halueval_qa_02501
halueval_qa_03459
```

Create a complete shared claim inventory, obtain a human inventory review, then
issue identical copies to two independent annotators. Give claims stable IDs,
atomic decontextualized text, sentence IDs, spans, and boolean check-worthiness.
Annotators must independently inspect evidence, supply three-way labels and
citations, and propose reference repairs. Adjudicate disagreements and retain
raw files and decision notes. Write the adjudicated five records to
`data/gold/pilot_dev.jsonl`; do not fabricate them from source variant names.
Gold citations must correspond to the declared frozen evidence/corpus.

The small pilot does not require completed test gold. The pilot split file uses
only these five dev IDs and the selected test IDs; it never runs the test records.
Later, complete and validate the full 50-dev/100-test gold using the
[gold workflow](gold_annotation_workflow.md). These human gold annotations are
separate from the later blinded assessments of model-generated repairs.

## Live pilot prerequisites and command

The local environment prepared here is:
`/tmp/hallucination-evaluation-env/bin/python`. It is temporary and can be
recreated with Python 3.12, CPU PyTorch, and `requirements-evaluation.txt`.
The NumPy/SciPy versions in that file are compatible with the pinned spaCy
runtime; unconstrained latest installations produced an incompatible pair.

To recreate a persistent local environment instead of using `/tmp`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-evaluation.txt
```

Substitute `.venv/bin/python` in the commands below when using that environment.

Before running the pipeline, provide:

1. The adjudicated five-record human gold file above.
2. A real FEVER BM25 index at `indexes/bm25_A.pkl`, or change the pilot config to
   its actual location **before freezing the run**. The existing
   `indexes/bm25_sample.pkl` is a sample and is not a valid benchmark corpus.
   See the existing index notebook/build script for corpus preparation; a full
   corpus build was not launched on this workstation (14 GiB disk free at check).
   A Colab/Drive or other adequately provisioned runtime can host the index.
3. `GROQ_API_KEY` configured in the process environment. No key was present in
   this checkout. Keep it out of chat, commits, and experiment outputs.
   The Groq model's service availability must be checked when credentials exist;
   a configured model name alone does not establish live API availability.

Then execute the already-authorized development pilot:

```bash
/tmp/hallucination-evaluation-env/bin/python scripts/run_benchmark.py \
  --config configs/evaluation_halueval_pilot.yaml \
  --inputs data/gold/pilot_dev.jsonl \
  --splits data/gold/pilot_splits.json \
  --task repair --split dev \
  --output-dir results/evaluation/halueval_live_pilot_v1
```

This runs B0 and the prompted pipeline on the same five dev records. Use the
[review workflow](independent_evaluation.md) to collect blinded repair assessments
and the [bundle workflow](evaluation_bundles.md) to produce an audited pilot
report. Resolve failures and evaluate assessor agreement on development data.
After settings are frozen and the full gold is available, run the final test
comparison with `configs/evaluation_halueval_v1.yaml` and all applicable baselines.
Freeze final source/split/gold/config hashes before the test run. Keep task runs
separate (repair, fixed-inventory verification, answer-level detection).

## Reproduce preparation

Clone the official repository, detach at the exact commit above, and ensure
`README.md`, `LICENSE`, and `data/qa_data.json` are checked out. The converter
only reads a local checkout; it verifies commit/content identity and never
fetches data or guesses a substitute revision.

```bash
python3 scripts/prepare_halueval_source.py \
  --source-repo /path/to/HaluEval \
  --revision b7253db3cdaa0ab2c382f92b26b390109174f77e \
  --output-dir data/processed/halueval_v1

python3 scripts/prepare_annotation_batch.py \
  --config configs/evaluation_halueval_v1.yaml \
  --output-dir data/annotation_batches/halueval_v1

python3 scripts/export_annotation_sentences.py \
  --batch-dir data/annotation_batches/halueval_v1 \
  --output-dir data/annotation_batches/halueval_v1_sentence_reference
```

Use the prepared environment for the sentence export. These commands refuse
existing output directories; choose a new version instead of overwriting human
work. Upstream source context and answer origins remain in private provenance,
not in model inputs or human verdict fields.


## Historical drafting stage (superseded by accepted annotations)

The first five records now have a proposed seven-claim inventory in
`data/annotation_batches/halueval_v1_operational_pilot/inventory/dev.jsonl`.
This working inventory remains unlabeled. Evidence-backed AI-proposed verdicts,
reference repairs, source metadata and detailed human-review questions are in
that pilot directory's `ai_draft/` folder; start with `ai_draft/review.md`.

These are **AI-assisted drafts**, not human submissions or adjudicated gold.
Human annotator A/B files remain blank and separate. In particular, review the
Ortega evidence-sufficiency question, the compound accounting question, and the
truthful-but-nonresponsive Borowska answer. The draft uses browser-retrieved
Wikipedia excerpts, not verified FEVER snapshot passages; bind citations to the
declared corpus before accepting final labels. Human reviewers should approve
the inventory before independently assigning labels. If they see and review AI
verdicts first, report that annotation method explicitly rather than claiming
independent annotation from scratch.


## Pilot human review accepted

The project user confirmed that human reviewers verified and agreed with the
five-record AI draft. Finalized annotations are now in `data/gold/pilot_dev.jsonl`,
with citation URLs/sections in `data/gold/pilot_dev.sources.json`, review provenance
and hashes in `data/gold/pilot_dev.manifest.json`, and membership in
`data/gold/pilot_splits.json`. The accepted annotations are human-reviewed; drafting history remains in the
private provenance manifest. Raw independent rater submissions and agreement statistics were not
supplied and have not been manufactured. The earlier draft section above is an
archive of how these proposals were prepared.

Accepted inventory: five records, seven claims (six Supported, one Unverifiable).
Explicit FEVER-corpus citation alignment is still unrecorded. The global
`data/splits.json` remains the full-benchmark placeholder; the finalized pilot
uses its own split file. The real retrieval index and local provider credential
are the remaining execution prerequisites. These gold records concern original
answers; future generated repairs still require separate independent assessment.


## Colab is the selected execution environment

The API key is available to the project user. Use the
[Colab pilot instructions](colab_reviewed_pilot.md) and
`notebooks/05_run_reviewed_pilot_colab.ipynb`. Upload the generated reviewed-input
ZIP separately from the Git clone and configure the key in Colab Secrets.
Citation URLs are now inline in every accepted evidence entry. The remaining
corpus check concerns FEVER's retrieved evidence, not missing citation fields.
