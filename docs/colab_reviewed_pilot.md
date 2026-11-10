# Run the reviewed pilot in Colab

The five pilot records are finalized as **human-reviewed annotations**. Each
`evidence` entry in `data/gold/pilot_dev.jsonl` now contains `title`, `text`,
`doc_id`, and a directly usable `url`. Detailed sections/access metadata remain
in `pilot_dev.sources.json`. The earlier layout separated URLs from excerpts;
there was no missing-citation requirement preventing execution.

Original drafting history is retained in the private provenance manifest rather
than repeated throughout the accepted labels and rationales.

## What to do next

1. Commit and push the current code so Colab can clone the same implementation.
2. Open `notebooks/05_run_reviewed_pilot_colab.ipynb` in Colab. Set `REPO_URL`
   if needed and preferably set `CODE_REF` to that commit. The pinned environment
   uses Python 3.12; a GPU is not needed for prompted API inference.
3. In Colab Secrets, add your existing key under `GROQ_API_KEY` and enable access
   for the notebook. The notebook reads it through `google.colab.userdata` and
   does not print it or save it into code.
4. Upload `results/evaluation/halueval_reviewed_pilot_inputs.zip` when prompted.
   It contains the five reviewed gold/provenance files. Gold is ignored by Git,
   so cloning the repository alone does not supply these files.
5. Run the FEVER download/index cell. It builds a streaming SQLite **FTS5 lexical
   index** and saves it to Google Drive. The dedicated
   `configs/evaluation_halueval_colab.yaml` profile uses this index for pipeline
   plus B0. Existing BM25 configurations remain available separately. The full
   baseline comparison must declare its final retrieval settings before test use.
6. Inspect `pilot_retrieval_review.json` on Drive to check what the frozen FEVER
   corpus retrieves for each approved claim. The current citations are Wikipedia
   web passages; this is the separate check that the chosen corpus provides the
   relevant evidence. The notebook does not silently replace labels or feed
   gold evidence to the pipeline.
7. Run the five-record pipeline+B0 pilot. Results, caches, reference metrics, and
   a new repair-review packet are saved under
   `MyDrive/hallucination_reviewed_pilot/`.
8. Review the **generated repairs**, then import/adjudicate those output ratings
   and create the final pilot report using the existing review/bundle workflow.
   The approved original-answer annotations do not need to be authored again.

The notebook checks that the configured Groq model is listed before launching
inference. If it is unavailable, select an available model and freeze that choice
before running. Use a new output directory for changed settings; `RESUME=True`
is for identical interrupted runs only.

## Preparation status

- Human review accepted: 5 records, 7 claims, with inline citation URLs.
- Annotator-facing wording cleaned; prior finalized files archived privately.
- Colab input archive and dedicated notebook/profile prepared.
- Notebook structure and Python syntax validated locally; remote Colab execution
  has not yet been performed.
- Local test suite: 220 tests and 35 subtests passed.

Source questions, citations, reviewer details, and traces remain private. The
100 selected test records are reserved for the later frozen test evaluation.


## FEVER download endpoint

The old S3 bucket URL can return HTTP 404. Notebook 05 now uses
`https://fever.ai/download/fever/wiki-pages.zip`, with the
[authors' Zenodo deposit](https://zenodo.org/records/4925954) as fallback. The
published MD5 is checked before a partial download becomes the final ZIP;
SHA-256/source metadata is saved beside the archive. This uses the same FEVER
corpus, not a different dataset. Re-run the updated index cell after this error;
there is no need to repeat annotation upload or environment setup in the same
active Colab session. Existing valid indexes and verified archives are reused.
The notebook retains the selected Python 3.12 environment and output paths.

## Inspection cell errors

A `CalledProcessError` alone only reports that the isolated Python process
failed. The updated inspection cell captures and displays its combined output,
including the underlying Python traceback and the last stage reached (gold
validation, retrieval, or Groq access). Rerun that cell to diagnose the error;
the index does not need to be rebuilt. Do not share API keys when reporting an
error.

If model availability fails, the inspection cell now prints the model IDs
returned by your Groq account. Set `MODEL` at the top of that cell to an available
text-generation model and rerun the cell. The benchmark explicitly receives the
same model via `--model`; its execution manifest records the selection. A changed
model requires a new run directory if a run has already started. Do not select
speech, transcription, or moderation models for this text-generation benchmark.

GPT-OSS calls use low reasoning effort and at least 4,096 completion tokens,
including reasoning, to leave room for final answer text. The effective token
limit is recorded in traces and cache keys. Empty Groq answers report their
finish reason and token usage; reasoning text is never substituted for a verdict.
After updating the client, use a new run directory (for example
`pilot_repair_gptoss_v2`) with `RESUME = False`: the earlier incomplete run's code
identity differs. Keep the old directory for diagnosis.

## Downloaded pilot analysis

The supplied Colab archive has been inspected locally. The successful repair run
has all ten outputs (five dev answers each for B0 and the pipeline); its progress
bundle passed offline audit, and reference metrics reproduced exactly. Analysis
and model-authored review drafts are stored privately at
`results/evaluation/colab_reviewed_pilot/analysis/report.md`, with citation notes,
per-item drafts, artifact hashes and a human-review handoff. The source archive
and original execution artifacts remain preserved.

The pilot exposes a diagnostic limitation: three final decomposition responses
mark the revised answer as not check-worthy, allowing self-verification SCR to
rise to one without supported corrected claims. Those scores must not be read as
successful factual repair. The model-authored drafts remain separate from the subsequently supplied human
ratings. The completed reviewer A, reviewer B and adjudicator submissions were
validated, imported and adjudicated. The final pilot bundle at
`results/evaluation/colab_reviewed_pilot/pilot_final_v1/` passed offline audit with
metric recomputation. Its `report.md` contains the final human-assessed outcomes;
`analysis/final_audit_results.json` records submission hashes and provenance.
Broader baseline and held-out test execution remain separate next phases.

## Next live development run

Use [the follow-up v3 handoff](colab_followup_v3.md) and the updated notebook for
phases 5–6. The conservative repair profile, generating-source snapshots, explicit
reasoning settings, stage diagnostics and small result export are implemented.
A fresh live run and new independent human repair ratings are still needed to
measure improvement; earlier pilot outputs and assessments remain preserved.
