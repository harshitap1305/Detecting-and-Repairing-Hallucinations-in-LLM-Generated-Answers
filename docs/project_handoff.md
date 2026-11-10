# Project handoff

Updated 11 November 2026. This is the current status and entry point for the
handoff. Older plans and Colab guides retain their historical context.

## Delivered and measured

The project implements a prompted claim-decomposition, retrieval, verification
and guarded-repair pipeline, plus a reproducible evaluation framework:

- Deterministic annotation preparation, citation/gold validation and split binding.
- Common benchmark adapters, prediction-safe inputs, immutable attempts, traces,
  source/environment snapshots and identity-checked resume.
- Independent repair review import/adjudication, answer-cluster statistics,
  generated reports and offline bundle audits.
- Bounded extraction checks, original-claim tracking, contextual retrieval,
  repair safeguards, API rate-limit recovery and Colab export workflows.

The latest measured experiment is **v4b**, using five HaluEval development
answers, seven gold claims (six Supported, one Unverifiable), and ten completed
outputs across B0 and the pipeline. User-supplied reviewers A/B and an adjudicator
covered all ten blinded items. Their outcome agreement was 1.0 over fourteen claim
ratings; role labels are pseudonyms, and checksums do not establish independence.

| Adjudicated result | B0 | Pipeline |
| --- | --- | --- |
| Correction, eligible answers | 0/1 | 0/1 |
| Preservation, eligible answers | 4/4 | 4/4 |
| Supported claims retained | 6 | 6 |
| Newly introduced factual errors | 0 | 0 |
| Mean completeness, 0–2 scale | 1.0 | 1.0 |

The pipeline preserved all five original answers. The final bundle passes offline
recomputation. **No factual repair improvement or full-benchmark completion is
claimed.** Historical aggregate tables in other reports remain unverified.

## Artifact locations and private transfer

These paths exist in the working copy but are ignored by Git. Transfer them
separately through an appropriate private channel; a clone is insufficient.

| Artifact | Local path |
| --- | --- |
| Exact pilot gold, citations, manifests and split IDs | `data/gold/` |
| Latest audited final bundle | `results/evaluation/colab_followup_v4b/pilot_final_v1/` |
| Raw v4b reviews and validated imports/adjudication | `results/evaluation/colab_followup_v4b/analysis/human_review/` |
| Review provenance, hashes and analysis | `results/evaluation/colab_followup_v4b/analysis/` |
| Existing portable bundle and review/analysis archive | `results/evaluation/colab_followup_v4b/pilot_followup_v4b_reviewed_results.zip` |
| Original v4b execution and immutable attempts | `results/evaluation/colab_followup_v4b/pilot_followup_v4b/` |
| Earlier adjudicated final pilot | `results/evaluation/colab_reviewed_pilot/pilot_final_v1/` |

For a compact transfer, supply the reviewed-results ZIP plus `data/gold/` and the
source revision. The ZIP already contains the final bundle and review materials;
there is no need to send both it and an identical extracted copy. Preserve the
original execution directories and historical runs in the owner's archive.
Do not include API keys, `.env`, or root inference caches.

The roughly 3.02 GiB `results/evaluation/reviewed_pilot_results(1).zip` contains
an index backup and is retained locally. No external disk was mounted during
cleanup, so it was not moved or deleted. Move it to durable external storage
when available; verify the copied archive before removing the local copy. It is
not needed for offline scoring. Live reruns need the FEVER index separately,
using the existing Colab/Drive copy or a verified rebuild.

## Set up and verify

Use Python 3.12. Follow the CPU dependency pattern used by CI:

```bash
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

`requirements-evaluation.txt` adds pinned pilot runtime dependencies and the
English spaCy model. `requirements-training.txt` supports the optional A/C
training path. Keep these distinct; a full training environment is unnecessary
for offline report auditing.

After restoring the private artifacts, audit the latest final bundle:

```bash
python scripts/audit_baseline.py \
  --bundle results/evaluation/colab_followup_v4b/pilot_final_v1 \
  --gold data/gold/pilot_dev.jsonl
```

The same command with
`--bundle results/evaluation/colab_reviewed_pilot/pilot_final_v1` audits the earlier
adjudicated pilot. Offline audits need neither API credentials nor the corpus
index. Preserve bundles intact: their embedded source/configuration/environment
snapshots describe generation, even if this checkout later changes.

For optional live work, use [the v4b Colab guide](colab_followup_v4b.md) and
notebook 05. Reuse the existing environment/index where available. A completed
run must not be overwritten; changed code/configuration needs a new run name.
Use Colab Secrets for credentials. Hosted-model availability must be checked in
the actual account before freezing a new run.

## Open work and boundaries

1. Improve relation-preserving extraction and compound-question ambiguity.
   The current checker can accept a vague battle-identification claim or an
   FTC-specific assertion that does not preserve the accepted answer reading.
2. Improve entity-lead selection: capitalized runs from flattened FEVER link
   annotations can generate long, repeated-name queries. The suggested cleanup
   was reverted at the user's request. **Planner version 2 remains installed;**
   no v4c profile or version-3 fix is included.
3. Broaden development evaluation with separately versioned, reviewed examples
   containing more evidence-backed correctable errors and preservation cases.
4. Complete the planned 50-development/100-test annotations, freeze settings,
   execute the declared systems and finish independent test assessment. Do not
   tune on the reserved test split or alter accepted pilot gold to fit outputs.
5. Optionally train/evaluate A/C variants using
   [the implemented training workflow](training_workflow.md). No trained-model
   checkpoint or measured trained-model superiority is claimed.

The original v4 execution remains incomplete (7/10). V3 is preserved as a complete
execution without its own human-rated final bundle. V4b's new human ratings bind
v4b outputs; do not silently transfer them to another run. Model-authored review
drafts remain separate and are not human submissions.

## Document map

- [Evaluation protocol](evaluation_protocol.md): tasks, splits and comparison rules.
- [Benchmark runner](benchmark_runner.md): execution and resume contract.
- [Independent evaluation](independent_evaluation.md): review and scoring commands.
- [Evaluation bundles](evaluation_bundles.md): final/progress reports and auditing.
- [Reproducible evaluation plan](reproducible_evaluation_plan.md): delivered scope.
- Root design/training proposals and general Colab advice: historical background;
  use this handoff and the specific workflow guides for current status.

## Final cleanup checks

On 11 November 2026, the unchanged runtime passed **387 tests plus 35 subtests**.
Both earlier and v4b adjudicated final bundles passed offline recomputation.
Notebooks 04 and 05 passed schema and Python-cell syntax validation.
Disposable Python/pytest caches and root inference-cache databases were removed;
raw reviews, source snapshots, run-local caches and result bundles were retained.
The local cleanup record is `results/evaluation/handoff_cleanup_record.json`.
