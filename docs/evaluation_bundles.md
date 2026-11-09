# Generated reports and auditable result bundles

Increment 4 packages saved executions, evaluation inputs, calculated metrics,
and task reports for offline verification. The generator never fills in missing
benchmark results or calls a model. A bundle describes one task and one split;
it does not establish that every project research question has been evaluated.
Real annotations, live pilot/test runs, and independent reviews remain increment 5.

## Build a final task report

For repair, first complete and adjudicate independent reviews using the
[review workflow](independent_evaluation.md). Then run:

```bash
python3 scripts/build_evaluation_report.py \
  --run-dir results/evaluation/dev-repair-v1 \
  --gold data/gold/dev.jsonl \
  --assessments data/review_batches/dev-v1/adjudicated.json \
  --output-dir results/evaluation/dev-repair-report-v1 \
  --n-bootstrap 1000 --seed 42
```

`--mode final` is the default. Final reports require complete execution across
all configured systems applicable to the task, exact gold IDs, valid committed
attempts and traces, and valid output views. Final **repair** reports also require
complete adjudicated human ratings, including both independent source reviews
and their decision provenance. A single-reviewer file is insufficient. Detection
runs need the complete human gold claim/answer labels; repair ratings do not
apply to them. Omit `--assessments` for detection bundles.

The command recalculates metrics from the actual artifacts rather than trusting
an arbitrary metrics JSON. It records the scoring code revision/source digest,
package/runtime environment, resample count and seed. The protocol text snapshot
is from reporting time; the original generation configuration and source identity
remain in the execution manifest. It snapshots artifacts
under the execution directory's lock, writes to a staging directory, validates
the copied structure, and publishes only the completed bundle directory. Existing
output directories are never overwritten. Choose a destination separate from
(and not inside or containing) the source execution directory.

Generated tables include actual split/source counts and label distribution,
coverage, retained failed-attempt counts, task-appropriate results and confidence
intervals, paired comparisons, and method limitations. Claim reports include
per-class metrics and confusion matrices. Repair reports separate independent
factual outcomes from reference-wording agreement. Undefined/inapplicable values
are N/A. Tables do not assert superiority simply because comparisons were run.
All numeric precision is retained in JSON; Markdown displays rounded values.

## Progress reports

```bash
python3 scripts/build_evaluation_report.py \
  --run-dir results/evaluation/dev-repair-v1 \
  --gold data/gold/dev.jsonl \
  --mode progress \
  --output-dir results/evaluation/dev-repair-progress-v1
```

Progress mode accepts structurally valid `incomplete` or complete runs. It shows
coverage, missing IDs in JSON, and recorded failed-attempt counts, with performance
metrics/comparisons explicitly unavailable. It never computes comparisons on a
surviving subset or presents a partial run as a final experiment. Interrupted runs
need consistent output views; resume the runner first if an abrupt process kill
left stale views. Progress bundles always remain progress bundles when audited,
even when their underlying execution is complete.

## Bundle layout and privacy

```text
<report-bundle>/
  manifest.json                 # version, scope, completion state, scoring provenance
  protocol.json                 # generation config, rubric, reporting-time protocol text
  inputs_manifest.json          # exact external gold hash, IDs, generation input hashes
  execution/
    manifest.json               # original generation manifest
    attempts/<cell>/<attempt>.json
    traces/<id>.jsonl
    outputs/<system>.jsonl
  evaluations/
    <system>.jsonl              # per-record output binding and applicable assessments
    assessments.json            # supplied human reviews/adjudication, when applicable
  errors.jsonl                  # failed-attempt history derived from attempts
  metrics.json
  comparisons.json
  report.md
  checksums.json                # exact file inventory and SHA-256 for every other file
```

The `execution/` snapshot deliberately keeps generation provenance separate from
reporting provenance. Caches, locks, and an extra copy of the gold file are not
included. The original run, review packet, and imported-review files can be moved
or removed after bundling; verification uses the snapshot and the exact external
gold JSONL. Preserve the raw human annotation/review archive separately for audit
of annotation quality. Original processed inputs/index/model artifacts are still
needed to repeat inference; their recorded hashes are provenance, not a promise
that this bundle contains those large or restricted files.

**Bundles are private reproduction artifacts.** They contain original answers,
evidence, raw model requests/responses, and reviewer details. The gold remains
external and access-controlled. The default `results/evaluation/` directory is
ignored by Git. To share findings, review dataset terms and share only the
aggregate `report.md` or other approved artifacts. A generated final task report
is not a license to publish underlying data.

## Audit and recompute offline

```bash
python3 scripts/audit_baseline.py \
  --bundle results/evaluation/dev-repair-report-v1 \
  --gold data/gold/dev.jsonl

# Equivalent full verification through the report CLI:
python3 scripts/build_evaluation_report.py \
  --verify results/evaluation/dev-repair-report-v1 \
  --gold data/gold/dev.jsonl
```

Both default to full offline recomputation, without changing bundle files.
They require NumPy/SciPy and the exact hashed gold file, not an API key,
retrieval corpus/index, spaCy model, or trained checkpoint. Use the matching recorded scoring code revision/source digest and package
environment if another implementation or numerical-library version changes results.
Source code and its dependencies are not embedded in the bundle.
The full audit:

1. Checks the exact file inventory and all hashes; rejects symlinks.
2. Checks gold bytes/content/IDs against the external-input manifest.
3. Validates the generation manifest, attempts, trace hashes, output identities,
   completeness, and the protocol/model/corpus scope.
4. Validates human coverage and adjudication, when applicable.
5. Recalculates metrics, comparisons, and per-record evaluation views, and checks
   them against saved JSON. Re-renders all report tables and verifies exact text.
6. Verifies retained errors and coverage counts against committed attempts.

Deleting an output, changing the gold, adding duplicate records, or editing scores
invalidates the bundle. Even if someone updates file checksums after editing
metrics, full recomputation detects that the scores do not follow from the saved
outputs and assessments. Checksums are not a cryptographic signature and cannot
establish that annotations or model outputs are truthful.

`--no-recompute` on `audit_baseline.py` performs the structure/integrity checks
and verifies that report text matches saved metrics, but skips numerical
resampling and per-record evaluation recomputation. Use the default full audit
for reported results. Without `--bundle`, the original minimum-baseline-artifact
check remains available and retains its prior behavior.

## CI and local checks

CI installs CPU PyTorch separately and the remaining test dependencies from
`requirements-test.txt`. Its dedicated offline lifecycle test covers deterministic
annotation preparation, isolated execution, interruption/resume, review import,
adjudication, report generation, and bundle auditing with explicitly synthetic
fixtures. Other regressions cover corruption, incomplete runs, altered gold,
rehashed fabricated scores, duplicate outputs, and verification after original
run/review directories are removed. It then runs the remaining CPU test suite.

```bash
python3 -m unittest tests.test_evaluation_bundle -v
python3 -m compileall -q src scripts tests
python3 -m pytest tests -q --ignore=tests/test_evaluation_bundle.py
git diff --check
```

CI fixtures are engineering checks, not measured research results. Historical
numerical tables remain unverified until replaced by a real held-out execution
and independent assessment bundle.
