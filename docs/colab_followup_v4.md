# Follow-up v4: run in the existing Colab session

Historical workflow: v4 remains incomplete (7/10). Its failures and outputs
are preserved. The subsequent [v4b recovery run](colab_followup_v4b.md) completed
10/10 and has an audited human-reviewed final bundle. See the
[current handoff](project_handoff.md). The cells below describe the original v4
workflow; use a fresh name for any new experiment.

Keep the same running notebook, Python environment, reviewed gold and FEVER
index. Do not upload a replacement notebook or restart the runtime. Preserve
both earlier pilot archives and all existing run directories.

## 1. Apply the code update

Add a cell to the existing notebook. Upload
`results/evaluation/pilot_followup_v4_code.zip` from the local repository when
prompted. This is the code ZIP, not a data/results ZIP; it contains no index,
gold, credentials or model results.

```python
from pathlib import Path
from google.colab import files
import hashlib, io, json, zipfile

REPO = Path('/content/hallucination_project')
uploaded = files.upload()
assert len(uploaded) == 1, 'Select exactly pilot_followup_v4_code.zip.'
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    manifest = json.loads(archive.read('source_manifest.json'))
    assert len(archive.namelist()) == len(manifest['files']) + 1
    assert set(archive.namelist()) == set(manifest['files']) | {'source_manifest.json'}
    payloads = {}
    for name, expected in manifest['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert (relative.parts[0] in {'src', 'scripts', 'configs', 'docs', 'notebooks'}
                or relative.name.startswith('requirements'))
        target = REPO / relative
        assert target.resolve().is_relative_to(REPO.resolve())
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == expected
        payloads[target] = data
    for target, data in payloads.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
print('Code updated; data, environment, index and previous runs preserved.')
```

To regenerate the local ZIP after any further source changes, use the evaluation
environment to run:

```bash
python scripts/package_colab_code.py --output results/evaluation/pilot_followup_v4_code.zip
```

## 2. Execute, postprocess and download

Add this cell below it. No dependency installation or index rebuild is needed in
the existing working session. The child interpreter imports the updated code in
a fresh process, so there is no need to reload old Python modules in the kernel.

```python
import os, subprocess
from google.colab import userdata

PYTHON = '/content/reviewed_pilot_env/bin/python'
RESULTS = Path('/content/hallucination_reviewed_pilot')
RUN_NAME = 'pilot_followup_v4'
RESUME = False
BACKUP = Path('/content') / f'{RUN_NAME}_results.zip'

assert Path(PYTHON).is_file(), 'Existing Python environment missing.'
assert (REPO / 'indexes/fever_fts.sqlite').is_file(), 'Existing index missing.'
assert (REPO / 'data/gold/pilot_dev.jsonl').is_file(), 'Reviewed gold missing.'
if not os.environ.get('GROQ_API_KEY'):
    os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')

command = [PYTHON, '-u', 'scripts/run_followup_pilot.py',
           '--results-root', str(RESULTS), '--run-name', RUN_NAME,
           '--options-file', 'configs/pilot_followup_v4.options.json',
           '--output-zip', str(BACKUP)]
previous = RESULTS / 'pilot_followup_v3'
if previous.exists():
    command += ['--previous-run', str(previous)]
if RESUME:
    command.append('--resume')
result = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
print(result.stdout, flush=True)
if BACKUP.exists():
    files.download(str(BACKUP))
result.check_returncode()
```

Use `RESUME = True` only for the same interrupted/incomplete run with unchanged
code, input, configuration and package environment. A changed experiment needs a
new run name. Do not delete the earlier directory or repeatedly rerun a failed
case to obtain a preferred score. A startup failure before a manifest is created
cannot produce a run backup; its printed error needs correction first.

## Artifacts and failure behavior

The runner always attempts a small result export after a manifest exists.
Complete executions produce reference metrics, stage diagnostics, a fresh blinded
review packet and an audited progress bundle. Incomplete executions return a
nonzero exit status but still export successful outputs, all committed failed
attempts, raw traces, structured extraction findings and coverage-only progress.
They produce no reference comparison or human-review packet from the surviving
subset. A partial progress bundle is never promoted to a final result.

Products live under `<run_name>_products/<revision>/`; `latest.json` identifies
the current revision. Each revision is bound to the execution manifest, gold and
optional previous-run manifest. Identical reprocessing preserves the same review
packet; resume creates new products when the execution state changes and keeps
old partial products. Changed products are rejected rather than silently reused.
The ZIP includes run/products directories and excludes the large index/caches.

Place the downloaded `pilot_followup_v4_results.zip` in local
`results/evaluation/`. Inspect these concrete conditions before human review:

- Entity leads contain useful body mentions, not index header labels; query and
  passage budgets remain bounded, including any continuation.
- Extracted claims cover the answer-bearing relationship and do not answer an
  unaddressed question clause. Structural attribution alone is not proof.
- Semantic checks pass for faithful extractions or report specific unresolved
  findings; no supported-but-mis-extracted inventory gets pipeline success.
- Corrections, abstentions, no-ops, rejected edits and introduced errors remain
  separate. Lexical similarity and model diagnostics do not establish factuality.

After inspecting a complete run, give reviewers only copies of its
`review/reviewer/` materials. Collect new independent ratings and adjudication;
old ratings cannot be transferred to new output fingerprints. Build and audit a
final repair bundle using `docs/independent_evaluation.md` and
`docs/evaluation_bundles.md`. Do not label model drafts as human reviews.

## Frozen v4 settings and limits

`configs/pilot_followup_v4.options.json` explicitly selects entity planner 2,
`semantic_v1` extraction with two generation attempts/two logical checks and a
64-entry claim/clause cap, `grounded_v1` repair with `rewrite-only` abstention,
three retrieval queries/twenty returned passages per target/stage, and two repair
iterations. Model/reasoning/completion-floor settings remain explicit. An installed
planner-version mismatch fails before model calls. The profile uses the same five
dev answers, B0 and pipeline; the reserved test split remains untouched.

Exact source/package snapshots and effective request settings remain part of
execution provenance. This pilot has only one correction-eligible gold answer.
Once the operational fixes are validated, a separately versioned dev extension
with human-annotated evidence-backed correctable errors and no-op cases is needed
to study repair effectiveness. That extension is conditional future work, not
an invented annotation set or a replacement for the original five records.
