# Baseline reproducibility status

As checked on 7 November 2026, this checkout contains the prompted pipeline code and
unit tests, but not the minimum experiment artifacts needed to reproduce the
numerical claims in `docs/evaluation_results_report.md` or
`docs/final_report_template.md`. **Those reported metrics are unverified from the
tracked repository artifacts.** This does not establish that an experiment was
never run; its inputs and outputs may be stored elsewhere.

## What is present

- `data/splits.json` exists, but both `dev_ids` and `test_ids` are empty.
- Pipeline, baseline, evaluation, and ablation scripts are present.
- On this Windows workspace, `python -m pytest -q` passed **101/101 tests** in
  32.54 seconds after setting `TEMP` and `TMP` to a writable project directory.
  The default sandbox temp location caused setup `PermissionError`s in nine tests.
  The tests use mocks and are not a reproduction of the published benchmark.

## What is absent from this checkout

- `data/gold/dev.jsonl` and `data/gold/test.jsonl` (gold annotations).
- `results/pipeline_output.jsonl` and `results/baseline_b0.jsonl` through
  `results/baseline_b7.jsonl` (per-record outputs for the reported comparison).
- A local trained-model checkpoint. The training plan proposes keeping
  checkpoints on Google Drive, so a separate path may exist.

There is also a split-count inconsistency to resolve: the annotation guidelines
specify 50 dev plus 100 test answers, while the evaluation report calls its
comparison a 150-answer test benchmark. The audit does not assume either count.

The original `main` revision (`f06797d`) also displayed preset benchmark scores
in `demo/app.py` and used a heuristic fallback when the live pipeline raised an
exception. Demo output from that revision must not be treated as experimental
evidence.

## Minimum artifact check

Run `python scripts/audit_baseline.py` from the repository root. It checks that
split IDs are populated and disjoint, that the gold and output JSONL files are
nonempty and have unique IDs, that gold IDs belong to their declared split, and
that all comparison outputs use exactly the gold-test IDs. It
returns a nonzero exit status if any required baseline artifact is missing or
misaligned. Pass `--checkpoint PATH --require-checkpoint` when checking a trained
model as well. Presence of these files is only a prerequisite; reported metrics
still need to be recomputed from them using the same configuration and code
revision.


## Versioned report bundles

The legacy minimum-artifact command above remains available. New executions can
be packaged and fully recomputed with the [bundle workflow](evaluation_bundles.md).
Use `python scripts/audit_baseline.py --bundle PATH --gold EXACT_GOLD_JSONL` to
verify a generated bundle. This tooling does not validate or replace the old
numerical tables without the corresponding real experiment artifacts.
