# Prepare the human gold set

The A+C comparison needs human-annotated development and test files. The
repository currently has empty `dev_ids` and `test_ids` in `data/splits.json`
and does not include `data/gold/dev.jsonl` or `data/gold/test.jsonl`. This
workflow prepares those inputs without making up labels or benchmark scores.

## Annotate and freeze the split

1. Select question/answer records before viewing model predictions. Assign
   each record to exactly one split. Follow
   [the annotation guidelines](annotation_guidelines.md), which propose 50 dev
   and 100 test answers. Resolve the existing report's different test count
   before quoting a benchmark size.
2. Put one `AnswerRecord` JSON object per line in `data/gold/dev.jsonl` and
   `data/gold/test.jsonl`. See [`src/schema.py`](../src/schema.py). Each record
   needs `id`, `source_dataset`, `split`, `question`, `answer`, a nonempty
   `claims` array, and a human `gold_repaired_answer`. An empty repair string is
   valid only when removing all text is appropriate.
3. Give every claim a stable, globally unique `id`, nonempty `text`, and a
   zero-based `sentence_id` in the original answer. `span`, when supplied, is
   `[start, end]` character offsets in that original answer (end exclusive).
   Set `check_worthy` to a boolean. Every check-worthy claim needs one of
   `Supported`, `Contradicted`, or `Unverifiable` in `label`.
4. For `Supported` and `Contradicted`, record the human-inspected citation in
   `evidence` with nonempty `doc_id`, `title`, and `text`. An `Unverifiable`
   claim may have empty evidence when no relevant passage was found. The
   validator checks the presence and shape of citations, not their factual
   correctness.
5. Populate `dev_ids` and `test_ids` in `data/splits.json` with the exact
   record IDs in the corresponding gold files. Keep questions/prompts disjoint
   across the two splits. Do not choose a checkpoint or tune thresholds on
   the test file.

Use two annotators and adjudicate disagreements where possible. Keep the raw
annotations and agreement calculation as separate artifacts; a valid JSONL
file alone does not establish annotation quality.

## Validate before training

Run from the repository root:

```bash
python scripts/validate_gold.py \
  --gold-dev data/gold/dev.jsonl \
  --gold-test data/gold/test.jsonl \
  --splits data/splits.json \
  --expected-dev 50 --expected-test 100 \
  --holdout-out data/gold/holdout_manifest.json
```

Omit the two `--expected-*` flags if the final, agreed split sizes differ.
The validator refuses missing records, duplicate IDs, wrong split membership,
unlabeled check-worthy claims, missing citations for decisive verdicts,
invalid sentence offsets, and dev/test prompt overlap. It prints the record
counts, verdict distribution, and hashes of the input files. It writes the
holdout manifest **only after** these checks pass and never overwrites an
existing manifest unless `--overwrite` is supplied.

The manifest includes test-only Wikipedia article titles from cited evidence,
plus test prompt IDs and question text. An `Unverifiable` test record without
a cited article needs an explicit entity key. Supply a JSON object mapping
test record IDs to lists of entity names, for example:

```json
{"test_record_id": ["entity name used by that test question"]}
```

Pass it as `--entity-map data/gold/test_entities.json`. The validator refuses
to write a leakage holdout if any check-worthy test claim has neither a cited
article nor a mapped entity for its record. Review every title and entity
manually: title matching and string exclusion cannot guarantee that
semantically related training examples are absent.

## Use the frozen files

Point the [Colab A+C workflow](training_workflow.md) at the generated holdout
manifest **before** preparing FEVER verifier or synthetic repair data. Save
the manifest and input hashes with the run. Use the dev file for checkpoint
selection or calibration; run the paired comparison on the untouched test
file only once the model choices are fixed:

```bash
python scripts/compare_training_paths.py \
  --inputs data/gold/test.jsonl \
  --gold-dev data/gold/dev.jsonl \
  --gold-test data/gold/test.jsonl \
  --splits data/splits.json \
  --verifier-checkpoint /path/to/verifier/final \
  --repair-checkpoint /path/to/repair/checkpoint-epoch-N \
  --index /path/to/fever_fts.sqlite \
  --retriever-method fts5 \
  --output-dir /path/to/ac_comparison
```

The comparison fixes the same human claim inventory across all four systems.
It still needs the actual checkpoints, index, and an LLM provider credential.
Keep any gold files private until the team has reviewed dataset terms and
decided whether to publish the annotations.
