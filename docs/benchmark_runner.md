# Execute and resume a benchmark

Increment 2 implements system execution and artifact validation. It does not
compute the corrected evaluation metrics, independent human assessments, or
final report planned for later increments. `execution_complete` means all
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
