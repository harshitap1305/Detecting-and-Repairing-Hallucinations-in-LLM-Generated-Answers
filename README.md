# Detecting and Repairing Hallucinations in LLM-Generated Answers

> **NLP Course Project** — claim-level hallucination detection and evidence-grounded repair.

## Current status

The project implements claim decomposition, contextual retrieval, three-way
verification, guarded localized repair, eight baseline systems (B0–B7), and a
reproducible execution/review/reporting workflow. Entity Planner v3 is installed;
optional training paths support a FEVER verifier and a Flan-T5 repair editor.

The [project report PDF](docs/report/project_report.pdf),
[LaTeX source](docs/report/project_report.tex), and
[report build instructions](docs/report/README.md) are the current submission
artifacts. The report includes four team-supplied figures covering baseline
comparison, retrieval, iteration diagnostics, and benchmark distribution.

Result sources are recorded separately:

- **Team-reported broader benchmark:** the team's Drive results describe 50
  development answers (59 claims: 28 Supported, 31 Contradicted) and 100 test
  answers (110 claims: 53 Supported, 57 Contradicted), completed B0 execution,
  dual-reviewer adjudication with Cohen's kappa 1.0000, and answer-cluster
  bootstrap evaluation. These summaries appear in the report; the corresponding
  private full-size gold and result bundles require separate transfer from Drive
  for local recomputation.
- **Locally audited v4b pilot:** five development answers, seven reviewed claims,
  and ten completed outputs across B0 and the pipeline. Offline recomputation
  passed on 16 November 2026. Both systems retained six supported claims, with
  zero introduced factual errors and a correction outcome of 0/1 eligible answer.
  The [11 November pilot handoff](docs/project_handoff.md) gives its private paths
  and audit procedure; its open-work list predates the later planner and benchmark
  updates.
- **Frozen revisions:** `benchmark-config-frozen-v1` and
  `benchmark-gold-frozen-v1` identify versioned repository revisions. Private
  annotations, model checkpoints, and result bundles are transferred separately.

For a short runnable Colab workflow, start with
[notebook 07](notebooks/07_run_fast_cloud_demo_colab.ipynb). Use
[notebook 06](notebooks/06_run_full_benchmark_colab.ipynb) for the full benchmark
workflow or [notebook 05](notebooks/05_run_reviewed_pilot_colab.ipynb) for the
reviewed pilot.

## Pipeline

```
question + LLM answer
        │
        ▼
[Stage 1] Claim Decomposition    →  atomic, decontextualized claims
        │
        ▼
[Stage 2] Evidence Retrieval     →  top-k Wikipedia passages per claim
        │
        ▼
[Stage 3] Claim Verification     →  Supported | Contradicted | Unverifiable
        │
        ▼
[Stage 4] Localized Repair       →  minimal edits to flagged sentences only
        │
        ▼
[Stage 5] Re-verification        →  reliability score before / after
        │
        ▼
Repaired answer + per-claim report
```

## Quickstart

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your_key_here"
```

If using a `.env` file, explicitly call `from dotenv import load_dotenv` and
`load_dotenv()` before inference. The LLM wrapper reads environment variables;
it does not automatically load `.env` files.

### 3. Build the retrieval index

From the repository root, download/extract FEVER pages and build the BM25 index:

```bash
# Download FEVER wiki-pages first:
wget https://fever.ai/download/fever/wiki-pages.zip
unzip wiki-pages.zip -d data/raw/fever/

# Build BM25 index (fast):
python scripts/build_index.py --tier A --bm25_only

# Build BM25 + FAISS index (full; needs GPU for speed):
python scripts/build_index.py --tier A --encoder BAAI/bge-small-en-v1.5
```

The full-corpus SQLite FTS5 alternative is documented in the
[training workflow](docs/training_workflow.md) and the Colab notebooks.

### 4. Run the pipeline on a single example

```python
from src.schema import AnswerRecord
from src.pipeline import run_pipeline

record = AnswerRecord(
    id="demo_001",
    source_dataset="manual",
    split="dev",
    question="Who invented the telephone?",
    answer="Alexander Graham Bell invented the telephone in 1876 in Boston. "
           "He was born in Edinburgh in 1847.",
)
result = run_pipeline(record, index_path="indexes/bm25_A.pkl", verbose=True)
print(result.repaired_answer)
print(result.summary())
```

### 5. Prepare datasets for batch inference

```bash
python scripts/download_data.py --output_dir data/processed \
  --splits_file data/local_download_splits.json
```

This writes a separate download split manifest so the command preserves the
benchmark's existing `data/splits.json`. For the pinned HaluEval benchmark,
follow the [source preparation instructions](docs/gold_annotation_workflow.md#pinned-halueval-source-preparation).

### 6. Run on a batch of records

```bash
python -c "
import json
from src.schema import AnswerRecord
from src.pipeline import run_pipeline_batch

records = [AnswerRecord.from_jsonl_line(line)
           for line in open('data/processed/halueval.jsonl')
           if line.strip()][:20]  # start with 20

run_pipeline_batch(records, output_path='results/phase3_pilot.jsonl',
                   index_path='indexes/bm25_A.pkl', verbose=True)
"
```

For formal comparisons, use the [benchmark runner](docs/benchmark_runner.md),
which records expected outputs, attempts, errors, and experiment identity.

### 7. Run tests

For a dedicated test environment, install the CPU dependencies used by CI:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

---

## Repository Structure

```
project/
├── README.md
├── requirements.txt
├── requirements-test.txt     ← CPU tests and offline checks
├── requirements-evaluation.txt ← pinned pilot runtime dependencies
├── requirements-training.txt ← additional A+C training dependencies
├── .env                    ← GROQ_API_KEY (not committed)
├── docs/
│   ├── spec.md             ← task definitions, label schema
│   ├── literature_notes.md
│   ├── baseline_status.md  ← evidence audit of reported results
│   ├── training_workflow.md ← A+C training and evaluation steps
│   ├── gold_annotation_workflow.md ← validate human annotations
│   └── report/              ← LaTeX report, compiled PDF, build instructions
├── configs/
│   ├── default.yaml        ← reference configuration
│   ├── *.options.json      ← benchmark/demo model and pipeline options
│   ├── evaluation*.yaml    ← benchmark tasks, sources, splits, retrieval
│   └── ablations/
├── data/
│   ├── raw/                ← downloaded datasets (not committed)
│   ├── processed/          ← unified-schema JSONL (not committed)
│   └── gold/               ← committed demo samples; private reviewed gold
├── indexes/                ← BM25/FAISS indexes (not committed)
├── src/
│   ├── schema.py           ← AnswerRecord, Claim, EvidencePassage
│   ├── llm.py              ← single LLM wrapper + SQLite cache
│   ├── decompose.py        ← Stage 1: claim decomposition
│   ├── retrieve.py         ← Stage 2: evidence retrieval
│   ├── verify.py           ← Stage 3: claim verification
│   ├── repair.py           ← Stage 4: answer repair
│   ├── pipeline.py         ← end-to-end orchestration
│   ├── evaluate.py         ← evaluation metrics
│   ├── experiments/        ← runner, review, scoring, provenance, report bundles
│   ├── training/           ← FEVER verifier and repair model code
│   └── baselines/          ← B0–B7 baseline implementations
├── scripts/
│   ├── download_data.py    ← fetch & convert all datasets
│   ├── build_index.py      ← build BM25 + FAISS indexes
│   ├── build_fever_fts.py  ← stream full FEVER pages to SQLite FTS5
│   ├── retrieve_fever_evidence.py ← resumable FEVER retrieval
│   ├── validate_gold.py    ← check annotations and holdout keys
│   ├── prepare_fever_verifier.py ← prepare Path A training data
│   ├── build_repair_data.py ← prepare Path C training data
│   ├── train_verifier.py / train_repair.py
│   ├── eval_verifier.py / eval_repair.py
│   ├── compare_training_paths.py ← paired held-out A/C comparison
│   ├── audit_baseline.py   ← check minimum baseline artifacts
│   └── run_eval.py         ← compute metrics on pipeline output
├── notebooks/
│   ├── 01_download_data.ipynb   ← Colab: download datasets
│   ├── 02_build_index.ipynb     ← Colab: build indexes
│   ├── 03_run_pipeline.ipynb    ← Colab: run & evaluate pipeline
│   ├── 04_train_A_C_colab.ipynb ← Colab: train A+C and save artifacts
│   ├── 05_run_reviewed_pilot_colab.ipynb ← reviewed five-answer pilot
│   ├── 06_run_full_benchmark_colab.ipynb ← full benchmark execution
│   └── 07_run_fast_cloud_demo_colab.ipynb ← 20-answer-per-split demo
├── results/                ← retained figures; private evaluation bundles
├── demo/
│   └── app.py              ← Gradio demo (Phase 9)
└── tests/
    ├── test_schema.py
    ├── test_llm.py
    ├── test_decompose.py
    ├── test_retrieve.py
    ├── test_verify.py
    └── test_pipeline.py
```

---

## Colab Workflow

Choose the notebook for the task:

| Notebook | Purpose | Storage |
|---|---|---|
| 01 → 02 → 03 | Dataset download, index construction, pipeline execution | Configure the notebook's Drive paths |
| 04 | Train and evaluate A+C models | `MyDrive/hallucination_project_ac/` |
| 05 | Reproduce the reviewed pilot | Upload private pilot inputs; export completed results |
| 06 | Full benchmark execution | `MyDrive/hallucination_full_benchmark/` |
| 07 | Fast demonstration with 20 dev and 20 test answers | Committed demo gold plus the notebook's configured runtime/Drive paths |

Follow each notebook's setup cells: paths and dependency environments differ.
The [v4b guide](docs/colab_followup_v4b.md) accompanies notebook 05. Preserve
run-local caches and immutable attempts when exporting a benchmark run.

For LLM pipeline or comparison cells, add `GROQ_API_KEY` to Colab Secrets and
read it with `from google.colab import userdata` and
`os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")`. Training the
verifier and repair model from prepared data does not need that API key.

## A+C model training

Path **A** fine-tunes a three-way FEVER evidence verifier. Path **C** trains a
Flan-T5 repair editor on verified synthetic corruptions and their source
sentences. Install `requirements-training.txt` in a GPU Colab runtime and follow
[the full workflow](docs/training_workflow.md) or
[the Colab notebook](notebooks/04_train_A_C_colab.ipynb). The workflow prepares
article-disjoint splits, saves data and checkpoints to Drive, and records exact
run inputs and package versions. The FEVER training route uses a streaming
SQLite FTS5 index and resumes interrupted claim retrieval. Before reporting a
paired course-project score, [validate the human gold set](docs/gold_annotation_workflow.md).

The pipeline accepts `verifier="trained"` with
`trained_verifier_checkpoint=...`, and `repair_backend="trained"` with
`trained_repair_checkpoint=...`. For a valid contribution claim, evaluate four
variants on the *same* held-out gold test records using
`scripts/compare_training_paths.py`: prompted baseline, A only, C only, and
A+C. Its required gold files and trained checkpoints are not included in this
checkout. Keep the test set separate from model selection and report measured
results only after that run completes.

---

## Configuration

Runtime settings are supplied through the Python API or a benchmark options
file; [`configs/default.yaml`](configs/default.yaml) is a reference configuration
and is not automatically loaded by `run_pipeline()`.

| Entry point | Configuration |
|---|---|
| Direct Python API | Keyword arguments to `run_pipeline()` in `src/pipeline.py` |
| Benchmark protocol | `configs/evaluation*.yaml`: tasks, sources, split sizes, corpus and retrieval |
| Full benchmark | `configs/full_benchmark_v1.options.json` |
| Fast demo | `configs/demo.options.json`; structural extraction checks |
| Reviewed-pilot profile | `configs/pilot_followup_v4b.options.json`; restore the saved manifest/source to reproduce historical settings |
| NLI thresholds | `configs/calibrated_thresholds.yaml`, loaded by verification code |

Current full-benchmark/demo profiles specify `qwen/qwen3.8-27b` through Groq,
`rewrite-only`, `grounded_v1`, and planner version 3. The direct API defaults to
`llama-3.3-70b-versatile`, prompted LLM verification, and `rewrite+hedge`.
The saved v4b execution used `openai/gpt-oss-120b` and planner version 2; changing
a current options file does not change that historical run. Record the resolved
model/settings in each new execution manifest.

---

## Datasets

The following datasets have loaders or preparation routes. The report identifies
the source and cohort used for each result; this table describes supported uses.

| Dataset | Use | Split |
|---|---|---|
| HaluEval QA | Answer candidates, reviewed claim verification and repair | dev/test |
| FELM world-knowledge | Segment-level detection | test only |
| WikiBio GPT-3 | Sentence-level detection | test only |
| Factcheck-Bench | Claim-level detection + repair gold | test only |
| FEVER claims + wiki-pages | Path A verifier training and Tier A retrieval | article-disjoint train/dev/eval |
| Wikipedia 20231101.en | Retrieval corpus (Tier B, Phase 5+) | — |

---

## License notes

- FELM: CC BY-NC-SA 4.0 (non-commercial only)
- Wikipedia: CC BY-SA 3.0 / GFDL
- HaluEval: MIT
- WikiBio: CC BY-SA 3.0
- Do not commit large raw data files

---

## References

- [FActScore](https://arxiv.org/abs/2305.14251) · [RARR](https://arxiv.org/abs/2210.08726) · [CoVe](https://arxiv.org/abs/2309.11495)
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) · [RefChecker](https://arxiv.org/abs/2405.14486)
- [Factcheck-GPT / Factcheck-Bench](https://arxiv.org/abs/2311.09000) · [FAVA](https://arxiv.org/abs/2401.06855)
- [HaluEval](https://arxiv.org/abs/2305.11747) · [FELM](https://arxiv.org/abs/2310.00741)
- [FEVER](https://fever.ai) · [TruthfulQA](https://arxiv.org/abs/2109.07958)

## Report and retained figures

Read the [project report](docs/report/project_report.pdf) or follow the
[LaTeX/Overleaf instructions](docs/report/README.md) to rebuild it. The report uses
four team-supplied chart assets in `results/figures/`: baseline comparison,
retrieval recall/latency, verification convergence, and the HaluEval B0 control.
The synthetic calibration heatmap and illustrative training curves were removed.
`python scripts/plot_results.py` regenerates the four retained charts from the
team-supplied aggregate values embedded in the script. It does not execute the
experiments or recompute metrics from raw predictions.

## Reproducible evaluation preparation

The [evaluation protocol](docs/evaluation_protocol.md) defines tasks, native
baseline capabilities, annotation handoff, and the planned scoring contract.
Fill actual dataset paths, exact source versions, and per-source quotas in
[`configs/evaluation.yaml`](configs/evaluation.yaml), then run:

```bash
python3 scripts/prepare_annotation_batch.py \
  --config configs/evaluation.yaml \
  --output-dir data/annotation_batches/v1
```

This exports deterministic blank annotation packets, prediction-safe inputs,
split candidates, and a provenance manifest. Source dev/test assignments are
preserved. New batches require human annotation before becoming benchmark gold. Execution,
independent scoring and report generation are implemented.

The [benchmark runner](docs/benchmark_runner.md) executes the configured systems
with isolated inputs, native output envelopes, request traces, immutable attempts,
and identity-checked resume. Its `execution_complete` status is separate from
independent evaluation and final benchmark completion.

[Offline scoring and independent human review](docs/independent_evaluation.md)
now support complete runner artifacts. Corrected three-way detection metrics,
answer-cluster confidence intervals, blinded repair review, adjudication, and
Holm-adjusted paired comparisons run without provider credentials or new inference.

[Generated task reports and result bundles](docs/evaluation_bundles.md) package
saved executions and independent assessments for offline recomputation. Use
`scripts/build_evaluation_report.py` to build a new bundle and
`scripts/audit_baseline.py --bundle ... --gold ...` to verify it. Final repair
reports require complete adjudicated reviews; progress reports show coverage.
The report attributes team-supplied aggregate charts separately from locally
recomputed task bundles.

[Real HaluEval preparation](docs/gold_annotation_workflow.md#pinned-halueval-source-preparation) produced a
50-development/100-test candidate cohort. Five development records have reviewed
gold, completed live executions and adjudicated repair assessments. The later
team-reported broader benchmark is summarized above; its private Drive artifacts are separate from these pilot files. See the
[pilot handoff](docs/project_handoff.md) for exact pilot-artifact locations.

For the reviewed five-record pilot, follow [the Colab instructions](docs/colab_followup_v4b.md)
and open `notebooks/05_run_reviewed_pilot_colab.ipynb`. Reviewed evidence passages
include inline citation URLs; private gold is uploaded separately from the code.
