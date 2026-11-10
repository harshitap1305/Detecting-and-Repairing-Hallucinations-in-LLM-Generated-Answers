# Detecting and Repairing Hallucinations in LLM-Generated Answers

> **NLP Course Project** — claim-level hallucination detection and evidence-grounded repair.

## Current status

The prompted pipeline and its tests are in the repository. This checkout does
not include the human gold dev/test annotations, populated split IDs, per-record
baseline outputs, or trained model checkpoints needed to reproduce the numbers
in the existing reports. Those numbers are **unverified from the tracked
artifacts**; see [the baseline audit](docs/baseline_status.md) and run
`python scripts/audit_baseline.py` to check the available files. The A+C
training and comparison code below is ready to run, but no trained-model score
is claimed until data, checkpoints, and held-out outputs are supplied.

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
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your_key_here"
# OR create a .env file:
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Run the pipeline on a single example

Build the FEVER BM25 index in step 6 first, then run this example.

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

### 4. Run unit tests

```bash
pytest tests/ -v
```

### 5. Download datasets

```bash
python scripts/download_data.py --output_dir data/processed
```

### 6. Build retrieval index (requires FEVER wiki-pages)

```bash
# Download FEVER wiki-pages first:
wget https://s3-eu-west-1.amazonaws.com/fever.ai/wiki-pages.zip
unzip wiki-pages.zip -d data/raw/fever/

# Build BM25 index (fast):
python scripts/build_index.py --tier A --bm25_only

# Build BM25 + FAISS index (full; needs GPU for speed):
python scripts/build_index.py --tier A --encoder BAAI/bge-small-en-v1.5
```

### 7. Run on a batch of records

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

---

## Repository Structure

```
project/
├── README.md
├── requirements.txt
├── requirements-training.txt ← additional A+C training dependencies
├── .env                    ← GROQ_API_KEY (not committed)
├── docs/
│   ├── spec.md             ← task definitions, label schema
│   ├── literature_notes.md
│   ├── baseline_status.md  ← evidence audit of reported results
│   ├── training_workflow.md ← A+C training and evaluation steps
│   └── gold_annotation_workflow.md ← validate human annotations
├── configs/
│   ├── default.yaml        ← all model/threshold/policy settings
│   └── ablations/
├── data/
│   ├── raw/                ← downloaded datasets (not committed)
│   ├── processed/          ← unified-schema JSONL (not committed)
│   └── gold/               ← human-annotated gold set
├── indexes/                ← BM25/FAISS indexes (not committed)
├── src/
│   ├── schema.py           ← AnswerRecord, Claim, EvidencePassage
│   ├── llm.py              ← single LLM wrapper + SQLite cache
│   ├── decompose.py        ← Stage 1: claim decomposition
│   ├── retrieve.py         ← Stage 2: evidence retrieval
│   ├── verify.py           ← Stage 3: claim verification
│   ├── repair.py           ← Stage 4: answer repair
│   ├── pipeline.py         ← end-to-end orchestration
│   ├── evaluate.py         ← metrics (Phase 8)
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
│   └── 04_train_A_C_colab.ipynb ← Colab: train A+C and save artifacts
├── results/
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

1. Mount Google Drive and set `DRIVE_BASE = "/content/drive/MyDrive/hallucination_project"`
2. Set `LLM_CACHE_PATH` to a Drive path so the cache survives disconnects
3. Run the baseline notebooks in order: `01` → `02` → `03`. Use `04` for the
   A+C model training workflow.
4. All indexes, processed data, and LLM cache live on Drive

```python
# Mount Drive for runs that save data, caches, or checkpoints:
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE_BASE = "/content/drive/MyDrive/hallucination_project"
os.environ["LLM_CACHE_PATH"] = f"{DRIVE_BASE}/llm_cache.db"
```

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

All settings live in [`configs/default.yaml`](configs/default.yaml). Key knobs:

| Setting | Default | Description |
|---|---|---|
| `llm.provider` | `groq` | LLM provider |
| `llm.pipeline_model` | `llama-3.3-70b-versatile` | Decompose + verify + repair |
| `llm.eval_model` | `gemma2-9b-it` | Independent evaluator (different family) |
| `retriever.default_method` | `bm25` | `bm25` / `dense` / `hybrid` |
| `retriever.top_k` | `5` | Passages per claim |
| `pipeline.repair_policy` | `rewrite+hedge` | What to do with unfixable claims |
| `pipeline.max_repair_iterations` | `2` | Re-verify loops |

---

## Datasets

| Dataset | Use | Split |
|---|---|---|
| HaluEval QA | Answer-level detection | dev/test |
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
preserved. It requires human annotation before any gold benchmark exists;
experiment execution and final evaluation are later implementation increments.

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
reports require complete adjudicated reviews; progress reports show coverage
without final performance claims. Historical tables remain unverified.

Step 5 has begun with [real HaluEval preparation and a human handoff](docs/halueval_experiment_handoff.md):
50 development and 100 test candidates, blank annotation packets, sentence
offsets, and a five-record B0 execution smoke test. Human gold, independent
reviews, a full retrieval index, and a live provider credential are still needed
for measured pipeline results. Preparation tools do not manufacture annotations.

For the reviewed five-record pilot, follow [the Colab instructions](docs/colab_reviewed_pilot.md)
and open `notebooks/05_run_reviewed_pilot_colab.ipynb`. Reviewed evidence passages
include inline citation URLs; private gold is uploaded separately from the code.
