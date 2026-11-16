# Detecting and Repairing Hallucinations in LLM-Generated Answers

An NLP course project that checks factual claims in an LLM-generated answer against
Wikipedia evidence and attempts targeted corrections while preserving supported
content. Given a question and an answer, the system produces a repaired answer,
claim-level verdicts, retrieved evidence, and diagnostics for each processing stage.

[Project report](docs/FINAL_PROJECT_REPORT.md) ·
[Colab demo](notebooks/07_run_fast_cloud_demo_colab.ipynb)

## How it works

```text
Question + original answer
          │
          ▼
1. Decompose   Extract atomic claims and align them with the original text
          │
          ▼
2. Retrieve   Find relevant Wikipedia passages for each claim
          │
          ▼
3. Verify     Assign Supported, Contradicted, or Unverifiable verdicts
          │
          ▼
4. Repair     Propose sentence-level edits and check preservation and grounding
          │
          ▼
5. Re-verify  Check the revised answer and track the original claims
          │
          ▼
Repaired answer + claim verdicts + evidence + diagnostics
```

The pipeline supports lexical retrieval with BM25 or SQLite FTS5, dense retrieval
with FAISS, and hybrid rank fusion. Verification can use a prompted LLM, an NLI
model, an ensemble, or a trained verifier. Repair checks supported content and
new factual terms before accepting an edit. Sentences outside the edited spans
retain their original text and whitespace.

## Run in Google Colab

Start with [notebook 07](notebooks/07_run_fast_cloud_demo_colab.ipynb) for the cloud
demo. Add `GROQ_API_KEY` to Colab Secrets and follow the notebook's setup cells.

| Notebook | Purpose |
|---|---|
| [01](notebooks/01_download_data.ipynb) | Download and prepare datasets |
| [02](notebooks/02_build_index.ipynb) | Build retrieval indexes |
| [03](notebooks/03_run_pipeline.ipynb) | Run the pipeline |
| [04](notebooks/04_train_A_C_colab.ipynb) | Train the verifier and repair editor |
| [05](notebooks/05_run_reviewed_pilot_colab.ipynb) | Run the reviewed pilot |
| [06](notebooks/06_run_full_benchmark_colab.ipynb) | Execute the full benchmark workflow |
| [07](notebooks/07_run_fast_cloud_demo_colab.ipynb) | Run the smaller cloud demo |

## Run locally

Use Python 3.12 and run commands from the repository root.

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GROQ_API_KEY="your_key_here"
```

Choose dependencies for the workflow you need:

| File | Environment |
|---|---|
| `requirements.txt` | General pipeline, retrieval, and Gradio demo |
| `requirements-evaluation.txt` | Pinned evaluation and annotation environment |
| `requirements-training.txt` | Verifier and repair-model training |
| `requirements-test.txt` | Automated tests and offline checks |

### Build a retrieval index

Reuse an existing FEVER SQLite FTS5 index, or build one from the Wikipedia archive:

```bash
mkdir -p data/raw/fever
wget -O data/raw/fever/wiki-pages.zip https://fever.ai/download/fever/wiki-pages.zip
python scripts/build_fever_fts.py \
  --wiki-pages-zip data/raw/fever/wiki-pages.zip \
  --output indexes/fever_fts.sqlite
```

For BM25 and FAISS indexes, use `scripts/build_index.py` or notebook 02.

### Check and repair an answer

```python
from src.schema import AnswerRecord
from src.pipeline import run_pipeline

record = AnswerRecord(
    id="example_001",
    source_dataset="manual",
    question="Where was Alexander Graham Bell born?",
    answer="Alexander Graham Bell was born in London, England.",
)

result = run_pipeline(
    record,
    index_path="indexes/fever_fts.sqlite",
    retriever_method="fts5",
    verbose=True,
)
print(result.repaired_answer)
print(result.summary())
```

Model, retrieval, verification, repair policy, and iteration settings can be
passed to `run_pipeline()`. Benchmark runs use the protocol YAML and runtime
options in `configs/`. `configs/default.yaml` provides a reference configuration.

### Launch the interface

```bash
python demo/app.py
```

The Gradio interface displays the original and repaired answers, claim verdicts,
evidence passages, and word-level changes.

## Baselines and training

Eight baseline implementations support comparisons of detection and repair strategies:

| ID | Approach |
|---|---|
| B0 | Original answer without repair |
| B1 | Answer-level hallucination detection |
| B2 | SelfCheck-style sampling consistency and hedging |
| B3 | LLM self-repair using internal knowledge |
| B4 | Chain-of-Verification-style revision |
| B5 | RARR-style whole-answer retrieval and rewriting |
| B6 | Detector-driven deletion of flagged sentences |
| B7 | Factcheck-style claim verification and unguarded revision |

The training workflow provides a FEVER-based three-way verifier (Path A) and a
Flan-T5 repair editor trained on verified synthetic corruptions (Path C). See the
[training guide](docs/training_workflow.md) for data preparation, training, and evaluation.

## Evaluation and tests

The benchmark workflow records model outputs, execution attempts, configuration,
and provenance. Evaluation tools compute detection metrics, reference-repair
agreement, independently reviewed repair outcomes, and answer-level confidence
intervals. Report bundles support offline metric recomputation.

- [Evaluation protocol](docs/evaluation_protocol.md)
- [Annotation guidelines](docs/annotation_guidelines.md) and [gold-data preparation](docs/gold_annotation_workflow.md)
- [Benchmark execution](docs/benchmark_runner.md)
- [Review and scoring](docs/independent_evaluation.md)
- [Report bundles and auditing](docs/evaluation_bundles.md)

Run the automated tests in a dedicated CPU environment:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

## Repository layout

```text
src/          Pipeline stages, baselines, training, and experiment framework
scripts/      Data preparation, indexing, execution, training, and evaluation
configs/      Model options, evaluation protocols, and ablation settings
notebooks/    Colab workflows
demo/        Gradio application
tests/       Automated tests
data/        Dataset files and split manifests
indexes/     Retrieval indexes
results/     Figures and experiment outputs
docs/        Technical documentation and project report
```

## Data and references

Dataset preparation supports HaluEval, FELM, WikiBio, Factcheck-Bench, and FEVER.
Raw datasets, indexes, credentials, and large generated artifacts are excluded
from Git. Follow each dataset's license and attribution requirements.

- [FActScore](https://aclanthology.org/2023.emnlp-main.741/)
- [RARR](https://aclanthology.org/2023.acl-long.910/)
- [SelfCheckGPT](https://aclanthology.org/2023.emnlp-main.557/)
- [Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/)
- [FEVER](https://fever.ai/)
