# Detecting and Repairing Hallucinations in LLM-Generated Answers

> **NLP Course Project** — claim-level hallucination detection and evidence-grounded repair.

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

### 3. Run the pipeline on a single example (no index needed)

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
├── .env                    ← GROQ_API_KEY (not committed)
├── docs/
│   ├── spec.md             ← task definitions, label schema
│   └── literature_notes.md
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
│   └── baselines/          ← B0–B7 baseline implementations
├── scripts/
│   ├── download_data.py    ← fetch & convert all datasets
│   ├── build_index.py      ← build BM25 + FAISS indexes
│   └── run_eval.py         ← compute metrics on pipeline output
├── notebooks/
│   ├── 01_download_data.ipynb   ← Colab: download datasets
│   ├── 02_build_index.ipynb     ← Colab: build indexes
│   └── 03_run_pipeline.ipynb    ← Colab: run & evaluate pipeline
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
3. Run notebooks in order: `01` → `02` → `03`
4. All indexes, processed data, and LLM cache live on Drive

```python
# Add at the top of every Colab notebook:
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE_BASE = "/content/drive/MyDrive/hallucination_project"
os.environ["GROQ_API_KEY"] = "your_key"
os.environ["LLM_CACHE_PATH"] = f"{DRIVE_BASE}/llm_cache.db"
```

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
| FEVER wiki-pages | Retrieval corpus (Tier A) | — |
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
