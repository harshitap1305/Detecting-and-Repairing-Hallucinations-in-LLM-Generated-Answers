# Running the Hallucination Repair Project on Google Colab

Practical sizing and workflow advice for the NLP course project (claim-level hallucination detection and repair).

The numbers below are approximate, and the free Colab tier changes over time. Check what runtime you actually get: Runtime → Change runtime type, then run `!nvidia-smi`, `!free -h` and `!df -h`.

## Typical free-Colab limits

- **GPU:** usually a T4 with 16 GB.
- **RAM:** about 12 GB, which is the first limit you will hit.
- **Disk:** temporary, and wiped when the session ends.
- **Sessions:** they disconnect when idle or after several hours, so save progress as you go.

## What this means for the design

### 1. Store everything on Google Drive

Mount Drive and keep indexes, embeddings and the LLM response cache there. Anything left on the local disk is lost when the session ends.

### 2. Use a smaller corpus than all of Wikipedia

Dense vectors take about 3 KB each at 768 dimensions (float32).

| Passages | 768-dim embeddings | 384-dim embeddings |
| --- | --- | --- |
| 1M | about 3 GB | about 1.5 GB |
| 3M | about 9 GB (too big for 12 GB RAM) | about 4.5 GB |

Recommendation: stay at **1–2M passages**. That is enough for Wikipedia-based claim checking.

### 3. Build the Wikipedia subset by streaming it

Do not download the full dump. Use `load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)` and keep:

- every article whose title matches an entity in your test and gold questions, and
- a random sample of other articles (about 300–500k) as distractors.

Truncate each kept article to its first 300–500 words, then chunk it. The same approach works for FEVER's `wiki-pages` file. It has millions of pages (from memory, about 5M), so for retrieval evaluation keep only the pages used as gold evidence for your FEVER dev claims, plus distractors.

### 4. Choose lightweight retrieval tools

- **Dense encoder:** a small or base model such as bge-small or bge-base, run in fp16 with batches of 128–256.
- **FAISS:** `faiss-cpu` with a flat inner-product index is fine up to about 1–2M vectors, and it installs more reliably than `faiss-gpu`.
- **BM25:** `rank-bm25` gets slow beyond about 100k passages. Use `bm25s` (a fast pure-Python library; check that it still installs) or Pyserini (needs Java: `apt install openjdk-17-jre-headless`).

### 5. Encode in shards with checkpoints

Encode 50–100k passages at a time and save each shard as a `.npy` file on Drive, with a record of which shards are done. After a disconnect, rerun the notebook and it skips finished shards. Encoding 1M passages on a T4 is likely well under a few hours, but measure it on 50k passages first and extrapolate.

### 6. Pick models that fit a T4

- **NLI:** DeBERTa-v3-large (about 435M parameters) fits easily. Verifying 200 answers × about 8 claims × 5 passages is only about 8,000 pairs per run, which is quick.
- **LLM for decomposition and repair:** this is the hard part. A 7–8B model needs 4-bit quantization (bitsandbytes) and is slow. Realistic options:
  - an **API model** for the final runs (a cheap, small one is enough for development);
  - a **smaller open model** (about 3B) running locally for development.
- Whichever you choose, keep the **evaluator LLM** a different family from the pipeline LLM.

## Minimal workflow

1. **Notebook 1:** download data, build the passage list, write it to Drive as Parquet.
2. **Notebook 2:** encode in shards, then build the FAISS and BM25 indexes and save them to Drive.
3. **Notebook 3:** run the pipeline, using a Drive-backed cache for every LLM call.

Keep the notebooks thin and put the logic in `src/*.py` in your GitHub repo. Clone the repo at the top of each notebook so the whole team runs the same code.
