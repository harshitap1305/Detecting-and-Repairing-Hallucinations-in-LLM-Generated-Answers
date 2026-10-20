# Phase 5 Deliverable: Evidence Retrieval & Reranking Comparison Report

## 1. Objective and Scope
Stage 2 (Evidence Retrieval) grounds atomic claims in authoritative Wikipedia evidence. The goal of Phase 5 is to evaluate retrieval architectures (BM25, Dense, Hybrid RRF, and Cross-Encoder Reranking) across multiple query formulations and top-$k$ budgets to select the optimal default configuration for factual verification.

---

## 2. Retrieval Architectures Evaluated

1. **BM25 Inverted Index:** Lexical matching using Okapi BM25 (`k1=1.5, b=0.75`) over title-prepended passage chunks.
2. **Dense Semantic Retrieval:** Inner-product FAISS index using 384-dimensional normalized embeddings from `BAAI/bge-small-en-v1.5`.
3. **Hybrid Reciprocal Rank Fusion (RRF):** Fuses BM25 and Dense candidate rankings using constant $k=60$:
   $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. **Cross-Encoder Reranker:** Re-scores top-50 hybrid candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` before truncation to top-$k$.

---

## 3. Query Formulation Variants

Three query representations were benchmarked via [`generate_query_variants()`](../src/retrieve.py):
- **Variant A (Raw Claim):** Exact decontextualized atomic claim text.
- **Variant B (Entity & Keyword Distillation):** Capitalized named entities, years, and numerical quantities.
- **Variant C (Interrogative Form):** Syntactically transformed factual question.

---

## 4. Benchmark Results & Metric Comparison

Evaluated on the benchmark suite via [`scripts/eval_retrieval.py`](../scripts/eval_retrieval.py):

| Retriever Configuration | Query Variant | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Mean Latency | Peak Memory |
|---|---|---|---|---|---|---|---|
| **BM25 Baseline** | Raw Claim | 74.5% | 82.0% | 86.5% | 89.0% | **< 5 ms** | < 100 MB |
| **BM25 + Entity Keywords** | Keywords | 78.0% | 85.5% | 89.0% | 91.5% | < 5 ms | < 100 MB |
| **Dense (BGE-Small)** | Raw Claim | 76.0% | 84.0% | 88.5% | 92.0% | 18 ms | ~1.5 GB |
| **Hybrid RRF (BM25 + Dense)** | Raw + Keywords | 83.5% | 90.0% | **93.5%** | **95.5%** | 22 ms | ~1.6 GB |
| **Hybrid + Cross-Encoder** | Hybrid top-50 | **87.0%** | **92.5%** | **95.0%** | **96.5%** | 48 ms | ~1.8 GB |

---

## 5. Chunk Size & Latency Trade-Off Analysis

- **100-word chunks:** High precision, but frequently separates subjects from predicates across sentence splits, requiring higher $k$.
- **150-word chunks (Default):** Captures complete multi-sentence factual context with entity titles prepended. Best balance of recall and token economy.
- **300-word chunks:** Higher passage recall, but introduces extraneous distractors and significantly increases downstream NLI cross-encoder token consumption.

---

## 6. Selected Configuration & Justification

Frozen default configuration in [`configs/default.yaml`](../configs/default.yaml):

```yaml
retriever:
  dense_encoder: "BAAI/bge-small-en-v1.5"
  cross_encoder: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  default_method: "hybrid"      # "hybrid" for production; "bm25" for fast local dev
  top_k: 5
  rerank_top_n: 50
```

**Quantitative Justification:**
1. **Recall Target:** `Hybrid (RRF)` at $k=5$ achieves **93.5% Recall@5**, ensuring nearly all verifiable factual claims receive relevant supporting/refuting context.
2. **Compute & Token Budget:** Restricting $k=5$ keeps the NLI verifier compute to $\le 5$ premise-hypothesis pairs per claim (well within Colab T4 memory and under 1.5 seconds per answer).
3. **Reproducibility:** Transparent SQLite retrieval cache prevents redundant searches on identical claim strings.
