# Detecting and Repairing Hallucinations in LLM Answers — Detailed Implementation Plan

**Pipeline:** question + answer → claim decomposition → evidence retrieval → claim verification → localized repair → re-verification → repaired answer + report.

Conventions in this document:
- **[VERIFIED]** = checked against the dataset or tool's public page while preparing this plan.
- **[VERIFY]** = from general knowledge; check the README or page before relying on it.
- Numbers marked *(estimate)* are planning estimates, not measurements. Replace them with measured values after the first pilot run.

---

## 0. Corrections and decisions to apply to the original proposal

| Issue | What the proposal says | What to do |
|---|---|---|
| FELM size | about 3.9k responses | FELM has **847 prompts / 4,425 segments** (3,640 correct, 785 incorrect) across 5 domains **[VERIFIED]**. Use only world-knowledge (and optionally science/tech) for a Wikipedia-evidence pipeline. |
| FEVER on Hugging Face | load via `datasets` | The HF `fever` loader is script-based and **no longer works** with current `datasets` **[VERIFIED]**. Download from fever.ai or use a parquet mirror. |
| HaluEval-QA | "main detection data" | Built from HotpotQA with short answers and synthetic (ChatGPT-written) hallucinations **[VERIFIED]**. Use for answer-level detection and sanity checks, not as the main long-form claim-level benchmark. |
| Wikipedia corpus | "dump or HF dataset" | `wikimedia/wikipedia`, `20231101.en`, about 6.4M articles **[VERIFIED]**. Too large for a full dense index on free GPUs *(estimate)*. Use a scoped corpus (Section 3.2). |
| Closest prior work | not listed | Add **Factcheck-GPT / Factcheck-Bench** (decomposition, retrieval, stance, *and revision*) as related work **and** baseline B7. |
| Evaluator circularity | repair evaluated by re-running own pipeline | Add an **independent evaluator** (different LLM family + human spot checks). Otherwise gains are inflated. |

---

## 1. Final scope and definitions (write these into `docs/spec.md` in Phase 0)

- **Task:** English, open-domain factual QA and short biography-style answers. Evidence source: Wikipedia.
- **Atomic claim:** one independently checkable factual statement, decontextualized (no unresolved pronouns, includes the entity name and needed time/place qualifiers).
- **Check-worthy:** factual, verifiable statements. **Skipped:** opinions, questions, instructions, hedges ("I think"), meta-text ("As an AI...").
- **Labels:**
  - `Supported`: at least one retrieved passage entails the claim and none contradicts it.
  - `Contradicted`: a retrieved passage conflicts with the claim.
  - `Unverifiable`: evidence is insufficient (this can also mean retrieval failed; track this separately as a "retrieval-miss" analysis tag on the gold set).
- **Correct repair of a claim:** the edited text (a) is supported by cited evidence, (b) keeps the original meaning wherever the original was correct, (c) does not add information not in the evidence, (d) stays fluent.
- **Repair policy (default):** Supported → keep. Contradicted with replacement evidence → rewrite. Contradicted without replacement → remove. Unverifiable → hedge (configurable: remove).

---

## 2. Repository, environment and conventions

```
project/
├── README.md
├── docs/spec.md                 # definitions from Section 1
├── configs/
│   ├── default.yaml             # models, k, thresholds, seeds
│   └── ablations/*.yaml
├── data/
│   ├── raw/                     # downloaded, not committed
│   ├── processed/               # unified-schema jsonl
│   └── gold/                    # your annotated gold set
├── indexes/                     # bm25/, faiss/ (not committed)
├── src/
│   ├── schema.py                # dataclasses: Claim, Evidence, AnswerRecord
│   ├── llm.py                   # single wrapper + disk cache
│   ├── decompose.py
│   ├── retrieve.py
│   ├── verify.py
│   ├── repair.py
│   ├── pipeline.py
│   ├── evaluate.py
│   └── baselines/ (b1_...py ... b7_...py)
├── scripts/                     # download_data.py, build_index.py, run_eval.py
├── notebooks/
├── results/                     # tables, figures, raw outputs (versioned by run id)
├── demo/app.py
└── tests/
```

**How:**
1. One `llm.py` function `call(model, prompt, temperature=0, seed=…)` that hashes `(model, prompt, params)` to a key and stores responses in a SQLite or jsonl cache. Every module uses it. This gives reproducibility and cost control.
2. `schema.py` defines one record format used everywhere:

```json
{
  "id": "felm_wk_0012",
  "source_dataset": "felm",
  "question": "...",
  "answer": "...",
  "claims": [{"id": "c1", "text": "...", "sentence_id": 0,
              "span": [12, 87], "label": "Supported|Contradicted|Unverifiable",
              "evidence": [{"doc": "Title", "text": "...", "score": 0.0}]}],
  "gold_repaired_answer": null,
  "split": "dev|test"
}
```
3. `configs/default.yaml` holds all knobs (LLM names, retriever type, k, NLI thresholds, repair policy, iteration cap, seed). No magic numbers in code.
4. Log each run with a run id and the config hash (W&B or a CSV).
5. Git: branch per module, PR review, a `tests/` file per module with 5–10 hand-written cases.

---

## 3. Data: exactly what to download, how, and what it is used for

### 3.1 Datasets

| Dataset | Access | Size / license | Use |
|---|---|---|---|
| **HaluEval** `RUCAIBox/HaluEval` (HF mirror `pminervini/HaluEval`, `qa_samples`) **[VERIFIED]** | GitHub `data/` or HF | 35K samples (5K general queries with ChatGPT responses + 30K task-specific); MIT | Answer-level detection; use the *general* 5K subset for long-form |
| **FELM** `hkust-nlp/felm` **[VERIFIED]** | HF / GitHub | 847 prompts, 4,425 segments; CC BY-NC-SA 4.0 | Segment-level detection (world-knowledge subset) |
| **WikiBio GPT-3 hallucination** `potsawee/wiki_bio_gpt3_hallucination` **[VERIFIED]** | HF, split `evaluation` | 238 passages; sentence labels accurate / minor_inaccurate / major_inaccurate; CC BY-SA 3.0 | Small sentence-level benchmark |
| **Factcheck-Bench** `yuxiaw/Factcheck-GPT` **[VERIFIED]** | GitHub | 94 ChatGPT responses, 678 atomic claims, claim-level labels, claim corrections. Code license Apache-2.0; **data license: check repo** | Claim-level detection **and** revision gold |
| **TruthfulQA** **[VERIFIED]** | HF (`truthfulqa/truthful_qa`, config `generation`; check current ID) | 817 questions, single `validation` split; Apache-2.0 | Question source for gold set |
| **FEVER** fever.ai **[VERIFIED]** | `train.jsonl`, `paper_dev.jsonl`, `paper_test.jsonl`, `wiki-pages.zip` from fever.ai | 185,445 claims (Supported/Refuted/NEI), June 2017 Wikipedia intros; CC BY-SA 3.0 | Verifier training/eval, retrieval recall@k |
| **Wikipedia** `wikimedia/wikipedia`, `20231101.en` **[VERIFIED]** | `load_dataset("wikimedia/wikipedia", "20231101.en")` | about 6.4M articles; CC BY-SA 3.0 / GFDL | Main retrieval corpus (scoped) |
| **FActScore knowledge DB** `enwiki-20230401.db` **[VERIFIED]** via `python -m factscore.download_data` | pip package | large SQLite DB | Optional alternative corpus + reference metric |
| **RefChecker** `amazon-science/refchecker` **[VERIFIED]** | GitHub | 2.1k annotated responses (300 prompts × 7 LLMs) | Optional extra detection data / tool |
| FAVA, VeriScore, SAFE/LongFact, HHEM, Poly-FEVER, FaithBench | **[VERIFY]** | — | Optional extensions only |

Config names for HF datasets (e.g. FELM domains, HaluEval subsets) should be listed with `datasets.get_dataset_config_names(...)` **[VERIFY]** before writing loaders.

### 3.2 Retrieval corpus strategy (critical for compute)

- **Tier A (dev + retrieval evaluation):** FEVER `wiki-pages` (2017 intros). Smaller; has gold evidence labels in FEVER for recall@k.
- **Tier B (main pipeline):** scoped subset of `20231101.en`:
  - Keep only the **lead section + first N words** of each article (e.g. first 400–600 words), **or**
  - Keep all articles for entities appearing in the test/gold sets (found by running decomposition first and extracting entities / TruthfulQA topics) **plus** a large set of "distractor" popular articles.
- **Passage chunking:** 100–200 words, sentence-aware, with `Title: ` prepended to every chunk. Store `{doc_id, title, chunk_id, text}`.
- **Index sizes:** estimate vectors × dimension × 4 bytes before building. Use FAISS `IVF` or `IVFPQ` rather than flat if above about 5M vectors *(estimate)*.

### 3.3 Label mapping for evaluation

| Source | Mapping |
|---|---|
| FEVER | Supported→Supported, Refuted→Contradicted, NotEnoughInfo→Unverifiable |
| WikiBio | accurate→Supported; minor/major_inaccurate→hallucinated (no Unverifiable class) |
| FELM | segment correct / incorrect; a segment is "hallucinated" if **any** of its claims is flagged |
| HaluEval | answer-level hallucinated yes/no |
| Factcheck-Bench | claim-level support/refute/etc.; map "refute" → Contradicted, "irrelevant/partial" → Unverifiable (document the exact mapping) |

### 3.4 Splits

- Split **by question/prompt**, never by claim.
- `dev` (tune prompts, thresholds, fine-tune) and `test` (frozen, run rarely).
- Use FEVER's own train/dev for verifier training; hold out its test.
- Keep Factcheck-Bench, WikiBio and FELM-test as **never-tuned** external benchmarks.

---

## 4. Our own gold set (repair evaluation)

**Goal:** about 150 answers (50 dev, 100 test), about 1,000–1,500 claims.

**How:**
1. **Questions:** about 40% TruthfulQA, about 30% FELM world-knowledge prompts, about 30% biography prompts for rare Wikipedia entities ("Tell me about X"). Rare entities produce more hallucinations.
2. **Generate answers** with two LLMs (a weaker one to get more errors, a stronger one), prompting for 3–6 sentences.
3. **Pre-label:** run the draft pipeline (decompose → retrieve → verify) to pre-fill claim labels and evidence.
4. **Annotate (spreadsheet or Label Studio):** per claim, the human sets the label, picks/pastes supporting or refuting evidence, and writes a gold corrected claim (or `REMOVE`). Then the annotator writes the gold repaired answer (original text with minimal edits).
5. **Guidelines:** a one-page rubric with 10 worked examples, written from `docs/spec.md`.
6. **Agreement:** two annotators label the same 30 answers, report Cohen's κ for the 3-way label; resolve disagreements by discussion; calibrate before annotating the rest.
7. **Effort:** about 1–2 minutes per claim after pre-labeling, so roughly 20–40 person-hours *(estimate)* across the team. Start in Week 2.
8. **Free extra gold:** Factcheck-Bench includes claim corrections; use it as an external repair check.

**Test set for the "≥200 answers" success criterion:** gold-test (100) + Factcheck-Bench (94) + FELM world-knowledge sample; WikiBio (238) for detection-only.

---

## 5. Phase-by-phase plan

Each phase: **Goal → Tasks (how) → Deliverables → Exit criteria**.

### Phase 0 — Scope, setup, definitions (Week 1)

**Goal:** everyone works from the same definitions and environment.

**Tasks**
1. Write `docs/spec.md` (Section 1). Review as a team; freeze.
2. Create repo, `requirements.txt` (`transformers`, `sentence-transformers`, `faiss-cpu/gpu`, `rank-bm25` or `pyserini`, `datasets`, `scikit-learn`, `pandas`, `numpy`, `gradio`, `spacy`, `wandb`).
3. Choose and freeze models in `configs/default.yaml`:
   - LLM-A (decompose + repair + LLM-judge verifier).
   - LLM-B (**different family**, independent evaluator only).
   - Dense encoder (e.g. bge-base or e5-base) **[VERIFY]**; cross-encoder reranker **[VERIFY]**.
   - NLI model: a DeBERTa-v3 checkpoint fine-tuned on MNLI/FEVER/ANLI **[VERIFY exact HF ID + license]**.
4. Implement `llm.py` with cache; test cache hits.
5. Set up experiment tracking and a shared annotation spreadsheet.
6. Assign owners (data/eval; retrieval; verification; repair/demo; gold-set annotation owner; baselines owner).

**Deliverables:** repo, spec, working LLM+cache, config file.
**Exit:** a script that takes one question+answer and prints it back through `schema.py` objects, with a cached LLM call.

### Phase 1 — Literature + baseline reproduction (Weeks 1–2)

**Goal:** understand prior systems and reuse what works.

**Tasks**
1. Read in this order: FActScore → RARR → CoVe → SelfCheckGPT → RefChecker → FacTool → Factcheck-Bench/GPT → FAVA → code-hallucination paper (design inspiration).
2. For each paper, fill a 5-row template: *task, granularity, evidence source, repair or not, limitation*.
3. Run **FActScore** (`pip install factscore`) on about 10 examples and **Factcheck-GPT** on about 10 examples. Record input/output formats, runtime, cost.
4. Write a 1–2 page related-work summary (this becomes the report section).

**Deliverables:** literature summary, notes on reusable prompts/formats, working baseline run.
**Exit:** you can explain in one sentence how your method differs from each of RARR, CoVe, RefChecker, Factcheck-GPT.

### Phase 2 — Data and indexes (Week 2)

**Goal:** all data in one schema; retrieval indexes built.

**Tasks**
1. `scripts/download_data.py`: fetch every dataset in Section 3.1; print counts; check licenses.
2. One converter per dataset → `data/processed/*.jsonl` in the unified schema. Add unit tests that validate every record.
3. Make dev/test splits (Section 3.4) with a fixed seed; save split ids to `data/splits.json`.
4. Build indexes:
   - BM25 over passages (Pyserini or `rank-bm25` for small corpora).
   - Dense: encode passages in batches on GPU, save embeddings to disk, build FAISS.
   - Do **Tier A first**, then Tier B.
5. Start gold-set creation (Section 4): choose questions, generate answers.

**Deliverables:** processed data, indexes, splits, first batch of generated answers.
**Exit:** retrieval returns sensible passages for 20 hand-picked claims.

### Phase 3 — Minimal end-to-end skeleton (Week 3)

**Goal:** the whole pipeline runs, even if poorly.

**Tasks**
1. Decompose: one LLM prompt returning JSON claims.
2. Retrieve: BM25 top-5 per claim.
3. Verify: LLM-judge returning `{label, rationale}`.
4. Repair: one LLM call per flagged sentence.
5. `pipeline.py` glues stages and writes a report object.
6. Run on 20 dev answers; read every output; list the top 5 problems.

**Deliverables:** `pipeline.py` v0, a 20-example output file.
**Exit:** one command goes from (question, answer) to (repaired answer, per-claim table).

### Phase 4 — Claim decomposition (Weeks 3–4)

**Goal:** reliable atomic, decontextualized claims with source mapping.

**Tasks**
1. **Sentence split** (spaCy), assign `sentence_id`.
2. **Prompt (few-shot, JSON-only):**
   - Instruction: break each sentence into the smallest independently verifiable facts; resolve pronouns using the question and prior sentences; keep names, dates, numbers; mark non-factual items `check_worthy=false`.
   - Output: `[{"claim": "...", "sentence_id": 0, "check_worthy": true}]`.
   - Include 5–8 few-shot examples covering: a list sentence, a pronoun, a hedged sentence, an opinion, a multi-fact sentence.
3. **Parse and validate JSON** (retry once on failure; drop and log on second failure).
4. **Span mapping:** claim → `sentence_id` is the guaranteed link. Optionally compute a character span by fuzzy matching claim tokens inside the sentence; if no match, fall back to the whole sentence. Repair works at sentence level, so sentence-level mapping is sufficient.
5. **Validation:** sample 50 answers; two people score each claim for *atomicity, completeness (nothing lost), decontextualization* (0/1). Report rates; iterate the prompt until above a threshold the team sets (e.g. 90%).
6. **Edge cases tested:** questions, imperative sentences, opinions, "As an AI language model…" boilerplate.

**Deliverables:** `decompose.py`, a decomposition quality report with failure examples.
**Exit:** measured atomicity/completeness/decontextualization rates reported.

### Phase 5 — Evidence retrieval (Weeks 4–5)

**Goal:** choose the best retrieval configuration with evidence.

**Tasks**
1. Implement `retrieve(claim, method, k)` for **BM25**, **dense**, **hybrid** (reciprocal rank fusion over both rankings), optional **cross-encoder rerank** of the top 50 candidates.
2. Query variants: claim text only vs. claim plus entity title vs. LLM-generated question form.
3. **Evaluate on FEVER dev:** for claims with gold evidence, compute recall@k (is any gold evidence sentence/page in the top-k passages?) for k = 1, 3, 5, 10. Report per retriever.
4. **Evaluate on gold-set claims:** manual check that the evidence contains the answer for a sample of 50 claims.
5. Tune chunk size (100 vs. 200 words) and k; pick defaults; freeze in config.
6. Caching: key = (method, query, k, index version).

**Deliverables:** `retrieve.py`, retrieval comparison table.
**Exit:** one default retrieval config justified by numbers.

### Phase 6 — Claim verification (Weeks 5–6)

**Goal:** compare three verifiers and a calibrated aggregation rule.

**Tasks**
1. **Verifier 1 — NLI:** for each (passage, claim) pair compute entailment/neutral/contradiction probabilities (premise = passage, hypothesis = claim). Truncate long passages sensibly.
2. **Verifier 2 — LLM judge:** prompt with claim + numbered passages; output JSON `{label, supporting_passage_ids, rationale}`; temperature 0.
3. **Verifier 3 — fine-tuned encoder:** fine-tune a small encoder on FEVER (input: claim + gold or retrieved evidence; output: 3 classes). Use dev split for early stopping; log accuracy/F1.
4. **Aggregation (NLI):** `Supported` if `max_entail ≥ τ_s` and `max_contra < τ_c`; `Contradicted` if `max_contra ≥ τ_c`; else `Unverifiable`. Confidence = the winning score. **Tune τ_s and τ_c on dev only** (grid search maximizing F1 on dev gold claims).
5. **Answer reliability score** = fraction of check-worthy claims labeled `Supported`.
6. **Evaluation:** FEVER dev (3-way accuracy/F1), gold-dev claims (3-way confusion matrix), detection benchmarks (WikiBio, FELM, Factcheck-Bench) for claim- or segment-level P/R/F1 on the "hallucinated" class.
7. Inspect 30 errors per verifier; categorize (retrieval miss, negation, numeric, entity confusion, long-passage truncation).

**Deliverables:** `verify.py`, verifier comparison results, frozen thresholds.
**Exit:** one default verifier (or ensemble) chosen with evidence.

### Phase 7 — Localization and repair (Weeks 6–7)

**Goal:** minimal, evidence-grounded edits that do not break correct content.

**Tasks**
1. **Localization:** group claims by `sentence_id`; mark sentences containing Contradicted/Unverifiable claims; highlight them in the report.
2. **Repair prompt (per flagged sentence):** give original sentence, the flagged claim(s) with their labels, and the evidence passages. Rules: use only provided evidence; change only the flagged claim's wording; keep every other word and fact; if the evidence gives no replacement, apply the policy (remove or hedge); return only the edited sentence (or an empty string if the sentence should be deleted).
3. **Programmatic guards (after each LLM edit):**
   - Re-extract claims from the edited sentence; check that every originally Supported claim in that sentence is still entailed (NLI). If not → reject edit.
   - Check that new numbers/dates/named entities in the edit appear in the evidence text. If not → reject edit.
   - On reject: retry once with a stricter prompt, then fall back to deletion/hedging.
4. **Reassembly:** replace sentences in original order; keep untouched sentences byte-identical.
5. **Re-verification loop:** run Stages 1–3 on the repaired answer; if unsupported claims remain, run one more repair iteration (max 2), then delete remaining unsupported claims. Log iterations and convergence.
6. **Manual review:** inspect 30+ repairs on dev; tag issues (over-edit, copied evidence verbatim, lost fluency, new error); refine prompt and guards.
7. **Policy variants** for the ablation: rewrite-only / rewrite+hedge / rewrite+remove.

**Deliverables:** `repair.py`, `pipeline.py` (final), issue log with fixes.
**Exit:** pipeline runs on all dev answers with logged iterations and no crashes.

### Phase 8 — Full evaluation, baselines, ablations (Weeks 7–8)

**Goal:** produce every table and figure for the report.

**Tasks**
1. **Freeze** config and commit hash; record model versions and prompts.
2. Run the final pipeline on test once (a second run only if there is a bug; document it).
3. Run baselines B0–B7 (Section 6) on the same inputs.
4. Run ablations (Section 7) on dev (and the final config on test).
5. Compute metrics (Section 8) with bootstrap CIs.
6. Human evaluation: sample 50–100 test outputs, two raters, 1–5 scale for factual correctness, preservation, fluency, minimality; report mean ± CI and agreement.
7. Error analysis: label every failure with one category (decomposition error, retrieval miss, verifier error, repair over-edit, repair introduced error, evaluation-label noise) and count.

**Deliverables:** results tables/figures, error analysis.
**Exit:** all RQs (RQ1–RQ4) answerable from the tables.

### Phase 9 — Demo, report, presentation (Weeks 7–8, overlapping)

**Tasks**
1. **Demo (Gradio):** inputs for question and answer; outputs: highlighted answer (green/yellow/red by verdict), per-claim table with evidence, repaired answer, diff view, reliability score before/after, toggles for retriever, verifier and policy.
2. **Report** sections: introduction, related work, method, datasets, experiments, results, error analysis, limitations, conclusion.
3. **README:** setup, data download, index building, reproduction commands, license notes.
4. **Slides + (optional) demo video.**

**Deliverables:** demo, report, slides, clean repo.

---

## 6. Baselines (what to build and how)

All baselines use the **same input answers**, the same LLM-A where an LLM is needed, and the same evaluation code.

| ID | Method | Implementation | Question it answers |
|---|---|---|---|
| **B0** | No repair | Output = original answer | Lower bound |
| **B1** | Answer-level detection | One prompt on the whole answer: "does it contain factual errors? yes/no" (plus token-probability or verbalized confidence for AUROC) | Does claim-level beat answer-level? |
| **B2** | SelfCheckGPT-style | `selfcheckgpt` package **[VERIFY version]**: sample N=5–10 answers at temperature 1.0, score each sentence by NLI/prompt consistency. **Detection only** | Value of external evidence over consistency |
| **B3** | LLM self-repair, no evidence | Prompt: "Find and fix factual errors in this answer." | Value of retrieval for repair |
| **B4** | CoVe-style | Plan verification questions → answer them independently (no retrieval) → revise | Self-verification vs. evidence-based |
| **B5** | RARR-style rewrite | Retrieve evidence for the answer (or per sentence), then rewrite the **entire** answer with evidence | Local edits vs. whole rewrite (RQ3) |
| **B6** | Delete-only | Your detector + remove all flagged claims/sentences | Rewrite vs. delete |
| **B7** | Factcheck-GPT | Official repo pipeline incl. revision; swap in the same retriever if feasible and document differences | Closest published system |
| **B8** (optional) | Oracle detection | Use gold labels as detector, then your repair | Upper bound on repair quality |

Fairness notes: give B3–B5 the same evidence budget where applicable; report cost (calls/tokens) per method.

---

## 7. Ablations (run on dev; final configuration on test)

| Axis | Variants | Primary metrics |
|---|---|---|
| Claim granularity | whole answer / sentence / atomic claim | detection F1, preservation |
| Retriever | BM25 / dense / hybrid / hybrid+rerank | recall@k, end-to-end F1 |
| Top-k | 1 / 3 / 5 / 10 | detection F1, cost |
| Verifier | NLI / LLM judge / fine-tuned / ensemble | F1, calibration |
| Repair policy | rewrite-only / +hedge / +remove | fix rate, over-edit rate |
| Re-verification | off / 1 / 2 iterations | supported-claim rate, convergence |
| Evidence source (optional) | Wikipedia only / + web search | Unverifiable rate |

Change one factor at a time from the default configuration. A full grid is too expensive.

---

## 8. Metrics: how to compute each

**Detection**
- *Claim level:* P/R/F1 for the class "hallucinated" = {Contradicted, Unverifiable}. Also report the 3-way confusion matrix on FEVER-dev and gold claims.
- *Segment/sentence level (FELM, WikiBio):* a unit is flagged if any of its claims is flagged.
- *Answer level:* accuracy, F1 and AUROC using reliability score = 1 − fraction supported.
- *Retrieval:* recall@k on FEVER.

**Repair**
- *Supported-claim rate (before/after):* re-run decomposition + retrieval + verification on original and repaired answers with the **independent evaluator** (LLM-B + NLI), not the repair components.
- *Preservation rate / over-edit rate:* decompose original and repaired answers. For each originally Supported claim, search the repaired claims for a match using **bidirectional NLI entailment** (or embedding similarity above a threshold chosen on 50 manually checked pairs). Preservation = matched / total; over-edit = 1 − preservation (for originally Supported claims).
- *Fix rate:* human judgment (or LLM-B pre-screen + human sample) of whether each hallucinated claim was correctly repaired against the gold corrected claim.
- *Drift:* token-level edit distance, ROUGE-L, BERTScore vs. original.
- *Convergence rate and iterations:* from logs.
- *Human eval:* 1–5 scales (correctness, preservation, fluency, minimality), two raters, report agreement.

**Statistics**
- Bootstrap 95% CIs (1,000 resamples) over answers, not claims (claims in one answer are correlated).
- Paired bootstrap or Wilcoxon signed-rank for pipeline-vs-baseline comparisons on the same answers.
- Fix seeds; save all prompts, model versions and thresholds with each run.

---

## 9. Timeline (8 weeks)

| Week | Focus | Milestone |
|---|---|---|
| 1 | Phase 0, start Phase 1 | Spec, repo, LLM+cache |
| 2 | Phase 1–2 | Literature summary, data + Tier-A index, gold-set generation started |
| 3 | Phase 3, start 4 | End-to-end skeleton v0 |
| 4 | Phase 4–5 | Decomposition validated; retrieval comparison |
| 5 | Phase 5–6 | Verifier results; thresholds tuned on dev |
| 6 | Phase 6–7 | Repair + re-verification working; gold set complete |
| 7 | Phase 8 | Test run, baselines, ablations; human eval starts |
| 8 | Phase 8–9 | Error analysis, demo, report, slides |

Adjust to your real deadline. If you fall behind, cut in this order: web search → optional ablations → fine-tuned verifier → reranker → gold set down to 100 answers.

---

## 10. Compute and cost control

- Use free GPUs for encoding and NLI. Encode Tier A once and cache embeddings.
- Estimate calls per answer: 1 decompose + about 8 verifications (if LLM judge) + a few repairs + re-verification. Measure on 20 answers, multiply, and set a budget before running baselines.
- Develop on small dev subsets (20–50 answers); run the full test set only for final reporting.
- Prefer an open model via Ollama/vLLM for development, and the stronger model for final runs, if budget is limited.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Retrieval misses true facts → false Unverifiable | Report Unverifiable separately; hedge policy; retrieval-miss analysis tag; optional web search |
| Poor decomposition propagates | Validate on 50 answers early; iterate prompt |
| LLM judge noisy/biased | Compare with NLI and fine-tuned verifier; calibrate on dev |
| Repair over-edits or adds errors | Strict prompt, programmatic guards, over-edit metric, re-verification |
| Evaluator circularity | Independent LLM-B + NLI, plus human checks |
| Benchmark leakage | Held-out test, own gold set; acknowledge in report |
| Annotation overruns | Start Week 2; LLM pre-labeling; cut to 100 answers if needed |
| Index too large | Scoped Tier-B corpus; IVF/PQ; Tier-A for dev |
| Licenses | Keep CC BY-NC-SA (FELM) data non-commercial; check Factcheck-Bench data license before redistributing; do not commit large raw data |
| Scope creep | Freeze scope in Phase 0; stretch goals are optional |

---

## 12. Expected deliverables checklist

- [ ] Repo with README and reproduction commands
- [ ] Unified processed datasets (respecting licenses) and the gold set (with annotation guidelines and κ)
- [ ] Decomposition quality report
- [ ] Retrieval and verifier comparison tables
- [ ] Final pipeline results (detection and repair), baselines B0–B7, ablations
- [ ] Human evaluation results
- [ ] Error analysis with counts per category
- [ ] Demo app (highlights, evidence, diff, before/after reliability)
- [ ] Final report and slides

---

## 13. Known limitations to state in the report

- Verification quality is bounded by retrieved evidence and Wikipedia coverage; the 2023 dump misses recent events.
- Scope is English factual claims; reasoning, math, and opinions are out of scope.
- The gold set is small and team-annotated.
- LLM components are non-deterministic across versions; mitigated with temperature 0, seeds, caching.
- Public benchmarks may overlap with LLM training data.
