# Suggested Next Steps: Training and Fine-Tuning Paths

Extension plan for the NLP course project *Detecting and Repairing Hallucinations in LLM-Generated Answers*. Do this **after** the base pipeline (prompted decomposition, retrieval, verification, repair, re-verification, baselines and ablations) works end to end.

Conventions:
- **[VERIFY]** = from general knowledge; check the model card, README or library docs before relying on it (model names, library APIs and defaults change).
- *(estimate)* = planning number, not measured. Replace it with a measured number after a small pilot.
- Code blocks are **sketches** to show the structure. Check library versions (`transformers`, `peft`, `trl`, `sentence-transformers`) before copying them.

---

## 1. Why add training, and what each path gives you

The base pipeline mostly uses prompts and pretrained models. For an NLP course, a trained component gives you an experiment you control: your own data, your own training run, your own evaluation, and a clear comparison with the prompted version.

| Path | What you train | Research question | Effort | Risk | Novelty |
|---|---|---|---|---|---|
| **A. Verifier** | Fine-tune a DeBERTa-style classifier on FEVER | Does a fine-tuned verifier beat off-the-shelf NLI and an LLM judge on claim detection? | Low | Low | Low–medium |
| **B. Retriever** | Fine-tune a bi-encoder (and/or a cross-encoder reranker) | Does fine-tuning on claim–evidence pairs improve recall@k and end-to-end F1? | Medium | Medium | Medium |
| **C. Repair model** | Fine-tune a small seq2seq or LoRA LLM to edit flagged sentences | Can a small trained editor match a prompted LLM while preserving correct content better and costing less? | Medium–high | Medium | **High** |
| **D. Decomposer** | Distill the LLM claim splitter into a small model | Can a small model replace API calls with little quality loss? | Medium | Low | Low–medium |
| **E. Meta-classifier** | Train logistic regression / gradient boosting on pipeline signals | Does combining NLI score, LLM verdict and retrieval score beat any single verifier? | Low | Low | Medium |
| **F. Preference tuning (stretch)** | DPO on top of the repair model | Does preference training reduce over-editing and new errors? | High | High | High |

**Recommended bundles** (pick one):

| Bundle | Paths | Good if |
|---|---|---|
| **Minimal** | A + E | Time is tight, or you want a safe, clearly "trained" contribution |
| **Standard (recommended)** | A + C | You want one strong, original result |
| **Ambitious** | A + B + C (+ F) | The team is strong, there are 3+ extra weeks, and compute is available |

If the course rubric says "train or fine-tune a model", confirm with your instructor that fine-tuning a pretrained model counts. It almost always does.

---

## 2. Rules that apply to every training path

1. **No test leakage.** Never train on test questions, test answers, or Wikipedia articles about test entities. Split training data **by Wikipedia article / entity** and by prompt. Keep the gold-test set, Factcheck-Bench test part, WikiBio and FELM-test untouched.
2. **Fix seeds** and log: model name and version, data version, hyperparameters, git commit, and training curves (W&B or CSV).
3. **Always compare to the non-trained version** from the base pipeline, on the same inputs, with the same metrics and bootstrap confidence intervals.
4. **Report training data size ablations** (for example 1k / 5k / 20k examples). They show whether the model is learning or just memorizing.
5. **Colab survival kit** (details in Section 9): checkpoint to Google Drive, resume after disconnects, small batches, mixed precision.
6. **Licenses.** FEVER annotations are CC BY-SA 3.0, FELM is CC BY-NC-SA 4.0, Wikipedia is CC BY-SA / GFDL. For a course project this is fine; if you publish trained models or data, keep attribution and share-alike, and check the license of every base model [VERIFY].

---

## 3. Path A — Fine-tuned verifier

### 3.1 Goal
Replace or complement the off-the-shelf NLI verifier with a classifier trained for **your** task: (evidence passages, claim) → {Supported, Contradicted, Unverifiable}.

### 3.2 Data
- **FEVER** train split (claims labeled Supported / Refuted / NotEnoughInfo). Map: Supported→Supported, Refuted→Contradicted, NEI→Unverifiable.
- **Evidence construction (important):**
  - Supported and Refuted claims: use the gold evidence sentence(s) as the positive premise.
  - NEI claims have **no gold evidence**. Use the **top-k passages from your own retriever** as the premise, otherwise the model learns "no evidence text → NEI".
  - For robustness, also train on retrieved passages for Supported/Refuted claims (sometimes the retriever misses the gold evidence), so the model sees realistic noisy input.
- **Optional extras:** MNLI/ANLI subsets for generalization [VERIFY availability], and **your gold-dev claims** (see Section 8) for a final small adaptation step. Do not use gold-test.
- **Size:** start with 20–50k examples *(estimate)*; the full FEVER train set is large, and a subset is enough for a course project.

### 3.3 Model
- Starting point options:
  - `microsoft/deberta-v3-base` (about 184M parameters, easy on a T4) [VERIFY], or
  - a DeBERTa-v3 checkpoint already fine-tuned on MNLI/FEVER/ANLI, if you want to continue from an NLI-trained model [VERIFY exact ID and license].
  - `deberta-v3-large` (about 435M parameters) also fits on a T4 with fp16 and small batches, but trains slower.
- Head: 3-class sequence classification.

### 3.4 Hyperparameters (starting points)
- Max length 256–384 tokens; learning rate 1e-5 to 2e-5; batch 16 (use gradient accumulation if memory is tight); 2–3 epochs; warmup 6–10%; weight decay 0.01; fp16.
- Pick the checkpoint by **dev macro-F1**, not training loss.

### 3.5 Sketch
```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

model_name = "microsoft/deberta-v3-base"  # [VERIFY]
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def encode(batch):
    # premise = evidence text, hypothesis = claim
    return tok(batch["evidence"], batch["claim"], truncation=True, max_length=320)

ds = load_dataset("json", data_files={"train": "data/verifier_train.jsonl",
                                      "dev": "data/verifier_dev.jsonl"})
ds = ds.map(encode, batched=True)

args = TrainingArguments(
    output_dir="/content/drive/MyDrive/ckpt/verifier",
    learning_rate=2e-5, per_device_train_batch_size=16,
    gradient_accumulation_steps=2, num_train_epochs=3,
    fp16=True, eval_strategy="steps", eval_steps=1000,
    save_steps=1000, save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="macro_f1", report_to="wandb", seed=42)
# Trainer(model, args, train_dataset=ds["train"], eval_dataset=ds["dev"], compute_metrics=...)
```

### 3.6 Evaluation
- FEVER dev: 3-way accuracy and macro-F1; confusion matrix.
- **Out-of-domain:** gold-test claims (LLM-generated text, not FEVER's artificial claims). Expect a drop. That drop is a finding.
- Claim-level detection on WikiBio / FELM / Factcheck-Bench (precision, recall, F1 for "hallucinated").
- Compare against: off-the-shelf NLI, LLM judge, and the ensemble. Report calibration (reliability diagram or expected calibration error).

### 3.7 Ablations
- Training with gold evidence only vs. gold + retrieved evidence.
- Base vs. large model; 10k vs. 50k examples.
- Starting from raw DeBERTa vs. from an MNLI-trained checkpoint.

### 3.8 Pitfalls
- **FEVER claims are artificial** (written by annotators from Wikipedia sentences), so a verifier trained only on FEVER may weaken on natural LLM claims. Test on the gold set to show this.
- NEI has no evidence: handled in 3.2.
- Long passages truncated: keep passages at 100–200 words and put the most relevant first.

---

## 4. Path B — Fine-tuned retriever

### 4.1 Goal
Improve evidence retrieval for claims, which feeds every downstream stage.

### 4.2 Data
- FEVER Supported/Refuted claims with gold evidence pages/sentences → positive passages.
- **Hard negatives:** BM25 top-ranked passages that are **not** gold (same topic, wrong fact). Also include negatives from other chunks of the same article.
- Training triples: (claim, positive passage, hard negative).
- Split by article.

### 4.3 Model options
1. **Bi-encoder** (`sentence-transformers`): continue training a small embedding model (bge-small/base or e5-base) [VERIFY], with `MultipleNegativesRankingLoss` plus hard negatives.
2. **Cross-encoder reranker:** train a small encoder to score (claim, passage) pairs; rerank the top 50 from the hybrid retriever. Often a bigger gain than fine-tuning the bi-encoder.

### 4.4 Hyperparameters (starting points)
- Batch size as large as memory allows (in-batch negatives help; 32–64); learning rate 2e-5; 1–2 epochs; max sequence length 192–256; warmup 10%.
- After training, **re-encode the corpus** (this takes the same time as the original indexing, so budget for it).

### 4.5 Evaluation
- FEVER dev recall@1/3/5/10 against the untouched pretrained retriever and BM25.
- End-to-end: detection F1 and supported-claim rate with the fine-tuned vs. pretrained retriever (everything else fixed).
- Gold-set claims: manual check of evidence quality on 50 claims.

### 4.6 Pitfalls
- FEVER retrieval is easier than real LLM claims. Report both.
- Index rebuilding on Colab is the slow step. Do it in shards (see the Colab guide) and only for the final model.

---

## 5. Path C — Trained repair model (recommended flagship)

### 5.1 Goal
Train a **small** model to perform the Stage-4 edit: given a flagged sentence, the flagged claim and its label, and evidence passages, output a minimally edited sentence. Compare with the prompted LLM repair from the base pipeline.

### 5.2 Why this is a good project
- The task is precise and measurable (the pipeline already has fix rate, preservation, over-edit rate).
- Training data can be generated **without human annotation**.
- The comparison is meaningful: quality, preservation, cost and latency vs. a prompted LLM.

### 5.3 Building the training data (core of this path)

**Step 1 — Source sentences.** Sample factual sentences from Wikipedia articles that are **not** about any test or gold-set entity. Keep sentences with named entities, numbers or dates (these are easy to corrupt and verify). Pair each with its source passage as evidence.

**Step 2 — Create corrupted versions** (the "hallucinated" input) with a mix of two methods:

| Method | How | Pros / cons |
|---|---|---|
| **Rule-based** | Swap an entity for another of the same type (NER + entity lists), shift a number or date, replace a relation word or an antonym | Free, deterministic, scalable, easy to label by construction; less natural |
| **LLM-generated** | Prompt an LLM: "Rewrite this sentence with exactly one factual error of type X; return JSON" | More natural errors; costs API calls (free-tier limits will slow this) |

Suggested mix: about 70% rule-based, 30% LLM-generated. Error types to cover: entity swap, wrong number/date, wrong relation, fabricated added detail, unverifiable embellishment (a plausible but unsupported claim).

**Step 3 — Verify the corruption with your own pipeline.** Keep an example only if (a) the original sentence is **Supported** by its evidence and (b) the corrupted claim is **Contradicted** (or Unverifiable for fabricated details). Discard the rest. This filter protects label quality.

**Step 4 — Build the input/target format.**

```
### Task
Fix only the unsupported claim, using only the evidence. Keep everything else unchanged.
### Sentence
<corrupted sentence>
### Flagged claim (label: Contradicted)
<claim>
### Evidence
[1] <passage 1>
[2] <passage 2>
### Edited sentence
```
Target = the original sentence (or a hedged / removed version, depending on policy).

**Step 5 — Include three kinds of examples:**
1. **Contradicted + evidence has the right fact** → target: corrected sentence.
2. **Unverifiable** → target: hedged sentence ("reportedly ...") or `<REMOVE>`, according to your policy setting.
3. **Nothing wrong** (a Supported sentence with no flagged claim) → target: **unchanged sentence**. About 15–25% of the data. These teach preservation and fight over-editing.

**Step 6 — Add a little real data.** Add the repairs from Factcheck-Bench **training part only** (split its 94 answers by prompt, for example 60 train / 34 test; decide the split once and record it) and your gold-dev answers. Real examples help with the synthetic-to-real shift. Hold out gold-test and the Factcheck-Bench test part.

**Size:** 5k examples for a first run, up to 20k for the final run *(estimate)*. Report an ablation over data size.

### 5.4 Model options

| Option | Notes |
|---|---|
| **Flan-T5-base (about 250M) or large (about 780M)** [VERIFY] | Full fine-tuning is feasible on a T4; stable, easy to evaluate; good first choice |
| **Small instruct LLM (about 0.5B–1.5B, e.g. Qwen2.5 or Llama 3.2 sizes)** [VERIFY names and licenses] with **LoRA** | More fluent output; more setup; QLoRA (4-bit) helps memory |

Start with Flan-T5-base to get a pipeline working. Move to a LoRA LLM only if time allows, and report both if you do.

### 5.5 Hyperparameters (starting points)
- **Flan-T5 full fine-tune:** learning rate 3e-4 to 1e-3 with Adafactor, or 1e-4 with AdamW; batch 8–16; input length 512; 3–5 epochs.
- **LoRA on an LLM:** rank 16, alpha 32, dropout 0.05, target attention projection layers; learning rate 1e-4 to 2e-4; batch 4 with gradient accumulation 8; 1–3 epochs; max sequence 512; gradient checkpointing on.
- Use `bf16` or `fp16` as the GPU supports; T4 typically needs fp16.

### 5.6 Sketch (LoRA)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig   # API changes between versions: check docs

base = "Qwen/Qwen2.5-1.5B-Instruct"     # [VERIFY]
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, quantization_config=BitsAndBytesConfig(load_in_4bit=True))  # optional QLoRA
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"],
                                         task_type="CAUSAL_LM"))
# Dataset: one text field = prompt + target. Compute loss on the target only
# (use a completion-only collator, or mask the prompt tokens).
```

### 5.7 Inference integration
- Plug the trained model into `repair.py` as another repair backend (`repair_backend: prompted | trained`).
- Keep the **same programmatic guards** (re-check claims in the edited sentence; reject new numbers and entities not in the evidence) and the **same re-verification loop**, so the only difference is the editor.

### 5.8 Evaluation
**Synthetic held-out test** (articles not in training): exact match and token-level F1 against the original sentence. This checks that the model learned the task.

**Real test (the important one):** gold-test, Factcheck-Bench test part, WikiBio/FELM flagged sentences where possible. Metrics from the base plan:
- supported-claim rate before/after (independent evaluator: different LLM + NLI),
- fix rate (human or LLM-B pre-screened plus human sample),
- **preservation and over-edit rate**,
- drift (edit distance, ROUGE-L, BERTScore),
- convergence rate of the re-verification loop,
- human evaluation (correctness, preservation, fluency, minimality).

Plus **cost and speed:** seconds per sentence, memory, and API cost avoided.

**Comparisons:** B3 (self-repair, no evidence), B5 (whole-answer rewrite), prompted evidence-based repair (your main pipeline), B6 (delete-only), and the trained repair model.

### 5.9 Ablations
- Data size (1k / 5k / 20k).
- Rule-based only vs. LLM-generated only vs. mixed corruptions.
- With vs. without the "nothing wrong" examples (shows the effect on over-editing).
- With vs. without the real-data examples (shows the synthetic-to-real gap).
- Model size (base vs. large, or 0.5B vs. 1.5B).
- With vs. without the programmatic guards.

### 5.10 Pitfalls and how to handle them
| Pitfall | Handling |
|---|---|
| Model copies evidence verbatim | Add training examples where the target keeps the original wording; penalize large edit distance in your analysis |
| Over-editing | "Nothing wrong" examples; guards; report over-edit rate |
| Synthetic-to-real gap | Add real examples; report the gap honestly |
| Rule-based corruptions create unnatural errors | Mix in LLM-generated ones; manually inspect 50 samples |
| Leakage via shared entities | Split by article; exclude all gold/test entities from training sources |
| Evaluator is the same family as the editor | Use an independent evaluator and human checks |

---

## 6. Path D — Distilled claim decomposer

### 6.1 Goal
Replace API calls in Stage 1 with a small fine-tuned model, keeping quality close to the LLM teacher.

### 6.2 Data
1. Collect 3–5k diverse sentences or short answers (model-generated answers to TruthfulQA / HaluEval-general prompts, WikiBio sentences, biography prompts). Do **not** include gold-test or test prompts.
2. Run the LLM decomposer (the teacher) on them to get JSON claims.
3. **Filter** teacher outputs automatically:
   - valid JSON,
   - each claim is entailed by the source sentence (NLI),
   - coverage check: every named entity and number in the sentence appears in some claim,
   - claims are not duplicates.
4. Manually inspect about 100 examples.

### 6.3 Model
Flan-T5-base or small (about 250M / 80M) [VERIFY], input = sentence plus the previous sentence for context, output = claims separated by a delimiter (simpler than JSON for small models; convert to JSON in code).

### 6.4 Evaluation
- On your 50 manually checked answers: atomicity, completeness, decontextualization rates, teacher vs. student.
- Downstream: detection F1 of the full pipeline with each decomposer (the metric that matters).
- Speed and cost per answer.

### 6.5 Note
This is solid engineering but the weakest research story. Choose it only if API limits are blocking you.

---

## 7. Path E — Meta-classifier over pipeline signals

### 7.1 Goal
Combine multiple verification signals into one better decision, using classical ML.

### 7.2 Features per claim
- max entailment, max contradiction and mean entailment over top-k passages (NLI);
- the fine-tuned verifier's class probabilities (if Path A is done);
- LLM-judge label (one-hot) and its confidence if available;
- retrieval scores (top-1 score, score gap between top-1 and top-2), BM25 vs. dense agreement;
- claim features: length, number of entities, has number/date, claim type;
- consistency across passages (how many support vs. contradict).

### 7.3 Training
- Label = gold claim label from gold-dev (about 50 answers, maybe 400 claims) plus FEVER-derived claims and Factcheck-Bench train part. Small data means a **simple model**: logistic regression, then gradient boosting if data allows.
- Use cross-validation on dev; choose class weights to favour precision or recall as needed.

### 7.4 Evaluation
- Claim-level F1 vs. each single verifier and a naive majority vote.
- Feature importance (shows which signal matters), calibration curve.
- Ablation by feature group.

### 7.5 Note
Cheap, fast, and gives a clear analysis chapter. Pairs well with Path A.

---

## 8. Path F — Preference tuning for repair (stretch)

Only attempt after Path C works.

1. **Preference pairs:** for each flagged sentence, **chosen** = the correct minimal edit (gold or best guard-passing output); **rejected** = a bad edit (over-edited sentence, a fluent edit that introduces a new unsupported claim, or a verbatim copy of the evidence). Generate rejected examples from the model's own bad samples or from baseline B5.
2. Train with DPO (`trl` DPOTrainer) on top of the LoRA repair model [VERIFY API]. Learning rate about 5e-6 to 5e-5 for LoRA setups, beta about 0.1.
3. Evaluate on over-edit rate, preservation and new-error rate vs. the SFT-only model.
4. Risk: unstable training, more tuning, and a small preference set. Only worth it if the team has spare time.

---

## 9. Colab practicalities for training

1. **Checkpoint to Drive.** Set `output_dir` on Drive, `save_steps` to a few hundred, `save_total_limit=2`, and resume with `trainer.train(resume_from_checkpoint=True)` after a disconnect.
2. **Memory.** fp16 or bf16, small per-device batch with gradient accumulation, gradient checkpointing, 4-bit loading for LLMs. If you hit out-of-memory errors, cut sequence length first.
3. **Data on Drive as JSONL**; load with `datasets`. Do not rebuild datasets in every session.
4. **Run time.** Estimate by training for 200 steps and extrapolating. Choose dataset size so a run takes at most a few hours *(estimate)*, since sessions can end unexpectedly.
5. **Logging.** Log to W&B or a CSV on Drive; save the config (YAML) next to each checkpoint.
6. **Install pins.** Write exact library versions to `requirements.txt` after the first working run; library APIs for `peft`/`trl`/`transformers` move quickly.
7. **Share trained models** with teammates through Drive or the Hugging Face Hub (private repo) so nobody retrains.
8. **Fallback.** If Colab quota runs out, Kaggle notebooks offer free GPU time too [VERIFY current limits].

---

## 10. Suggested schedule after the base plan

Assume the base pipeline, baselines and ablations are finished. Adjust to your real deadline.

### Standard bundle (A + C): 4 weeks

| Week | Work | Output |
|---|---|---|
| 1 | Path A: build verifier data, train, evaluate on FEVER dev and gold set. Start repair-data generation (Path C steps 1–3) in parallel | Verifier results; first 2k filtered synthetic examples |
| 2 | Path C: finish data (5k+), format, train Flan-T5-base; synthetic held-out evaluation; plug into `repair.py` | First trained repair model; synthetic test numbers |
| 3 | Path C: real-data evaluation vs. prompted repair and baselines; ablations (data size, corruption mix, "nothing wrong" examples) | Result tables for RQ comparisons |
| 4 | Human evaluation on a sample; error analysis; update report, slides and demo (add a "repair backend" toggle) | Final report sections, demo update |

### Minimal bundle (A + E): 2 weeks
Week 1: Path A. Week 2: Path E, ablations, report.

### Ambitious bundle (A + B + C + F): 6–7 weeks
Add Path B in parallel with Path C data generation (weeks 1–2), then Path F in the last 2 weeks if Path C is stable.

If time gets tight, drop in this order: Path F → Path D → Path B → LoRA LLM variant of Path C (keep Flan-T5) → Path A's large model.

---

## 11. How to frame it in the report

Possible research questions to add:
- **RQ5:** Does a fine-tuned verifier trained on FEVER generalize to natural LLM-generated claims better or worse than off-the-shelf NLI?
- **RQ6:** Can a small trained repair model match a prompted LLM in fix rate while improving preservation and reducing cost?
- **RQ7:** How much of the performance comes from synthetic training data, and how large is the synthetic-to-real gap?

Report sections to add: *Training data construction* (with filtering statistics and examples), *Training setup* (model, hyperparameters, compute), *Results vs. non-trained components*, *Data-size and data-mix ablations*, *Failure analysis of the trained model*.

Be honest about limits: synthetic errors differ from real hallucinations, the gold set is small, and the base models may have seen the public benchmarks.

---

## 12. Decision guide

| If your situation is... | Choose |
|---|---|
| Little time or compute, need something clearly "trained" | A + E |
| Want one strong original result | A + C |
| API limits keep blocking you | D (distill the decomposer) and use local models |
| Retrieval is your measured bottleneck (many Unverifiable false alarms) | B |
| Team is strong and ahead of schedule | A + C, then F |
| Rubric stresses classical ML techniques | E (plus A) |

Start by measuring where the base pipeline fails most (decomposition, retrieval, verification or repair) from your error analysis, and put the training effort there.
