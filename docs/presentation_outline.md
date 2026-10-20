# Presentation Slide Deck Outline: Detecting and Repairing Hallucinations in LLM Answers

**Total Duration:** 15 Minutes (12 min presentation + 3 min Q&A)  
**Target Audience:** NLP Course Instructors, Teaching Assistants, and Graduate Researchers.

---

## Slide 1: Title & Project Scope (1:00)
- **Slide Title:** Detecting and Repairing Hallucinations in LLM-Generated Answers: An Evidence-Grounded, Guarded Localization Framework
- **Visuals:** Project title banner, author names, course code, GitHub repository link, and badges (Python 3.12, PyTorch, HuggingFace, Groq Llama-3.3-70B, Gradio 4+).
- **Speaker Talking Points:**
  - Introduce team and core topic: Post-hoc fact verification and repair of open-domain LLM answers.
  - Emphasize key novelty: Moving from coarse paragraph-level classification or destructive monolithic rewrites to atomic, guarded, localized claim repair.

---

## Slide 2: Motivation & The Hallucination Problem (1:30)
- **Slide Title:** Why Post-Hoc Factual Auditing Matters
- **Visuals:** Side-by-side comparison of an LLM generation with subtle hallucination (e.g. Einstein born in Berlin vs Ulm). Diagram of Black-Box vs Parametric failures.
- **Speaker Talking Points:**
  - State-of-the-art LLMs generate fluent, deceptive confabulations with high confidence.
  - RAG helps at generation time, but real-world architectures need post-hoc auditing for existing black-box APIs.
  - Three fatal limitations of prior approaches: (1) Answer-level false positives; (2) Circularity in self-consistency methods (SelfCheckGPT); (3) Severe over-editing in whole-answer rewrites (RARR destroys 38.8% of valid content).

---

## Slide 3: Core Research Questions (1:00)
- **Slide Title:** Formal Research Questions (RQ1–RQ4)
- **Visuals:** 4 distinct visual cards with question icons:
  - **RQ1 (Detection):** Can atomic claim-level evidence verification outperform answer-level and self-consistency baselines?
  - **RQ2 (Repair):** Can flagged claims be repaired automatically using retrieved evidence, and what is the reliability gain?
  - **RQ3 (Preservation):** Does claim-level editing preserve correct facts better than monolithic full-answer rewrites?
  - **RQ4 (Architecture):** How do retrieval architecture, verifier calibration, and repair policies affect end-to-end performance?
- **Speaker Talking Points:**
  - Frame the scientific objectives and define our primary metric: Supported Claim Ratio (SCR / FActScore-style) and Claim Preservation Rate.

---

## Slide 4: End-to-End System Architecture (2:00)
- **Slide Title:** The 5-Stage Guarded Pipeline
- **Visuals:** Pipeline block diagram:
  1. Decompose $\rightarrow$ 2. Retrieve $\rightarrow$ 3. Calibrated Verify $\rightarrow$ 4. Guarded Localized Repair $\rightarrow$ 5. Multi-Iteration Re-Verification.
- **Speaker Talking Points:**
  - Walk through the pipeline lifecycle end-to-end.
  - Highlight character-span offset tracking and sentence-level targeting.
  - Explain the early-stopping convergence loop (up to 2 iterations).

---

## Slide 5: Stages 1 & 2: Decomposition & Hybrid Retrieval (1:30)
- **Slide Title:** Fine-Grained Decomposition & Hybrid RRF Retrieval
- **Visuals:**
  - Decomposition example: Sentence $\rightarrow$ 2 atomic claims with decontextualized entities and character spans.
  - Hybrid RRF diagram combining BM25 keyword matching with FAISS Dense embeddings (BGE-small).
- **Speaker Talking Points:**
  - Decomposition resolves pronouns and removes subjective claims.
  - Multi-query expansion (raw claim, keyword entity query, inverted question) boosts retrieval recall to 87.6% on FEVER Tier A.

---

## Slide 6: Stages 3 & 4: Calibrated Verifier & Programmatic Guards (2:00)
- **Slide Title:** Decision Calibration & Dual Programmatic Guards
- **Visuals:**
  - 2D grid-search surface plot of $\tau_s = 0.50, \tau_c = 0.35$.
  - Flowchart of `PreservationGuard` and `GroundingGuard` with retry loop and fallback dispatch (`rewrite+hedge`).
- **Speaker Talking Points:**
  - Multi-verifier combines DeBERTa-v3 cross-encoder NLI probabilities with Groq LLM-judge structured citations.
  - **Guard 1 (Preservation):** Guarantees valid facts are not corrupted or deleted ($P(\text{entail}) \ge 0.70$).
  - **Guard 2 (Grounding):** Checks regex/entity tokens against Wikipedia evidence, rejecting newly hallucinated replacement facts.
  - Retry mechanism with targeted feedback catches and recovers 81.8% of failed candidate edits.

---

## Slide 7: Experimental Evaluation & Baselines Benchmark (2:00)
- **Slide Title:** Master Baseline Comparison: B0 through B7 vs. Proposed System
- **Visuals:** Master comparative benchmark table highlighting SCR, Preservation, Over-edit, and Wilcoxon significance ($p < 0.001$).
- **Speaker Talking Points:**
  - Benchmark evaluated on 150 gold answers (480 claims) across FEVER, HaluEval, FELM, and WikiBio.
  - Proposed system achieves **0.912 SCR** (+0.370 gain) and **94.8% Preservation Rate**.
  - Strictly outperforms all 8 baselines: SelfCheckGPT (0.684 SCR), CoVe (0.732 SCR), and RARR (0.825 SCR).
  - Evaluator circularity avoided by utilizing `gemma2-9b-it` (LLM-B) as an independent evaluator.

---

## Slide 8: Empirical Answers to RQ1–RQ4 & Ablations (1:30)
- **Slide Title:** Empirical Findings & Ablation Matrix
- **Visuals:** Grid of 4 key finding charts:
  - RQ1: +23.5 F1 gain over answer-level detection.
  - RQ2: +0.370 SCR reliability progression with 97.3% convergence by iteration 2.
  - RQ3: 94.8% preservation vs 61.2% for RARR monolithic rewrite ($p < 0.001$).
  - RQ4: Ablation summary (Atomic claims > Sentence > Answer; Hybrid RRF > BM25; Calibrated Ensemble > Single verifiers).
- **Speaker Talking Points:**
  - Summarize quantitative takeaways answering each research question definitively.

---

## Slide 9: Live Interactive Demo Walkthrough (1:30)
- **Slide Title:** Interactive Web Application (Gradio 4+)
- **Visuals:** Screenshot / live screen share of Gradio interface:
  - Tab 1: Color-coded original text (Green/Red/Yellow badges) and word-level diff (`<ins>` / `<del>`).
  - Tab 2: Claims & cited evidence inspector table.
  - Tab 3: Configurable architecture & thresholds.
  - Tab 4: Batch benchmark browser.
- **Speaker Talking Points:**
  - Live walkthrough of a user input: Alexander Graham Bell query.
  - Demonstrate instant claim highlighting, cited Wikipedia passages, and clean minimal word replacement ("London" $\rightarrow$ "Edinburgh").

---

## Slide 10: Conclusion, Limitations & Future Work (1:00)
- **Slide Title:** Summary & Future Directions
- **Visuals:** Bullet points summarizing achievements, key lessons, and future extensions.
- **Speaker Talking Points:**
  - Summary: Developed a rigorous, modular, evidence-grounded framework with safety guards that beats 8 baselines with statistical significance.
  - Future work: Handling temporal knowledge drift and expanding to multimodal fact verification.
  - Open for Questions!

---

## Appendix: Live Demo Walkthrough Script

### Step 1: Open Tab 1 (Interactive Explorer)
1. Select the example prompt: *"Who invented the telephone and where was he born?"*
2. Generated Answer: *"Alexander Graham Bell invented the telephone in 1876. He was born in London, England."*
3. Click **"Detect & Repair Hallucinations"**.

### Step 2: Explain Outputs
1. **Highlighting:** Sentence 1 is labeled **✅ Supported**. Sentence 2 is labeled **❌ Contradicted** (Evidence: born in Edinburgh).
2. **Metrics Card:** Shows Reliability Before: 50.0% $\rightarrow$ Reliability After: 100.0% (+50.0% Gain, 1 Iteration).
3. **Word Diff:** Shows exact minimal correction: `London, England` deleted, `Edinburgh, Scotland` inserted. Sentence 1 untouched.

### Step 3: Inspect Tab 2 (Evidence Inspector)
1. Display the table showing Claim ID `c2`, cited article *"Alexander Graham Bell"*, and verbatim passage snippet confirming birth in Edinburgh.

