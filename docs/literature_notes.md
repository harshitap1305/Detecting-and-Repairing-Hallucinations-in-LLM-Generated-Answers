# Phase 1: Literature Survey and Related Work Notes

## 1. Summary of Prior Systems & Comparative Template

| System | Task & Goal | Claim Granularity | Evidence Source | Repair Support? | Main Limitation | Differentiation from Our Work |
|---|---|---|---|---|---|---|
| **FActScore** (Min et al., 2023) | Factual precision evaluation of biographies | Atomic facts | Wikipedia database | ❌ Scoring metric only; no repair | Cannot localize or repair erroneous spans | We provide an active detection, localization, and localized repair loop; FActScore is evaluation-only. |
| **RARR** (Gao et al., 2022) | Research and revise LLM outputs | Sentence/segment queries | Search engine (web) | ✅ Re-prompts model to revise full answer | Rewrites the whole response; prone to semantic drift and unnecessary edits | We localize edits to specific flagged sentences, preserving untouched correct sentences byte-for-byte. |
| **CoVe** (Dhuliawala et al., 2023) | Chain-of-Verification self-correction | Generated verification questions | Closed-book (LLM's internal weights) | ✅ Self-revision | Depends on the same hallucinatory model weights without grounding | We ground all verification in external, retrieved factual passages (Wikipedia/FEVER). |
| **SelfCheckGPT** (Manakul et al., 2023) | Black-box zero-resource hallucination detection | Sentence level | Stochastic sampling consistency | ❌ Detection only | Stochastically consistent hallucinations are not caught | We check external factuality against ground-truth evidence, catching confident popular falsehoods. |
| **RefChecker** (Hu et al., 2024) | Fine-grained reference-based checking | Knowledge triplets (S, P, O) | User-provided reference documents | ❌ Detection / benchmark only | Heavy triplet extraction; assumes gold references provided; no repair | We automate open-domain evidence retrieval and perform localized evidence-grounded repair. |
| **FacTool** (Chern et al., 2023) | Tool-augmented factuality checking | Extracted claims | Search engines / Python / Google Scholar | ❌ Detection only; no repair | Complex multi-tool setup; does not produce a repaired answer | We implement an integrated repair and re-verification loop. |
| **Factcheck-Bench / Factcheck-GPT** (Wang et al., 2023) | Fact-checking benchmark & revision pipeline | Atomic claims | Web search / Wikipedia | ✅ Claim correction & answer revision | Closest prior baseline; evaluates detection and revision | We use Factcheck-Bench as baseline B7 and test set; we evaluate preservation rate and independent evaluator circularity. |
| **FAVA** (Mishra et al., 2024) | Fine-grained hallucination taxonomy & editing | Token/span error types | Retrieved documents | ✅ Span-level editing | Heavy specialized model fine-tuning; error taxonomy dependent | We use a modular, transparent pipeline with explicit claim verification and programmatic guards. |
| **Code Hallucinations** (Khati et al., FORGE 2026) | Detecting and correcting hallucinations in code | AST structural units | Ground-truth library definitions / APIs | ✅ Deterministic local patch | Domain specific (code AST vs. natural language) | **Architectural inspiration:** decompose into units, verify against ground truth, edit locally, evaluate detection and fix separately. |

---

## 2. Key Architectural Decisions Borrowed & Adapted

1. **Decontextualization with Few-Shot Prompting (adapted from FActScore / Factcheck-GPT):**
   - Pronouns must be resolved into canonical named entities before retrieval so queries are self-contained.
   - Non-factual expressions (hedges, opinions, discourse markers) are marked `check_worthy = False`.

2. **Localization & Minimal Edits (adapted from RARR & Khati et al.):**
   - Rather than asking the LLM to rewrite the whole answer, edits are mapped to `sentence_id`.
   - Untouched sentences remain byte-identical, preventing semantic drift and stylistic corruption.

3. **Multi-Verifier Design (inspired by RefChecker & Factcheck-Bench):**
   - Verifier 1: Fast LLM judge (zero-shot structured JSON output) via Groq.
   - Verifier 2: DeBERTa-v3 cross-encoder NLI (calibrated thresholds $\tau_s, \tau_c$).

4. **Independent Evaluator Protocol (mitigating evaluator circularity):**
   - Pipeline uses LLM-A (e.g. LLaMA 3.3 70B via Groq) for decomposition, verification, and repair.
   - Evaluation uses LLM-B (e.g. Gemma 2 9B or external model) + NLI + human spot checks to prevent self-preference bias.

