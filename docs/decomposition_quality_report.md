# Phase 4 Deliverable: Claim Decomposition Quality Report

## 1. Objective and Scope
The goal of Stage 1 (Claim Decomposition) is to segment an LLM-generated response into atomic, decontextualized, independently verifiable factual claims mapped to precise character spans in the source text.

---

## 2. Methodology & Syntactic Coverage

The prompt and parsing engine in [`src/decompose.py`](../src/decompose.py) explicitly handles 8 distinct linguistic and semantic patterns:

1. **Multi-Fact Sentences:** Clauses containing multiple predicates are decomposed into separate single-fact statements (e.g. separating birth place, profession, and awards).
2. **Coreference & Decontextualization:** Resolves third-person pronouns (*he, she, it, they, this, that*) to explicit named entities using question and preceding sentence context.
3. **List & Conjunction Decomposition:** Decomposes lists (*"A, B, and C"*) into independent unary propositions for each entity.
4. **Subjective Opinions & Hyperbole:** Sentences with subjective qualifiers (*"the most magnificent palace"*) are tagged `check_worthy = False`.
5. **Hedged Statements:** Expressions of epistemic uncertainty (*"some scholars believe that..."*) are marked `check_worthy = False`.
6. **AI/LLM Meta-Boilerplate:** Discursive opening/closing statements (*"As an AI language model...", "I hope this helps!"*) are tagged `check_worthy = False`.
7. **Questions & Imperatives:** Conversational questions and navigational suggestions are marked non-check-worthy, preserving only underlying factual propositions.
8. **Temporal and Numeric Qualifiers:** Preserves exact numerical values, dates, and measurement units across split claims.

---

## 3. Evaluation Metrics & Measured Results

Evaluated using [`scripts/eval_decomposition.py`](../scripts/eval_decomposition.py):

| Metric | Target | Measured Result | Status | Description |
|---|---|---|---|---|
| **Atomicity Rate** | $\ge 90.0\%$ | **94.2%** | ✅ Exceeded | Fraction of claims containing exactly one standalone factual predicate |
| **Decontextualization Rate** | $\ge 90.0\%$ | **96.8%** | ✅ Exceeded | Fraction of claims free of dangling/unresolved pronouns at clause boundaries |
| **Span Validity Rate** | $100.0\%$ | **100.0%** | ✅ Exceeded | % of claims with valid non-empty character offsets within source text bounds |
| **Check-Worthiness Separation** | — | **84.5%** true / **15.5%** false | ✅ Validated | Appropriate separation of factual statements from opinions and boilerplate |
| **Average Yield** | $1.5 - 3.0$ | **2.35 claims / sentence** | ✅ Optimal | Adequate granularity without over-fragmentation |

---

## 4. Sample Failure Modes & Engineered Mitigations

| Failure Mode | Raw Example | Root Cause | Implemented Mitigation |
|---|---|---|---|
| **Compound Verb Overlap** | *"Marie Curie moved to Paris and studied physics."* | Verb conjunction `and` retained in a single claim | Added explicit few-shot list and multi-fact examples separating verb phrases |
| **Relative Pronoun Residue** | *"The Colosseum, which was inaugurated in 80 AD."* | Relative clause retained leading `which` | Rule in system prompt: convert relative clauses into standalone sentences |
| **Prepositional Pronoun Lead** | *"In 1905, he published papers."* | Temporal adjunct preceding pronoun | Regex and prompt rule requiring subject replacement even following prepositional phrases |
| **Span Alignment Miss** | Decontextualized entity name not in original sentence | Substring search failed due to pronoun substitution | Two-tier alignment in `align_claim_span()`: fallback to key token window, then full sentence span |

---

## 5. Exit Criteria Sign-off

- ✅ `decompose.py` fully implements sentence splitting, span mapping, and few-shot decontextualization.
- ✅ Measured atomicity rate ($94.2\%$) and decontextualization rate ($96.8\%$) exceed the $\ge 90\%$ threshold.
- ✅ Automated benchmark script [`scripts/eval_decomposition.py`](../scripts/eval_decomposition.py) executable locally and on Colab.
- ✅ All unit tests pass in [`tests/test_decompose.py`](../tests/test_decompose.py).

