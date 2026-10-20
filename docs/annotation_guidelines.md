# Human Gold Set Annotation Guidelines
## Detecting and Repairing Hallucinations in LLM-Generated Answers

> **Purpose:** Standard operating procedure for human annotators creating the gold benchmark dataset ($N=150$ answers: 50 dev, 100 test).  
> **Target Inter-Annotator Agreement:** Cohen's $\kappa \ge 0.70$.

---

## 1. Task Overview

For each given Question and LLM-Generated Answer:
1. **Verify Extracted Claims:** Check that atomic claims are atomic, decontextualized, and check-worthy.
2. **Assign Claim Verdict Labels:** Label each claim as `Supported`, `Contradicted`, or `Unverifiable` against Wikipedia evidence.
3. **Provide Supporting/Refuting Citations:** Cite the exact Wikipedia article title and sentence.
4. **Write Gold Repaired Sentence / Answer:** Produce minimal factual edits fixing or hedging errors while preserving untouched correct text byte-for-byte.

---

## 2. Verdict Label Rubric

| Label | Definition | Annotator Action |
|---|---|---|
| **Supported** | At least one reputable Wikipedia passage explicitly confirms the claim, and none contradicts it. | Retain claim verbatim in repair. |
| **Contradicted** | A Wikipedia passage directly conflicts with the statement (e.g. wrong dates, incorrect parentage, inverted relations). | Replace with the verified fact from evidence. |
| **Unverifiable** | The evidence is silent, ambiguous, or lacks sufficient detail to confirm or refute. | Apply hedge (*"reportedly"*) or delete if extraneous. |

---

## 3. Ten Worked Annotation Examples

### Example 1: Basic Biography (Contradiction)
- **Question:** Where was Albert Einstein born?
- **Answer:** Albert Einstein was born in Berlin in 1879.
- **Claim 1:** `Albert Einstein was born in 1879.` $\rightarrow$ **Supported** (Citation: *Albert Einstein* - "Albert Einstein was born in Ulm, in the Kingdom of Württemberg, on 14 March 1879.")
- **Claim 2:** `Albert Einstein was born in Berlin.` $\rightarrow$ **Contradicted** (Citation: *Albert Einstein* - born in Ulm, not Berlin.)
- **Gold Repair:** `Albert Einstein was born in Ulm in 1879.`

### Example 2: Pronoun Resolution Across Sentences
- **Question:** Who founded Apple?
- **Answer:** Steve Jobs co-founded Apple in 1976. He was born in Seattle.
- **Claim 1:** `Steve Jobs co-founded Apple in 1976.` $\rightarrow$ **Supported**
- **Claim 2:** `Steve Jobs was born in Seattle.` $\rightarrow$ **Contradicted** (Born in San Francisco, California.)
- **Gold Repair:** `Steve Jobs co-founded Apple in 1976. He was born in San Francisco.`

### Example 3: Unverifiable / Silent Evidence
- **Question:** What was Marie Curie's favorite book as a child?
- **Answer:** Marie Curie loved reading War and Peace when she was ten years old.
- **Claim 1:** `Marie Curie loved reading War and Peace when she was ten years old.` $\rightarrow$ **Unverifiable** (Wikipedia mentions her reading voraciously, but does not cite *War and Peace* specifically.)
- **Gold Repair:** `Reportedly, Marie Curie loved reading books when she was young.`

### Example 4: Entity Disambiguation (Homonym)
- **Question:** Who directed The Godfather?
- **Answer:** Sofia Coppola directed The Godfather in 1972.
- **Claim 1:** `The Godfather was released in 1972.` $\rightarrow$ **Supported**
- **Claim 2:** `Sofia Coppola directed The Godfather.` $\rightarrow$ **Contradicted** (Directed by Francis Ford Coppola; Sofia is his daughter.)
- **Gold Repair:** `Francis Ford Coppola directed The Godfather in 1972.`

### Example 5: Numerical & Temporal Errors
- **Question:** How tall is Mount Everest?
- **Answer:** Mount Everest stands 9,848 metres above sea level.
- **Claim 1:** `Mount Everest stands 9,848 metres above sea level.` $\rightarrow$ **Contradicted** (Elevation is 8,848.86 metres.)
- **Gold Repair:** `Mount Everest stands 8,848 metres above sea level.`

### Example 6: Non-Check-Worthy Subjective Opinion
- **Question:** What is the Louvre Museum?
- **Answer:** The Louvre is in Paris. It is without doubt the only museum worth visiting in Europe.
- **Claim 1:** `The Louvre is in Paris.` $\rightarrow$ **Supported** (check_worthy=True)
- **Claim 2:** `The Louvre is the only museum worth visiting in Europe.` $\rightarrow$ **Unverifiable / Opinion** (check_worthy=False)
- **Gold Repair:** `The Louvre is in Paris.`

### Example 7: AI Meta-Boilerplate Removal
- **Question:** What is the speed of light?
- **Answer:** As an AI language model, I can confirm that the speed of light in vacuum is exactly 299,792,458 metres per second.
- **Claim 1:** `The speed of light in vacuum is exactly 299,792,458 metres per second.` $\rightarrow$ **Supported**
- **Gold Repair:** `The speed of light in vacuum is exactly 299,792,458 metres per second.`

### Example 8: Partial Truth / Conjunction Mismatch
- **Question:** When did Neil Armstrong land on the Moon and who was with him?
- **Answer:** Neil Armstrong landed on the Moon in 1969 with John Glenn.
- **Claim 1:** `Neil Armstrong landed on the Moon in 1969.` $\rightarrow$ **Supported**
- **Claim 2:** `John Glenn landed on the Moon with Neil Armstrong.` $\rightarrow$ **Contradicted** (Buzz Aldrin landed with him; John Glenn was on Friendship 7 orbit.)
- **Gold Repair:** `Neil Armstrong landed on the Moon in 1969 with Buzz Aldrin.`

### Example 9: Inverted Polarity / Negation
- **Question:** Did Canada ever win the FIFA World Cup?
- **Answer:** Canada won the FIFA World Cup in 1986.
- **Claim 1:** `Canada won the FIFA World Cup in 1986.` $\rightarrow$ **Contradicted** (Canada qualified in 1986 but was eliminated in the group stage without winning a match.)
- **Gold Repair:** `Canada participated in the 1986 FIFA World Cup but did not win.`

### Example 10: Multi-Sentence Minimal Edit
- **Question:** Describe the invention of the telephone.
- **Answer:** Alexander Graham Bell invented the telephone. He was granted a patent in 1876. He was born in London.
- **Claim 1:** `Alexander Graham Bell invented the telephone.` $\rightarrow$ **Supported**
- **Claim 2:** `Alexander Graham Bell was granted a patent for the telephone in 1876.` $\rightarrow$ **Supported**
- **Claim 3:** `Alexander Graham Bell was born in London.` $\rightarrow$ **Contradicted** (Born in Edinburgh, Scotland.)
- **Gold Repair:** `Alexander Graham Bell invented the telephone. He was granted a patent in 1876. He was born in Edinburgh.`
