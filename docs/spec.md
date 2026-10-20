# Project Specification — Hallucination Detection and Repair

## Task
English, open-domain factual QA and short biography-style answers.  
Evidence source: Wikipedia.  
- **Tier A (dev):** FEVER wiki-pages (2017 Wikipedia intros, gold evidence labels available)  
- **Tier B (main):** Scoped subset of `wikimedia/wikipedia` 20231101.en

## Definitions

### Atomic claim
One independently checkable factual statement, **decontextualized** — no unresolved pronouns; includes the entity name and any needed time/place qualifiers.

**Example:**  
- ❌ "He was born in 1950 in Chicago." (pronoun unresolved)  
- ✅ "Albert Einstein was born in 1950 in Chicago." (decontextualized)

### Check-worthy
Factual, verifiable statements only.  
**Skipped (check_worthy=false):** opinions, questions, instructions, hedges ("I think", "I believe"), meta-text ("As an AI language model…").

### Verdict labels
| Label | Definition |
|---|---|
| `Supported` | At least one retrieved passage entails the claim and none contradicts it |
| `Contradicted` | A retrieved passage directly conflicts with the claim |
| `Unverifiable` | Evidence is insufficient — may mean retrieval failed; track as "retrieval-miss" separately |

### Correct repair
The edited text must:
1. Be supported by the cited evidence
2. Keep the original meaning wherever the original was correct
3. Not add information not present in the evidence
4. Remain fluent and relevant to the question

### Repair policy (default: `rewrite+hedge`)
| Verdict | Action |
|---|---|
| `Supported` | Keep as-is |
| `Contradicted` + replacement evidence exists | Rewrite using evidence |
| `Contradicted` + no replacement evidence | Remove or hedge with "reportedly" |
| `Unverifiable` | Hedge with "reportedly" or "according to some sources" |

## Label mapping from source datasets
| Source | Mapping |
|---|---|
| FEVER | Supported→Supported, Refuted→Contradicted, NotEnoughInfo→Unverifiable |
| WikiBio | accurate→Supported; minor/major_inaccurate→Contradicted |
| FELM | segment correct→Supported; incorrect→Contradicted |
| HaluEval | answer-level hallucinated yes/no → answer-level flag |
| Factcheck-Bench | support→Supported; refute→Contradicted; irrelevant/partial→Unverifiable |

## Research Questions
- **RQ1:** How accurately can an evidence-grounded, claim-level pipeline detect hallucinated claims vs. baselines?
- **RQ2:** Can flagged claims be repaired automatically using retrieved evidence, and how often is the repair correct?
- **RQ3:** Does claim-level editing preserve correct content better than rewriting the whole answer?
- **RQ4:** How do retrieval quality and verifier choice affect end-to-end performance?

## Success criteria
- Pipeline runs end-to-end on ≥200 test answers
- Outperforms ≥2 baselines on claim-level detection F1
- Repair increases supported-claim fraction vs. unrepaired answers
- Preservation of correct claims measured and reported
- Documented error analysis
