"""
repair.py — Stage 4: Repair flagged claims with evidence-grounded, minimal edits.

Phase 3 (skeleton): per-sentence LLM repair.
Phase 7 (full):     programmatic guards, policy variants, re-verification loop.

Public API:
    repair(record, model, provider, policy) -> AnswerRecord
"""
from __future__ import annotations

from typing import Dict, List

from src.schema import AnswerRecord, Claim
import src.llm as llm


# ─── Prompt ───────────────────────────────────────────────────────────────────

_REPAIR_SYSTEM = """You are an editor fixing factual errors in a single sentence.

RULES:
1. Fix ONLY the claims labelled "Contradicted" or "Unverifiable".
2. Use ONLY information from the provided evidence passages.
3. Keep every other word and fact IDENTICAL to the original.
4. Do NOT add new information that is not in the evidence.
5. If no correct replacement is available in the evidence:
   - For "rewrite+hedge": add "reportedly" or "according to some sources" before the uncertain claim.
   - For "rewrite+remove": remove the flagged claim from the sentence entirely.
   - For "rewrite-only": attempt a rewrite; if impossible, leave the sentence unchanged.
6. Return ONLY the corrected sentence — no explanation, no quotes, no extra text.
7. If the entire sentence should be removed, return exactly: REMOVE"""


def _build_repair_prompt(
    sentence: str,
    flagged_claims: List[Claim],
    policy: str,
) -> str:
    claims_str = "\n".join(
        f"  - [{c.label}] \"{c.text}\"  → reason: {c.label_rationale}"
        for c in flagged_claims
    )
    # Collect evidence across all flagged claims in this sentence (deduplicated)
    seen_doc_ids = set()
    evidence_list = []
    for c in flagged_claims:
        for e in c.evidence:
            if e.doc_id not in seen_doc_ids:
                evidence_list.append(e)
                seen_doc_ids.add(e.doc_id)

    evidence_str = "\n".join(
        f"  [{i + 1}] (Title: {e.title}) {e.text}"
        for i, e in enumerate(evidence_list[:6])  # cap at 6 passages
    ) or "  (none available)"

    return (
        f"{_REPAIR_SYSTEM}\n\n"
        f"Repair policy: {policy}\n\n"
        f"Original sentence:\n  {sentence}\n\n"
        f"Flagged claims in this sentence:\n{claims_str}\n\n"
        f"Evidence passages:\n{evidence_str}\n\n"
        f"Corrected sentence:"
    )


# ─── Grouping ─────────────────────────────────────────────────────────────────

def _group_flagged_by_sentence(record: AnswerRecord) -> Dict[int, List[Claim]]:
    """Group claims that need repair by their sentence_id."""
    groups: Dict[int, List[Claim]] = {}
    for claim in record.claims:
        if claim.check_worthy and claim.label in ("Contradicted", "Unverifiable"):
            groups.setdefault(claim.sentence_id, []).append(claim)
    return groups


# ─── Public API ───────────────────────────────────────────────────────────────

def repair(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    policy: str = "rewrite+hedge",
) -> AnswerRecord:
    """
    Repair flagged sentences in the answer.
    Mutates record.repaired_answer and record.repair_iterations.

    Args:
        record:  AnswerRecord with claims verified (claim.label populated).
        model:   LLM model name.
        provider: LLM provider.
        policy:  "rewrite+hedge" | "rewrite+remove" | "rewrite-only"
    """
    from src.decompose import split_sentences

    sentences = split_sentences(record.answer)
    flagged = _group_flagged_by_sentence(record)

    if not flagged:
        # Nothing to repair
        record.repaired_answer = record.answer
        record.repair_iterations = 0
        return record

    repaired_sentences: List[str] = []
    for sid, sentence in enumerate(sentences):
        if sid not in flagged:
            repaired_sentences.append(sentence)
            continue

        prompt = _build_repair_prompt(sentence, flagged[sid], policy)
        response = llm.call(
            prompt=prompt, model=model, provider=provider, temperature=0
        ).strip()

        if response.upper() == "REMOVE" or not response:
            # Sentence removed per policy
            continue
        repaired_sentences.append(response)

    record.repaired_answer = " ".join(repaired_sentences)
    record.repair_iterations = 1
    return record
