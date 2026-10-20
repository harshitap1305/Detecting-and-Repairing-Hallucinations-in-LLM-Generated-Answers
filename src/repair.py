"""
repair.py — Stage 4: Localized, evidence-grounded repair with programmatic safety guards.

Phase 7 Implementation:
  - Sentence-level localization (isolating edits to flagged sentences only)
  - Programmatic Safety Guards:
      * Guard 1 (Preservation Guard): Prevents deleting or corrupting previously Supported claims
      * Guard 2 (Grounding Guard): Rejects edits introducing ungrounded entities or numbers
  - Error-corrective retry prompt on guard rejection
  - Policy Dispatch:
      * "rewrite+hedge": Rewrite with evidence; fallback to epistemic hedge ("reportedly")
      * "rewrite+remove": Rewrite with evidence; fallback to deletion (REMOVE)
      * "rewrite-only": Rewrite with evidence; fallback to original sentence
  - Byte-identical preservation for untouched sentences

Public API:
    repair(record, model, provider, policy) -> AnswerRecord
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from src.schema import AnswerRecord, Claim, EvidencePassage
import src.llm as llm
from src.guards import validate_repair_guards
from src.decompose import split_sentences


# ─── Prompts ───────────────────────────────────────────────────────────────────

_REPAIR_SYSTEM = """You are an expert factual editor. Your objective is to perform MINIMAL, SURGICAL edits to fix factual errors in a sentence using verified evidence.

CORE EDITING RULES:
1. Fix ONLY the claims marked Contradicted or Unverifiable.
2. Use ONLY facts explicitly provided in the evidence passages.
3. Keep every other word, phrase, and supported fact EXACTLY IDENTICAL to the original.
4. Never introduce new names, dates, numbers, or places not found in the evidence.
5. If the evidence provides no replacement fact:
   - For "rewrite+hedge": Prepend an uncertainty hedge like "Reportedly," or "According to some sources,".
   - For "rewrite+remove": Omit the unverified claim or return exactly: REMOVE.
   - For "rewrite-only": Keep the original sentence.
6. Return ONLY the edited sentence without explanations, quotes, or markdown."""


def _build_repair_prompt(
    sentence: str,
    flagged_claims: List[Claim],
    supported_claims: List[Claim],
    evidence_passages: List[EvidencePassage],
    policy: str,
    rejection_feedback: Optional[str] = None,
) -> str:
    flagged_str = "\n".join(
        f"  - [{c.label}] \"{c.text}\" (Reason: {c.label_rationale})"
        for c in flagged_claims
    )
    supported_str = "\n".join(
        f"  - [MUST PRESERVE] \"{c.text}\""
        for c in supported_claims
    ) if supported_claims else "  (none in this sentence)"

    evidence_str = "\n".join(
        f"  [{i + 1}] Title: {e.title}\n      {e.text}"
        for i, e in enumerate(evidence_passages[:6])
    ) if evidence_passages else "  (no evidence available)"

    prompt = (
        f"{_REPAIR_SYSTEM}\n\n"
        f"Repair Policy: {policy}\n\n"
        f"Original sentence:\n  {sentence}\n\n"
        f"Supported claims that MUST NOT be altered:\n{supported_str}\n\n"
        f"Flagged claims that MUST be corrected:\n{flagged_str}\n\n"
        f"Verified evidence passages:\n{evidence_str}\n\n"
    )

    if rejection_feedback:
        prompt += (
            f"PREVIOUS ATTEMPT FAILED SAFETY CHECK:\n"
            f"  {rejection_feedback}\n"
            f"Please revise your edit to strictly follow all grounding and preservation rules.\n\n"
        )

    prompt += "Corrected sentence:"
    return prompt


# ─── Grouping and Localization ────────────────────────────────────────────────

def _group_claims_by_sentence(record: AnswerRecord) -> Tuple[Dict[int, List[Claim]], Dict[int, List[Claim]]]:
    """Group claims into flagged (need repair) and supported (must preserve) by sentence_id."""
    flagged: Dict[int, List[Claim]] = {}
    supported: Dict[int, List[Claim]] = {}

    for claim in record.claims:
        if not claim.check_worthy:
            continue
        sid = claim.sentence_id
        if claim.label in ("Contradicted", "Unverifiable"):
            flagged.setdefault(sid, []).append(claim)
        elif claim.label == "Supported":
            supported.setdefault(sid, []).append(claim)

    return flagged, supported


def _collect_sentence_evidence(claims: List[Claim]) -> List[EvidencePassage]:
    """Deduplicate evidence passages across all claims in a sentence."""
    seen_doc_ids: Set[str] = set()
    collected: List[EvidencePassage] = []
    for c in claims:
        for ep in c.evidence:
            if ep.doc_id not in seen_doc_ids:
                collected.append(ep)
                seen_doc_ids.add(ep.doc_id)
    return collected


def _apply_fallback_policy(sentence: str, policy: str) -> str:
    """Apply configured fallback when repair cannot be grounded or verified."""
    if policy == "rewrite+remove":
        return ""  # Sentence deleted
    if policy == "rewrite+hedge":
        # Prepend epistemic hedge if not already hedged
        if not re.match(r'^(reportedly|according to|it is claimed|allegedly)', sentence.lower()):
            return f"Reportedly, {sentence[0].lower()}{sentence[1:]}"
        return sentence
    # "rewrite-only": keep original
    return sentence


# ─── Public API ───────────────────────────────────────────────────────────────

def repair(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    policy: str = "rewrite+hedge",
    max_retries: int = 1,
) -> AnswerRecord:
    """
    Repair flagged sentences in AnswerRecord using evidence-grounded minimal edits with safety guards.

    Args:
        record:       AnswerRecord with verified claims (claim.label populated).
        model:        LLM model for repair generation.
        provider:     LLM provider ("groq" | "openai" | "local").
        policy:       "rewrite+hedge" | "rewrite+remove" | "rewrite-only".
        max_retries:  Maximum retries per sentence on guard rejection.
    """
    sentences = split_sentences(record.answer)
    flagged_by_sid, supported_by_sid = _group_claims_by_sentence(record)

    if not flagged_by_sid:
        # No errors detected: answer is fully preserved
        record.repaired_answer = record.answer
        record.repair_iterations = 0
        return record

    repaired_sentences: List[str] = []

    for sid, sentence in enumerate(sentences):
        # 1. Untouched sentence: guarantee byte-identical preservation
        if sid not in flagged_by_sid:
            repaired_sentences.append(sentence)
            continue

        flagged_here = flagged_by_sid[sid]
        supported_here = supported_by_sid.get(sid, [])
        evidence_here = _collect_sentence_evidence(flagged_here)

        accepted_edit = None
        rejection_reason = None

        # 2. Attempt repair with guard validation loop
        for attempt in range(max_retries + 1):
            prompt = _build_repair_prompt(
                sentence=sentence,
                flagged_claims=flagged_here,
                supported_claims=supported_here,
                evidence_passages=evidence_here,
                policy=policy,
                rejection_feedback=rejection_reason,
            )
            # Use temperature 0 on first attempt, slightly higher on retry
            temp = 0.0 if attempt == 0 else 0.2
            candidate = llm.call(
                prompt=prompt, model=model, provider=provider, temperature=temp
            ).strip()

            # Handle explicit deletion signal
            if candidate.upper() == "REMOVE" or not candidate:
                accepted_edit = ""
                break

            # Strip accidental quotes
            if (candidate.startswith('"') and candidate.endswith('"')) or (candidate.startswith("'") and candidate.endswith("'")):
                candidate = candidate[1:-1].strip()

            # 3. Evaluate Programmatic Safety Guards
            guards_ok, reason = validate_repair_guards(
                original_sentence=sentence,
                candidate_edited_sentence=candidate,
                supported_claims=supported_here,
                evidence_passages=evidence_here,
            )

            if guards_ok:
                accepted_edit = candidate
                break
            else:
                rejection_reason = reason
                if attempt == max_retries:
                    # All retries exhausted: apply fallback policy
                    accepted_edit = _apply_fallback_policy(sentence, policy)

        if accepted_edit:
            repaired_sentences.append(accepted_edit)

    record.repaired_answer = " ".join(s for s in repaired_sentences if s.strip())
    record.repair_iterations = max(1, record.repair_iterations)
    return record
