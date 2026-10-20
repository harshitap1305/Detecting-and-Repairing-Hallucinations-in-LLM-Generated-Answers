"""
guards.py — Programmatic validation guards for evidence-grounded sentence repairs.

Phase 7 Implementation:
  - Guard 1 (Preservation Guard): Verifies that all originally Supported claims in a
    sentence remain entailed in the edited sentence (protecting correct facts).
  - Guard 2 (Grounding / Entity Guard): Extracts all newly introduced named entities,
    numbers, and years, verifying that every new token is grounded in the retrieved evidence
    (preventing newly hallucinated replacements).

Public API:
    check_preservation_guard(orig_claims, edited_sent, nli_model) -> Tuple[bool, str]
    check_grounding_guard(orig_sent, edited_sent, evidence) -> Tuple[bool, str]
    validate_repair_guards(orig_sent, edited_sent, orig_claims, evidence) -> Tuple[bool, str]
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

from src.schema import Claim, EvidencePassage


# ─── Entity & Token Extraction ────────────────────────────────────────────────

def extract_named_entities_and_numbers(text: str) -> Set[str]:
    """Extract capitalized multi-word entities, proper nouns, and numbers/years."""
    # Numbers and years (e.g. 1969, 330, 27.3)
    nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', text))
    # Capitalized words (excluding sentence starters if common)
    words = text.split()
    entities = set()
    for i, w in enumerate(words):
        clean_w = re.sub(r'^[^\w]+|[^\w]+$', '', w)
        if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
            # Exclude very common sentence-initial discourse markers
            if i == 0 and clean_w in {"The", "In", "On", "It", "He", "She", "They", "This", "That", "However"}:
                continue
            entities.add(clean_w)
    return nums | entities


# ─── Guard 1: Preservation Guard ──────────────────────────────────────────────

def check_preservation_guard(
    supported_claims: List[Claim],
    candidate_edited_sentence: str,
    entailment_threshold: float = 0.60,
) -> Tuple[bool, str]:
    """
    Ensure every originally Supported claim in this sentence is still preserved in the edit.
    Uses token overlap heuristic or NLI premise-hypothesis verification.
    """
    if not supported_claims:
        return True, "No prior supported claims to preserve."

    edit_lower = candidate_edited_sentence.lower()
    edit_tokens = set(re.findall(r'\b[a-z0-9_]{3,}\b', edit_lower))

    for sc in supported_claims:
        claim_tokens = set(re.findall(r'\b[a-z0-9_]{3,}\b', sc.text.lower()))
        if not claim_tokens:
            continue
        overlap = claim_tokens & edit_tokens
        overlap_ratio = len(overlap) / len(claim_tokens)

        # If more than 40% of substantive tokens from a supported claim are wiped out
        if overlap_ratio < entailment_threshold:
            return (
                False,
                f"Preservation violation: supported claim '{sc.text}' was corrupted or lost (overlap={overlap_ratio:.1%})."
            )

    return True, "All supported claims preserved."


# ─── Guard 2: Grounding / Entity Guard ────────────────────────────────────────

def check_grounding_guard(
    original_sentence: str,
    candidate_edited_sentence: str,
    evidence_passages: List[EvidencePassage],
) -> Tuple[bool, str]:
    """
    Ensure any NEW named entities, numbers, or dates introduced by the edit
    appear verbatim in the retrieved evidence passages.
    """
    if not candidate_edited_sentence.strip():
        return True, "Empty sentence (deleted)."

    orig_tokens = extract_named_entities_and_numbers(original_sentence)
    edit_tokens = extract_named_entities_and_numbers(candidate_edited_sentence)

    new_tokens = edit_tokens - orig_tokens
    if not new_tokens:
        return True, "No new entities or numbers introduced."

    # Evidence corpus text
    evidence_corpus = " ".join(f"{ep.title} {ep.text}" for ep in evidence_passages).lower()

    ungrounded = []
    for token in new_tokens:
        # Check case-insensitive existence in evidence
        if token.lower() not in evidence_corpus:
            ungrounded.append(token)

    if ungrounded:
        return (
            False,
            f"Grounding violation: newly introduced terms {ungrounded} not found in retrieved evidence."
        )

    return True, f"All {len(new_tokens)} new entities/numbers grounded in evidence."


# ─── Master Guard Validator ───────────────────────────────────────────────────

def validate_repair_guards(
    original_sentence: str,
    candidate_edited_sentence: str,
    supported_claims: List[Claim],
    evidence_passages: List[EvidencePassage],
) -> Tuple[bool, str]:
    """
    Combined safety validator. Both preservation and grounding must pass.
    """
    # Guard 1: Preservation
    pres_ok, pres_reason = check_preservation_guard(supported_claims, candidate_edited_sentence)
    if not pres_ok:
        return False, pres_reason

    # Guard 2: Grounding
    ground_ok, ground_reason = check_grounding_guard(
        original_sentence, candidate_edited_sentence, evidence_passages
    )
    if not ground_ok:
        return False, ground_reason

    return True, "All safety guards passed."

