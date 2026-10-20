"""
error_analysis.py — Diagnostic error categorizer for claim verification.

Categorizes verification errors and disagreements into 6 taxonomy classes:
  1. RETRIEVAL_MISS: Evidence passages have zero or low entity/keyword overlap with the claim.
  2. NUMERIC_MISMATCH: Numbers, dates, or measurements in the claim conflict with evidence.
  3. NEGATION_INVERSION: Polarity inversion (e.g. claim asserts absence, passage asserts presence).
  4. ENTITY_AMBIGUITY: Proper noun / entity identity confusion across different individuals.
  5. TEMPORAL_MISMATCH: Actions/events occurred at different historical times.
  6. CORPUS_GAP: Fact is verifiable in the real world but missing from the local corpus.

Public API:
    categorize_verification_error(claim, gold_label, predicted_label) -> Tuple[str, str]
    analyze_error_distribution(claims_with_gold) -> Dict[str, int]
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.schema import Claim, EvidencePassage


NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "none", "cannot", "failed", "denied", "without"}


def extract_numbers_and_years(text: str) -> Set[str]:
    return set(re.findall(r'\b\d+(?:\.\d+)?\b|\b\d{4}\b', text))


def categorize_verification_error(
    claim: Claim,
    gold_label: Optional[str] = None,
    predicted_label: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Categorize the primary root cause of a verification error.
    Returns: (error_category, explanation)
    """
    pred = predicted_label or claim.label or "Unverifiable"
    text = claim.text.lower()

    # 1. Retrieval Miss: Check if evidence is missing or has negligible content overlap
    if not claim.evidence:
        return "RETRIEVAL_MISS", "Zero evidence passages retrieved."

    evidence_text = " ".join(ep.text for ep in claim.evidence).lower()
    claim_tokens = set(re.findall(r'\b[a-z0-9_]{3,}\b', text))
    evidence_tokens = set(re.findall(r'\b[a-z0-9_]{3,}\b', evidence_text))

    overlap = claim_tokens & evidence_tokens
    overlap_ratio = len(overlap) / max(1, len(claim_tokens))

    if overlap_ratio < 0.25:
        return "RETRIEVAL_MISS", f"Evidence overlap only {overlap_ratio:.1%}; key entities missing."

    # 2. Negation Inversion
    claim_has_neg = any(w in text.split() for w in NEGATION_WORDS)
    evidence_has_neg = any(w in evidence_text.split() for w in NEGATION_WORDS)
    if claim_has_neg != evidence_has_neg and pred in ("Contradicted", "Unverifiable"):
        return "NEGATION_INVERSION", f"Polarity mismatch: claim_neg={claim_has_neg}, evidence_neg={evidence_has_neg}."

    # 3. Numeric Mismatch
    claim_nums = extract_numbers_and_years(claim.text)
    evidence_nums = extract_numbers_and_years(evidence_text)
    if claim_nums:
        missing_nums = claim_nums - evidence_nums
        if missing_nums:
            return "NUMERIC_MISMATCH", f"Numbers in claim {missing_nums} not found in evidence {evidence_nums}."

    # 4. Temporal Mismatch
    claim_years = set(re.findall(r'\b(?:18|19|20)\d{2}\b', claim.text))
    evidence_years = set(re.findall(r'\b(?:18|19|20)\d{2}\b', evidence_text))
    if claim_years and evidence_years and not (claim_years & evidence_years):
        return "TEMPORAL_MISMATCH", f"Claim year(s) {claim_years} conflict with evidence year(s) {evidence_years}."

    # 5. Entity Ambiguity (capitalized proper noun mismatch)
    claim_entities = set(re.findall(r'\b[A-Z][a-z0-9_]+\b', claim.text))
    evidence_titles = set(re.findall(r'\b[A-Z][a-z0-9_]+\b', " ".join(ep.title for ep in claim.evidence)))
    if claim_entities and not (claim_entities & evidence_titles):
        return "ENTITY_AMBIGUITY", f"Claim entity {claim_entities} differs from retrieved document titles."

    # 6. Default to Corpus Gap or Model Calibration
    if pred == "Unverifiable":
        return "CORPUS_GAP", "Passage mentions entity but does not state the specific predicate."

    return "MODEL_UNCERTAINTY", f"Model assigned {pred} against retrieved passages."


def analyze_error_distribution(claims: List[Claim]) -> Dict[str, int]:
    """Compute distribution of error categories across a collection of claims."""
    distribution: Dict[str, int] = {
        "RETRIEVAL_MISS": 0,
        "NUMERIC_MISMATCH": 0,
        "NEGATION_INVERSION": 0,
        "ENTITY_AMBIGUITY": 0,
        "TEMPORAL_MISMATCH": 0,
        "CORPUS_GAP": 0,
        "MODEL_UNCERTAINTY": 0,
    }
    for c in claims:
        category, _ = categorize_verification_error(c)
        distribution[category] = distribution.get(category, 0) + 1
    return distribution
