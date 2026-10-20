"""
verify.py — Stage 3: Verify each claim against retrieved evidence.

Phase 3 (skeleton): LLM-judge verifier (simplest; works without NLI model).
Phase 6 (full):     NLI verifier, fine-tuned encoder, threshold calibration, ensembling.

Public API:
    verify(record, model, provider) -> AnswerRecord   (mutates claim.label fields)
"""
from __future__ import annotations

import json
import re
from typing import List

from src.schema import AnswerRecord, Claim, VerdictLabel
import src.llm as llm


# ─── LLM-judge verifier ───────────────────────────────────────────────────────

_VERIFY_SYSTEM = """You are a precise fact-checker. Given a claim and evidence passages, decide if the claim is supported.

LABELS:
- "Supported": at least one passage clearly supports the claim AND none contradicts it.
- "Contradicted": a passage directly conflicts with the claim.
- "Unverifiable": passages lack enough information to judge (this includes when evidence is empty).

RULES:
- Base your decision ONLY on the provided passages, not on your own knowledge.
- Output ONLY valid JSON — no markdown, no extra text.

OUTPUT FORMAT:
{"label": "Supported" | "Contradicted" | "Unverifiable", "rationale": "<one sentence>", "supporting_passage_ids": [<int>, ...]}"""


def _build_verify_prompt(claim_text: str, evidence: list) -> str:
    if not evidence:
        passages_str = "(no evidence retrieved)"
    else:
        passages_str = "\n".join(
            f"[{i + 1}] Title: {e.title}\n    {e.text}"
            for i, e in enumerate(evidence)
        )
    return (
        f"{_VERIFY_SYSTEM}\n\n"
        f"Claim: {claim_text}\n\n"
        f"Evidence passages:\n{passages_str}\n\n"
        f"JSON:"
    )


def _parse_verdict(response: str) -> tuple[VerdictLabel, str, list]:
    """Parse LLM response into (label, rationale, supporting_ids)."""
    response = re.sub(r"```(?:json)?", "", response).strip()
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return "Unverifiable", "Could not parse LLM response.", []
    try:
        d = json.loads(match.group())
        label = d.get("label", "Unverifiable")
        if label not in ("Supported", "Contradicted", "Unverifiable"):
            label = "Unverifiable"
        rationale = str(d.get("rationale", ""))
        supporting_ids = d.get("supporting_passage_ids", [])
        return label, rationale, supporting_ids
    except json.JSONDecodeError:
        return "Unverifiable", "JSON decode error.", []


def verify_claim_llm(
    claim: Claim,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> Claim:
    """Verify a single claim using an LLM judge. Mutates claim.label in place."""
    if not claim.check_worthy:
        claim.label = "Unverifiable"
        claim.label_rationale = "Marked as not check-worthy."
        return claim

    if not claim.evidence:
        claim.label = "Unverifiable"
        claim.label_rationale = "No evidence retrieved."
        return claim

    prompt = _build_verify_prompt(claim.text, claim.evidence)
    response = llm.call(prompt=prompt, model=model, provider=provider, temperature=0)
    label, rationale, _ = _parse_verdict(response)
    claim.label = label
    claim.label_rationale = rationale
    return claim


# ─── NLI verifier (Phase 6) ───────────────────────────────────────────────────

_nli_model = None

def _get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small"):
    global _nli_model
    if _nli_model is None:
        from transformers import pipeline
        _nli_model = pipeline("text-classification", model=model_name,
                               top_k=None, device=-1)
    return _nli_model


def verify_claim_nli(
    claim: Claim,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    threshold_supported: float = 0.7,
    threshold_contradicted: float = 0.5,
) -> Claim:
    """
    NLI-based verification. Phase 6 implementation.
    Premise = each evidence passage; hypothesis = claim text.
    Aggregates scores over all passages.
    """
    if not claim.check_worthy or not claim.evidence:
        claim.label = "Unverifiable"
        claim.label_rationale = "No evidence or not check-worthy."
        return claim

    nli = _get_nli_model(model_name)
    max_entail = 0.0
    max_contra = 0.0

    for ep in claim.evidence:
        # Truncate to avoid exceeding model max length
        premise = ep.text[:512]
        hypothesis = claim.text[:256]
        preds = nli(f"{premise} [SEP] {hypothesis}")
        # preds is a list of dicts: [{"label": "entailment", "score": ...}, ...]
        for pred in (preds[0] if isinstance(preds[0], list) else preds):
            lbl = pred["label"].lower()
            sc = pred["score"]
            if "entail" in lbl:
                max_entail = max(max_entail, sc)
            elif "contra" in lbl:
                max_contra = max(max_contra, sc)

    if max_entail >= threshold_supported and max_contra < threshold_contradicted:
        claim.label = "Supported"
        claim.label_rationale = f"Max entailment score: {max_entail:.3f}"
    elif max_contra >= threshold_contradicted:
        claim.label = "Contradicted"
        claim.label_rationale = f"Max contradiction score: {max_contra:.3f}"
    else:
        claim.label = "Unverifiable"
        claim.label_rationale = (
            f"Entail={max_entail:.3f}, Contra={max_contra:.3f} — below thresholds."
        )
    return claim


# ─── Public API ───────────────────────────────────────────────────────────────

def verify(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    verifier: str = "llm",         # "llm" | "nli" (Phase 6)
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    threshold_supported: float = 0.7,
    threshold_contradicted: float = 0.5,
) -> AnswerRecord:
    """
    Verify all check-worthy claims in a record.
    Mutates claim.label fields and record.reliability_score_before.

    Args:
        record:   AnswerRecord with claims and evidence already populated.
        model:    LLM model name (used when verifier="llm").
        provider: LLM provider (used when verifier="llm").
        verifier: "llm" | "nli"
    """
    supported_count = 0
    check_worthy_count = 0

    for claim in record.claims:
        if not claim.check_worthy:
            continue
        check_worthy_count += 1

        if verifier == "llm":
            verify_claim_llm(claim, model=model, provider=provider)
        elif verifier == "nli":
            verify_claim_nli(
                claim, model_name=nli_model_name,
                threshold_supported=threshold_supported,
                threshold_contradicted=threshold_contradicted,
            )
        else:
            raise ValueError(f"Unknown verifier: {verifier!r}")

        if claim.label == "Supported":
            supported_count += 1

    if check_worthy_count > 0:
        record.reliability_score_before = supported_count / check_worthy_count
    else:
        record.reliability_score_before = 1.0   # no check-worthy claims → nothing wrong

    return record
