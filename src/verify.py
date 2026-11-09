"""
verify.py — Stage 3: Claim verification with NLI, LLM Judge, and Calibrated Ensembling.

Phase 6 Implementation:
  - Verifier 1: NLI Cross-Encoder (DeBERTa-v3) with calibrated decision boundaries (tau_s, tau_c)
  - Verifier 2: Structured zero-shot LLM Judge (Groq LLaMA 3.3 70B / Gemma 2)
  - Verifier 3: Calibrated Ensemble combining NLI distribution & LLM Judge citations
  - Automated threshold configuration loader (configs/calibrated_thresholds.yaml)
  - Answer reliability scoring

Public API:
    aggregate_verdict(max_entail, max_contra, tau_s, tau_c) -> Tuple[VerdictLabel, str]
    verify_claim_nli(claim, model_name, tau_s, tau_c) -> Claim
    verify_claim_llm(claim, model, provider) -> Claim
    verify_claim_ensemble(claim, model, provider, ...) -> Claim
    verify(record, model, provider, verifier, ...) -> AnswerRecord
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.schema import AnswerRecord, Claim, VerdictLabel
import src.llm as llm
from src.experiments.runtime import strict


# ─── Calibrated Thresholds Loader ─────────────────────────────────────────────

DEFAULT_TAU_S = 0.70  # Entailment threshold
DEFAULT_TAU_C = 0.50  # Contradiction threshold


def load_calibrated_thresholds(config_path: str = "configs/calibrated_thresholds.yaml") -> Tuple[float, float]:
    """Load calibrated thresholds tau_s, tau_c from YAML if available."""
    p = Path(config_path)
    if p.exists():
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                tau_s = float(data.get("tau_s", DEFAULT_TAU_S))
                tau_c = float(data.get("tau_c", DEFAULT_TAU_C))
                return tau_s, tau_c
        except Exception:
            pass
    return DEFAULT_TAU_S, DEFAULT_TAU_C


def aggregate_verdict(
    max_entail: float,
    max_contra: float,
    tau_s: float = DEFAULT_TAU_S,
    tau_c: float = DEFAULT_TAU_C,
) -> Tuple[VerdictLabel, str]:
    """
    Calibrated decision rule over NLI probabilities:
      - Supported: max_entail >= tau_s and max_contra < tau_c
      - Contradicted: max_contra >= tau_c
      - Unverifiable: otherwise
    """
    if max_contra >= tau_c:
        return "Contradicted", f"Contradiction score {max_contra:.3f} >= threshold {tau_c:.2f}"
    if max_entail >= tau_s:
        return "Supported", f"Entailment score {max_entail:.3f} >= threshold {tau_s:.2f}"
    return "Unverifiable", f"Entail={max_entail:.3f} < {tau_s:.2f}, Contra={max_contra:.3f} < {tau_c:.2f}"


# ─── Verifier 1: NLI Cross-Encoder ────────────────────────────────────────────

_nli_model = None
_nli_model_name = None
_mock_nli_scorer = None  # Hook for unit testing without downloading 400MB models


def set_mock_nli_scorer(scorer_fn):
    """Set a mock NLI scoring function for testing: fn(premise, hypothesis) -> (entail, contra)."""
    global _mock_nli_scorer
    _mock_nli_scorer = scorer_fn


def _get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small"):
    global _nli_model, _nli_model_name
    if _nli_model is None or _nli_model_name != model_name:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        _nli_model = (tokenizer, model)
        _nli_model_name = model_name
    return _nli_model


def verify_claim_nli(
    claim: Claim,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    tau_s: float = DEFAULT_TAU_S,
    tau_c: float = DEFAULT_TAU_C,
) -> Claim:
    """Verify an atomic claim using DeBERTa-v3 NLI premise-hypothesis pairs."""
    if not claim.check_worthy:
        claim.label = "Unverifiable"
        claim.label_rationale = "Claim marked as non-check-worthy."
        return claim

    if not claim.evidence:
        claim.label = "Unverifiable"
        claim.label_rationale = "No evidence passages retrieved."
        return claim

    max_entail = 0.0
    max_contra = 0.0

    if _mock_nli_scorer is not None:
        for ep in claim.evidence:
            ent, con = _mock_nli_scorer(ep.text, claim.text)
            max_entail = max(max_entail, ent)
            max_contra = max(max_contra, con)
    else:
        import torch

        tokenizer, nli = _get_nli_model(model_name)
        label_names = {int(i): label.lower() for i, label in nli.config.id2label.items()}
        if not any("entail" in name for name in label_names.values()) or not any(
            "contra" in name for name in label_names.values()
        ):
            raise ValueError(f"NLI model {model_name!r} lacks entailment/contradiction labels")

        for ep in claim.evidence:
            # Encode a pair, matching the NLI model's training input format.
            features = tokenizer(
                ep.text, claim.text, truncation=True, max_length=384,
                return_tensors="pt",
            )
            device = getattr(nli, "device", None)
            if device is not None:
                features = {name: tensor.to(device) for name, tensor in features.items()}
            with torch.no_grad():
                probabilities = torch.softmax(nli(**features).logits[0], dim=-1)
            for index, probability in enumerate(probabilities.tolist()):
                name = label_names[index]
                if "entail" in name:
                    max_entail = max(max_entail, probability)
                elif "contra" in name:
                    max_contra = max(max_contra, probability)

    label, rationale = aggregate_verdict(max_entail, max_contra, tau_s=tau_s, tau_c=tau_c)
    claim.label = label
    claim.label_rationale = rationale
    return claim


# ─── Verifier 2: LLM Judge ────────────────────────────────────────────────────

_VERIFY_SYSTEM = """You are an objective fact-checking judge. Determine if the claim is Supported, Contradicted, or Unverifiable based solely on the provided evidence.

VERDICT RULES:
- "Supported": At least one passage directly confirms the claim and none contradicts it.
- "Contradicted": A passage directly conflicts with the claim (e.g. conflicting dates, entities, actions).
- "Unverifiable": Passages do not have sufficient information to confirm or refute.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{"label": "Supported" | "Contradicted" | "Unverifiable", "rationale": "<one concise sentence>", "supporting_passage_ids": [<int>]}"""


def _build_verify_prompt(claim_text: str, evidence: list) -> str:
    passages_str = "\n".join(
        f"[{i + 1}] Title: {e.title}\n    {e.text}"
        for i, e in enumerate(evidence)
    ) if evidence else "(no evidence available)"

    return (
        f"{_VERIFY_SYSTEM}\n\n"
        f"Claim: {claim_text}\n\n"
        f"Evidence passages:\n{passages_str}\n\n"
        f"JSON output:"
    )


def _parse_verdict(response: str) -> Tuple[VerdictLabel, str, List[int]]:
    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        if strict.get():
            raise ValueError("Malformed verifier JSON")
        return "Unverifiable", "Could not parse JSON response from LLM judge.", []
    try:
        d = json.loads(match.group())
        if strict.get() and (not isinstance(d, dict) or d.get("label") not in
                             ("Supported", "Contradicted", "Unverifiable")):
            raise ValueError("Invalid verifier label")
        lbl = d.get("label", "Unverifiable")
        if lbl not in ("Supported", "Contradicted", "Unverifiable"):
            lbl = "Unverifiable"
        rationale = str(d.get("rationale", ""))
        supporting_ids = d.get("supporting_passage_ids", [])
        return lbl, rationale, supporting_ids
    except Exception:
        if strict.get():
            raise
        return "Unverifiable", "Malformed JSON from judge.", []


def verify_claim_llm(
    claim: Claim,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> Claim:
    """Verify an atomic claim using LLM Judge."""
    if not claim.check_worthy:
        claim.label = "Unverifiable"
        claim.label_rationale = "Non-check-worthy statement."
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


# ─── Verifier 3: Calibrated Ensemble ──────────────────────────────────────────

def verify_claim_ensemble(
    claim: Claim,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    tau_s: float = DEFAULT_TAU_S,
    tau_c: float = DEFAULT_TAU_C,
) -> Claim:
    """
    Calibrated Ensemble:
      1. Computes NLI distribution.
      2. Computes LLM Judge verdict.
      3. If both agree -> final verdict with high confidence.
      4. If Contradicted by either -> flags as Contradicted (safety priority).
      5. If NLI is neutral/borderline -> defers to LLM Judge rationale.
    """
    if not claim.check_worthy or not claim.evidence:
        claim.label = "Unverifiable"
        claim.label_rationale = "No evidence or non-check-worthy."
        return claim

    # Run NLI
    nli_claim = Claim(id=claim.id, text=claim.text, sentence_id=claim.sentence_id, evidence=claim.evidence)
    verify_claim_nli(nli_claim, model_name=nli_model_name, tau_s=tau_s, tau_c=tau_c)

    # Run LLM
    llm_claim = Claim(id=claim.id, text=claim.text, sentence_id=claim.sentence_id, evidence=claim.evidence)
    verify_claim_llm(llm_claim, model=model, provider=provider)

    # Fusion logic
    if nli_claim.label == llm_claim.label:
        claim.label = nli_claim.label
        claim.label_rationale = f"Ensemble consensus: {llm_claim.label_rationale}"
    elif nli_claim.label == "Contradicted" or llm_claim.label == "Contradicted":
        claim.label = "Contradicted"
        claim.label_rationale = f"Flagged by contradiction guard: NLI={nli_claim.label}, LLM={llm_claim.label}"
    elif llm_claim.label == "Supported":
        claim.label = "Supported"
        claim.label_rationale = f"LLM verified: {llm_claim.label_rationale}"
    else:
        claim.label = "Unverifiable"
        claim.label_rationale = f"Ensemble uncertainty: NLI={nli_claim.label}, LLM={llm_claim.label}"

    return claim


# ─── Fine-tuned verifier (Path A) ────────────────────────────────────────────

def verify_claim_trained(claim: Claim, checkpoint: str) -> Claim:
    """Verify a claim with a locally saved three-class classifier checkpoint."""
    if not claim.check_worthy or not claim.evidence:
        claim.label = "Unverifiable"
        claim.label_rationale = "No evidence or non-check-worthy claim."
        return claim

    from src.training.verifier_model import predict_verdict

    label, probabilities = predict_verdict(
        [f"Title: {passage.title}\n{passage.text}" for passage in claim.evidence],
        claim.text, checkpoint,
    )
    claim.label = label
    claim.label_rationale = (
        "Fine-tuned verifier: "
        f"P(Supported)={probabilities[0]:.3f}, "
        f"P(Contradicted)={probabilities[1]:.3f}, "
        f"P(Unverifiable)={probabilities[2]:.3f}"
    )
    return claim


# ─── Public API ───────────────────────────────────────────────────────────────

def verify(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    verifier: str = "llm",          # "llm" | "nli" | "ensemble" | "trained"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    trained_model_path: Optional[str] = None,
    tau_s: Optional[float] = None,
    tau_c: Optional[float] = None,
) -> AnswerRecord:
    """
    Verify all check-worthy claims in an AnswerRecord and compute reliability score.
    """
    if tau_s is None or tau_c is None:
        loaded_s, loaded_c = load_calibrated_thresholds()
        tau_s = tau_s if tau_s is not None else loaded_s
        tau_c = tau_c if tau_c is not None else loaded_c

    supported_count = 0
    check_worthy_count = 0

    for claim in record.claims:
        if not claim.check_worthy:
            continue
        check_worthy_count += 1

        if verifier == "llm":
            verify_claim_llm(claim, model=model, provider=provider)
        elif verifier == "nli":
            verify_claim_nli(claim, model_name=nli_model_name, tau_s=tau_s, tau_c=tau_c)
        elif verifier == "ensemble":
            verify_claim_ensemble(
                claim, model=model, provider=provider,
                nli_model_name=nli_model_name, tau_s=tau_s, tau_c=tau_c
            )
        elif verifier == "trained":
            if not trained_model_path:
                raise ValueError("trained_model_path is required for verifier='trained'")
            verify_claim_trained(claim, checkpoint=trained_model_path)
        else:
            raise ValueError(f"Unknown verifier type: {verifier!r}")

        if claim.label == "Supported":
            supported_count += 1

    if check_worthy_count > 0:
        record.reliability_score_before = supported_count / check_worthy_count
    else:
        record.reliability_score_before = 1.0

    return record
