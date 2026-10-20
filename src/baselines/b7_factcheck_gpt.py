"""
b7_factcheck_gpt.py — Baseline B7: Factcheck-GPT Pipeline (Wang et al., 2023).

Standard published fact-checking benchmark pipeline:
1. Decomposes answer into claims.
2. Retrieves top-k evidence passages using BM25.
3. Classifies claim stance (Support / Refute / Not enough info).
4. Un-guarded direct revision of non-supported claims (no programmatic guards, no retry loop).
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from src.decompose import decompose
from src.llm import call as llm_call
from src.retrieve import retrieve_bm25
from src.schema import AnswerRecord, Claim


_STANCE_PROMPT = """You are a fact-checking judge following the Factcheck-GPT standard.

Claim: "{claim}"
Evidence Passages:
{passages}

Classify the evidence stance on the claim into one of:
- "Support" (evidence directly verifies claim)
- "Refute" (evidence contradicts claim)
- "Not enough info" (evidence is insufficient)

Respond with JSON: {{"stance": "Support" | "Refute" | "Not enough info", "reason": "..."}}"""

_REVISION_PROMPT = """You are a text revision assistant. Revise the following answer based on fact-checking findings.

Original Answer: {answer}

Fact-checking results:
{findings}

Instructions:
1. For claims marked Refute or Not enough info, edit or replace them with verified information.
2. Maintain fluency and coherence.
3. Output ONLY the revised answer text without extra preamble.

Revised Answer:"""


def classify_claim_stance(
    claim_text: str,
    passages: list,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> str:
    """Classify stance of passages toward claim."""
    if not passages:
        return "Not enough info"

    passages_text = "\n".join(
        f"[{i + 1}] {p.text}" for i, p in enumerate(passages[:5])
    )
    prompt = _STANCE_PROMPT.format(claim=claim_text, passages=passages_text)
    resp = llm_call(prompt=prompt, model=model, provider=provider, temperature=0.0)

    match = re.search(r"\{.*\}", resp, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            stance = data.get("stance", "Not enough info")
            if stance in ("Support", "Refute", "Not enough info"):
                return stance
        except Exception:
            pass
    return "Not enough info"


def run_b7(
    record: AnswerRecord,
    index_path: Optional[str] = None,
    top_k: int = 5,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> AnswerRecord:
    """
    Run Factcheck-GPT baseline on an AnswerRecord.
    """
    # 1. Claim decomposition
    if not record.claims:
        record = decompose(record, model=model, provider=provider)

    findings: List[str] = []
    # 2. Retrieve & stance classify
    for claim in record.check_worthy_claims:
        if index_path and not claim.evidence:
            try:
                claim.evidence = retrieve_bm25(claim.text, index_path, top_k=top_k)
            except Exception:
                claim.evidence = []

        stance = classify_claim_stance(claim.text, claim.evidence, model=model, provider=provider)
        # Map to canonical schema labels
        if stance == "Support":
            claim.label = "Supported"
        elif stance == "Refute":
            claim.label = "Contradicted"
        else:
            claim.label = "Unverifiable"

        findings.append(f"- Claim: \"{claim.text}\" -> Stance: {stance}")

    # 3. Direct un-guarded revision
    findings_text = "\n".join(findings) if findings else "(No specific claim findings)"
    prompt = _REVISION_PROMPT.format(answer=record.answer, findings=findings_text)
    revised = llm_call(prompt=prompt, model=model, provider=provider, temperature=0.0).strip()

    record.repaired_answer = revised
    record.repair_iterations = 1
    return record

