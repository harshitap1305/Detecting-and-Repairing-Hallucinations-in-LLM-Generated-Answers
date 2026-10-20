"""
decompose.py — Stage 1: Break an answer into atomic, decontextualized claims.

Phase 3 (skeleton): few-shot LLM prompt, JSON output, basic retry on parse failure.
Phase 4 (full):     span mapping, per-claim quality validation, edge-case handling.

Public API:
    decompose(record, model, provider) -> AnswerRecord   (mutates record.claims)
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import List

from src.schema import AnswerRecord, Claim
import src.llm as llm

# ─── spaCy (lazy load with regex fallback) ────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                               check=True)
                _nlp = spacy.load("en_core_web_sm")
        except (ImportError, ModuleNotFoundError, Exception):
            _nlp = None
    return _nlp


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy, with regex fallback if unavailable."""
    if not text.strip():
        return []
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # Fallback when spaCy is not installed
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a fact-checking assistant. Break the given answer into atomic, self-contained factual claims.

RULES:
1. Each claim must contain exactly ONE independently verifiable fact.
2. Replace ALL pronouns with the actual entity name (decontextualize).
3. Preserve dates, numbers, and named entities exactly.
4. Mark opinions, hedges ("I think", "I believe"), questions, and meta-text ("As an AI...") with check_worthy=false.
5. Output ONLY a valid JSON array — no markdown fences, no extra text.

OUTPUT FORMAT:
[
  {"claim": "...", "sentence_id": <int>, "check_worthy": <bool>},
  ...
]"""

_FEW_SHOT = """EXAMPLE:
Question: Who was Marie Curie?
Answer: Marie Curie was a Polish physicist. She won two Nobel Prizes and I think she was the greatest scientist ever.

Output:
[
  {"claim": "Marie Curie was a Polish physicist.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Marie Curie won two Nobel Prizes.", "sentence_id": 1, "check_worthy": true},
  {"claim": "Marie Curie was the greatest scientist ever.", "sentence_id": 1, "check_worthy": false}
]

EXAMPLE:
Question: Tell me about the Eiffel Tower.
Answer: The Eiffel Tower was built in 1889. It stands 330 metres tall and is located in Berlin.

Output:
[
  {"claim": "The Eiffel Tower was built in 1889.", "sentence_id": 0, "check_worthy": true},
  {"claim": "The Eiffel Tower stands 330 metres tall.", "sentence_id": 1, "check_worthy": true},
  {"claim": "The Eiffel Tower is located in Berlin.", "sentence_id": 1, "check_worthy": true}
]"""


def _build_prompt(question: str, answer: str) -> str:
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{_FEW_SHOT}\n\n"
        f"NOW DECOMPOSE:\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Output:"
    )


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_claims_json(response: str) -> List[dict] | None:
    """Extract and parse a JSON array from an LLM response string."""
    # Strip markdown code fences if present
    response = re.sub(r"```(?:json)?", "", response).strip()
    # Find the first [...] block
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def decompose(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    max_retries: int = 2,
    min_claim_length: int = 5,
) -> AnswerRecord:
    """
    Decompose record.answer into atomic claims.
    Mutates record.claims in place and returns the record.

    Args:
        record:           The AnswerRecord to process.
        model:            LLM model name.
        provider:         LLM provider ("groq" | "openai" | "local").
        max_retries:      Number of re-tries on JSON parse failure.
        min_claim_length: Discard claims shorter than this many characters.
    """
    prompt = _build_prompt(record.question, record.answer)
    raw_claims: List[dict] = []

    for attempt in range(max_retries):
        # Use slightly higher temperature on retry to get different output
        temp = 0.0 if attempt == 0 else 0.3
        response = llm.call(
            prompt=prompt, model=model, provider=provider,
            temperature=temp, use_cache=(attempt == 0),
        )
        parsed = _parse_claims_json(response)
        if parsed is not None:
            raw_claims = parsed
            break
        if attempt == max_retries - 1:
            print(f"[decompose] ⚠ JSON parse failed for {record.id} after {max_retries} attempts.")
            print(f"  Last response snippet: {response[:300]}")

    claims: List[Claim] = []
    for i, rc in enumerate(raw_claims):
        if not isinstance(rc, dict):
            continue
        text = str(rc.get("claim", "")).strip()
        if len(text) < min_claim_length:
            continue
        claim = Claim(
            id=f"{record.id}_c{i + 1}",
            text=text,
            sentence_id=int(rc.get("sentence_id", 0)),
            check_worthy=bool(rc.get("check_worthy", True)),
        )
        claims.append(claim)

    record.claims = claims
    return record
