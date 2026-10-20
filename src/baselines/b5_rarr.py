"""
b5_rarr.py — Baseline B5: RARR-style Monolithic Rewrite (Gao et al., 2023).

Retrofit Attribution using Research and Revision:
- Retrieves evidence for the entire query/answer text (rather than atomic decomposed claims).
- Rewrites the entire answer in a single monolithic prompt conditioned on all evidence.
- Directly addresses Research Question RQ3: localized claim editing vs. whole-answer rewrite.
"""
from __future__ import annotations

import re
from typing import List, Optional

from src.llm import call as llm_call
from src.retrieve import retrieve_bm25
from src.schema import AnswerRecord, EvidencePassage


_RARR_PROMPT = """You are a factual editor revising an answer using retrieved reference passages.

Question: {question}
Original Answer: {answer}

Retrieved Reference Passages:
{passages}

Instructions:
1. Rewrite the ENTIRE answer so that all facts stated are supported by the reference passages.
2. If any fact in the original answer is contradicted or not supported by the passages, correct or remove it.
3. Keep the answer fluent and cohesive.
4. Output ONLY the rewritten answer text.

Rewritten Answer:"""


def run_b5(
    record: AnswerRecord,
    index_path: Optional[str] = None,
    top_k: int = 5,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    passages: Optional[List[EvidencePassage]] = None,
) -> AnswerRecord:
    """
    Run RARR-style monolithic retrieval-augmented rewrite on an AnswerRecord.
    """
    if passages is None and index_path is not None:
        # Retrieve passages using the whole question + answer as the search query
        query = f"{record.question} {record.answer}"
        try:
            passages = retrieve_bm25(query, index_path, top_k=top_k)
        except Exception:
            passages = []

    if passages:
        passages_text = "\n\n".join(
            f"[{i + 1}] (Title: {p.title}) {p.text}"
            for i, p in enumerate(passages[:top_k])
        )
    else:
        passages_text = "(No reference passages retrieved)"

    prompt = _RARR_PROMPT.format(
        question=record.question,
        answer=record.answer,
        passages=passages_text,
    )

    rewritten = llm_call(prompt=prompt, model=model, provider=provider, temperature=0.0).strip()
    record.repaired_answer = rewritten
    record.repair_iterations = 1
    return record
