"""
b4_cove.py — Baseline B4: Chain-of-Verification (CoVe; Dhuliawala et al., 2023).

Self-verification baseline without external retrieval:
1. Plan verification: generate factual verification questions checking the claims.
2. Execute verification: answer each question independently (factored / unconditioned).
3. Final revision: revise original answer based on verification outcomes.
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

from src.llm import call as llm_call
from src.schema import AnswerRecord


_GENERATE_QUESTIONS_PROMPT = """You are a verification planner.
Given the question and the candidate answer, generate 2 to 4 factual verification questions
to verify the key facts, dates, names, or locations mentioned in the answer.

Original Question: {question}
Candidate Answer: {answer}

Output ONLY a JSON list of strings, for example:
["When was X born?", "Where did Y occur?"]"""

_ANSWER_QUESTION_PROMPT = """Answer the following question factually, directly, and concisely:

Question: {verification_question}
Answer:"""

_REVISE_PROMPT = """You are a factual editor revising an answer using independently verified facts.

Original Question: {question}
Original Answer: {answer}

Verification Results:
{qa_pairs}

Instructions:
1. Review the original answer against the verified facts.
2. If any fact in the original answer conflicts with the verification answers, correct it.
3. Keep parts that are consistent.
4. Output ONLY the revised answer text.

Revised Answer:"""


def generate_verification_questions(
    question: str,
    answer: str,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> List[str]:
    """Step 1: Generate verification questions checking the candidate answer."""
    prompt = _GENERATE_QUESTIONS_PROMPT.format(question=question, answer=answer)
    response = llm_call(prompt=prompt, model=model, provider=provider, temperature=0.0)

    # Extract JSON list
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group())
            if isinstance(questions, list):
                return [str(q).strip() for q in questions if str(q).strip()]
        except Exception:
            pass

    # Fallback heuristic: split lines
    lines = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in response.split("\n")
        if "?" in line
    ]
    return lines[:4] if lines else [f"Is the factual information about {question} accurate?"]


def answer_verification_questions(
    questions: List[str],
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> List[Tuple[str, str]]:
    """Step 2: Answer verification questions independently (unconditioned)."""
    qa_pairs: List[Tuple[str, str]] = []
    for q in questions:
        prompt = _ANSWER_QUESTION_PROMPT.format(verification_question=q)
        ans = llm_call(prompt=prompt, model=model, provider=provider, temperature=0.0).strip()
        qa_pairs.append((q, ans))
    return qa_pairs


def revise_answer_with_cove(
    question: str,
    answer: str,
    qa_pairs: List[Tuple[str, str]],
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> str:
    """Step 3: Revise the initial answer based on verified Q&A pairs."""
    qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    prompt = _REVISE_PROMPT.format(
        question=question,
        answer=answer,
        qa_pairs=qa_text,
    )
    revised = llm_call(prompt=prompt, model=model, provider=provider, temperature=0.0).strip()
    return revised


def run_b4(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> AnswerRecord:
    """Run Chain-of-Verification (CoVe) on an AnswerRecord."""
    questions = generate_verification_questions(
        question=record.question,
        answer=record.answer,
        model=model,
        provider=provider,
    )
    qa_pairs = answer_verification_questions(
        questions=questions,
        model=model,
        provider=provider,
    )
    revised = revise_answer_with_cove(
        question=record.question,
        answer=record.answer,
        qa_pairs=qa_pairs,
        model=model,
        provider=provider,
    )

    record.repaired_answer = revised
    record.repair_iterations = 1
    return record

