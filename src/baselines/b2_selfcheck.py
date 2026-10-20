"""
b2_selfcheck.py — Baseline B2: SelfCheckGPT (Manakul et al., 2023).

Measures factual hallucination using internal consistency across stochastically
sampled LLM outputs, without external retrieval.

Core steps:
1. Sample N stochastic responses (temperature = 1.0) for the question.
2. For each sentence in the candidate answer, compute consistency against sampled responses.
3. Mark sentences with low consistency as hallucinated and apply fallback hedging.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

from src.decompose import split_sentences
from src.llm import call as llm_call
from src.schema import AnswerRecord, Claim


_SAMPLE_PROMPT = """Question: {question}

Please provide a direct, factual answer:"""

_CONSISTENCY_PROMPT = """You are evaluating factual consistency.
Context samples generated for the same question:
{samples}

Candidate statement to evaluate:
"{statement}"

Based on the context samples, is this statement:
- "Consistent": supported by or consistent with the majority of samples
- "Inconsistent": contradicts or is unsupported by the samples

Output JSON: {{"verdict": "Consistent" | "Inconsistent", "score": <float 0.0 to 1.0 consistency score>}}"""


_mock_sampler: Optional[Callable[[str, int], List[str]]] = None


def set_mock_selfcheck_sampler(sampler_fn: Optional[Callable[[str, int], List[str]]]) -> None:
    """Mock hook for testing without calling live LLMs."""
    global _mock_sampler
    _mock_sampler = sampler_fn


def generate_stochastic_samples(
    question: str,
    n_samples: int = 5,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> List[str]:
    """Generate n stochastic alternative answers at temperature 1.0."""
    if _mock_sampler is not None:
        return _mock_sampler(question, n_samples)

    samples: List[str] = []
    prompt = _SAMPLE_PROMPT.format(question=question)
    for i in range(n_samples):
        # Use different seeds to ensure stochastic diversity
        sample = llm_call(
            prompt=prompt,
            model=model,
            temperature=1.0,
            seed=42 + i * 17,
            provider=provider,
            use_cache=False,
        ).strip()
        if sample:
            samples.append(sample)
    return samples


def evaluate_sentence_consistency(
    sentence: str,
    samples: List[str],
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> float:
    """
    Score consistency of sentence with samples in [0.0, 1.0].
    Higher = more consistent (less hallucinated).
    """
    if not samples:
        return 0.5

    samples_formatted = "\n\n".join(f"[Sample {i + 1}]: {s}" for i, s in enumerate(samples))
    prompt = _CONSISTENCY_PROMPT.format(samples=samples_formatted, statement=sentence)

    response = llm_call(
        prompt=prompt,
        model=model,
        temperature=0.0,
        provider=provider,
        use_cache=True,
    )

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            verdict = data.get("verdict", "").lower()
            score = float(data.get("score", 0.5))
            if "inconsistent" in verdict and score > 0.5:
                # Invert if model gave inconsistency score
                score = 1.0 - score
            return max(0.0, min(1.0, score))
        except Exception:
            pass

    # Fallback lexical overlap if JSON parsing fails
    sent_tokens = set(sentence.lower().split())
    if not sent_tokens:
        return 0.5
    sample_tokens = set(" ".join(samples).lower().split())
    overlap = len(sent_tokens & sample_tokens) / len(sent_tokens)
    return overlap


def run_b2(
    record: AnswerRecord,
    n_samples: int = 5,
    threshold: float = 0.5,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    samples: Optional[List[str]] = None,
) -> AnswerRecord:
    """
    Run Baseline B2 (SelfCheckGPT) on an AnswerRecord.
    Identifies sentences with consistency score < threshold,
    hedges inconsistent sentences, and records consistency scores.
    """
    if samples is None:
        samples = generate_stochastic_samples(
            question=record.question,
            n_samples=n_samples,
            model=model,
            provider=provider,
        )

    sentences = split_sentences(record.answer)
    repaired_sentences: List[str] = []
    consistency_scores: List[float] = []

    for sid, sentence in enumerate(sentences):
        score = evaluate_sentence_consistency(
            sentence=sentence,
            samples=samples,
            model=model,
            provider=provider,
        )
        consistency_scores.append(score)

        if score < threshold:
            # Low consistency -> hedge with reportedly
            hedged = f"Reportedly, {sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence}"
            repaired_sentences.append(hedged)
        else:
            repaired_sentences.append(sentence)

    record.repaired_answer = " ".join(repaired_sentences)
    record.repair_iterations = 1

    # Estimate reliability score from mean consistency
    mean_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
    record.reliability_score_after = mean_score
    return record

