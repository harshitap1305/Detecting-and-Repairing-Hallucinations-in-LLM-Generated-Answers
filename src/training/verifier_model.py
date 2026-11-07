"""Inference helpers for a Path A FEVER-trained three-way verifier.

The checkpoint must have ``id2label`` set to Supported, Contradicted and
Unverifiable. Input is always a (premise, claim) pair, matching the training
script. Heavy ML imports and model loading happen only when inference runs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from src.training.verifier_data import LABELS, format_premise


@lru_cache(maxsize=2)
def _load_checkpoint(checkpoint: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    labels = [model.config.id2label[i] for i in range(3)]
    if labels != list(LABELS):
        raise ValueError(f"Checkpoint label order {labels!r} does not match {LABELS!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    max_length = int(getattr(model.config, "verifier_max_length", 320))
    return tokenizer, model, device, max_length


def predict_probabilities(pairs: Sequence[tuple[str, str]], checkpoint: str,
                          *, batch_size: int = 16) -> list[list[float]]:
    """Return calibrated-*unadjusted* softmax scores in ``LABELS`` order.

    These are model probabilities, not a calibrated reliability claim. Use a
    held-out calibration set before interpreting them as confidence values.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not pairs:
        return []
    for premise, claim in pairs:
        if not premise.strip() or not claim.strip():
            raise ValueError("Premise and claim must be nonempty")
    import torch

    tokenizer, model, device, max_length = _load_checkpoint(checkpoint)
    scores: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            inputs = tokenizer([premise for premise, _ in chunk],
                               [claim for _, claim in chunk],
                               padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            scores.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    return scores


def predict_verdict(evidence_texts: list[str], claim_text: str,
                    checkpoint: str) -> tuple[str, list[float]]:
    """Predict one claim using evidence passages and a saved Path A checkpoint.

    Returns ``(label, [P(Supported), P(Contradicted), P(Unverifiable)])``.
    A caller with no retrieved passages should assign Unverifiable directly.
    """
    premise = format_premise(evidence_texts)
    probabilities = predict_probabilities([(premise, claim_text)], checkpoint)[0]
    best = max(range(3), key=probabilities.__getitem__)
    return LABELS[best], probabilities
