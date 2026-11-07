"""Prompt and lazy inference adapter for a locally fine-tuned repair editor.

The same prompt builder is used by training and the live repair stage. Loading a
checkpoint is deliberately lazy: importing the base pipeline never downloads a
model or allocates GPU memory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


def _field(item: Any, name: str, default: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get(name, default) or default)
    return str(getattr(item, name, default) or default)


def format_repair_prompt(
    sentence: str,
    flagged_claims: Iterable[Any],
    evidence_passages: Iterable[Any],
    policy: str = "rewrite+hedge",
) -> str:
    """Return the exact input format used for both fine-tuning and inference."""
    claims = list(flagged_claims)
    passages = list(evidence_passages)
    if claims:
        claim_text = "\n".join(
            f"- [{_field(c, 'label', 'Contradicted')}] {_field(c, 'text')}"
            for c in claims
        )
    else:
        claim_text = "(none; return the sentence unchanged)"
    if passages:
        evidence_text = "\n".join(
            f"[{i}] {_field(p, 'title')}: {_field(p, 'text')[:500]}"
            for i, p in enumerate(passages[:3], 1)
        )
    else:
        evidence_text = "(no evidence)"
    return (
        "### Task\n"
        "Fix only the unsupported claim using the evidence. Preserve all other "
        "facts and wording. If no claim is flagged, copy the sentence exactly. "
        f"Fallback policy: {policy}.\n"
        f"### Sentence\n{sentence}\n"
        f"### Flagged claims\n{claim_text}\n"
        f"### Evidence\n{evidence_text}\n"
        "### Edited sentence\n"
    )


@lru_cache(maxsize=4)
def _load_checkpoint(checkpoint: str, device: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.eval()
    return tokenizer, model


def generate_edit(
    sentence: str,
    flagged_claims: Iterable[Any],
    evidence_passages: Iterable[Any],
    checkpoint: str,
    policy: str = "rewrite+hedge",
    max_new_tokens: int = 128,
) -> str:
    """Generate one candidate edit; callers must still apply the normal guards.

    A directory path is required so selecting the trained backend cannot silently
    use the untrained base model.
    """
    claims = list(flagged_claims)
    if not claims:
        return sentence
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Trained repair checkpoint not found: {checkpoint}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _load_checkpoint(str(checkpoint_path.resolve()), device)
    prompt = format_repair_prompt(sentence, claims, evidence_passages, policy)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()
