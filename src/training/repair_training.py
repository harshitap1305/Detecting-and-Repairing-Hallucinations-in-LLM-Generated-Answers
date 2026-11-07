"""Dataset validation and independent text metrics for the trained editor."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from src.training.repair_model import format_repair_prompt


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_examples(path: Path, expected_split: str | None = None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("id", "article_id", "split", "sentence", "target", "evidence", "error_type")
            if any(k not in row for k in required):
                raise ValueError(f"{path}:{line_no}: missing required repair field")
            if expected_split and row["split"] != expected_split:
                raise ValueError(f"{path}:{line_no}: expected split {expected_split}, got {row['split']}")
            if not isinstance(row["evidence"], list) or not row["evidence"]:
                raise ValueError(f"{path}:{line_no}: evidence must be a nonempty list")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def assert_disjoint_splits(*splits: list[dict]) -> None:
    """Refuse article, prompt, or exact input overlap across supplied splits."""
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            for field in ("article_id", "entity_group", "prompt_id", "sentence"):
                left_values = {str(r.get(field, "")).strip().casefold() for r in left}
                right_values = {str(r.get(field, "")).strip().casefold() for r in right}
                left_values.discard("")
                right_values.discard("")
                overlap = left_values & right_values
                if overlap:
                    raise ValueError(f"Train/dev/test leakage in {field}: {next(iter(overlap))}")


def example_prompt(row: dict, policy: str = "rewrite+hedge") -> str:
    flagged = []
    if row.get("flagged_claim"):
        flagged = [{"text": row["flagged_claim"], "label": row["flagged_label"]}]
    return format_repair_prompt(row["sentence"], flagged, row["evidence"], policy)


def token_f1(reference: str, prediction: str) -> float:
    ref = Counter(re.findall(r"\w+|[^\w\s]", reference.casefold()))
    pred = Counter(re.findall(r"\w+|[^\w\s]", prediction.casefold()))
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = sum((ref & pred).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_predictions(rows: list[dict], predictions: list[str]) -> dict:
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Prediction count must match a nonempty example set")
    n = len(rows)
    unchanged = [(r, p) for r, p in zip(rows, predictions) if r["error_type"] == "unchanged"]
    return {
        "count": n,
        "exact_match": sum(p.strip() == r["target"].strip() for r, p in zip(rows, predictions)) / n,
        "token_f1": sum(token_f1(r["target"], p) for r, p in zip(rows, predictions)) / n,
        "unchanged_count": len(unchanged),
        "unchanged_accuracy": (
            sum(p == r["sentence"] for r, p in unchanged) / len(unchanged)
            if unchanged else None
        ),
    }


def predict_rows(model, tokenizer, rows: list[dict], device: str,
                 batch_size: int = 8, max_source_length: int = 512,
                 max_new_tokens: int = 128, policy: str = "rewrite+hedge") -> tuple[list[str], float]:
    """Greedy generation with elapsed seconds, for dev and held-out evaluation."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [example_prompt(row, policy) for row in batch]
            inputs = tokenizer(prompts, padding=True, truncation=True,
                               max_length=max_source_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [p.strip() for p in predictions], time.perf_counter() - started
