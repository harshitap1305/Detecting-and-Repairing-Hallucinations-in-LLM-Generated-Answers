"""Deterministic candidate generation and verified filtering for Path C.

Source input may be FEVER ``wiki-*.jsonl`` pages or JSONL rows containing
``article_id``, ``sentence``, and ``evidence``. No synthetic row is admitted to
training until the original and corrupted text have explicit verifier labels.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator


LABELS = {"Supported", "Contradicted", "Unverifiable"}
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_COPULA = re.compile(r"\b(is|was|were|are|has|had)\b", re.IGNORECASE)


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.replace("_", " ").lower()).split())


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def article_split(article_id: str, seed: int = 42) -> str:
    """Stable 80/10/10 train/dev/test assignment at article granularity."""
    bucket = _stable_int(f"{seed}:{_normal(article_id)}") % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def load_holdouts(path: Path) -> dict[str, set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    holdouts = {
        name: {_normal(str(v)) for v in raw.get(name, []) if _normal(str(v))}
        for name in ("article_ids", "entities", "prompt_ids", "prompts")
    }
    if not any(holdouts.values()):
        raise ValueError("Holdout manifest is empty; provide test articles, entities, or prompts before building training data")
    return holdouts


def _contains_term(haystack: str, term: str) -> bool:
    normalized = f" {_normal(haystack)} "
    return f" {term} " in normalized


def is_held_out(source: dict, holdouts: dict[str, set[str]]) -> bool:
    article = _normal(str(source["article_id"]))
    if article in holdouts["article_ids"]:
        return True
    prompt_id = _normal(str(source.get("prompt_id", "")))
    if prompt_id and prompt_id in holdouts["prompt_ids"]:
        return True
    prompt = _normal(str(source.get("prompt", "")))
    if prompt and prompt in holdouts["prompts"]:
        return True
    context = " ".join(str(source.get(k, "")) for k in ("article_id", "entity_group", "sentence", "evidence"))
    return any(_contains_term(context, entity) for entity in holdouts["entities"])


def iter_wiki_sentences(wiki_dir: Path) -> Iterator[dict]:
    """Stream FEVER page sentences with neighbouring lines as evidence."""
    files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No FEVER wiki-*.jsonl files in {wiki_dir}")
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                page = json.loads(raw)
                article_id = str(page.get("id", "")).strip()
                if not article_id:
                    continue
                lines = []
                for raw_line in str(page.get("lines", "")).splitlines():
                    fields = raw_line.split("\t")
                    if len(fields) > 1 and fields[1].strip():
                        lines.append((fields[0], fields[1].strip()))
                for index, (line_id, sentence) in enumerate(lines):
                    if not 7 <= len(sentence.split()) <= 60:
                        continue
                    # Include immediate context, but retain the article title.
                    context = " ".join(text for _, text in lines[max(0, index-1):index+2])
                    yield {
                        "article_id": article_id,
                        "entity_group": article_id,
                        "source_id": f"{article_id}:{line_id}",
                        "sentence": sentence,
                        "evidence": f"{article_id.replace('_', ' ')}. {context}",
                    }


def iter_source_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not all(str(row.get(k, "")).strip() for k in ("article_id", "sentence", "evidence")):
                raise ValueError(f"{path}:{line_no}: article_id, sentence, and evidence are required")
            row.setdefault("source_id", f"{row['article_id']}:{line_no}")
            yield row


def corrupt_sentence(source: dict, seed: int) -> tuple[str, str] | None:
    """Create one reproducible error; later verifier filtering decides its label."""
    sentence = str(source["sentence"])
    rng = random.Random(_stable_int(f"{seed}:{source['source_id']}"))
    options: list[tuple[str, str]] = []
    year_match = _YEAR.search(sentence)
    if year_match:
        year = int(year_match.group())
        shifted = year + rng.choice((-5, -3, -1, 1, 3, 5))
        options.append(("year_shift", sentence[:year_match.start()] + str(shifted) + sentence[year_match.end():]))
    number_match = next((m for m in _NUMBER.finditer(sentence) if not _YEAR.fullmatch(m.group())), None)
    if number_match:
        number = number_match.group()
        if "." not in number:
            value = int(number)
            changed = max(0, value + rng.choice((-2, -1, 1, 2)))
            if changed != value:
                options.append(("number_shift", sentence[:number_match.start()] + str(changed) + sentence[number_match.end():]))
    copula_match = _COPULA.search(sentence)
    if copula_match and not re.match(r"\s+not\b", sentence[copula_match.end():], flags=re.IGNORECASE):
        options.append(("negation", sentence[:copula_match.end()] + " not" + sentence[copula_match.end():]))
    # Same-type replacements require an externally curated entity pool.
    for entity in source.get("entities", []):
        text = str(entity.get("text", ""))
        label = str(entity.get("label", ""))
        pool = source.get("entity_pool", {}).get(label, [])
        alternatives = [str(v) for v in pool if _normal(str(v)) != _normal(text)]
        if text and alternatives:
            match = re.search(r"\b" + re.escape(text) + r"\b", sentence)
            if match:
                replacement = alternatives[rng.randrange(len(alternatives))]
                options.append(("entity_swap", sentence[:match.start()] + replacement + sentence[match.end():]))
    if not options:
        return None
    return options[rng.randrange(len(options))]


def generate_candidates(
    sources: Iterable[dict], holdouts: dict[str, set[str]], seed: int = 42,
    max_sources: int | None = None,
) -> Iterator[dict]:
    accepted_sources = 0
    for source in sources:
        if is_held_out(source, holdouts):
            continue
        sentence = str(source["sentence"]).strip()
        if not sentence or not str(source["evidence"]).strip():
            continue
        if max_sources is not None and accepted_sources >= max_sources:
            break
        accepted_sources += 1
        source_id = str(source["source_id"])
        common = {
            "article_id": str(source["article_id"]),
            "entity_group": str(source.get("entity_group", source["article_id"])),
            "source_id": source_id,
            "prompt_id": str(source.get("prompt_id", "")),
            "split": article_split(str(source["article_id"]), seed),
            "original_sentence": sentence,
            "evidence": [{"title": str(source["article_id"]).replace("_", " "), "text": str(source["evidence"])}],
        }
        yield {**common, "id": f"{source_id}:unchanged", "sentence": sentence,
               "error_type": "unchanged"}
        result = corrupt_sentence(source, seed)
        if result:
            error_type, corrupted = result
            yield {**common, "id": f"{source_id}:{error_type}", "sentence": corrupted,
                   "error_type": error_type}


def filter_verified_candidates(
    candidates: Iterable[dict], verifier: Callable[[dict], tuple[str, str]],
    unchanged_fraction: float = 0.2, seed: int = 42,
) -> tuple[dict[str, list[dict]], Counter]:
    """Admit only rows with explicit Supported original and verified corruption."""
    if not 0 <= unchanged_fraction < 1:
        raise ValueError("unchanged_fraction must be in [0, 1)")
    kept: dict[str, list[dict]] = defaultdict(list)
    unchanged: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    for row in candidates:
        counts["candidates"] += 1
        original_label, candidate_label = verifier(row)
        if original_label not in LABELS or candidate_label not in LABELS:
            raise ValueError(f"Invalid verifier labels for {row['id']}")
        if original_label != "Supported":
            counts["original_not_supported"] += 1
            continue
        is_unchanged = row["error_type"] == "unchanged"
        if (is_unchanged and candidate_label != "Supported") or (
            not is_unchanged and candidate_label not in {"Contradicted", "Unverifiable"}
        ):
            counts["corruption_not_verified"] += 1
            continue
        example = {
            "id": row["id"], "article_id": row["article_id"],
            "source_id": row.get("source_id", ""),
            "entity_group": row.get("entity_group", row["article_id"]),
            "prompt_id": row.get("prompt_id", ""), "split": row["split"],
            "sentence": row["sentence"],
            "flagged_claim": "" if is_unchanged else row["sentence"],
            "flagged_label": "" if is_unchanged else candidate_label,
            "evidence": row["evidence"], "target": row["original_sentence"],
            "error_type": row["error_type"],
        }
        (unchanged if is_unchanged else kept)[row["split"]].append(example)
        counts[f"verified_{row['error_type']}"] += 1
    for split in ("train", "dev", "test"):
        edited_count = len(kept[split])
        unchanged_limit = int(edited_count * unchanged_fraction / max(1e-9, 1 - unchanged_fraction))
        options = sorted(unchanged[split], key=lambda r: _stable_int(f"{seed}:{r['id']}"))
        kept[split].extend(options[:unchanged_limit])
        kept[split].sort(key=lambda r: _stable_int(f"{seed}:{r['id']}"))
        counts[f"output_{split}"] = len(kept[split])
    return dict(kept), counts


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
