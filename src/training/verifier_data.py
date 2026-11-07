"""Prepare evidence-grounded FEVER examples for a three-way verifier.

Inputs are the original FEVER ``train.jsonl``, FEVER ``wiki-pages`` JSONL files
(a directory, one file, or the original zip), and retrieval output in JSONL:

    {"claim_id": 123, "passages": [{"title": "Page", "text": "...", "score": 1.2}]}

Retrieval must be produced independently of the label. NEI claims have no gold
evidence, so they are only kept when retrieved evidence is available. A
Supported/Refuted claim uses a complete gold evidence set, optionally with
retrieved noise from the same article. We never relabel a retrieval miss as
NEI: it could contain unannotated support or contradiction.

Samples are partitioned by connected Wikipedia article and optional explicit
entity keys. This prevents article/entity text from crossing train and dev.
FEVER's original NEI source article is not provided, so its top retrieved
article is the grouping proxy unless ``source_page`` or ``entity`` is supplied.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LABELS = ("Supported", "Contradicted", "Unverifiable")
FEVER_LABELS = {
    "SUPPORTS": "Supported",
    "REFUTES": "Contradicted",
    "NOT ENOUGH INFO": "Unverifiable",
}


def normalize_title(title: str) -> str:
    """Normalize FEVER underscores and whitespace without erasing identity."""
    return " ".join(title.replace("_", " ").split()).casefold()


@contextmanager
def _text_stream(path: Path, member: str | None = None) -> Iterator[io.TextIOBase]:
    if member is not None:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8") as stream:
                    yield stream
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield stream
    else:
        with path.open(encoding="utf-8") as stream:
            yield stream


def _jsonl_sources(path: Path) -> list[tuple[Path, str | None]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and
                       (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz")))
        if not files:
            raise ValueError(f"No JSONL files found under {path}")
        return [(p, None) for p in files]
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if n.endswith(".jsonl"))
        if not members:
            raise ValueError(f"No JSONL files found in {path}")
        return [(path, name) for name in members]
    return [(path, None)]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL with source line numbers in validation errors."""
    for file_path, member in _jsonl_sources(path):
        with _text_stream(file_path, member) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {file_path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {file_path}:{line_number}")
                yield row


def format_premise(evidence_texts: list[str]) -> str:
    """Use the same passage framing in preparation and trained inference."""
    clean = [text.strip() for text in evidence_texts if text.strip()]
    if not clean:
        raise ValueError("At least one nonempty evidence passage is required")
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(clean, 1))


def _gold_groups(row: dict[str, Any]) -> list[list[tuple[str, int]]]:
    result: list[list[tuple[str, int]]] = []
    for group in row.get("evidence") or []:
        refs: list[tuple[str, int]] = []
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, list) or len(item) < 4 or not isinstance(item[2], str):
                continue
            try:
                sentence_id = int(item[3])
            except (TypeError, ValueError):
                continue
            if item[2].strip() and sentence_id >= 0:
                ref = (item[2], sentence_id)
                if ref not in refs:
                    refs.append(ref)
        if refs and refs not in result:
            result.append(refs)
    return result


def _read_retrieval(path: Path, max_passages: int,
                    wanted_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    retrieval: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        claim_id = row.get("claim_id", row.get("id"))
        if claim_id is None:
            raise ValueError("Retrieval row lacks claim_id/id")
        key = str(claim_id)
        if key not in wanted_ids:
            continue
        if key in retrieval:
            raise ValueError(f"Duplicate retrieval claim_id: {key}")
        raw = row.get("passages", row.get("evidence", []))
        if not isinstance(raw, list):
            raise ValueError(f"Retrieval passages must be a list for claim {key}")
        passages: list[dict[str, Any]] = []
        for passage in raw[:max_passages]:
            if not isinstance(passage, dict):
                continue
            title, text = passage.get("title"), passage.get("text")
            if isinstance(title, str) and title.strip() and isinstance(text, str) and text.strip():
                passages.append({"title": title.strip(), "text": text.strip()})
        retrieval[key] = passages
    return retrieval


def _read_wiki_sentences(path: Path, refs: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    wanted: dict[str, set[int]] = defaultdict(set)
    for title, sentence_id in refs:
        wanted[normalize_title(title)].add(sentence_id)
    found: dict[tuple[str, int], str] = {}
    for page in iter_jsonl(path):
        page_id = page.get("id")
        if not isinstance(page_id, str):
            continue
        title = normalize_title(page_id)
        if title not in wanted:
            continue
        for line in str(page.get("lines", "")).splitlines():
            number, separator, rest = line.partition("\t")
            if not separator:
                continue
            try:
                sentence_id = int(number)
            except ValueError:
                continue
            if sentence_id in wanted[title]:
                # FEVER lines may have tab-separated hyperlink metadata after text.
                sentence = rest.split("\t", 1)[0].strip()
                if sentence:
                    found[(title, sentence_id)] = sentence
        if len(found) == len(refs):
            break
    return found


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def split_by_articles(rows: list[dict[str, Any]], dev_fraction: float, seed: int
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split connected article/entity/claim groups, rejecting any overlap."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    uf = _UnionFind()
    for row in rows:
        titles = row.get("article_ids") or []
        if not titles:
            raise ValueError(f"Sample {row.get('id')} has no article_id")
        keys = [f"article:{normalize_title(title)}" for title in titles]
        if row.get("entity_key"):
            keys.append(f"entity:{normalize_title(row['entity_key'])}")
        keys.append(f"claim:{row['claim_id']}")
        for key in keys[1:]:
            uf.union(keys[0], key)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[uf.find(f"claim:{row['claim_id']}")].append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two disconnected article/entity groups for train/dev")
    random.Random(seed).shuffle(keys)
    target_dev = max(1, round(len(rows) * dev_fraction))
    dev_keys: set[str] = set()
    dev_count = 0
    for key in keys[:-1]:
        if dev_count >= target_dev:
            break
        dev_keys.add(key)
        dev_count += len(groups[key])
    train = [row for key in keys if key not in dev_keys for row in groups[key]]
    dev = [row for key in keys if key in dev_keys for row in groups[key]]
    if not train or not dev:
        raise ValueError("Could not form nonempty train and dev sets")
    train_articles = {normalize_title(t) for row in train for t in row["article_ids"]}
    dev_articles = {normalize_title(t) for row in dev for t in row["article_ids"]}
    if train_articles & dev_articles:
        raise AssertionError("Article leakage between train and dev")
    return train, dev


def prepare_examples(claims_path: Path, wiki_path: Path, retrieved_path: Path,
                     *, max_claims: int | None = None, max_passages: int = 5,
                     max_noise_passages: int = 2, dev_fraction: float = 0.2,
                     seed: int = 42, excluded_keys: set[str] | None = None
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build FEVER claim/premise pairs without manufacturing evidence labels."""
    if max_claims is not None and max_claims <= 0:
        raise ValueError("max_claims must be positive")
    if max_passages <= 0 or max_noise_passages < 0:
        raise ValueError("max_passages must be positive; max_noise_passages nonnegative")
    claims: list[dict[str, Any]] = []
    ids: set[str] = set()
    needed_refs: set[tuple[str, int]] = set()
    for row in iter_jsonl(claims_path):
        claim_id = row.get("id")
        claim, raw_label = row.get("claim"), row.get("label")
        if claim_id is None or not isinstance(claim, str) or not claim.strip() or raw_label not in FEVER_LABELS:
            raise ValueError(f"Invalid FEVER claim row: id={claim_id!r}, label={raw_label!r}")
        key = str(claim_id)
        if key in ids:
            raise ValueError(f"Duplicate FEVER claim id: {key}")
        ids.add(key)
        groups = _gold_groups(row) if raw_label != "NOT ENOUGH INFO" else []
        for group in groups:
            needed_refs.update((normalize_title(title), sentence_id) for title, sentence_id in group)
        claims.append({"id": key, "claim": claim.strip(), "label": FEVER_LABELS[raw_label],
                       "groups": groups, "entity_key": row.get("source_page") or row.get("entity")})
        if max_claims is not None and len(claims) >= max_claims:
            break
    if not claims:
        raise ValueError("No FEVER claims found")
    retrieved = _read_retrieval(retrieved_path, max_passages, ids)
    wiki = _read_wiki_sentences(wiki_path, needed_refs) if needed_refs else {}
    samples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for claim in claims:
        claim_id, label = claim["id"], claim["label"]
        passages = retrieved.get(claim_id, [])
        entity_key = claim["entity_key"] if isinstance(claim["entity_key"], str) else None
        if label == "Unverifiable":
            if not passages:
                skipped["nei_no_retrieval"] += 1
                continue
            anchor = normalize_title(passages[0]["title"])
            same_article = [p for p in passages if normalize_title(p["title"]) == anchor]
            texts = [f"Title: {p['title']}\n{p['text']}" for p in same_article]
            samples.append({"id": f"fever:{claim_id}:nei", "claim_id": claim_id,
                            "claim": claim["claim"], "premise": format_premise(texts),
                            "label": LABELS.index(label), "label_name": label,
                            "source": "retrieved_nei", "article_ids": [passages[0]["title"]],
                            "entity_key": entity_key})
            continue
        good_group: list[tuple[str, int]] | None = None
        for group in claim["groups"]:
            if all((normalize_title(title), sid) in wiki for title, sid in group):
                good_group = group
                break
        if good_group is None:
            skipped["gold_evidence_missing"] += 1
            continue
        titles = list(dict.fromkeys(title for title, _ in good_group))
        texts = [f"Title: {title}\n{wiki[(normalize_title(title), sid)]}"
                 for title, sid in good_group]
        gold_sentences = {wiki[(normalize_title(title), sid)].casefold()
                          for title, sid in good_group}
        for passage in passages:
            if len(texts) >= len(good_group) + max_noise_passages:
                break
            if (normalize_title(passage["title"]) in {normalize_title(t) for t in titles}
                    and passage["text"].casefold() not in gold_sentences):
                texts.append(f"Title: {passage['title']}\n{passage['text']}")
        samples.append({"id": f"fever:{claim_id}:gold", "claim_id": claim_id,
                        "claim": claim["claim"], "premise": format_premise(texts),
                        "label": LABELS.index(label), "label_name": label,
                        "source": "gold_plus_retrieved" if len(texts) > len(good_group) else "gold",
                        "article_ids": titles, "entity_key": entity_key})
    if not samples:
        raise ValueError("No usable FEVER examples; check wiki and retrieval inputs")
    excluded = {normalize_title(key) for key in (excluded_keys or set()) if key.strip()}
    if excluded:
        filtered = []
        for sample in samples:
            sample_keys = {normalize_title(title) for title in sample["article_ids"]}
            if sample["entity_key"]:
                sample_keys.add(normalize_title(sample["entity_key"]))
            claim_text = normalize_title(sample["claim"])
            mentions_excluded = any(re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", claim_text)
                                    for key in excluded)
            if sample_keys & excluded or mentions_excluded:
                skipped["heldout_entity_or_article"] += 1
            else:
                filtered.append(sample)
        samples = filtered
    train, dev = split_by_articles(samples, dev_fraction, seed)
    counts = lambda data: dict(Counter(row["label_name"] for row in data))
    manifest = {"seed": seed, "dev_fraction": dev_fraction,
                "source": "FEVER train only", "label_order": list(LABELS),
                "input_claims": len(claims), "usable_samples": len(samples),
                "skipped": dict(skipped), "train": {"count": len(train), "labels": counts(train)},
                "dev": {"count": len(dev), "labels": counts(dev)},
                "article_overlap": 0, "heldout_exclusions_applied": bool(excluded),
                "excluded_key_count": len(excluded),
                "inputs": {"claims": str(claims_path), "wiki": str(wiki_path),
                           "retrieval": str(retrieved_path)}}
    return train, dev, manifest


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(output_dir: Path, train: list[dict[str, Any]], dev: list[dict[str, Any]],
                  manifest: dict[str, Any], *, overwrite: bool = False) -> None:
    """Persist prepared rows and provenance; refuse accidental replacement."""
    paths = {"train": output_dir / "train.jsonl", "dev": output_dir / "dev.jsonl",
             "manifest": output_dir / "manifest.json"}
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Output exists under {output_dir}; pass --overwrite to replace")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with paths[name].open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {**manifest, "git_commit": _git_commit()}
    for name in ("claims", "retrieval"):
        path = Path(manifest["inputs"][name])
        if path.is_file():
            manifest["inputs"][name + "_sha256"] = _sha256(path)
    with paths["manifest"].open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
