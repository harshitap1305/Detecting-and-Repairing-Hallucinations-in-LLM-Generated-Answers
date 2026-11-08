"""Build and query a streaming SQLite FTS5 index of FEVER Wikipedia passages.

Unlike build_index.py's in-memory BM25 pickle, this command can stream the
complete wiki-pages directory. It also accepts an existing passages_A.jsonl.
Both inputs produce the same doc_id/title/text/chunk_id passage schema.

    python scripts/build_fever_fts.py --wiki-pages-dir data/raw/fever/wiki-pages \
      --output indexes/fever_fts.sqlite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import EvidencePassage


SCHEMA_VERSION = "1"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with",
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Match build_index.py's word chunking without retaining all passages."""
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller")
    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start += chunk_size - overlap


def wiki_passages(wiki_dir: Path, chunk_size: int, overlap: int) -> Iterator[dict]:
    files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No FEVER wiki-*.jsonl files in {wiki_dir}")
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    page = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(page, dict) or not isinstance(page.get("id"), str):
                    raise ValueError(f"Invalid wiki page at {path}:{line_number}")
                title = page["id"].strip()
                if not title:
                    continue
                lines = str(page.get("lines", "")).splitlines()
                text = " ".join(
                    rest.strip()
                    for raw_line in lines
                    for _, separator, rest in [raw_line.partition("\t")]
                    if separator and rest.strip()
                )
                for chunk_id, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                    yield {
                        "doc_id": f"{title}_{chunk_id}",
                        "title": title,
                        "text": f"Title: {title}. {chunk}",
                        "chunk_id": chunk_id,
                    }


def file_passages(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected passage object at {path}:{line_number}")
            yield row


def _validated_passage(row: dict) -> tuple[str, str, str, int]:
    doc_id, title, text = (row.get(key) for key in ("doc_id", "title", "text"))
    chunk_id = row.get("chunk_id", 0)
    if (not all(isinstance(value, str) and value.strip()
                for value in (doc_id, title, text))
            or type(chunk_id) is not int or chunk_id < 0):
        raise ValueError(f"Invalid passage: {row!r}")
    return doc_id, title, text, chunk_id


def build_index(rows: Iterator[dict], output: Path, *, source: str,
                max_passages: int | None = None, batch_size: int = 2000,
                overwrite: bool = False) -> dict:
    if max_passages is not None and max_passages < 1:
        raise ValueError("max_passages must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Index exists: {output}")
    if partial.exists() and not overwrite:
        raise FileExistsError(f"Interrupted index build exists: {partial}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    digest = hashlib.sha256()
    count = 0
    connection = sqlite3.connect(partial)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE passages (id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL UNIQUE, "
            "title TEXT NOT NULL, text TEXT NOT NULL, chunk_id INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE passage_fts USING fts5("
            "title, text, content='passages', content_rowid='id', tokenize='unicode61')"
        )
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        batch: list[tuple[str, str, str, int]] = []
        for row in rows:
            passage = _validated_passage(row)
            digest.update(json.dumps(passage, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
            batch.append(passage)
            count += 1
            if len(batch) >= batch_size:
                connection.executemany(
                    "INSERT INTO passages(doc_id,title,text,chunk_id) VALUES(?,?,?,?)", batch
                )
                connection.commit()
                batch.clear()
            if max_passages is not None and count >= max_passages:
                break
        if batch:
            connection.executemany(
                "INSERT INTO passages(doc_id,title,text,chunk_id) VALUES(?,?,?,?)", batch
            )
            connection.commit()
        if count == 0:
            raise ValueError("No passages were indexed")
        connection.execute("INSERT INTO passage_fts(passage_fts) VALUES('rebuild')")
        info = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "source_sha256": digest.hexdigest(),
            "passage_count": str(count),
        }
        connection.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", info.items())
        connection.commit()
    except Exception:
        connection.close()
        if partial.exists():
            partial.unlink()
        raise
    else:
        connection.close()
    partial.replace(output)
    return {"output": str(output), **info}


def claim_query(claim: str, max_terms: int = 24) -> str:
    """Convert arbitrary claim text into a safe, bounded FTS5 OR expression."""
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    terms = list(dict.fromkeys(
        term.casefold() for term in re.findall(r"[^\W_]+", claim)
        if term.strip()
    ))
    content_terms = [term for term in terms if term not in _STOPWORDS]
    selected = (content_terms or terms)[:max_terms]
    # Regex tokens contain no FTS syntax, and quoting protects reserved words.
    return " OR ".join(f'"{term}"' for term in selected)


class FtsRetriever:
    """Read-only, rank-ordered retrieval with a stable row-id tie break."""

    def __init__(self, index_path: Path):
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.connection = sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro",
                                          uri=True)
        self.connection.row_factory = sqlite3.Row
        try:
            self.info = {
                row["key"]: row["value"]
                for row in self.connection.execute("SELECT key,value FROM metadata")
            }
            if (self.info.get("schema_version") != SCHEMA_VERSION
                    or int(self.info.get("passage_count", "0")) < 1
                    or not self.info.get("source_sha256")):
                raise ValueError(f"Unsupported or incomplete FTS5 index: {index_path}")
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FtsRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def retrieve(self, claim: str, top_k: int = 5) -> list[EvidencePassage]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = claim_query(claim)
        if not query:
            return []
        rows = self.connection.execute(
            "SELECT p.doc_id,p.title,p.text,p.chunk_id,"
            "bm25(passage_fts,4.0,1.0) AS rank "
            "FROM passage_fts JOIN passages p ON p.id=passage_fts.rowid "
            "WHERE passage_fts MATCH ? ORDER BY rank ASC,p.id ASC LIMIT ?",
            (query, top_k),
        )
        return [
            EvidencePassage(doc_id=row["doc_id"], title=row["title"], text=row["text"],
                            chunk_id=row["chunk_id"], score=-float(row["rank"]))
            for row in rows
        ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wiki-pages-dir", type=Path)
    source.add_argument("--passages-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=150)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--max-passages", type=int)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("chunk-size must be positive and overlap smaller")
    rows = (wiki_passages(args.wiki_pages_dir, args.chunk_size, args.chunk_overlap)
            if args.wiki_pages_dir else file_passages(args.passages_file))
    source_name = str(args.wiki_pages_dir or args.passages_file)
    info = build_index(rows, args.output, source=source_name,
                       max_passages=args.max_passages, batch_size=args.batch_size,
                       overwrite=args.overwrite)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
