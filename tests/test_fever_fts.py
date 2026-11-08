"""Streaming FEVER FTS5 indexing and resumable retrieval checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_fever_fts import (
    FtsRetriever, build_index, claim_query, file_passages, wiki_passages,
)
from scripts.retrieve_fever_evidence import main as retrieve_main
from src.retrieve import retrieve
from src.schema import AnswerRecord, Claim


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stream_wiki_index_and_ranked_retrieval(tmp_path):
    wiki = tmp_path / "wiki-pages"
    _jsonl(wiki / "wiki-001.jsonl", [
        {"id": "Alexander_Graham_Bell",
         "lines": "0\tAlexander Graham Bell invented the telephone.\n"
                  "1\tHe was born in Edinburgh, Scotland."},
        {"id": "Thomas_Edison",
         "lines": "0\tThomas Edison invented the phonograph."},
    ])
    db = tmp_path / "fever.sqlite"
    info = build_index(wiki_passages(wiki, chunk_size=7, overlap=2), db,
                       source=str(wiki), batch_size=2)
    assert info["passage_count"] == "3"
    with FtsRetriever(db) as retriever:
        assert retriever.info["source_sha256"] == info["source_sha256"]
        hits = retriever.retrieve('Did Bell invent a "telephone" (really)?', top_k=2)
        assert hits
        assert hits[0].title == "Alexander_Graham_Bell"
        assert hits[0].doc_id == "Alexander_Graham_Bell_0"
        assert hits[0].text.startswith("Title: Alexander_Graham_Bell.")
        assert hits[0].score >= 0
    assert claim_query('Bell" OR * (telephone)') == '"bell" OR "telephone"'
    assert claim_query("!!!") == ""


def test_passage_jsonl_index_has_stable_tie_break_and_refuses_replacement(tmp_path):
    source = tmp_path / "passages_A.jsonl"
    _jsonl(source, [
        {"doc_id": "first", "title": "Alpha", "text": "Title: Alpha. Same text.",
         "chunk_id": 0},
        {"doc_id": "second", "title": "Alpha", "text": "Title: Alpha. Same text.",
         "chunk_id": 1},
        {"doc_id": "third", "title": "Beta", "text": "Title: Beta. Different text.",
         "chunk_id": 0},
    ])
    db = tmp_path / "index.sqlite"
    build_index(file_passages(source), db, source=str(source))
    with FtsRetriever(db) as retriever:
        assert [hit.doc_id for hit in retriever.retrieve("Alpha", 2)] == [
            "first", "second"
        ]
        assert retriever.retrieve("???") == []
    with pytest.raises(FileExistsError):
        build_index(file_passages(source), db, source=str(source))


def test_fts_retrieval_output_resumes_exact_prefix_and_ignores_labels(tmp_path):
    wiki = tmp_path / "wiki-pages"
    _jsonl(wiki / "wiki-001.jsonl", [
        {"id": "Alpha", "lines": "0\tAlpha was founded in 1900."},
        {"id": "Beta", "lines": "0\tBeta was founded in 1901."},
    ])
    db = tmp_path / "index.sqlite"
    build_index(wiki_passages(wiki, 150, 20), db, source=str(wiki))
    claims = tmp_path / "train.jsonl"
    _jsonl(claims, [
        {"id": 1, "claim": "Alpha was founded in 1900.", "label": "SUPPORTS"},
        {"id": 2, "claim": "Beta was founded in 1901.", "label": "REFUTES"},
        {"id": 3, "claim": "No source for this claim.", "label": "NOT ENOUGH INFO"},
    ])
    output = tmp_path / "retrieved.jsonl"
    args = ["--fever-train", str(claims), "--index-path", str(db),
            "--method", "fts5", "--top-k", "2", "--output", str(output)]
    retrieve_main(args)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["claim_id"] for row in rows] == [1, 2, 3]
    assert rows[0]["passages"][0]["title"] == "Alpha"
    assert all("label" not in row for row in rows)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]),
                      encoding="utf-8")
    retrieve_main(args + ["--resume"])
    resumed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert resumed == rows
    meta = json.loads((tmp_path / "retrieved.jsonl.meta.json").read_text(
        encoding="utf-8"))
    assert meta["complete"] is True
    assert meta["retrieved_rows"] == 3
    with pytest.raises(ValueError, match="settings differ"):
        retrieve_main(args[:-2] + ["--top-k", "1", "--output", str(output),
                                   "--resume"])
    output.write_text('{"claim_id": 1, "passages": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="partial row"):
        retrieve_main(args + ["--resume"])


def test_duplicate_passage_id_fails_without_publishing_index(tmp_path):
    passages = [
        {"doc_id": "same", "title": "Alpha", "text": "First", "chunk_id": 0},
        {"doc_id": "same", "title": "Beta", "text": "Second", "chunk_id": 0},
    ]
    db = tmp_path / "duplicate.sqlite"
    with pytest.raises(Exception, match="UNIQUE constraint"):
        build_index(iter(passages), db, source="fixture", batch_size=1)
    assert not db.exists()
    assert not (tmp_path / "duplicate.sqlite.partial").exists()


def test_fts_index_runs_through_pipeline_retrieval(tmp_path):
    db = tmp_path / "index.sqlite"
    build_index(iter([{"doc_id": "Paris_0", "title": "Paris",
                       "text": "Title: Paris. Paris is in France.", "chunk_id": 0}]),
                db, source="fixture")
    record = AnswerRecord(id="r1", source_dataset="gold", split="test",
                          question="Where is Paris?", answer="Paris is in France.",
                          claims=[Claim(id="r1_c1", text="Paris is in France.",
                                        sentence_id=0)])
    result = retrieve(record, str(db), method="fts5", use_cache=False)
    assert result.claims[0].evidence[0].title == "Paris"
