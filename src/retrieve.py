"""
retrieve.py — Stage 2: Evidence retrieval & reranking over Wikipedia passages.

Phase 5 Implementation:
  - Multi-query generation (raw claim, entity/keyword query, question form)
  - BM25 retrieval over pre-built inverted index
  - Dense retrieval over FAISS inner-product index with BGE embeddings
  - Hybrid retrieval using Reciprocal Rank Fusion (RRF, k=60)
  - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  - Transparent retrieval caching for reproducible, fast execution

Public API:
    generate_query_variants(claim_text: str) -> List[str]
    retrieve_bm25(query: str, index_path: str, top_k: int) -> List[EvidencePassage]
    retrieve_dense(query: str, index_path: str, passages_path: str, top_k: int) -> List[EvidencePassage]
    retrieve_hybrid(query: str, bm25_path: str, faiss_path: str, passages_path: str, top_k: int) -> List[EvidencePassage]
    rerank_cross_encoder(query: str, candidates: List[EvidencePassage], top_k: int) -> List[EvidencePassage]
    retrieve(record: AnswerRecord, index_path: str, method: str, top_k: int, ...) -> AnswerRecord
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.schema import AnswerRecord, Claim, EvidencePassage


# ─── Module-Level Stores ──────────────────────────────────────────────────────

_bm25_store: Dict[str, Tuple] = {}     # {path: (bm25_obj, passages)}
_dense_store: Dict[str, Tuple] = {}    # {path: (encoder, faiss_index, passages)}
_cross_encoder = None                  # Lazy loaded CrossEncoder


def _load_bm25(index_path: str) -> Tuple:
    if index_path not in _bm25_store:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        _bm25_store[index_path] = (data["bm25"], data["passages"])
    return _bm25_store[index_path]


# ─── Retrieval Cache ──────────────────────────────────────────────────────────

def _cache_path() -> str:
    return os.environ.get("RETRIEVAL_CACHE_PATH", "retrieval_cache.db")


def _init_retrieval_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS retrieval_cache "
        "(key TEXT PRIMARY KEY, passages_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.commit()
    return conn


_retrieval_conn: Optional[sqlite3.Connection] = None


def _get_retrieval_conn() -> sqlite3.Connection:
    global _retrieval_conn
    if _retrieval_conn is None:
        _retrieval_conn = _init_retrieval_db(_cache_path())
    return _retrieval_conn


def _make_retrieval_key(method: str, query: str, top_k: int, rerank: bool) -> str:
    payload = f"{method}:{query.strip().lower()}:{top_k}:{rerank}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get_passages(key: str) -> Optional[List[EvidencePassage]]:
    try:
        row = _get_retrieval_conn().execute(
            "SELECT passages_json FROM retrieval_cache WHERE key=?", (key,)
        ).fetchone()
        if row:
            raw_list = json.loads(row[0])
            return [EvidencePassage(**item) for item in raw_list]
    except Exception:
        pass
    return None


def _cache_set_passages(key: str, passages: List[EvidencePassage]) -> None:
    try:
        raw_list = [p.__dict__ for p in passages]
        _get_retrieval_conn().execute(
            "INSERT OR REPLACE INTO retrieval_cache (key, passages_json) VALUES (?, ?)",
            (key, json.dumps(raw_list)),
        )
        _get_retrieval_conn().commit()
    except Exception:
        pass


# ─── Query Formulation ────────────────────────────────────────────────────────

def generate_query_variants(claim_text: str) -> List[str]:
    """
    Generate multiple query representations for a claim:
      1. Raw claim text
      2. Keywords / Entity query (nouns, numbers, capitalized terms)
      3. Pseudo-question form
    """
    variants = [claim_text.strip()]

    # Extract capitalized tokens, numbers, and key terms
    tokens = re.findall(r'\b[A-Z][a-z0-9_]*\b|\b\d{4}\b|\b\d+\b', claim_text)
    if tokens:
        keyword_query = " ".join(dict.fromkeys(tokens))
        if keyword_query.lower() != claim_text.lower():
            variants.append(keyword_query)

    # Question variant (simple syntactic inversion)
    # E.g. "Albert Einstein was born in 1879" -> "When was Albert Einstein born?"
    m_born = re.search(r'([A-Z][\w\s]+?)\s+was born in\s+(\d{4}|[\w\s]+)', claim_text)
    if m_born:
        variants.append(f"When and where was {m_born.group(1).strip()} born?")

    return variants


# ─── BM25 Retrieval ───────────────────────────────────────────────────────────

def retrieve_bm25(
    query: str,
    index_path: str,
    top_k: int = 5,
) -> List[EvidencePassage]:
    """Retrieve top-k passages using BM25 index."""
    bm25, passages = _load_bm25(index_path)
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        p = passages[int(idx)]
        results.append(EvidencePassage(
            doc_id=p["doc_id"],
            title=p["title"],
            text=p["text"],
            score=float(scores[idx]),
            chunk_id=int(p.get("chunk_id", 0)),
        ))
    return results


# ─── Dense Retrieval ──────────────────────────────────────────────────────────

def retrieve_dense(
    query: str,
    faiss_index_path: str,
    passages_path: str,
    top_k: int = 5,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
) -> List[EvidencePassage]:
    """Retrieve top-k passages using FAISS dense index with normalized embeddings."""
    import numpy as np

    cache_key = faiss_index_path
    if cache_key not in _dense_store:
        import faiss
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(encoder_name)
        index = faiss.read_index(faiss_index_path)
        passages = []
        with open(passages_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    passages.append(json.loads(line))
        _dense_store[cache_key] = (model, index, passages)

    model, index, passages = _dense_store[cache_key]
    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(passages):
            continue
        p = passages[idx]
        results.append(EvidencePassage(
            doc_id=p["doc_id"],
            title=p["title"],
            text=p["text"],
            score=float(score),
            chunk_id=int(p.get("chunk_id", 0)),
        ))
    return results


# ─── Hybrid Fusion (RRF) ──────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    bm25_results: List[EvidencePassage],
    dense_results: List[EvidencePassage],
    k: int = 60,
) -> List[EvidencePassage]:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
        RRF_Score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}
    passages_by_id: Dict[str, EvidencePassage] = {}

    for rank, ep in enumerate(bm25_results, 1):
        scores[ep.doc_id] = scores.get(ep.doc_id, 0.0) + 1.0 / (k + rank)
        passages_by_id[ep.doc_id] = ep

    for rank, ep in enumerate(dense_results, 1):
        scores[ep.doc_id] = scores.get(ep.doc_id, 0.0) + 1.0 / (k + rank)
        passages_by_id.setdefault(ep.doc_id, ep)

    sorted_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    results = []
    for doc_id in sorted_ids:
        ep = passages_by_id[doc_id]
        ep.score = scores[doc_id]
        results.append(ep)
    return results


def retrieve_hybrid(
    query: str,
    bm25_index_path: str,
    faiss_index_path: str,
    passages_path: str,
    top_k: int = 5,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
) -> List[EvidencePassage]:
    """Combine BM25 and Dense retrieval via Reciprocal Rank Fusion."""
    bm25_candidates = retrieve_bm25(query, bm25_index_path, top_k=top_k * 3)
    dense_candidates = retrieve_dense(
        query, faiss_index_path, passages_path,
        top_k=top_k * 3, encoder_name=encoder_name
    )
    fused = _reciprocal_rank_fusion(bm25_candidates, dense_candidates)
    return fused[:top_k]


# ─── Cross-Encoder Reranking ──────────────────────────────────────────────────

def rerank_cross_encoder(
    query: str,
    candidates: List[EvidencePassage],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[EvidencePassage]:
    """
    Rerank candidate passages using a cross-encoder model.
    Falls back to original ranking if cross-encoder is unavailable.
    """
    if not candidates:
        return []

    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(model_name)
        except Exception:
            _cross_encoder = None

    if _cross_encoder is None:
        # Fallback: maintain candidate ranking
        return candidates[:top_k]

    pairs = [[query, c.text] for c in candidates]
    scores = _cross_encoder.predict(pairs)

    scored_candidates = []
    for ep, score in zip(candidates, scores):
        ep.score = float(score)
        scored_candidates.append(ep)

    scored_candidates.sort(key=lambda x: x.score, reverse=True)
    return scored_candidates[:top_k]


# ─── Public API ───────────────────────────────────────────────────────────────

def retrieve(
    record: AnswerRecord,
    index_path: str,
    method: str = "bm25",
    top_k: int = 5,
    faiss_index_path: Optional[str] = None,
    passages_path: Optional[str] = None,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_cache: bool = True,
) -> AnswerRecord:
    """
    Retrieve top-k evidence passages for each check-worthy claim in an AnswerRecord.

    Supported methods:
      - "bm25"
      - "dense"
      - "hybrid"
      - "bm25+rerank"
      - "hybrid+rerank"
    """
    rerank = "rerank" in method
    base_method = method.replace("+rerank", "")

    for claim in record.claims:
        if not claim.check_worthy:
            continue

        cache_key = _make_retrieval_key(method, claim.text, top_k, rerank)
        if use_cache:
            cached = _cache_get_passages(cache_key)
            if cached is not None:
                claim.evidence = cached
                continue

        # Candidate retrieval
        fetch_k = top_k * 4 if rerank else top_k

        if base_method == "bm25":
            candidates = retrieve_bm25(claim.text, index_path, top_k=fetch_k)
        elif base_method == "dense":
            assert faiss_index_path and passages_path, "dense retrieval requires faiss_index_path and passages_path"
            candidates = retrieve_dense(
                claim.text, faiss_index_path, passages_path,
                top_k=fetch_k, encoder_name=encoder_name
            )
        elif base_method == "hybrid":
            assert faiss_index_path and passages_path, "hybrid retrieval requires faiss_index_path and passages_path"
            candidates = retrieve_hybrid(
                claim.text, index_path, faiss_index_path, passages_path,
                top_k=fetch_k, encoder_name=encoder_name
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")

        # Optional Reranking
        if rerank:
            passages = rerank_cross_encoder(
                claim.text, candidates, top_k=top_k, model_name=cross_encoder_name
            )
        else:
            passages = candidates[:top_k]

        claim.evidence = passages
        if use_cache:
            _cache_set_passages(cache_key, passages)

    return record
