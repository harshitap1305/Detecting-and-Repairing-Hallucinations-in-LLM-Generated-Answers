"""
retrieve.py — Stage 2: Retrieve top-k evidence passages for each claim.

Phase 3 (skeleton): BM25 retrieval from a pre-built rank-bm25 index.
Phase 5 (full):     Dense retrieval, hybrid (RRF), cross-encoder reranking.

Index format (saved by scripts/build_index.py):
    pickle file containing {"bm25": BM25Okapi, "passages": List[dict]}
    Each passage dict: {"doc_id", "title", "chunk_id", "text"}

Public API:
    retrieve(record, index_path, method, top_k) -> AnswerRecord
    retrieve_bm25(query, index_path, top_k)     -> List[EvidencePassage]
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.schema import AnswerRecord, EvidencePassage


# ─── Index store (load once per process) ──────────────────────────────────────

_bm25_store: Dict[str, Tuple] = {}   # {path: (BM25Okapi, passages)}
_dense_store: Dict[str, Tuple] = {}  # {path: (SentenceTransformer, np.ndarray, passages)}


def _load_bm25(index_path: str) -> Tuple:
    if index_path not in _bm25_store:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        _bm25_store[index_path] = (data["bm25"], data["passages"])
    return _bm25_store[index_path]


# ─── BM25 retrieval ───────────────────────────────────────────────────────────

def retrieve_bm25(
    query: str,
    index_path: str,
    top_k: int = 5,
) -> List[EvidencePassage]:
    """BM25 retrieval using a pre-built rank-bm25 index."""
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


# ─── Dense retrieval (Phase 5) ────────────────────────────────────────────────

def retrieve_dense(
    query: str,
    index_path: str,       # path to FAISS .bin file
    passages_path: str,    # path to passages .jsonl
    top_k: int = 5,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
) -> List[EvidencePassage]:
    """
    Dense retrieval using FAISS + sentence-transformers encoder.
    Phase 5 implementation — skeleton here so imports don't break.
    """
    import json
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    cache_key = index_path
    if cache_key not in _dense_store:
        model = SentenceTransformer(encoder_name)
        index = faiss.read_index(index_path)
        passages = []
        with open(passages_path) as f:
            for line in f:
                passages.append(json.loads(line))
        _dense_store[cache_key] = (model, index, passages)

    model, index, passages = _dense_store[cache_key]
    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        p = passages[idx]
        results.append(EvidencePassage(
            doc_id=p["doc_id"], title=p["title"],
            text=p["text"], score=float(score),
            chunk_id=int(p.get("chunk_id", 0)),
        ))
    return results


# ─── Hybrid retrieval (Phase 5) ───────────────────────────────────────────────

def _reciprocal_rank_fusion(
    bm25_results: List[EvidencePassage],
    dense_results: List[EvidencePassage],
    k: int = 60,
) -> List[EvidencePassage]:
    """Combine two ranked lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    passages_by_id: Dict[str, EvidencePassage] = {}

    for rank, ep in enumerate(bm25_results, 1):
        scores[ep.doc_id] = scores.get(ep.doc_id, 0) + 1 / (k + rank)
        passages_by_id[ep.doc_id] = ep

    for rank, ep in enumerate(dense_results, 1):
        scores[ep.doc_id] = scores.get(ep.doc_id, 0) + 1 / (k + rank)
        passages_by_id.setdefault(ep.doc_id, ep)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
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
    """Hybrid retrieval: BM25 + dense, combined with RRF. Phase 5."""
    bm25_results = retrieve_bm25(query, bm25_index_path, top_k=top_k * 2)
    dense_results = retrieve_dense(query, faiss_index_path, passages_path,
                                   top_k=top_k * 2, encoder_name=encoder_name)
    fused = _reciprocal_rank_fusion(bm25_results, dense_results)
    return fused[:top_k]


# ─── Public API ───────────────────────────────────────────────────────────────

def retrieve(
    record: AnswerRecord,
    index_path: str,
    method: str = "bm25",
    top_k: int = 5,
    # Dense / hybrid kwargs (Phase 5)
    faiss_index_path: Optional[str] = None,
    passages_path: Optional[str] = None,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
) -> AnswerRecord:
    """
    Retrieve evidence passages for each check-worthy claim in record.
    Mutates record.claims[*].evidence in place and returns record.

    Args:
        record:          AnswerRecord with claims already populated.
        index_path:      Path to BM25 .pkl or FAISS .bin index.
        method:          "bm25" | "dense" | "hybrid"
        top_k:           Number of passages to retrieve per claim.
    """
    for claim in record.claims:
        if not claim.check_worthy:
            continue

        if method == "bm25":
            claim.evidence = retrieve_bm25(claim.text, index_path, top_k=top_k)
        elif method == "dense":
            assert faiss_index_path and passages_path, (
                "dense retrieval requires faiss_index_path and passages_path"
            )
            claim.evidence = retrieve_dense(
                claim.text, faiss_index_path, passages_path,
                top_k=top_k, encoder_name=encoder_name,
            )
        elif method == "hybrid":
            assert faiss_index_path and passages_path, (
                "hybrid retrieval requires faiss_index_path and passages_path"
            )
            claim.evidence = retrieve_hybrid(
                claim.text, index_path, faiss_index_path, passages_path,
                top_k=top_k, encoder_name=encoder_name,
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")

    return record
