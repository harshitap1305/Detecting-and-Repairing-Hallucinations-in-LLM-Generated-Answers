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

import copy
import hashlib
import json
import os
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.schema import AnswerRecord, Claim, EvidencePassage
from src.experiments.runtime import strict, recorder


# ─── Module-Level Stores ──────────────────────────────────────────────────────

_bm25_store: Dict[tuple, Tuple] = {}     # {file identity: (bm25_obj, passages)}
_dense_store: Dict[tuple, Tuple] = {}    # {index/passages/encoder identity: loaded artifacts}
_cross_encoder = None                  # Lazy loaded CrossEncoder
_cross_encoder_name = None


def _load_bm25(index_path: str) -> Tuple:
    identity = _index_identity(index_path)
    if identity not in _bm25_store:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        _bm25_store[identity] = (data["bm25"], data["passages"])
    return _bm25_store[identity]


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


def _index_identity(path: Optional[str]) -> tuple | None:
    """Bind cached evidence to the actual on-disk index version."""
    if path is None:
        return None
    file = Path(path).resolve()
    stat = file.stat()
    return str(file), stat.st_size, stat.st_mtime_ns


def _scoped_retrieval_key(method: str, query: str, top_k: int,
                          index_path: str, faiss_index_path: Optional[str],
                          passages_path: Optional[str], encoder_name: str,
                          cross_encoder_name: str, strategy_options: Optional[dict] = None) -> str:
    payload = json.dumps({
        "method": method, "query": query.strip().casefold(), "top_k": top_k,
        "strategy_options": strategy_options,
        "index": _index_identity(index_path),
        "faiss_index": _index_identity(faiss_index_path),
        "passages": _index_identity(passages_path),
        "encoder": encoder_name if method.startswith(("dense", "hybrid")) else None,
        "cross_encoder": cross_encoder_name if "+rerank" in method else None,
    }, sort_keys=True)
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


def retrieve_fts5(query: str, index_path: str, top_k: int = 5) -> List[EvidencePassage]:
    """Retrieve from a streaming-built FEVER SQLite FTS5 index."""
    from scripts.build_fever_fts import FtsRetriever

    with FtsRetriever(Path(index_path)) as retriever:
        return retriever.retrieve(query, top_k=top_k)


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

    cache_key = (_index_identity(faiss_index_path), _index_identity(passages_path), encoder_name)
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

    global _cross_encoder, _cross_encoder_name
    if _cross_encoder is None or _cross_encoder_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(model_name)
            _cross_encoder_name = model_name
        except Exception:
            if strict.get():
                raise
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


# ─── Bounded contextual query planning ────────────────────────────────────────

_QUERY_STOP = set("a an the is are was were be been of in on at to for and or by with as it its this that which what who when where how does did do has have had".split())


def validate_retrieval_options(strategy, max_queries, max_passages, top_k):
    if strategy not in ("claim_only", "contextual_v1"):
        raise ValueError("query_strategy must be claim_only or contextual_v1")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if type(max_queries) is not int or not 1 <= max_queries <= 8:
        raise ValueError("retrieval_max_queries must be between 1 and 8")
    if type(max_passages) is not int or not top_k <= max_passages <= 200:
        raise ValueError("retrieval_max_passages must be between top_k and 200")


def _query_terms(text):
    return {word for word in re.findall(r"\w+", text.casefold()) if word not in _QUERY_STOP}


def _context_reason(claim, candidates, top_k):
    # Surface evidence-gap signals are query planning, not factual judgments.
    if claim.label == "Unverifiable":
        return "previously_unverifiable"
    if len(claim.text.split()) <= 5:
        return "short_claim"
    if len(candidates) < top_k:
        return "few_initial_passages"
    evidence_terms = _query_terms(" ".join(p.title + " " + p.text for p in candidates))
    if _query_terms(claim.text) - evidence_terms:
        return "uncovered_claim_terms"
    return None


def _linked_entities(passages, already_known, context=""):
    """Deterministic entity hypotheses drawn only from inspected passages.

    A textual mention is a retrieval lead, not proof of a relationship. Support
    wiki-style links and proper-name phrases; keep the exact originating span.
    """
    seen = set()
    for passage in passages:
        patterns = (r"\[\[([^]\n|]+)(?:\|[^]\n]+)?\]\]",
                    r"\b([A-Z][\w'-]+(?:[ ]+(?:[A-Z][\w'-]+|of|the|and)){0,4})")
        matches = sorted(((priority, m) for priority, pattern in enumerate(patterns)
                          for m in re.finditer(pattern, passage.text)),
                         key=lambda pair: (pair[0], pair[1].start(), -len(pair[1].group(1))))
        for _, match in matches:
            entity = match.group(1).strip()
            normalized = entity.casefold().replace('_', ' ')
            if (len(entity) < 3 or normalized in seen or normalized in already_known
                    or normalized in _QUERY_STOP
                    or re.search(r"\b" + re.escape(normalized) + r"\b", context.casefold())):
                continue
            seen.add(normalized)
            yield {"entity": entity, "source_doc_id": passage.doc_id,
                   "source_span": [match.start(1), match.end(1)]}


def _fetch_query(query, index_path, method, fetch_k, faiss_index_path, passages_path, encoder_name):
    if method == "bm25":
        return retrieve_bm25(query, index_path, top_k=fetch_k)
    if method == "fts5":
        return retrieve_fts5(query, index_path, top_k=fetch_k)
    if method == "dense":
        if not faiss_index_path or not passages_path:
            raise ValueError("dense retrieval requires faiss_index_path and passages_path")
        return retrieve_dense(query, faiss_index_path, passages_path, top_k=fetch_k,
                              encoder_name=encoder_name)
    if method == "hybrid":
        if not faiss_index_path or not passages_path:
            raise ValueError("hybrid retrieval requires faiss_index_path and passages_path")
        return retrieve_hybrid(query, index_path, faiss_index_path, passages_path,
                               top_k=fetch_k, encoder_name=encoder_name)
    raise ValueError(f"Unknown retrieval method: {method!r}")


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
    query_strategy: str = "claim_only",
    max_queries: int = 3,
    max_passages: int = 20,
    continue_unverifiable: bool = False,
) -> AnswerRecord:
    """Retrieve fixed claim targets with bounded, traced context/link fallbacks.

    Caps apply per claim per call, counting returned passages before deduplication.
    Linked queries have one hop from initial/context passages; no recursive crawl.
    Scores across queries are fused by rank, never by incomparable raw scores.
    """
    validate_retrieval_options(query_strategy, max_queries, max_passages, top_k)
    rerank = "+rerank" in method
    base_method = method.replace("+rerank", "")
    if base_method not in ("bm25", "fts5", "dense", "hybrid"):
        raise ValueError(f"Unknown retrieval method: {method!r}")
    settings = {"version": 1, "strategy": query_strategy, "max_queries": max_queries,
                "max_passages": max_passages, "top_k": top_k}
    scope = json.dumps({"method": method, "index": _index_identity(index_path),
                        "faiss": _index_identity(faiss_index_path),
                        "passages": _index_identity(passages_path), "question": record.question,
                        "encoder": encoder_name, "reranker": cross_encoder_name}, sort_keys=True) if record.check_worthy_claims else "no_targets"
    settings["scope_sha256"] = hashlib.sha256(scope.encode()).hexdigest()
    previous = record.retrieval_diagnostics if continue_unverifiable else None
    if continue_unverifiable and (not previous or any(previous.get(k) != v for k, v in settings.items())):
        raise ValueError("Cannot continue retrieval with a changed strategy, budget, question or index")
    previous_claims = {d["claim_id"]: d for d in previous["claims"]} if previous else {}
    diagnostics = {**settings, "claims": []}
    for claim in record.check_worthy_claims:
        old = previous_claims.get(claim.id)
        if old and old["target"] != claim.text:
            raise ValueError("Cannot continue retrieval for a changed claim target")
        if old and claim.label != "Unverifiable":
            diagnostics["claims"].append(copy.deepcopy(old))
            continue
        query_log = copy.deepcopy(old["queries"]) if old else []
        rankings = [[EvidencePassage(**p) for p in q["passages"]] for q in query_log]
        seen_queries = {q["query"].casefold() for q in query_log}
        fetched = sum(len(rows) for rows in rankings)
        fetch_k = min(top_k * 4 if rerank else top_k, max_passages)

        def run_query(query, kind, link=None):
            nonlocal fetched
            query = " ".join(query.split())
            if (not query or query.casefold() in seen_queries or len(query_log) >= max_queries
                    or fetched >= max_passages):
                return
            seen_queries.add(query.casefold())
            remaining = max_passages - fetched
            if query_strategy == "contextual_v1":
                # Reserve a share for each remaining query, redistributing any
                # unused returned-hit allowance without losing integer remainder.
                slots = max_queries - len(query_log)
                remaining = (remaining + slots - 1) // slots
            limit = min(fetch_k, remaining)
            key = _scoped_retrieval_key(
                method, query, limit, index_path, faiss_index_path, passages_path,
                encoder_name, cross_encoder_name, {**settings, "question": record.question},
            )
            rows = _cache_get_passages(key) if use_cache else None
            cache_hit = rows is not None
            if rows is None:
                rows = _fetch_query(query, index_path, base_method, limit,
                                    faiss_index_path, passages_path, encoder_name)[:limit]
                if use_cache:
                    _cache_set_passages(key, rows)
            rows = copy.deepcopy(rows[:limit])
            fetched += len(rows)
            rankings.append(rows)
            query_log.append({"query": query, "kind": kind, "link": link,
                              "cache_hit": cache_hit, "limit": limit,
                              "passages": [p.__dict__.copy() for p in rows]})

        if not query_log:
            run_query(claim.text, "claim")
        trigger = _context_reason(claim, rankings[0] if rankings else [], top_k)
        if query_strategy == "contextual_v1" and trigger:
            run_query(f"{record.question} {claim.text}", "question_context")
            seeds = [p for q, rows in zip(query_log, rankings)
                     if q["kind"] != "linked_entity" for p in rows]
            known = {p.title.casefold().replace('_', ' ') for p in seeds}
            for lead in _linked_entities(seeds, known, record.question + " " + claim.text):
                # Attribute terms remain from the fixed target/question. An
                # entity mention is not asserted as a verified bridge fact.
                run_query(f"{lead['entity']} {claim.text} {record.question}", "linked_entity", lead)
                if len(query_log) >= max_queries or fetched >= max_passages:
                    break
        scores, passages = {}, {}
        for rows in rankings:
            local_seen = set()
            for rank, passage in enumerate(rows, 1):
                key = (passage.doc_id, passage.chunk_id)
                if key in local_seen:
                    continue
                local_seen.add(key)
                scores[key] = scores.get(key, 0.) + 1. / (60 + rank)
                passages.setdefault(key, passage)
        ordered = sorted(passages, key=lambda key: (-scores[key], key))
        candidates = [passages[key] for key in ordered]
        if len(rankings) > 1:
            for key in ordered:
                passages[key].score = scores[key]
        claim.evidence = (rerank_cross_encoder(claim.text, candidates, top_k, cross_encoder_name)
                          if rerank else candidates[:top_k])
        diagnostics['claims'].append({"claim_id": claim.id, "target": claim.text,
            "trigger": trigger if query_strategy == "contextual_v1" else None,
            "queries": query_log, "fetched_passages": fetched, "unique_candidates": len(candidates),
            "selected": [{"doc_id": p.doc_id, "chunk_id": p.chunk_id} for p in claim.evidence]})
    record.retrieval_diagnostics = diagnostics
    callback = recorder.get()
    if callback:
        callback({"event": "retrieval", "record_id": record.id, "diagnostics": diagnostics})
    return record
