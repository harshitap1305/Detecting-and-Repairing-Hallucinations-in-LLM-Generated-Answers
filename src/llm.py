"""
llm.py — Single LLM access point with transparent SQLite disk cache.

Supports:
  - Groq API  (default; fast & free tier; set GROQ_API_KEY)
  - OpenAI API (set OPENAI_API_KEY)
  - Local HuggingFace model via transformers pipeline

Cache key = SHA-256 of (model, prompt, temperature, max_tokens, seed).
On cache hit → return stored response, no API call, no cost.

Usage:
    from src.llm import call
    text = call("What is the capital of France?")
    text = call("...", model="llama-3.3-70b-versatile", provider="groq")

Colab tip:
    Set env var LLM_CACHE_PATH to a Google Drive path so cache survives
    session disconnects:
        import os
        os.environ["LLM_CACHE_PATH"] = "/content/drive/MyDrive/project/llm_cache.db"
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional


# ─── Cache ────────────────────────────────────────────────────────────────────

def _cache_path() -> str:
    return os.environ.get("LLM_CACHE_PATH", "llm_cache.db")


def _make_key(model: str, prompt: str, temperature: float,
              max_tokens: int, seed: int) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "temperature": temperature,
         "max_tokens": max_tokens, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, model TEXT, response TEXT, created_at TEXT)"
    )
    conn.commit()
    return conn


# Module-level connection (lazy init, one per process)
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    db_path = _cache_path()
    # Reinitialise if path changed (e.g. Colab sets env var after import)
    if _conn is None:
        _conn = _init_db(db_path)
    return _conn


def _cache_get(key: str) -> Optional[str]:
    try:
        row = _get_conn().execute(
            "SELECT response FROM cache WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def _cache_set(key: str, model: str, response: str) -> None:
    try:
        _get_conn().execute(
            "INSERT OR REPLACE INTO cache (key, model, response, created_at) "
            "VALUES (?, ?, ?, ?)",
            (key, model, response, datetime.now(timezone.utc).isoformat()),
        )
        _get_conn().commit()
    except sqlite3.Error as e:
        print(f"[llm cache] Warning: could not save to cache: {e}")


def cache_stats() -> dict:
    """Return cache statistics."""
    try:
        conn = _get_conn()
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        by_model = conn.execute(
            "SELECT model, COUNT(*) FROM cache GROUP BY model"
        ).fetchall()
        return {"total": total, "by_model": dict(by_model)}
    except sqlite3.Error:
        return {"total": 0, "by_model": {}}


# ─── LLM Backends ─────────────────────────────────────────────────────────────

def _call_groq(model: str, prompt: str, temperature: float,
               max_tokens: int, seed: int) -> str:
    """Call Groq API. Requires GROQ_API_KEY env var."""
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    return resp.choices[0].message.content


def _call_openai(model: str, prompt: str, temperature: float,
                 max_tokens: int, seed: int) -> str:
    """Call OpenAI API. Requires OPENAI_API_KEY env var."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    return resp.choices[0].message.content


def _call_local(model: str, prompt: str, temperature: float,
                max_tokens: int) -> str:
    """
    Local HuggingFace model. Loads lazily and caches the pipeline in memory.
    Use only for development; prefer API for final runs.
    """
    import torch
    from transformers import pipeline as hf_pipeline

    if not hasattr(_call_local, "_pipes"):
        _call_local._pipes = {}
    if model not in _call_local._pipes:
        print(f"[llm] Loading local model {model} (first call only)…")
        _call_local._pipes[model] = hf_pipeline(
            "text-generation", model=model,
            torch_dtype=torch.float16, device_map="auto",
        )
    pipe = _call_local._pipes[model]
    out = pipe(
        prompt,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 1e-4),   # pipeline requires > 0
        do_sample=(temperature > 0),
        return_full_text=False,
    )
    return out[0]["generated_text"]


# ─── Public API ───────────────────────────────────────────────────────────────

def call(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 42,
    provider: str = "groq",
    use_cache: bool = True,
) -> str:
    """
    Make an LLM call with transparent disk caching.

    Args:
        prompt:      The user message / full prompt string.
        model:       Model name (must match the provider's accepted names).
        temperature: Sampling temperature. Use 0 for deterministic outputs.
        max_tokens:  Maximum tokens to generate.
        seed:        Random seed for reproducibility.
        provider:    "groq" | "openai" | "local"
        use_cache:   If True, check cache before calling; store result after.

    Returns:
        The model's text response as a plain string.
    """
    key = _make_key(model, prompt, temperature, max_tokens, seed)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    if provider == "groq":
        response = _call_groq(model, prompt, temperature, max_tokens, seed)
    elif provider == "openai":
        response = _call_openai(model, prompt, temperature, max_tokens, seed)
    elif provider == "local":
        response = _call_local(model, prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'groq', 'openai', or 'local'.")

    if use_cache:
        _cache_set(key, model, response)

    return response


def reset_connection() -> None:
    """Force re-initialise the DB connection (call after changing LLM_CACHE_PATH)."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None
