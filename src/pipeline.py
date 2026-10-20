"""
pipeline.py — Orchestrates all five stages end to end.

Usage (local / Colab):
    from src.schema import AnswerRecord
    from src.pipeline import run_pipeline

    record = AnswerRecord(
        id="demo_001", source_dataset="manual", split="dev",
        question="Who invented the telephone?",
        answer="Alexander Graham Bell invented the telephone in 1876 in Boston.",
    )
    result = run_pipeline(record, index_path="indexes/bm25_A.pkl")
    print(result.repaired_answer)
    print(result.summary())
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.schema import AnswerRecord
from src.decompose import decompose
from src.retrieve import retrieve
from src.verify import verify
from src.repair import repair


def run_pipeline(
    record: AnswerRecord,
    # ── retrieval ──
    index_path: str = "indexes/bm25_A.pkl",
    retriever_method: str = "bm25",
    top_k: int = 5,
    # ── LLM ──
    llm_model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    verifier: str = "llm",          # "llm" | "nli"
    # ── repair ──
    repair_policy: str = "rewrite+hedge",
    max_repair_iterations: int = 2,  # Phase 7: loop; Phase 3: single pass
    # ── misc ──
    verbose: bool = False,
) -> AnswerRecord:
    """
    Run the full hallucination detection and repair pipeline on one AnswerRecord.

    Stages:
        1. Decompose  → atomic claims
        2. Retrieve   → top-k evidence passages per claim
        3. Verify     → Supported | Contradicted | Unverifiable
        4. Repair     → minimal edits to flagged sentences
        5. Re-verify  → reliability score on repaired answer

    Returns the mutated AnswerRecord with all fields populated.
    """
    record.run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )

    def _log(msg: str):
        if verbose:
            print(f"[pipeline:{record.id}] {msg}")

    # ── Stage 1: Decompose ─────────────────────────────────────────────────────
    _log("Stage 1 — decomposing…")
    record = decompose(record, model=llm_model, provider=provider)
    _log(f"  → {len(record.claims)} claims "
         f"({len(record.check_worthy_claims)} check-worthy)")

    # ── Stage 2: Retrieve ──────────────────────────────────────────────────────
    _log("Stage 2 — retrieving evidence…")
    record = retrieve(record, index_path=index_path,
                      method=retriever_method, top_k=top_k)
    _log(f"  → evidence retrieved for {len(record.check_worthy_claims)} claims")

    # ── Stage 3: Verify ────────────────────────────────────────────────────────
    _log("Stage 3 — verifying claims…")
    record = verify(record, model=llm_model, provider=provider, verifier=verifier)
    label_counts = {lbl: 0 for lbl in ("Supported", "Contradicted", "Unverifiable")}
    for c in record.check_worthy_claims:
        label_counts[c.label or "Unverifiable"] += 1
    _log(f"  → {label_counts} | reliability={record.reliability_score_before:.2f}")

    # ── Stage 4: Repair ────────────────────────────────────────────────────────
    _log("Stage 4 — repairing flagged claims…")
    record = repair(record, model=llm_model, provider=provider, policy=repair_policy)
    n_flagged = len(record.hallucinated_claims)
    _log(f"  → {n_flagged} flagged claim(s) repaired")

    # ── Stage 5: Re-verify ─────────────────────────────────────────────────────
    # Phase 3: single-pass re-verify on repaired answer (no loop yet)
    # Phase 7: loop up to max_repair_iterations
    _log("Stage 5 — re-verifying repaired answer…")
    if record.repaired_answer and record.repaired_answer != record.answer:
        rev_record = AnswerRecord(
            id=record.id + "_rev",
            source_dataset=record.source_dataset,
            split=record.split,
            question=record.question,
            answer=record.repaired_answer,
        )
        rev_record = decompose(rev_record, model=llm_model, provider=provider)
        rev_record = retrieve(rev_record, index_path=index_path,
                              method=retriever_method, top_k=top_k)
        rev_record = verify(rev_record, model=llm_model, provider=provider,
                            verifier=verifier)
        record.reliability_score_after = rev_record.reliability_score_before
    else:
        record.reliability_score_after = record.reliability_score_before

    _log(
        f"Done. Reliability: {record.reliability_score_before:.2f} → "
        f"{record.reliability_score_after:.2f}"
    )
    return record


def run_pipeline_batch(
    records: List[AnswerRecord],
    output_path: str = "results/pipeline_output.jsonl",
    log_every: int = 5,
    **pipeline_kwargs,
) -> List[AnswerRecord]:
    """
    Run the pipeline on a list of records.
    Writes results to JSONL incrementally (safe against disconnects).

    Args:
        records:      List of AnswerRecord to process.
        output_path:  Where to write JSONL results.
        log_every:    Print progress every N records.
        **pipeline_kwargs: Passed to run_pipeline.

    Returns:
        List of processed AnswerRecord (successful only).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results: List[AnswerRecord] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            try:
                result = run_pipeline(record, **pipeline_kwargs)
                results.append(result)
                f.write(result.to_jsonl_line() + "\n")
                f.flush()
                if (i + 1) % log_every == 0:
                    print(f"  ✓ {i + 1}/{len(records)} — {result.summary()}")
            except Exception as exc:
                print(f"  ✗ ERROR on {record.id}: {exc}")

    print(f"\nBatch complete: {len(results)}/{len(records)} successful → {output_path}")
    return results
