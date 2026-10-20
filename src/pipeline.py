"""
pipeline.py — Orchestrates all five stages with multi-iteration convergence re-verification.

Phase 7 Implementation:
  - Stage 1: Atomic Claim Decomposition
  - Stage 2: Hybrid Evidence Retrieval
  - Stage 3: Calibrated Multi-Verifier
  - Stage 4: Guarded Localized Repair
  - Stage 5: Iterative Re-Verification Loop (up to max_repair_iterations with early stopping)
  - Reliability delta scoring & batch processing

Public API:
    run_pipeline(record, ...) -> AnswerRecord
    run_pipeline_batch(records, ...) -> List[AnswerRecord]
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
    faiss_index_path: Optional[str] = None,
    passages_path: Optional[str] = None,
    # ── LLM & Verifier ──
    llm_model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    verifier: str = "llm",          # "llm" | "nli" | "ensemble"
    # ── Repair & Loop ──
    repair_policy: str = "rewrite+hedge",
    max_repair_iterations: int = 2,
    # ── Logging ──
    verbose: bool = False,
) -> AnswerRecord:
    """
    Run the full 5-stage detection and repair pipeline with iterative convergence re-verification.

    Returns the mutated AnswerRecord with repaired_answer and reliability deltas populated.
    """
    record.run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )

    def _log(msg: str):
        if verbose:
            print(f"[pipeline:{record.id}] {msg}")

    # ── Initial Pass (Stages 1–3) ──────────────────────────────────────────────
    _log("Initial Pass: Decomposing...")
    record = decompose(record, model=llm_model, provider=provider)
    _log(f"  Extracted {len(record.claims)} claims ({len(record.check_worthy_claims)} check-worthy)")

    _log("Initial Pass: Retrieving evidence...")
    record = retrieve(
        record,
        index_path=index_path,
        method=retriever_method,
        top_k=top_k,
        faiss_index_path=faiss_index_path,
        passages_path=passages_path,
    )

    _log("Initial Pass: Verifying claims...")
    record = verify(record, model=llm_model, provider=provider, verifier=verifier)
    _log(f"  Reliability BEFORE = {record.reliability_score_before:.2f}")

    # ── Iterative Repair & Re-verification Loop (Stages 4 & 5) ─────────────────
    current_answer = record.answer
    total_iterations_run = 0

    for iteration in range(1, max_repair_iterations + 1):
        total_iterations_run = iteration
        hallucinated = record.hallucinated_claims

        if not hallucinated:
            _log(f"Iteration {iteration}: All claims Supported! Early stopping convergence.")
            break

        _log(f"Iteration {iteration}: Repairing {len(hallucinated)} hallucinated claims...")
        record = repair(record, model=llm_model, provider=provider, policy=repair_policy)

        # Check if the repair made any changes
        if not record.repaired_answer or record.repaired_answer == current_answer:
            _log(f"Iteration {iteration}: Repaired answer identical to previous. Convergence reached.")
            break

        current_answer = record.repaired_answer

        # If more iterations remain, re-run verification on new candidate text
        if iteration < max_repair_iterations:
            _log(f"Iteration {iteration}: Re-verifying candidate repair for next iteration...")
            rev_candidate = AnswerRecord(
                id=f"{record.id}_iter{iteration}",
                source_dataset=record.source_dataset,
                split=record.split,
                question=record.question,
                answer=current_answer,
            )
            rev_candidate = decompose(rev_candidate, model=llm_model, provider=provider)
            rev_candidate = retrieve(
                rev_candidate,
                index_path=index_path,
                method=retriever_method,
                top_k=top_k,
                faiss_index_path=faiss_index_path,
                passages_path=passages_path,
            )
            rev_candidate = verify(rev_candidate, model=llm_model, provider=provider, verifier=verifier)

            # Update claims to the newly verified claims
            record.claims = rev_candidate.claims

    # ── Final Re-Verification Pass ─────────────────────────────────────────────
    if record.repaired_answer and record.repaired_answer != record.answer:
        _log("Final Re-verification on converged answer...")
        final_check = AnswerRecord(
            id=f"{record.id}_final",
            source_dataset=record.source_dataset,
            split=record.split,
            question=record.question,
            answer=record.repaired_answer,
        )
        final_check = decompose(final_check, model=llm_model, provider=provider)
        final_check = retrieve(
            final_check,
            index_path=index_path,
            method=retriever_method,
            top_k=top_k,
            faiss_index_path=faiss_index_path,
            passages_path=passages_path,
        )
        final_check = verify(final_check, model=llm_model, provider=provider, verifier=verifier)
        record.reliability_score_after = final_check.reliability_score_before
    else:
        record.repaired_answer = record.answer
        record.reliability_score_after = record.reliability_score_before

    record.repair_iterations = total_iterations_run
    _log(f"Pipeline complete. Iterations: {total_iterations_run} | Reliability: {record.reliability_score_before:.2f} -> {record.reliability_score_after:.2f}")
    return record


def run_pipeline_batch(
    records: List[AnswerRecord],
    output_path: str = "results/pipeline_output.jsonl",
    log_every: int = 5,
    **pipeline_kwargs,
) -> List[AnswerRecord]:
    """Run pipeline on a list of records with streaming JSONL persistence."""
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
                    print(f"  [batch] {i + 1}/{len(records)} - {result.summary()}")
            except Exception as exc:
                print(f"  [batch] ERROR on {record.id}: {exc}")

    print(f"\nBatch complete: {len(results)}/{len(records)} successful -> {output_path}")
    return results
