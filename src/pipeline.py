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
from src.verify import verify, verification_summary
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
    verifier: str = "llm",          # "llm" | "nli" | "ensemble" | "trained"
    trained_verifier_checkpoint: Optional[str] = None,
    use_existing_claims: bool = False,
    # ── Repair & Loop ──
    repair_policy: str = "rewrite+hedge",
    repair_backend: str = "prompted",  # "prompted" | "trained"
    trained_repair_checkpoint: Optional[str] = None,
    max_repair_iterations: int = 2,
    # ── Logging ──
    verbose: bool = False,
) -> AnswerRecord:
    """
    Run the full 5-stage detection and repair pipeline with iterative convergence re-verification.

    Returns the mutated AnswerRecord with repaired_answer and reliability deltas populated.
    """
    if type(max_repair_iterations) is not int or max_repair_iterations < 0:
        raise ValueError("max_repair_iterations must be a nonnegative integer")
    if verifier == "trained" and not trained_verifier_checkpoint:
        raise ValueError("trained_verifier_checkpoint is required for verifier='trained'")
    if repair_backend == "trained" and not trained_repair_checkpoint:
        raise ValueError("trained_repair_checkpoint is required for repair_backend='trained'")

    record.run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )

    def _log(msg: str):
        if verbose:
            print(f"[pipeline:{record.id}] {msg}")

    # ── Initial Pass (Stages 1–3) ──────────────────────────────────────────────
    if use_existing_claims:
        if not record.claims:
            raise ValueError("use_existing_claims requires a nonempty claim inventory")
        _log("Initial Pass: Using supplied claim inventory...")
    else:
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
    record = verify(
        record, model=llm_model, provider=provider, verifier=verifier,
        trained_model_path=trained_verifier_checkpoint,
    )
    before = verification_summary(record)
    record.reliability_score_before = before["score"]
    def score_text(value):
        return "N/A" if value is None else f"{value:.2f}"
    _log(f"  Reliability BEFORE = {score_text(record.reliability_score_before)}")

    # ── Iterative Repair & Re-verification Loop (Stages 4 & 5) ─────────────────
    current_answer = record.answer
    working_record = record
    total_iterations_run = 0
    stop_reason = "verification_only" if max_repair_iterations == 0 else "budget_exhausted"

    for iteration in range(1, max_repair_iterations + 1):
        hallucinated = working_record.hallucinated_claims

        if not working_record.check_worthy_claims:
            stop_reason = "empty_answer" if not current_answer.strip() else "no_check_worthy_claims"
            _log(f"Iteration {iteration}: {stop_reason}; factual verification is unavailable.")
            break
        if not hallucinated:
            stop_reason = ("all_current_claims_supported" if all(
                c.label == "Supported" for c in working_record.check_worthy_claims
            ) else "unresolved_verification")
            _log(f"Iteration {iteration}: {stop_reason}.")
            break

        total_iterations_run = iteration
        _log(f"Iteration {iteration}: Repairing {len(hallucinated)} hallucinated claims...")
        working_record = repair(
            working_record, model=llm_model, provider=provider, policy=repair_policy,
            backend=repair_backend, trained_model_path=trained_repair_checkpoint,
        )

        # Check if the repair made any changes
        if working_record.repaired_answer is None or working_record.repaired_answer == current_answer:
            stop_reason = "unchanged_answer"
            _log(f"Iteration {iteration}: No change; stopping without a correction claim.")
            break

        current_answer = working_record.repaired_answer

        if not current_answer.strip():
            stop_reason = "empty_answer"
            _log(f"Iteration {iteration}: All flagged text removed.")
            break

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
            rev_candidate = verify(
                rev_candidate, model=llm_model, provider=provider, verifier=verifier,
                trained_model_path=trained_verifier_checkpoint,
            )

            # The next repair must use the revised answer and its own sentence
            # IDs. Keep the original record's claims for paired evaluation.
            working_record = rev_candidate

    record.repaired_answer = current_answer

    # ── Final Re-Verification Pass ─────────────────────────────────────────────
    if not record.repaired_answer.strip():
        after = verification_summary(AnswerRecord(
            id=record.id, source_dataset=record.source_dataset,
            question=record.question, answer=record.repaired_answer,
        ))
    elif record.repaired_answer is not None and record.repaired_answer != record.answer:
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
        final_check = verify(
            final_check, model=llm_model, provider=provider, verifier=verifier,
            trained_model_path=trained_verifier_checkpoint,
        )
        after = verification_summary(final_check)
    else:
        record.repaired_answer = record.answer
        after = dict(before)

    record.reliability_score_after = after["score"]
    if after["reason"]:
        stop_reason = after["reason"]
    elif stop_reason == "budget_exhausted" and after["supported"] == after["check_worthy"]:
        stop_reason = "all_current_claims_supported"
    record.verification_diagnostics = {
        "semantics_version": 2, "before": before, "after": after, "stop_reason": stop_reason,
    }
    record.repair_iterations = total_iterations_run
    _log(f"Pipeline complete. Iterations: {total_iterations_run} | Stop: {stop_reason} | "
         f"Reliability: {score_text(record.reliability_score_before)} -> {score_text(record.reliability_score_after)}")
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
