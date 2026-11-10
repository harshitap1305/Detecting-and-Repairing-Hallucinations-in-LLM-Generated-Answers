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

import copy
import json
from dataclasses import asdict
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.schema import AnswerRecord
from src.decompose import decompose
from src.retrieve import retrieve, validate_retrieval_options
from src.verify import verify, verification_summary
from src.repair import repair
from src.claim_tracking import track_claims
from src.extraction_check import extraction_coverage


def run_pipeline(
    record: AnswerRecord,
    # ── retrieval ──
    index_path: str = "indexes/bm25_A.pkl",
    retriever_method: str = "bm25",
    top_k: int = 5,
    faiss_index_path: Optional[str] = None,
    passages_path: Optional[str] = None,
    query_strategy: str = "contextual_v1",
    retrieval_max_queries: int = 3,
    retrieval_max_passages: int = 20,
    tracking_max_claims: int = 64,
    extraction_check: str = "semantic_v1",
    extraction_max_attempts: int = 2,
    extraction_max_checks: int = 2,
    extraction_max_claims: int = 64,
    # ── LLM & Verifier ──
    llm_model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    verifier: str = "llm",          # "llm" | "nli" | "ensemble" | "trained"
    trained_verifier_checkpoint: Optional[str] = None,
    use_existing_claims: bool = False,
    # ── Repair & Loop ──
    repair_policy: str = "rewrite+hedge",
    repair_guard: str = "grounded_v1",
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

    if type(tracking_max_claims) is not int or not 1 <= tracking_max_claims <= 512:
        raise ValueError("tracking_max_claims must be between 1 and 512")
    validate_retrieval_options(query_strategy, retrieval_max_queries, retrieval_max_passages, top_k)
    if extraction_check not in ("structural", "semantic_v1"):
        raise ValueError("extraction_check must be structural or semantic_v1")
    for name, value, maximum in (("extraction_max_attempts", extraction_max_attempts, 4),
                                 ("extraction_max_checks", extraction_max_checks, 4),
                                 ("extraction_max_claims", extraction_max_claims, 128)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
    extraction_options = dict(extraction_check=extraction_check, max_retries=extraction_max_attempts,
                              max_checks=extraction_max_checks, max_check_claims=extraction_max_claims)
    retrieval_options = dict(
        index_path=index_path, method=retriever_method, top_k=top_k,
        faiss_index_path=faiss_index_path, passages_path=passages_path,
        query_strategy=query_strategy, max_queries=retrieval_max_queries,
        max_passages=retrieval_max_passages,
    )
    record.run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )

    def _log(msg: str):
        if verbose:
            print(f"[pipeline:{record.id}] {msg}")

    def ground(candidate):
        candidate = retrieve(candidate, **retrieval_options)
        candidate = verify(
            candidate, model=llm_model, provider=provider, verifier=verifier,
            trained_model_path=trained_verifier_checkpoint,
        )
        diagnostics = candidate.retrieval_diagnostics
        if query_strategy == "contextual_v1" and diagnostics:
            previous = {d["claim_id"]: len(d["queries"]) for d in diagnostics["claims"]}
            if any(c.label == "Unverifiable" and any(
                    d["claim_id"] == c.id and len(d["queries"]) < retrieval_max_queries
                    and d["fetched_passages"] < retrieval_max_passages
                    for d in diagnostics["claims"]) for c in candidate.check_worthy_claims):
                candidate = retrieve(candidate, **retrieval_options, continue_unverifiable=True)
                expanded = {d["claim_id"] for d in candidate.retrieval_diagnostics["claims"]
                            if len(d["queries"]) > previous.get(d["claim_id"], 0)}
                if expanded:
                    # Only expanded targets get a second verdict, at most once
                    # per grounding stage; do not change the claim inventory.
                    subset = AnswerRecord(id=candidate.id, source_dataset=candidate.source_dataset,
                                          question=candidate.question, answer=candidate.answer,
                                          claims=[c for c in candidate.claims if c.id in expanded])
                    verify(subset, model=llm_model, provider=provider, verifier=verifier,
                           trained_model_path=trained_verifier_checkpoint)
                    summary = verification_summary(candidate)
                    candidate.reliability_score_before = summary["score"]
                    candidate.verification_diagnostics = {"semantics_version": 2, "before": summary}
        return candidate

    # ── Initial Pass (Stages 1–3) ──────────────────────────────────────────────
    if use_existing_claims:
        if not record.claims:
            raise ValueError("use_existing_claims requires a nonempty claim inventory")
        record.decomposition_diagnostics = {"semantics_version": 4, "status": "supplied_inventory",
                                            "reason": "supplied_inventory_not_extraction_assessed"}
        _log("Initial Pass: Using supplied claim inventory...")
    else:
        _log("Initial Pass: Decomposing...")
        record = decompose(record, model=llm_model, provider=provider, **extraction_options)
    _log(f"  Extracted {len(record.claims)} claims ({len(record.check_worthy_claims)} check-worthy)")

    _log("Initial Pass: Retrieving and verifying evidence...")
    record = ground(record)
    before = verification_summary(record)
    record.reliability_score_before = before["score"]
    def score_text(value):
        return "N/A" if value is None else f"{value:.2f}"
    _log(f"  Reliability BEFORE = {score_text(record.reliability_score_before)}")

    # ── Iterative Repair & Re-verification Loop (Stages 4 & 5) ─────────────────
    current_answer = record.answer
    original = copy.deepcopy(record)
    before_extraction = extraction_coverage(original)
    repair_history = []
    working_record = copy.deepcopy(record)
    history = []

    def snapshot(candidate, stage):
        tracking = track_claims(original, candidate, model=llm_model, provider=provider,
                                max_claims=tracking_max_claims)
        candidate_extraction = extraction_coverage(candidate)
        tracking["target_verification_success"] = tracking["factual_success"]
        tracking["extraction_coverage"] = {"original": before_extraction, "candidate": candidate_extraction}
        tracking["factual_success"] = bool(tracking["factual_success"]
            and before_extraction["status"] == "passed" and candidate_extraction["status"] == "passed")
        history.append({"stage": stage, "answer": candidate.answer,
                        "claims": [asdict(c) for c in candidate.claims],
                        "decomposition": copy.deepcopy(candidate.decomposition_diagnostics),
                        "verification": verification_summary(candidate),
                        "retrieval": copy.deepcopy(candidate.retrieval_diagnostics),
                        "tracking": tracking})
        return tracking

    current_tracking = snapshot(working_record, "initial")
    total_iterations_run = 0
    stop_reason = "verification_only" if max_repair_iterations == 0 else "budget_exhausted"

    for iteration in range(1, max_repair_iterations + 1):
        hallucinated = working_record.hallucinated_claims

        if (extraction_check == "semantic_v1" and working_record.answer.strip()
                and ((not use_existing_claims and before_extraction["status"] != "passed")
                     or (working_record.answer != original.answer
                         and extraction_coverage(working_record)["status"] != "passed"))):
            stop_reason = "unresolved_extraction_coverage"
            _log(f"Iteration {iteration}: Unresolved extraction; abstaining from repair.")
            break
        if not working_record.check_worthy_claims:
            stop_reason = "empty_answer" if not current_answer.strip() else "no_check_worthy_claims"
            _log(f"Iteration {iteration}: {stop_reason}; factual verification is unavailable.")
            break
        if not hallucinated:
            stop_reason = ("all_targets_verified" if current_tracking["factual_success"]
                           else "unresolved_target_coverage")
            _log(f"Iteration {iteration}: {stop_reason}.")
            break

        total_iterations_run = iteration
        _log(f"Iteration {iteration}: Repairing {len(hallucinated)} hallucinated claims...")
        working_record = repair(
            working_record, model=llm_model, provider=provider, policy=repair_policy,
            backend=repair_backend, trained_model_path=trained_repair_checkpoint, guard_mode=repair_guard,
        )

        repair_history.append(copy.deepcopy(working_record.repair_diagnostics))
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
            rev_candidate = decompose(rev_candidate, model=llm_model, provider=provider, **extraction_options)
            rev_candidate = ground(rev_candidate)

            # The next repair must use the revised answer and its own sentence
            # IDs. Keep the original record's claims for paired evaluation.
            working_record = rev_candidate
            current_tracking = snapshot(working_record, f"iteration-{iteration}")

    record.repaired_answer = current_answer

    # ── Final Re-Verification Pass ─────────────────────────────────────────────
    if not record.repaired_answer.strip():
        final_check = AnswerRecord(
            id=record.id, source_dataset=record.source_dataset,
            question=record.question, answer=record.repaired_answer,
        )
        after = verification_summary(final_check)
        current_tracking = snapshot(final_check, "final")
    elif working_record.answer == current_answer and current_answer != original.answer:
        # This exact candidate was already decomposed/retrieved/verified above.
        after = verification_summary(working_record)
    elif record.repaired_answer is not None and record.repaired_answer != record.answer:
        _log("Final Re-verification on converged answer...")
        final_check = AnswerRecord(
            id=f"{record.id}_final",
            source_dataset=record.source_dataset,
            split=record.split,
            question=record.question,
            answer=record.repaired_answer,
        )
        final_check = decompose(final_check, model=llm_model, provider=provider, **extraction_options)
        final_check = ground(final_check)
        after = verification_summary(final_check)
        current_tracking = snapshot(final_check, "final")
    else:
        record.repaired_answer = record.answer
        after = dict(before)

    record.reliability_score_after = after["score"]
    if after["reason"]:
        stop_reason = after["reason"]
    elif stop_reason == "budget_exhausted" and current_tracking["factual_success"]:
        stop_reason = "all_targets_verified"
    elif (stop_reason in ("budget_exhausted", "all_targets_verified")
          and not current_tracking["factual_success"] and after["supported"] == after["check_worthy"]):
        stop_reason = "unresolved_target_coverage"
    after_extraction = current_tracking["extraction_coverage"]["candidate"]
    if ((original.decomposition_diagnostics or {}).get("status") == "unresolved_coverage"
            or (history[-1].get("decomposition") or {}).get("status") == "unresolved_coverage"
            or (after["check_worthy"] and after["supported"] == after["check_worthy"]
                and (before_extraction["status"] != "passed" or after_extraction["status"] != "passed"))):
        stop_reason = "unresolved_extraction_coverage"
    record.claims = copy.deepcopy(original.claims)
    record.claim_tracking = {"semantics_version": 1,
                             "original_claims": [asdict(c) for c in original.claims],
                             "stages": history, "final": current_tracking}
    record.verification_diagnostics = {
        "semantics_version": 2, "before": before, "after": after, "stop_reason": stop_reason,
        "extraction_coverage": {"before": before_extraction, "after": after_extraction},
    }
    record.repair_diagnostics = {"semantics_version": 1, "guard_mode": repair_guard, "iterations": repair_history}
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
