"""Native baseline adapters. Gold never enters this layer."""
from __future__ import annotations

from dataclasses import asdict
from src.schema import AnswerRecord


def execute(system: dict, record: AnswerRecord, task: str, options: dict) -> dict:
    from src.baselines import run_b0, run_b1, run_b2, run_b3, run_b4, run_b5, run_b6, run_b7
    from src.pipeline import run_pipeline
    from src.retrieve import retrieve_bm25

    name = system["id"]
    common = {"model": options["model"], "provider": options["provider"]}
    native = {}
    evidence = []
    pipeline_options = {
        "llm_model": options["model"], "provider": options["provider"],
        "verifier": options["verifier"], "index_path": options["index_path"],
        "retriever_method": system["retrieval"], "top_k": options["top_k"],
        "faiss_index_path": options.get("faiss_index_path"),
        "passages_path": options.get("passages_path"),
        "repair_policy": options["repair_policy"],
        "query_strategy": options.get("query_strategy", "contextual_v1"),
        "retrieval_max_queries": options.get("retrieval_max_queries", 3),
        "retrieval_max_passages": options.get("retrieval_max_passages", 20),
        "tracking_max_claims": options.get("tracking_max_claims", 64),
        "use_existing_claims": task == "claim_verification",
        "max_repair_iterations": options["max_repair_iterations"] if task == "repair" else 0,
    }
    if name in ("pipeline", "b6"):
        if name == "b6":
            pipeline_options["max_repair_iterations"] = 0
        record = run_pipeline(record, **pipeline_options)
        native["self_verification_scr"] = {
            "before": record.reliability_score_before, "after": record.reliability_score_after}
        if record.verification_diagnostics is not None:
            native["self_verification_scr"]["details"] = record.verification_diagnostics
        if record.claim_tracking is not None:
            native["claim_tracking"] = record.claim_tracking["final"]
        if name == "b6":
            detector_tracking = record.claim_tracking
            record = run_b6(record)
            # These diagnostics describe the detector input, not B6's deleted
            # candidate, which has not been decomposed or reverified.
            record.claim_tracking = None
            record.reliability_score_after = None
            if record.verification_diagnostics:
                record.verification_diagnostics = {
                    "semantics_version": 2,
                    "before": record.verification_diagnostics["before"],
                    "stop_reason": "delete_only_not_reverified",
                }
            native = {"detector_self_verification_scr": record.reliability_score_before,
                      "verification_details": record.verification_diagnostics,
                      "detector_claim_tracking": detector_tracking["final"] if detector_tracking else None}
    elif name == "b0":
        record = run_b0(record)
    elif name == "b1":
        native["answer_detection"] = run_b1(record, **common)
    elif name == "b2":
        record = run_b2(record, n_samples=options["n_samples"],
                        threshold=options["consistency_threshold"], score_report=native, **common)
        native["mean_sentence_consistency"] = record.reliability_score_after
        # Keep native consistency out of fields conventionally used as factual SCR.
        record.reliability_score_after = None
    elif name == "b3":
        record = run_b3(record, **common)
    elif name == "b4":
        record = run_b4(record, **common)
    elif name == "b5":
        passages = retrieve_bm25(f"{record.question} {record.answer}", options["index_path"],
                                 top_k=options["top_k"])
        record = run_b5(record, passages=passages, **common)
        evidence = [asdict(passage) for passage in passages]
    elif name == "b7":
        record = run_b7(record, index_path=options["index_path"], top_k=options["top_k"],
                        revise=task == "repair", **common)
    else:
        raise ValueError(f"Unsupported system: {name}")
    return {"record": record.to_dict(), "native": native, "retrieved_evidence": evidence}
