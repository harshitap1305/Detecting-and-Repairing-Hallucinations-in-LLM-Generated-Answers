"""
Baseline B6: Delete-only baseline.
Uses our claim-level detector, but instead of repairing, it simply deletes any sentence containing flagged claims.
Answers RQ3 / ablation: rewrite vs. delete.
"""
from src.schema import AnswerRecord
from src.decompose import split_sentences

def run_b6(record: AnswerRecord) -> AnswerRecord:
    flagged_sids = {
        c.sentence_id
        for c in record.claims
        if c.check_worthy and c.label in ("Contradicted", "Unverifiable")
    }
    sentences = split_sentences(record.answer)
    kept_sentences = [
        s for sid, s in enumerate(sentences)
        if sid not in flagged_sids
    ]
    record.repaired_answer = " ".join(kept_sentences)
    record.repair_iterations = 1
    return record

