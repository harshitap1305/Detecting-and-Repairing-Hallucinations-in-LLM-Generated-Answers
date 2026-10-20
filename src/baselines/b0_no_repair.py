"""
Baseline B0: No repair (lower bound).
Returns the original answer untouched.
"""
from src.schema import AnswerRecord

def run_b0(record: AnswerRecord) -> AnswerRecord:
    record.repaired_answer = record.answer
    record.repair_iterations = 0
    return record

