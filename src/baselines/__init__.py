"""
baselines — Comparison baselines B0 through B7 for hallucination detection and repair.

B0: No repair (lower bound)
B1: Answer-level binary hallucination detection
B2: SelfCheckGPT (sampling internal consistency across N outputs)
B3: Self-repair (LLM repair without external evidence)
B4: Chain-of-Verification (CoVe; verification questions & revision)
B5: RARR-style monolithic retrieval-augmented rewrite
B6: Delete-only (remove sentences containing flagged claims)
B7: Factcheck-GPT (standard published pipeline & direct revision)
"""
from src.baselines.b0_no_repair import run_b0
from src.baselines.b1_answer_level import run_b1
from src.baselines.b2_selfcheck import run_b2
from src.baselines.b3_self_repair import run_b3
from src.baselines.b4_cove import run_b4
from src.baselines.b5_rarr import run_b5
from src.baselines.b6_delete_only import run_b6
from src.baselines.b7_factcheck_gpt import run_b7

__all__ = [
    "run_b0",
    "run_b1",
    "run_b2",
    "run_b3",
    "run_b4",
    "run_b5",
    "run_b6",
    "run_b7",
]
