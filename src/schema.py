"""
schema.py — Canonical dataclasses for the hallucination detection & repair pipeline.

All modules import from here. Never define a competing record format elsewhere.

Record hierarchy:
  AnswerRecord
    └── Claim[]
          └── EvidencePassage[]
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


VerdictLabel = Literal["Supported", "Contradicted", "Unverifiable"]


@dataclass
class EvidencePassage:
    """One retrieved passage supporting or contradicting a claim."""
    doc_id: str        # "{title}_{chunk_id}"
    title: str
    text: str
    score: float = 0.0   # retrieval score (higher = more relevant)
    chunk_id: int = 0


@dataclass
class Claim:
    """
    One atomic, decontextualized, check-worthy claim extracted from an answer.
    - 'id' is globally unique: "{record_id}_c{n}"
    - 'sentence_id' is the 0-indexed sentence this claim came from in the original answer
    - 'span' is optional [start, end] char offsets in the original answer
    """
    id: str
    text: str
    sentence_id: int
    span: Optional[List[int]] = None
    check_worthy: bool = True
    label: Optional[VerdictLabel] = None
    label_rationale: str = ""
    evidence: List[EvidencePassage] = field(default_factory=list)
    repaired_text: Optional[str] = None  # corrected claim text after repair


@dataclass
class AnswerRecord:
    """
    The unified record used throughout the pipeline.
    One record = one (question, answer) pair with all intermediate & final outputs.

    JSON schema (written to processed/*.jsonl):
    {
        "id": "felm_wk_0012",
        "source_dataset": "felm",
        "question": "...",
        "answer": "...",
        "split": "dev",
        "claims": [{"id": "...", "text": "...", "sentence_id": 0, ...}],
        "repaired_answer": null,
        "gold_repaired_answer": null,
        "reliability_score_before": null,
        "reliability_score_after": null,
        "repair_iterations": 0,
        "run_id": null
    }
    """
    id: str
    source_dataset: str       # "felm" | "halueval" | "wikibio" | "factcheck_bench" | "gold"
    question: str
    answer: str               # original LLM-generated answer
    split: Literal["dev", "test"] = "dev"
    claims: List[Claim] = field(default_factory=list)
    repaired_answer: Optional[str] = None
    gold_repaired_answer: Optional[str] = None  # human gold (if available)
    reliability_score_before: Optional[float] = None  # fraction of claims Supported, before repair
    reliability_score_after: Optional[float] = None   # same, after repair
    repair_iterations: int = 0
    run_id: Optional[str] = None

    # ── serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnswerRecord":
        d = dict(d)  # shallow copy to avoid mutating caller's dict
        raw_claims = d.pop("claims", [])
        claims = []
        for c in raw_claims:
            c = dict(c)
            raw_evidence = c.pop("evidence", [])
            evidence = [EvidencePassage(**e) for e in raw_evidence]
            claims.append(Claim(**c, evidence=evidence))
        return cls(**d, claims=claims)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "AnswerRecord":
        return cls.from_dict(json.loads(line))

    # ── convenience ────────────────────────────────────────────────────────────

    @property
    def check_worthy_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.check_worthy]

    @property
    def hallucinated_claims(self) -> List[Claim]:
        return [c for c in self.claims
                if c.check_worthy and c.label in ("Contradicted", "Unverifiable")]

    def summary(self) -> str:
        n_cw = len(self.check_worthy_claims)
        n_sup = sum(1 for c in self.check_worthy_claims if c.label == "Supported")
        n_con = sum(1 for c in self.check_worthy_claims if c.label == "Contradicted")
        n_unv = sum(1 for c in self.check_worthy_claims if c.label == "Unverifiable")
        return (
            f"Record {self.id} | {n_cw} check-worthy claims | "
            f"S={n_sup} C={n_con} U={n_unv} | "
            f"Reliability before={self.reliability_score_before} after={self.reliability_score_after}"
        )
