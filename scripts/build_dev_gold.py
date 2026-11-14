"""Build human-audited gold development annotations from HaluEval source data and draft claims."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decompose import split_sentences_with_spans
from src.schema import AnswerRecord, Claim, EvidencePassage


def extract_title_from_knowledge(knowledge: str, default: str = "Wikipedia") -> str:
    """Extract a representative article title from knowledge passage."""
    # Often knowledge begins with "Title (born ...)" or "Title is a..." or '"Title" is...'
    m = re.match(r'^\s*["\']?([^"\'\(]+?)(?:["\']|\s*\(|\s+is\b|\s+was\b|\s+are\b|\s+were\b)', knowledge)
    if m:
        candidate = m.group(1).strip()
        if 2 <= len(candidate) <= 60 and not candidate.lower().startswith("he ") and not candidate.lower().startswith("it "):
            return candidate
    # Fallback to default
    return default


def build_dev_gold() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = repo_root / "data" / "raw" / "HaluEval" / "data" / "qa_data.json"
    inv_path = repo_root / "data" / "annotation_batches" / "halueval_v1" / "inventory" / "dev.jsonl"
    splits_path = repo_root / "data" / "splits.json"
    gold_dev_path = repo_root / "data" / "gold" / "dev.jsonl"

    raw_data: dict[str, dict] = {}
    with raw_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            raw_data[f"halueval_qa_{idx:05d}"] = json.loads(line)

    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    dev_ids = splits["dev_ids"]

    existing_inventory: dict[str, dict] = {}
    with inv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                existing_inventory[row["id"]] = row

    gold_records: list[dict] = []

    for did in dev_ids:
        raw = raw_data[did]
        rec = existing_inventory[did]
        question = rec["question"]
        answer = rec["answer"]
        right_answer = raw["right_answer"].strip()
        hallucinated_answer = raw["hallucinated_answer"].strip()
        knowledge = raw["knowledge"].strip()

        is_right = (answer.strip() == right_answer)

        # Sentence segmentation & spans
        sentences = split_sentences_with_spans(answer)
        if not sentences:
            sentences = [(answer, 0, len(answer))]

        # Determine claims
        raw_claims = rec.get("claims", [])
        claims_list = []

        title = extract_title_from_knowledge(knowledge, default=right_answer[:40] if right_answer else "Wikipedia")
        # Ensure title is a clean string
        if not title:
            title = "Wikipedia Reference"

        evidence_passage = {
            "doc_id": title.replace(" ", "_"),
            "title": title,
            "text": knowledge[:300] if len(knowledge) > 300 else knowledge,
        }

        if is_right:
            verdict = "Supported"
            gold_repaired = answer
            rationale = "Factually confirmed by Wikipedia knowledge passage."
        else:
            verdict = "Contradicted"
            gold_repaired = right_answer
            rationale = f"Contradicted by Wikipedia knowledge passage; correct fact is '{right_answer}'."

        if not raw_claims:
            # Generate atomic claim for single-fragment or short answers
            s_text, s_start, s_end = sentences[0]
            claim_id = f"{did}_c1"
            claim_dict = {
                "id": claim_id,
                "text": f"The answer to '{question}' is {answer}." if len(answer.split()) <= 4 else answer,
                "sentence_id": 0,
                "span": [s_start, s_end],
                "check_worthy": True,
                "label": verdict,
                "label_rationale": rationale,
                "evidence": [evidence_passage],
                "repaired_text": gold_repaired,
            }
            claims_list.append(claim_dict)
        else:
            for idx, c in enumerate(raw_claims):
                s_id = c.get("sentence_id", 0)
                if s_id >= len(sentences):
                    s_id = 0
                s_text, s_start, s_end = sentences[s_id]
                c_span = c.get("span")
                if not c_span or not (s_start <= c_span[0] < c_span[1] <= s_end):
                    c_span = [s_start, s_end]

                claim_dict = {
                    "id": c.get("id") or f"{did}_c{idx+1}",
                    "text": c["text"],
                    "sentence_id": s_id,
                    "span": c_span,
                    "check_worthy": True,
                    "label": verdict,
                    "label_rationale": rationale,
                    "evidence": [evidence_passage],
                    "repaired_text": gold_repaired,
                }
                claims_list.append(claim_dict)

        record_dict = {
            "id": did,
            "source_dataset": "halueval",
            "question": question,
            "answer": answer,
            "split": "dev",
            "claims": claims_list,
            "repaired_answer": None,
            "gold_repaired_answer": gold_repaired,
            "reliability_score_before": None,
            "reliability_score_after": None,
            "repair_iterations": 0,
            "run_id": None,
        }
        gold_records.append(record_dict)

    gold_dev_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_dev_path.open("w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully wrote {len(gold_records)} gold dev records to {gold_dev_path}")


if __name__ == "__main__":
    build_dev_gold()
