"""Draft candidate claim inventory for human annotation using decomposition stage.

Usage:
    python scripts/draft_claim_inventory.py \\
        --inputs data/annotation_batches/halueval_v1/inputs/dev.jsonl \\
        --output data/annotation_batches/halueval_v1/inventory/dev_draft.jsonl \\
        --model llama-3.3-70b-versatile \\
        --provider groq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import AnswerRecord
from src.decompose import decompose


def draft_inventory(
    inputs_path: Path,
    output_path: Path,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    extraction_check: str = "semantic_v1",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume capability: collect already drafted IDs
    done_ids: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        if "id" in row:
                            done_ids.add(row["id"])
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: found {len(done_ids)} existing records in {output_path}")

    total = 0
    drafted = 0
    errors = 0

    with inputs_path.open("r", encoding="utf-8") as in_f, output_path.open("a", encoding="utf-8") as out_f:
        for line_no, line in enumerate(in_f, 1):
            if not line.strip():
                continue
            total += 1
            record = AnswerRecord.from_jsonl_line(line)
            if record.id in done_ids:
                continue

            print(f"[{total}] Decomposing record {record.id}...")
            import time
            success = False
            for attempt in range(1, 4):
                try:
                    record = decompose(
                        record,
                        model=model,
                        provider=provider,
                        extraction_check=extraction_check,
                    )
                    out_f.write(record.to_jsonl_line() + "\n")
                    out_f.flush()
                    drafted += 1
                    cw_count = len(record.check_worthy_claims)
                    print(f"    -> {len(record.claims)} claims ({cw_count} check-worthy)")
                    success = True
                    break
                except Exception as exc:
                    if attempt < 3:
                        wait = attempt * 3
                        print(f"    [WARN] Attempt {attempt} failed ({exc}). Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        errors += 1
                        print(f"    [ERROR] Failed to decompose {record.id} after 3 attempts: {exc}")

    print(f"\nCompleted: {drafted} newly drafted, {len(done_ids)} resumed, {errors} errors out of {total} records.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Input records JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output draft inventory JSONL")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="Decomposition model")
    parser.add_argument("--provider", type=str, default="groq", help="LLM provider")
    parser.add_argument("--extraction-check", type=str, default="semantic_v1",
                        choices=["structural", "semantic_v1"], help="Extraction audit check mode")
    args = parser.parse_args()

    draft_inventory(
        inputs_path=args.inputs,
        output_path=args.output,
        model=args.model,
        provider=args.provider,
        extraction_check=args.extraction_check,
    )


if __name__ == "__main__":
    main()

