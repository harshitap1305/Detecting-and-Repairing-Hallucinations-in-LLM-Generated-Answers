"""
download_data.py — Download and convert all datasets into the unified schema.

Datasets:
  - HaluEval QA subset     (answer-level detection)
  - FELM world-knowledge   (segment-level detection)
  - WikiBio GPT-3          (sentence-level detection)
  - Factcheck-Bench        (claim-level detection + revision gold)

Usage:
    python scripts/download_data.py --output_dir data/processed --seed 42

Colab:
    !python scripts/download_data.py \\
        --output_dir /content/drive/MyDrive/hallucination_project/data/processed
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import AnswerRecord


# ─── Writers ──────────────────────────────────────────────────────────────────

def _write_jsonl(records: list[AnswerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_jsonl_line() + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


# ─── HaluEval ─────────────────────────────────────────────────────────────────

def load_halueval(out_dir: Path, seed: int) -> list[AnswerRecord]:
    """
    HaluEval QA subset.
    HF: pminervini/HaluEval, config qa_samples
    Fallback: RUCAIBox/HaluEval
    Use hallucinated_answer as the answer to detect (it contains hallucinations).
    """
    print("\nLoading HaluEval…")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data",
                              trust_remote_code=True)
        except Exception:
            ds = load_dataset("RUCAIBox/HaluEval", "qa", split="data",
                              trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ Could not load HaluEval: {e}")
        return []

    rng = random.Random(seed)
    records: list[AnswerRecord] = []
    items = list(ds)
    rng.shuffle(items)

    for i, ex in enumerate(items):
        # HaluEval fields: question, right_answer, hallucinated_answer, hallucination
        question = ex.get("question", ex.get("query", ""))
        answer = ex.get("hallucinated_answer", ex.get("answer", ""))
        gold = ex.get("right_answer", ex.get("correct_answer", None))

        split = "test" if i % 5 == 0 else "dev"   # 80 / 20
        rec = AnswerRecord(
            id=f"halueval_{i:05d}",
            source_dataset="halueval",
            split=split,
            question=question,
            answer=answer,
            gold_repaired_answer=gold,
        )
        records.append(rec)

    _write_jsonl(records, out_dir / "halueval.jsonl")
    dev_n = sum(1 for r in records if r.split == "dev")
    test_n = len(records) - dev_n
    print(f"  HaluEval: {len(records):,} total  (dev={dev_n}, test={test_n})")
    return records


# ─── FELM ─────────────────────────────────────────────────────────────────────

def load_felm(out_dir: Path, seed: int) -> list[AnswerRecord]:
    """
    FELM — segment-level labels; world_knowledge domain only.
    HF: hkust-nlp/felm
    Held as external benchmark (split=test only; never tune on it).
    """
    print("\nLoading FELM…")
    try:
        from datasets import load_dataset, get_dataset_config_names
        try:
            configs = get_dataset_config_names("hkust-nlp/felm")
            print(f"  FELM configs: {configs}")
        except Exception:
            configs = []

        # Try world_knowledge config first, then default
        ds = None
        for cfg in (["world_knowledge"] + [c for c in configs if c != "world_knowledge"]):
            try:
                ds = load_dataset("hkust-nlp/felm", cfg, split="test",
                                  trust_remote_code=True)
                print(f"  Loaded config: {cfg}")
                break
            except Exception:
                continue
        if ds is None:
            ds = load_dataset("hkust-nlp/felm", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ Could not load FELM: {e}")
        return []

    records: list[AnswerRecord] = []
    for i, ex in enumerate(ds):
        # FELM fields vary by config; try common names
        question = ex.get("prompt", ex.get("question", f"felm_q_{i}"))
        answer = ex.get("response", ex.get("answer", ex.get("text", "")))
        if not answer:
            continue
        rec = AnswerRecord(
            id=f"felm_wk_{i:05d}",
            source_dataset="felm",
            split="test",   # always test; this is a held-out benchmark
            question=question,
            answer=answer,
        )
        records.append(rec)

    _write_jsonl(records, out_dir / "felm.jsonl")
    print(f"  FELM: {len(records):,} records (all test / held-out benchmark)")
    return records


# ─── WikiBio ──────────────────────────────────────────────────────────────────

def load_wikibio(out_dir: Path, seed: int) -> list[AnswerRecord]:
    """
    WikiBio GPT-3 hallucination dataset.
    HF: potsawee/wiki_bio_gpt3_hallucination, split=evaluation
    sentence labels: accurate / minor_inaccurate / major_inaccurate
    """
    print("\nLoading WikiBio…")
    try:
        from datasets import load_dataset
        ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation",
                          trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ Could not load WikiBio: {e}")
        return []

    records: list[AnswerRecord] = []
    for i, ex in enumerate(ds):
        gpt3_text = ex.get("gpt3_text", ex.get("generated_text", ""))
        if not gpt3_text:
            continue
        wiki_ref = ex.get("wiki_bio_text", "")
        # Construct a pseudo-question from the wiki reference (first 50 chars)
        question = f"Write a biography of {wiki_ref[:60].strip()}."
        rec = AnswerRecord(
            id=f"wikibio_{i:05d}",
            source_dataset="wikibio",
            split="test",   # external benchmark; never tune on it
            question=question,
            answer=gpt3_text,
        )
        records.append(rec)

    _write_jsonl(records, out_dir / "wikibio.jsonl")
    print(f"  WikiBio: {len(records):,} records (all test / held-out benchmark)")
    return records


# ─── Factcheck-Bench ──────────────────────────────────────────────────────────

def load_factcheck_bench(out_dir: Path, seed: int) -> list[AnswerRecord]:
    """
    Factcheck-Bench (Factcheck-GPT).
    Primary source: https://github.com/yuxiaw/Factcheck-GPT
    Manually download data/factcheck_gpt_benchmark.json and place at
    data/raw/factcheck_bench.json
    HF mirror (if available): yuxiaw/Factcheck-GPT
    """
    print("\nLoading Factcheck-Bench…")
    raw_path = Path("data/raw/factcheck_bench.json")

    # Try HF first
    try:
        from datasets import load_dataset
        ds = load_dataset("yuxiaw/Factcheck-GPT", split="train",
                          trust_remote_code=True)
        items = list(ds)
        print(f"  Loaded from HuggingFace ({len(items)} items)")
    except Exception:
        # Fallback to local file
        if not raw_path.exists():
            print(f"  ⚠ Factcheck-Bench not found.")
            print(f"    Download from https://github.com/yuxiaw/Factcheck-GPT")
            print(f"    and place at: {raw_path.absolute()}")
            return []
        with open(raw_path, encoding="utf-8") as f:
            items = json.load(f)
        print(f"  Loaded from local file ({len(items)} items)")

    records: list[AnswerRecord] = []
    for i, ex in enumerate(items):
        if isinstance(ex, dict):
            question = ex.get("question", ex.get("prompt", f"fcb_q_{i}"))
            answer = ex.get("answer", ex.get("response", ex.get("text", "")))
        else:
            continue
        if not answer:
            continue
        rec = AnswerRecord(
            id=f"fcb_{i:05d}",
            source_dataset="factcheck_bench",
            split="test",  # external benchmark
            question=question,
            answer=answer,
        )
        records.append(rec)

    _write_jsonl(records, out_dir / "factcheck_bench.jsonl")
    print(f"  Factcheck-Bench: {len(records):,} records (all test / held-out benchmark)")
    return records


# ─── Splits file ──────────────────────────────────────────────────────────────

def save_splits(all_records: list[AnswerRecord], splits_path: Path) -> None:
    dev_ids = [r.id for r in all_records if r.split == "dev"]
    test_ids = [r.id for r in all_records if r.split == "test"]
    splits = {
        "seed": 42,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "note": "Split by question/prompt. Held-out benchmarks: FELM, WikiBio, Factcheck-Bench (all test).",
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSplits saved → {splits_path}  (dev={len(dev_ids)}, test={len(test_ids)})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process hallucination benchmark datasets."
    )
    parser.add_argument("--output_dir", default="data/processed",
                        help="Directory for processed .jsonl files")
    parser.add_argument("--splits_file", default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", nargs="+",
        default=["halueval", "felm", "wikibio", "factcheck_bench"],
        choices=["halueval", "felm", "wikibio", "factcheck_bench"],
        help="Which datasets to download",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    random.seed(args.seed)

    all_records: list[AnswerRecord] = []
    loaders = {
        "halueval": load_halueval,
        "felm": load_felm,
        "wikibio": load_wikibio,
        "factcheck_bench": load_factcheck_bench,
    }

    for name in args.datasets:
        records = loaders[name](out_dir, args.seed)
        all_records.extend(records)

    save_splits(all_records, Path(args.splits_file))
    print("\n✅ All requested datasets processed.")
    print("Next step: build indexes with scripts/build_index.py")


if __name__ == "__main__":
    main()
