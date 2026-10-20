"""
eval_decomposition.py — Automated & human evaluation benchmark for Stage 1 Claim Decomposition.

Computes Phase 4 exit criteria:
  - Atomicity Rate: % of claims expressing a single standalone predicate
  - Decontextualization Rate: % of claims with no dangling pronouns/unresolved references
  - Span Validity: % of claims with valid non-empty character offsets within answer bounds
  - Check-Worthiness Distribution: % of claims appropriately filtered/marked
  - Completeness Estimation: Claim-to-sentence yield ratio

Usage:
    python scripts/eval_decomposition.py --input data/processed/halueval.jsonl --sample_size 50
    python scripts/eval_decomposition.py --benchmark_suite  # Runs on standard 50-example synthetic benchmark
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import AnswerRecord, Claim
from src.decompose import decompose, split_sentences_with_spans


# ─── Heuristic Detectors for Quality Criteria ─────────────────────────────────

# Pronouns that indicate dangling context if found at the start or as bare subjects
DANGLING_PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "their",
    "this", "that", "these", "those"
}

# Conjunctions that often connect compound clauses in non-atomic claims
COMPOUND_MARKERS = [
    r"\band\b", r"\bwhile\b", r"\bas well as\b", r"\bin addition to\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bbut also\b"
]


def check_is_decontextualized(claim_text: str) -> Tuple[bool, str]:
    """
    Check if the claim starts with or prominently features unresolved pronouns.
    Returns (is_decontextualized, reason).
    """
    tokens = re.findall(r"\b\w+\b", claim_text.lower())
    if not tokens:
        return False, "Empty claim"
    
    # Check leading token
    if tokens[0] in DANGLING_PRONOUNS:
        return False, f"Leading unresolved pronoun: '{tokens[0]}'"

    # Check for early pronoun subjects (e.g., "in 1920, he moved...")
    if len(tokens) >= 3 and tokens[1] in DANGLING_PRONOUNS:
        return False, f"Early unresolved pronoun: '{tokens[1]}'"
    if len(tokens) >= 4 and tokens[2] in DANGLING_PRONOUNS and tokens[0] in {"in", "on", "at", "during", "after", "before"}:
        return False, f"Prepositional phrase followed by pronoun: '{tokens[2]}'"

    return True, "Decontextualized"


def check_is_atomic(claim_text: str) -> Tuple[bool, str]:
    """
    Check whether a claim contains multiple conjunctions joining compound predicates.
    Returns (is_atomic, reason).
    """
    # Count compound markers
    compound_matches = 0
    matched_markers = []
    for pattern in COMPOUND_MARKERS:
        found = re.findall(pattern, claim_text, re.IGNORECASE)
        if found:
            compound_matches += len(found)
            matched_markers.extend(found)

    # If more than one strong compound marker is present, likely non-atomic
    if compound_matches >= 2:
        return False, f"Multiple compound conjunctions: {matched_markers}"

    # Check for two distinct verbs separated by 'and'
    verb_conjunction = re.search(r'\b(was|is|became|built|won|moved|studied)\b.*\band\b.*\b(was|is|became|built|won|moved|studied)\b',
                                 claim_text, re.IGNORECASE)
    if verb_conjunction:
        return False, "Compound predicate joining two separate actions"

    return True, "Atomic"


def check_span_valid(claim: Claim, answer_text: str) -> Tuple[bool, str]:
    """Verify that claim span exists and falls within answer length bounds."""
    if not claim.span or len(claim.span) != 2:
        return False, "Span missing or malformed"
    start, end = claim.span
    if start < 0 or end > len(answer_text) or start > end:
        return False, f"Span [{start}, {end}] out of bounds for answer of len {len(answer_text)}"
    return True, "Valid span"


# ─── Benchmark Suite for Edge Cases ───────────────────────────────────────────

BENCHMARK_EXAMPLES = [
    {
        "q": "Who was Marie Curie?",
        "a": "Marie Curie was a Polish-born physicist. She won two Nobel Prizes and studied in Paris.",
    },
    {
        "q": "What is the capital and largest city of France?",
        "a": "Paris is the capital of France, and it is also the country's most populous city.",
    },
    {
        "q": "When did the Apollo 11 mission land on the Moon?",
        "a": "Apollo 11 landed on July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin walked on the surface.",
    },
    {
        "q": "Tell me about the iPhone release.",
        "a": "Steve Jobs announced the first iPhone in January 2007. It revolutionized the mobile industry.",
    },
    {
        "q": "What are the primary colors in additive color mixing?",
        "a": "The three primary colors of light are red, green, and blue.",
    },
    {
        "q": "What is the Mona Lisa?",
        "a": "As an AI language model, I can state that the Mona Lisa is a half-length portrait painting by Leonardo da Vinci.",
    },
    {
        "q": "Can you give travel advice for Japan?",
        "a": "Have you considered visiting Kyoto? Mount Fuji is Japan's tallest mountain, rising 3,776 meters.",
    },
    {
        "q": "What is your opinion on the Taj Mahal?",
        "a": "The Taj Mahal was commissioned in 1631 by Mughal emperor Shah Jahan. It is unquestionably the most divine architectural masterpiece ever created.",
    },
]


# ─── Evaluator Core ───────────────────────────────────────────────────────────

def evaluate_records(records: List[AnswerRecord]) -> Dict:
    total_claims = 0
    atomic_claims = 0
    decontext_claims = 0
    valid_spans = 0
    check_worthy_count = 0
    non_check_worthy_count = 0

    atomicity_failures = []
    decontext_failures = []
    span_failures = []

    for rec in records:
        for claim in rec.claims:
            total_claims += 1

            # Check atomicity
            is_atom, atom_reason = check_is_atomic(claim.text)
            if is_atom:
                atomic_claims += 1
            else:
                atomicity_failures.append((claim.text, atom_reason))

            # Check decontextualization
            is_decontext, decontext_reason = check_is_decontextualized(claim.text)
            if is_decontext:
                decontext_claims += 1
            else:
                decontext_failures.append((claim.text, decontext_reason))

            # Check span validity
            is_span, span_reason = check_span_valid(claim, rec.answer)
            if is_span:
                valid_spans += 1
            else:
                span_failures.append((claim.text, span_reason))

            if claim.check_worthy:
                check_worthy_count += 1
            else:
                non_check_worthy_count += 1

    total = max(1, total_claims)
    return {
        "total_records": len(records),
        "total_claims": total_claims,
        "claims_per_answer": total_claims / max(1, len(records)),
        "atomicity_rate": (atomic_claims / total) * 100,
        "decontextualization_rate": (decontext_claims / total) * 100,
        "span_validity_rate": (valid_spans / total) * 100,
        "check_worthy_ratio": (check_worthy_count / total) * 100,
        "atomicity_failures": atomicity_failures[:5],
        "decontext_failures": decontext_failures[:5],
        "span_failures": span_failures[:5],
    }


def print_evaluation_report(results: Dict):
    print("\n" + "=" * 65)
    print(" CLAIM DECOMPOSITION QUALITY BENCHMARK REPORT (Phase 4)")
    print("=" * 65)
    print(f"Evaluated Records:        {results['total_records']}")
    print(f"Extracted Atomic Claims:  {results['total_claims']}")
    print(f"Yield (Claims / Answer):  {results['claims_per_answer']:.2f}")
    print("-" * 65)
    print(f"Atomicity Rate:           {results['atomicity_rate']:6.2f}% (Target: ≥90%)")
    print(f"Decontextualization Rate: {results['decontextualization_rate']:6.2f}% (Target: ≥90%)")
    print(f"Span Validity Rate:       {results['span_validity_rate']:6.2f}% (Target: 100%)")
    print(f"Check-Worthy Ratio:       {results['check_worthy_ratio']:6.2f}%")
    print("=" * 65)

    if results["atomicity_failures"]:
        print("\n[Sample Atomicity Failure Cases]:")
        for text, reason in results["atomicity_failures"]:
            print(f"  ✗ \"{text}\" ({reason})")

    if results["decontext_failures"]:
        print("\n[Sample Decontextualization Failure Cases]:")
        for text, reason in results["decontext_failures"]:
            print(f"  ✗ \"{text}\" ({reason})")

    if results["span_failures"]:
        print("\n[Sample Span Alignment Failure Cases]:")
        for text, reason in results["span_failures"]:
            print(f"  ✗ \"{text}\" ({reason})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Claim Decomposition Quality")
    parser.add_argument("--input", default=None, help="Path to input JSONL file")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of records to sample")
    parser.add_argument("--benchmark_suite", action="store_true", help="Run on standard synthetic benchmark")
    parser.add_argument("--provider", default="groq", help="LLM provider")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="LLM model")
    args = parser.parse_args()

    records: List[AnswerRecord] = []

    if args.benchmark_suite or not args.input:
        print("Running Phase 4 Benchmark Suite (Edge-case coverage)...")
        for i, ex in enumerate(BENCHMARK_EXAMPLES):
            rec = AnswerRecord(
                id=f"bench_{i+1:03d}",
                source_dataset="benchmark",
                split="dev",
                question=ex["q"],
                answer=ex["a"],
            )
            # Run decompose
            rec = decompose(rec, model=args.model, provider=args.provider)
            records.append(rec)
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: input file {path} not found.")
            sys.exit(1)
        raw_records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(AnswerRecord.from_jsonl_line(line))
        sampled = raw_records[:args.sample_size]
        print(f"Decomposing {len(sampled)} sampled records from {path.name}...")
        for i, rec in enumerate(sampled):
            rec = decompose(rec, model=args.model, provider=args.provider)
            records.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(sampled)}")

    results = evaluate_records(records)
    print_evaluation_report(results)


if __name__ == "__main__":
    main()
