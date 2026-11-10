"""Run a paired held-out comparison of the prompted, Path A, Path C and A+C systems.

No data or model is downloaded here. Every input, gold file, checkpoint and index
must be supplied explicitly. Results are published only after all four systems
finish and pass completeness checks. All variants receive the same human-annotated
initial claim inventory, with labels and evidence removed. This isolates verifier
and repair changes from variation in LLM claim decomposition. Final re-verification
may still decompose the edited answer; its SCR is a system metric, not an
independent repair judgment.

Example (the gold-test file may also supply the held-out question/answer inputs)::

    python scripts/compare_training_paths.py \
      --inputs data/gold/test.jsonl --gold-dev data/gold/dev.jsonl \
      --gold-test data/gold/test.jsonl --splits data/splits.json \
      --verifier-checkpoint checkpoints/verifier \
      --repair-checkpoint checkpoints/repair --index indexes/bm25_A.pkl \
      --output-dir results/ac_comparison
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_baseline import read_splits
from src.decompose import split_sentences_with_spans
from src.evaluate import bootstrap_confidence_interval, compute_rouge_l, evaluate_dataset
from src.pipeline import run_pipeline
from src.schema import AnswerRecord, Claim


LABELS = {"Supported", "Contradicted", "Unverifiable"}
VARIANTS = {
    "prompted_baseline": ("ensemble", "prompted"),
    "trained_verifier": ("trained", "prompted"),
    "trained_repair": ("ensemble", "trained"),
    "trained_both": ("trained", "trained"),
}


@dataclass(frozen=True)
class Config:
    inputs: Path
    gold_dev: Path
    gold_test: Path
    splits: Path
    verifier_checkpoint: Path
    repair_checkpoint: Path
    index: Path
    output_dir: Path
    repo_root: Path = Path(__file__).resolve().parent.parent
    retriever_method: str = "bm25"
    faiss_index: Path | None = None
    passages: Path | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    provider: str = "groq"
    repair_policy: str = "rewrite+hedge"
    top_k: int = 5
    max_repair_iterations: int = 2
    n_bootstrap: int = 1000
    seed: int = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path, split: str) -> list[AnswerRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {split} JSONL is missing: {path}")
    records: list[AnswerRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = AnswerRecord.from_jsonl_line(line)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
            if not isinstance(record.id, str) or not record.id.strip():
                raise ValueError(f"Missing record ID at {path}:{line_number}")
            if record.id in seen:
                raise ValueError(f"Duplicate record ID {record.id} in {path}")
            if record.split != split:
                raise ValueError(f"Record {record.id} in {path} is not split={split}")
            if not record.question.strip() or not record.answer.strip():
                raise ValueError(f"Record {record.id} in {path} has empty input text")
            seen.add(record.id)
            records.append(record)
    if not records:
        raise ValueError(f"Required JSONL is empty: {path}")
    return records


def validate_gold(records: list[AnswerRecord], path: Path) -> None:
    for record in records:
        if not isinstance(record.gold_repaired_answer, str):
            raise ValueError(f"Gold repair missing for {record.id} in {path}")
        if not record.claims or not record.check_worthy_claims:
            raise ValueError(f"Gold claims missing for {record.id} in {path}")
        claim_ids = [claim.id for claim in record.claims]
        if any(not isinstance(claim.id, str) or not claim.id.strip() for claim in record.claims):
            raise ValueError(f"Empty gold claim ID for {record.id} in {path}")
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError(f"Duplicate gold claim ID for {record.id} in {path}")
        spans = split_sentences_with_spans(record.answer)
        for claim in record.claims:
            if not isinstance(claim.text, str) or not claim.text.strip():
                raise ValueError(f"Empty gold claim text for {record.id} in {path}")
            if type(claim.sentence_id) is not int or not 0 <= claim.sentence_id < len(spans):
                raise ValueError(f"Gold claim sentence_id outside answer for {record.id} in {path}")
            if claim.span is not None:
                start, end = spans[claim.sentence_id][1:]
                if (not isinstance(claim.span, list) or len(claim.span) != 2
                        or any(type(offset) is not int for offset in claim.span)
                        or not start <= claim.span[0] < claim.span[1] <= end):
                    raise ValueError(f"Invalid gold claim span for {record.id} in {path}")
        if any(claim.label not in LABELS for claim in record.check_worthy_claims):
            raise ValueError(f"Unlabeled gold check-worthy claim for {record.id} in {path}")


def checkpoint_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint directory/config.json missing: {path}")
    weights = sorted(
        file for file in path.rglob("*")
        if file.is_file() and (file.suffix == ".safetensors" or file.name in {
            "pytorch_model.bin", "adapter_model.bin"
        })
    )
    if not weights:
        raise FileNotFoundError(f"No trained weights found in checkpoint: {path}")
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256(path / "config.json"),
        "weights": [
            {"file": str(file.relative_to(path)), "bytes": file.stat().st_size,
             "sha256": sha256(file)} for file in weights
        ],
    }


def validate_inputs(config: Config) -> tuple[list[AnswerRecord], list[AnswerRecord], dict]:
    if config.top_k < 1 or config.max_repair_iterations < 1 or config.n_bootstrap < 1:
        raise ValueError("top_k, max_repair_iterations and n_bootstrap must be positive")
    if config.retriever_method not in {"bm25", "fts5", "dense", "hybrid"}:
        raise ValueError("retriever_method must be bm25, fts5, dense or hybrid")
    if not config.index.is_file():
        raise FileNotFoundError(f"Retrieval index missing: {config.index}")
    if config.retriever_method == "fts5":
        from scripts.build_fever_fts import FtsRetriever
        with FtsRetriever(config.index):
            pass
    if config.retriever_method in {"dense", "hybrid"} and (
        config.faiss_index is None or not config.faiss_index.is_file()
        or config.passages is None or not config.passages.is_file()
    ):
        raise FileNotFoundError("Dense/hybrid retrieval requires --faiss-index and --passages")

    splits, note = read_splits(config.splits)
    if splits is None:
        raise ValueError(f"Invalid split file {config.splits}: {note}")
    inputs = load_records(config.inputs, "test")
    gold_dev = load_records(config.gold_dev, "dev")
    gold_test = load_records(config.gold_test, "test")
    validate_gold(gold_dev, config.gold_dev)
    validate_gold(gold_test, config.gold_test)
    input_map = {record.id: record for record in inputs}
    gold_map = {record.id: record for record in gold_test}
    if set(input_map) != set(gold_map):
        raise ValueError("Held-out input IDs must match gold-test IDs exactly")
    if not set(gold_map) <= splits["test"] or not {r.id for r in gold_dev} <= splits["dev"]:
        raise ValueError("Gold IDs must belong to their declared dev/test splits")
    dev_questions = {record.question.strip().casefold() for record in gold_dev}
    test_questions = {record.question.strip().casefold() for record in gold_test}
    if dev_questions & test_questions:
        raise ValueError("Dev and test gold files share a question/prompt")
    for record_id, record in input_map.items():
        gold = gold_map[record_id]
        if (record.question, record.answer) != (gold.question, gold.answer):
            raise ValueError(f"Held-out input differs from gold-test for {record_id}")

    checkpoints = {
        "verifier": checkpoint_manifest(config.verifier_checkpoint),
        "repair": checkpoint_manifest(config.repair_checkpoint),
    }
    return inputs, gold_test, checkpoints


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_output(result: AnswerRecord, source: AnswerRecord,
                    gold: AnswerRecord) -> None:
    if not isinstance(result, AnswerRecord):
        raise ValueError(f"Pipeline returned no AnswerRecord for {source.id}")
    if (result.id, result.split, result.question, result.answer) != (
        source.id, "test", source.question, source.answer
    ):
        raise ValueError(f"Pipeline changed held-out identity/input for {source.id}")
    if not isinstance(result.repaired_answer, str):
        raise ValueError(f"Missing repaired answer for {source.id}")
    if not result.run_id:
        raise ValueError(f"Missing run ID for {source.id}")
    for name in ("reliability_score_before", "reliability_score_after"):
        score = getattr(result, name)
        if score is None:
            stage = "before" if name.endswith("before") else "after"
            diagnostic = (result.verification_diagnostics or {}).get(stage, {})
            if diagnostic.get("reason") in ("empty_answer", "no_check_worthy_claims") and diagnostic.get("check_worthy") == 0:
                continue
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid {name} for {source.id}")
    predicted_inventory = [(c.id, c.text, c.sentence_id, c.span, c.check_worthy)
                           for c in result.claims]
    gold_inventory = [(c.id, c.text, c.sentence_id, c.span, c.check_worthy)
                      for c in gold.claims]
    if predicted_inventory != gold_inventory:
        raise ValueError(f"Pipeline changed fixed gold claim inventory for {source.id}")
    if any(claim.label not in LABELS for claim in result.check_worthy_claims):
        raise ValueError(f"Unlabeled predicted check-worthy claim for {source.id}")


def reference_repair_metrics(records: list[AnswerRecord], gold_map: dict[str, AnswerRecord],
                             n_bootstrap: int, seed: int) -> dict[str, Any]:
    """Score edited answers against human reference repairs, including deletions."""
    exact = []
    rouge = []
    for record in records:
        reference = gold_map[record.id].gold_repaired_answer
        if not isinstance(record.repaired_answer, str) or not isinstance(reference, str):
            raise ValueError(f"Missing repaired answer/reference for {record.id}")
        exact.append(float(record.repaired_answer.strip() == reference.strip()))
        rouge.append(compute_rouge_l(record.repaired_answer, reference)["f1"])
    exact_mean, exact_low, exact_high = bootstrap_confidence_interval(
        exact, n_bootstrap=n_bootstrap, seed=seed,
    )
    rouge_mean, rouge_low, rouge_high = bootstrap_confidence_interval(
        rouge, n_bootstrap=n_bootstrap, seed=seed,
    )
    return {
        "reference": "human gold_repaired_answer",
        "exact_match": {"mean": exact_mean, "ci_lower": exact_low,
                        "ci_upper": exact_high, "raw": exact},
        "rouge_l_f1": {"mean": rouge_mean, "ci_lower": rouge_low,
                       "ci_upper": rouge_high, "raw": rouge},
    }


def git_info(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=root, text=True,
                                capture_output=True, check=True)
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain"))}


def package_versions() -> dict[str, str | None]:
    packages = ("numpy", "torch", "transformers", "datasets", "scipy",
                "sentence-transformers", "groq", "rank-bm25", "spacy",
                "en-core-web-sm")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_comparison(config: Config) -> Path:
    inputs, gold_test, checkpoints = validate_inputs(config)
    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")
    source_git = git_info(config.repo_root)
    gold_map = {record.id: record for record in gold_test}
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ac-compare-", dir=config.output_dir.parent))
    try:
        outputs: dict[str, list[AnswerRecord]] = {}
        for variant, (verifier, repair_backend) in VARIANTS.items():
            seed_everything(config.seed)
            records: list[AnswerRecord] = []
            for source in inputs:
                gold = gold_map[source.id]
                fixed_claims = [
                    Claim(id=claim.id, text=claim.text,
                          sentence_id=claim.sentence_id, span=copy.deepcopy(claim.span),
                          check_worthy=claim.check_worthy)
                    for claim in gold.claims
                ]
                fresh = AnswerRecord(id=source.id, source_dataset=source.source_dataset,
                                     split="test", question=source.question, answer=source.answer,
                                     claims=fixed_claims)
                try:
                    result = run_pipeline(
                        fresh, use_existing_claims=True, index_path=str(config.index),
                        retriever_method=config.retriever_method, top_k=config.top_k,
                        faiss_index_path=str(config.faiss_index) if config.faiss_index else None,
                        passages_path=str(config.passages) if config.passages else None,
                        llm_model=config.llm_model, provider=config.provider,
                        verifier=verifier,
                        trained_verifier_checkpoint=(str(config.verifier_checkpoint)
                                                     if verifier == "trained" else None),
                        repair_policy=config.repair_policy, repair_backend=repair_backend,
                        trained_repair_checkpoint=(str(config.repair_checkpoint)
                                                   if repair_backend == "trained" else None),
                        max_repair_iterations=config.max_repair_iterations,
                    )
                    validate_output(result, source, gold_map[source.id])
                except Exception as exc:
                    raise RuntimeError(f"{variant} failed on {source.id}: {exc}") from exc
                records.append(result)
            if [record.id for record in records] != [record.id for record in inputs]:
                raise RuntimeError(f"Incomplete output for {variant}")
            outputs[variant] = records

        # No metric is computed until every variant has a complete, aligned output.
        metrics = {}
        for variant, records in outputs.items():
            detection_gold = copy.deepcopy(gold_test)
            for gold in detection_gold:
                for claim in gold.claims:
                    if not claim.check_worthy:
                        claim.label = None
            measured = evaluate_dataset(records, gold_records=detection_gold,
                                        n_bootstrap=config.n_bootstrap, seed=config.seed)
            if (not isinstance(measured, dict) or measured.get("n_records") != len(inputs)
                    or measured.get("detection_vs_gold") is None):
                raise RuntimeError(f"Incomplete gold evaluation for {variant}")
            measured["repair_vs_gold"] = reference_repair_metrics(
                records, gold_map, config.n_bootstrap, config.seed,
            )
            metrics[variant] = measured

        baseline_repair = metrics["prompted_baseline"]["repair_vs_gold"]
        for variant, measured in metrics.items():
            if variant == "prompted_baseline":
                continue
            measured["paired_delta_vs_prompted"] = {}
            for name in ("exact_match", "rouge_l_f1"):
                baseline_values = baseline_repair[name]["raw"]
                variant_values = measured["repair_vs_gold"][name]["raw"]
                deltas = [value - baseline for value, baseline in
                          zip(variant_values, baseline_values)]
                mean, low, high = bootstrap_confidence_interval(
                    deltas, n_bootstrap=config.n_bootstrap, seed=config.seed,
                )
                measured["paired_delta_vs_prompted"][name] = {
                    "mean": mean, "ci_lower": low, "ci_upper": high, "raw": deltas,
                }

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": source_git,
            "python": platform.python_version(),
            "package_versions": package_versions(),
            "seed": config.seed,
            "config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in asdict(config).items()},
            "input_hashes": {name: sha256(path) for name, path in {
                "inputs": config.inputs, "gold_dev": config.gold_dev,
                "gold_test": config.gold_test, "splits": config.splits,
                "index": config.index,
            }.items()},
            "checkpoints": checkpoints,
            "test_record_ids": [record.id for record in inputs],
            "initial_claim_inventory": "human gold-test claims without labels or evidence",
            "variants": {name: {"verifier": mode[0], "repair_backend": mode[1]}
                         for name, mode in VARIANTS.items()},
            "metrics_note": (
                "Detection uses human gold claim labels; repair_vs_gold compares with "
                "human reference answers. Pipeline SCR and preservation rely on the "
                "system's own verdicts and are not independent repair judgments."
            ),
        }
        for variant, records in outputs.items():
            (staging / f"{variant}.jsonl").write_text(
                "".join(record.to_jsonl_line() + "\n" for record in records),
                encoding="utf-8",
            )
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(config.output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return config.output_dir


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "gold-dev", "gold-test", "splits", "verifier-checkpoint",
                 "repair-checkpoint", "index", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--retriever-method", choices=("bm25", "fts5", "dense", "hybrid"),
                        default="bm25")
    parser.add_argument("--faiss-index", type=Path)
    parser.add_argument("--passages", type=Path)
    parser.add_argument("--llm-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--repair-policy", default="rewrite+hedge")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-repair-iterations", type=int, default=2)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return Config(**vars(parser.parse_args()))


def main() -> int:
    try:
        output_dir = run_comparison(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Comparison aborted: {exc}", file=sys.stderr)
        return 1
    print(f"Complete paired comparison saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
