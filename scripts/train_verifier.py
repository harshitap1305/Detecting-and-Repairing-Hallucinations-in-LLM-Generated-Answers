"""Fine-tune a three-way FEVER verifier and select by dev macro-F1.

Requires PyTorch, Transformers, Datasets and SentencePiece. Run on a GPU for
the intended experiment; CPU is supported for tiny smoke runs only. See the
official Transformers Trainer and Datasets JSON loading documentation for
the mutable APIs used here.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verifier_data import LABELS, iter_jsonl, normalize_title


def validate_splits(train_file: Path, dev_file: Path) -> tuple[int, int]:
    """Catch malformed rows, duplicate IDs and article leakage before ML work."""
    seen_ids: set[str] = set()
    article_sets: list[set[str]] = []
    entity_sets: list[set[str]] = []
    counts: list[int] = []
    for path in (train_file, dev_file):
        articles: set[str] = set()
        entities: set[str] = set()
        count = 0
        for row in iter_jsonl(path):
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or sample_id in seen_ids:
                raise ValueError(f"Invalid or duplicate sample id: {sample_id!r}")
            seen_ids.add(sample_id)
            if not isinstance(row.get("premise"), str) or not row["premise"].strip():
                raise ValueError(f"Empty premise for {sample_id}")
            if not isinstance(row.get("claim"), str) or not row["claim"].strip():
                raise ValueError(f"Empty claim for {sample_id}")
            label = row.get("label")
            if type(label) is not int or not 0 <= label < len(LABELS):
                raise ValueError(f"Invalid label for {sample_id}: {label!r}")
            if row.get("label_name") != LABELS[label]:
                raise ValueError(f"Inconsistent label_name for {sample_id}")
            titles = row.get("article_ids")
            if not isinstance(titles, list) or not titles or not all(isinstance(t, str) and t.strip() for t in titles):
                raise ValueError(f"Missing article_ids for {sample_id}")
            articles.update(normalize_title(title) for title in titles)
            entity_key = row.get("entity_key")
            if entity_key:
                if not isinstance(entity_key, str):
                    raise ValueError(f"Invalid entity_key for {sample_id}")
                entities.add(normalize_title(entity_key))
            count += 1
        if not count:
            raise ValueError(f"No examples in {path}")
        counts.append(count)
        article_sets.append(articles)
        entity_sets.append(entities)
    overlap = article_sets[0] & article_sets[1]
    if overlap:
        raise ValueError(f"Article leakage between train/dev: {sorted(overlap)[:5]}")
    entity_overlap = entity_sets[0] & entity_sets[1]
    if entity_overlap:
        raise ValueError(f"Entity leakage between train/dev: {sorted(entity_overlap)[:5]}")
    return counts[0], counts[1]


def compute_metrics(eval_prediction) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    gold = eval_prediction.label_ids
    return {"accuracy": float(accuracy_score(gold, predictions)),
            "macro_f1": float(f1_score(gold, predictions, labels=[0, 1, 2],
                                       average="macro", zero_division=0))}


def _training_arguments(cls, args, fp16: bool, bf16: bool):
    parameters = inspect.signature(cls).parameters
    kwargs = dict(output_dir=str(args.output_dir), learning_rate=args.learning_rate,
                  per_device_train_batch_size=args.train_batch_size,
                  per_device_eval_batch_size=args.eval_batch_size,
                  gradient_accumulation_steps=args.gradient_accumulation,
                  num_train_epochs=args.epochs, warmup_ratio=args.warmup_ratio,
                  weight_decay=args.weight_decay, save_strategy="epoch",
                  logging_strategy="steps", logging_steps=args.logging_steps,
                  save_total_limit=2, load_best_model_at_end=True,
                  metric_for_best_model="macro_f1", greater_is_better=True,
                  seed=args.seed, data_seed=args.seed, fp16=fp16, bf16=bf16,
                  report_to=[])
    # Transformers 4.40 called this evaluation_strategy; current docs use eval_strategy.
    kwargs["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = "epoch"
    return cls(**kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--max-train-examples", type=int,
                        help="Deterministic train-size ablation; shuffles by seed before selecting N")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint")
    args = parser.parse_args(argv)
    if (args.max_length <= 0 or (args.max_train_examples is not None and args.max_train_examples <= 0)
            or args.train_batch_size <= 0 or args.eval_batch_size <= 0
            or args.gradient_accumulation <= 0 or args.epochs <= 0 or args.learning_rate <= 0
            or args.logging_steps <= 0 or not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0):
        parser.error("Invalid positive batch/length/learning parameters")
    train_count, dev_count = validate_splits(args.train_file, args.dev_file)
    if (args.output_dir / "final" / "config.json").exists() and not args.resume:
        raise FileExistsError(f"Trained checkpoint already exists in {args.output_dir}")

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    if args.precision in ("fp16", "bf16") and not torch.cuda.is_available():
        raise ValueError(f"{args.precision} requires a CUDA GPU")
    fp16 = args.precision == "fp16" or (args.precision == "auto" and torch.cuda.is_available())
    bf16 = args.precision == "bf16"
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3,
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    model.config.verifier_max_length = args.max_length
    dataset = load_dataset("json", data_files={"train": str(args.train_file),
                                               "dev": str(args.dev_file)})
    if args.max_train_examples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(
            range(min(args.max_train_examples, len(dataset["train"]))))
        train_count = len(dataset["train"])

    def encode(batch):
        encoded = tokenizer(batch["premise"], batch["claim"], truncation=True,
                            max_length=args.max_length)
        encoded["labels"] = batch["label"]
        return encoded

    encoded = dataset.map(encode, batched=True,
                          remove_columns=dataset["train"].column_names)
    training_args = _training_arguments(TrainingArguments, args, fp16, bf16)
    trainer_kwargs = dict(model=model, args=training_args,
                          train_dataset=encoded["train"], eval_dataset=encoded["dev"],
                          data_collator=DataCollatorWithPadding(tokenizer),
                          compute_metrics=compute_metrics)
    parameter = "processing_class" if "processing_class" in inspect.signature(Trainer).parameters else "tokenizer"
    trainer_kwargs[parameter] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"model_name": args.model_name, "max_length": args.max_length,
              "train_examples": train_count, "dev_examples": dev_count,
              "seed": args.seed, "hyperparameters": vars(args),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("transformers", "datasets", "torch", "scikit-learn")}}
    try:
        config["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        config["git_commit"] = None
    config["hyperparameters"] = {key: str(value) if isinstance(value, Path) else value
                                  for key, value in config["hyperparameters"].items()}
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    (args.output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")
    metrics = trainer.evaluate()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (args.output_dir / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(final_dir), "dev_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
