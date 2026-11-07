"""Fine-tune a local Flan-T5 repair editor on verified Path C examples.

Uses a small explicit PyTorch loop so this command does not depend on a
particular Transformers Trainer/Accelerate release. Save ``--output-dir`` on
Google Drive to retain checkpoints across Colab sessions. No test file is read.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.repair_training import (
    assert_disjoint_splits, evaluate_predictions, example_prompt, file_sha256,
    load_examples, predict_rows,
)


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _collator(tokenizer, max_source_length: int, max_target_length: int, policy: str):
    def collate(rows: list[dict]) -> dict:
        prompts = [example_prompt(r, policy) for r in rows]
        targets = [r["target"] for r in rows]
        inputs = tokenizer(prompts, padding=True, truncation=True,
                           max_length=max_source_length, return_tensors="pt")
        labels = tokenizer(text_target=targets, padding=True, truncation=True,
                           max_length=max_target_length, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--policy", default="rewrite+hedge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-dev", type=int)
    parser.add_argument("--resume-from", type=Path,
                        help="checkpoint-epoch-N directory from a previous run")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, grad-accum and learning-rate must be positive")
    if args.max_source_length < 16 or args.max_target_length < 2:
        parser.error("token lengths are too short")
    if args.max_train is not None and args.max_train < 1:
        parser.error("--max-train must be positive")
    if args.max_dev is not None and args.max_dev < 1:
        parser.error("--max-dev must be positive")

    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    train_rows = load_examples(args.train_data, "train")
    dev_rows = load_examples(args.dev_data, "dev")
    assert_disjoint_splits(train_rows, dev_rows)
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    if args.max_dev:
        dev_rows = dev_rows[:args.max_dev]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if device == "cpu" and precision != "fp32":
        parser.error("fp16/bf16 training requires CUDA; use --precision fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("This CUDA device does not support bf16")

    hashes = {"train": file_sha256(args.train_data), "dev": file_sha256(args.dev_data)}
    config_path = args.output_dir / "run_config.json"
    if args.resume_from:
        if not args.resume_from.is_dir():
            parser.error(f"Missing resume checkpoint: {args.resume_from}")
        if not config_path.is_file():
            parser.error(f"Missing run_config.json in {args.output_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["data_sha256"] != hashes or previous["seed"] != args.seed:
            parser.error("Resume data or seed differs from the saved run")
        model_source = str(args.resume_from)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error(f"Output directory is not empty: {args.output_dir}")
        model_source = args.model_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    start_epoch = 1
    if args.resume_from:
        state = torch.load(args.resume_from / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        if precision == "fp16" and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
    else:
        config = vars(args).copy()
        config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        config.update({"device": device, "resolved_precision": precision,
                       "torch_version": torch.__version__,
                       "transformers_version": transformers.__version__,
                       "git_commit": _git_commit(), "data_sha256": hashes,
                       "train_count": len(train_rows), "dev_count": len(dev_rows)})
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if start_epoch > args.epochs:
        parser.error("Resume checkpoint already reached the requested epoch count")

    log_path = args.output_dir / "training_curve.csv"
    write_header = not log_path.exists()
    best_score = (-1.0, -1.0)
    if (args.output_dir / "best.json").exists():
        prior = json.loads((args.output_dir / "best.json").read_text(encoding="utf-8"))
        best_score = (prior["exact_match"], prior["token_f1"])
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "dev_exact_match", "dev_token_f1", "dev_seconds"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True,
                                generator=generator, collate_fn=_collator(
                                    tokenizer, args.max_source_length, args.max_target_length, args.policy))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for index, batch in enumerate(loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                window_remaining = len(loader) - (index // args.grad_accum) * args.grad_accum
                denominator = min(args.grad_accum, window_remaining)
                use_amp = precision in ("fp16", "bf16")
                with torch.autocast(device_type="cuda", dtype=(torch.float16 if precision == "fp16" else torch.bfloat16), enabled=use_amp):
                    loss = model(**batch).loss
                loss_total += float(loss.detach().cpu())
                scaler.scale(loss / denominator).backward()
                if (index + 1) % args.grad_accum == 0 or index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            predictions, seconds = predict_rows(
                model, tokenizer, dev_rows, device, batch_size=args.batch_size,
                max_source_length=args.max_source_length,
                max_new_tokens=args.max_target_length, policy=args.policy)
            metrics = evaluate_predictions(dev_rows, predictions)
            row = {"epoch": epoch, "train_loss": loss_total / len(loader),
                   "dev_exact_match": metrics["exact_match"],
                   "dev_token_f1": metrics["token_f1"], "dev_seconds": seconds}
            writer.writerow(row)
            log_file.flush()
            checkpoint = args.output_dir / f"checkpoint-epoch-{epoch}"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            checkpoint.mkdir()
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if precision == "fp16" else {}},
                       checkpoint / "training_state.pt")
            score = (metrics["exact_match"], metrics["token_f1"])
            if score > best_score:
                best_score = score
                (args.output_dir / "best.json").write_text(json.dumps({
                    "checkpoint": str(checkpoint.resolve()), "epoch": epoch,
                    "exact_match": score[0], "token_f1": score[1],
                }, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
