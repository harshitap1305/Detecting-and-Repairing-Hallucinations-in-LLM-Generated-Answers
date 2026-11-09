"""Check the minimum artifacts needed to reproduce the reported baseline comparison.

This checks file presence, JSONL IDs, and split alignment. It does not calculate
metrics or assert that a published result has been independently reproduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULT_NAMES = ("pipeline_output.jsonl",) + tuple(
    f"baseline_b{i}.jsonl" for i in range(8)
)


def read_jsonl_ids(path: Path) -> tuple[set[str] | None, str]:
    if not path.is_file():
        return None, "missing"

    ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = record.get("id") if isinstance(record, dict) else None
                if not isinstance(record_id, str) or not record_id.strip():
                    return None, f"line {line_number}: missing nonempty id"
                if record_id in ids:
                    return None, f"line {line_number}: duplicate id {record_id}"
                ids.add(record_id)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"

    if not ids:
        return None, "empty"
    return ids, f"{len(ids)} records"


def read_splits(path: Path) -> tuple[dict[str, set[str]] | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(data, dict):
        return None, "expected a JSON object"

    splits: dict[str, set[str]] = {}
    for name in ("dev", "test"):
        values = data.get(f"{name}_ids")
        if not isinstance(values, list) or not values:
            return None, f"{name}_ids missing or empty"
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return None, f"{name}_ids contains an invalid id"
        if len(set(values)) != len(values):
            return None, f"{name}_ids contains duplicate ids"
        splits[name] = set(values)

    if splits["dev"] & splits["test"]:
        return None, "dev_ids and test_ids overlap"
    return splits, f"{len(splits['dev'])} dev, {len(splits['test'])} test ids"


def audit(root: Path, checkpoint: Path | None = None) -> tuple[list[str], bool, bool]:
    lines: list[str] = []
    splits, split_note = read_splits(root / "data/splits.json")
    lines.append(f"data/splits.json: {split_note}")
    complete = splits is not None

    gold_ids: dict[str, set[str]] = {}
    for split_name in ("dev", "test"):
        relative = f"data/gold/{split_name}.jsonl"
        ids, note = read_jsonl_ids(root / relative)
        if ids is not None and splits is not None and not ids <= splits[split_name]:
            note += f"; IDs absent from {split_name}_ids"
            ids = None
        if ids is not None:
            gold_ids[split_name] = ids
        lines.append(f"{relative}: {note}")
        complete &= ids is not None

    for name in RESULT_NAMES:
        relative = f"results/{name}"
        ids, note = read_jsonl_ids(root / relative)
        if ids is not None and "test" in gold_ids and ids != gold_ids["test"]:
            note += "; IDs differ from gold test records"
            ids = None
        lines.append(f"{relative}: {note}")
        complete &= ids is not None

    if checkpoint is None:
        lines.append("trained checkpoint: not supplied (use --checkpoint PATH)")
        checkpoint_present = False
    else:
        checkpoint_present = checkpoint.is_dir() and any(
            path.is_file()
            for pattern in ("*.safetensors", "pytorch_model.bin", "adapter_model.bin")
            for path in checkpoint.rglob(pattern)
        )
        lines.append(
            f"trained checkpoint {checkpoint}: "
            + ("weight file present" if checkpoint_present else "missing weight file")
        )

    return lines, complete, checkpoint_present


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--checkpoint", type=Path, help="Optional trained model directory")
    parser.add_argument(
        "--require-checkpoint", action="store_true",
        help="Also require a trained model weight file for a zero exit status",
    )
    parser.add_argument("--bundle", type=Path, help="Audit a versioned report bundle instead of legacy artifacts")
    parser.add_argument("--gold", type=Path, help="Exact external gold JSONL required for bundle auditing")
    parser.add_argument("--no-recompute", action="store_true", help="Check bundle structure/integrity without resampling metrics")
    args = parser.parse_args(argv)
    if args.bundle:
        if not args.gold or args.checkpoint or args.require_checkpoint:
            parser.error("Bundle mode requires --gold and does not accept checkpoint flags")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.experiments.reporting import audit_bundle
        try:
            result = audit_bundle(args.bundle, args.gold, recompute=not args.no_recompute)
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            print(f"Bundle audit failed: {exc}")
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.gold or args.no_recompute:
        parser.error("--gold and --no-recompute require --bundle")

    lines, baseline_present, checkpoint_present = audit(
        args.repo_root.resolve(), args.checkpoint.resolve() if args.checkpoint else None
    )
    print("Minimum baseline evidence audit (no metrics calculated)")
    for line in lines:
        print(f"- {line}")
    passed = baseline_present and (checkpoint_present or not args.require_checkpoint)
    print("Minimum artifacts: " + ("present" if passed else "INCOMPLETE"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
