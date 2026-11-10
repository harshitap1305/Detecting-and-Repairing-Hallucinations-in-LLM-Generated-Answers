"""Execute declared systems; save attempts and resume only identical experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments.runner import DEFAULTS, run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "inputs", "splits", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--options-file", type=Path, help="Versioned JSON runtime options; explicit CLI flags override")
    parser.add_argument("--task", choices=("repair", "claim_verification", "answer_detection"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--resume", action="store_true")
    for name, value in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), default=argparse.SUPPRESS,
                            type=int if type(value) is int else float if type(value) is float else str)
    args = vars(parser.parse_args(argv))
    options = {name: args.pop(name) for name in DEFAULTS if name in args}
    options_file = args.pop("options_file")
    args.update(config_path=args.pop("config"), inputs_path=args.pop("inputs"),
                splits_path=args.pop("splits"), inventory_path=args.pop("inventory"))
    try:
        if options_file:
            frozen = json.loads(options_file.read_text())
            if not isinstance(frozen, dict):
                raise ValueError("Options file must be a JSON object")
            options = {**frozen, **options}
        manifest = run_benchmark(**args, options=options)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Benchmark aborted: {exc}", file=sys.stderr)
        return 1
    print(f"{manifest['status']}: {manifest['successful_outputs']}/{manifest['expected_outputs']} outputs")
    return 0 if manifest["status"] == "execution_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
