"""Export blinded packets, import human ratings, or adjudicate independent reviews."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments.assessment import export_packet, import_ratings, validate_assessments, adjudicate
from src.experiments.artifacts import canonical
from src.experiments.scoring import load_gold, load_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "import", "adjudicate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        gold = load_gold(args.gold)
        run = load_run(args.run_dir, gold)
        if args.action == "export":
            if not args.packet_dir:
                raise ValueError("export requires --packet-dir")
            export_packet(run, gold, args.packet_dir, args.seed)
            print(f"Review packet saved: {args.packet_dir}; share only reviewer/")
            return 0
        if not args.output or args.output.exists():
            raise ValueError("Supply a new --output path; existing files are not overwritten")
        if args.action == "import":
            if not args.packet_dir or not args.ratings or not args.reviewer:
                raise ValueError("import requires --packet-dir, --ratings and --reviewer")
            result = import_ratings(args.packet_dir, args.ratings, run, gold, args.reviewer)
            validate_assessments(result, run, gold)
        else:
            if not all((args.first, args.second, args.final)):
                raise ValueError("adjudicate requires --first, --second and --final imported reviews")
            reviews = [validate_assessments(json.loads(path.read_text()), run, gold)
                       for path in (args.first, args.second, args.final)]
            result = adjudicate(*reviews)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(canonical(result) + b"\n")
        print(f"Human review saved: {args.output}")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
