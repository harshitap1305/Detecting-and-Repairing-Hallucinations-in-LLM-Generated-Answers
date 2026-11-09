"""Build an offline, auditable final-task or progress report bundle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments.reporting import build_bundle, audit_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--assessments", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("final", "progress"), default="final")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", type=Path, help="Recompute and verify an existing bundle instead of building")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            if args.run_dir or args.output_dir or args.assessments:
                raise ValueError("--verify cannot be combined with build paths")
            result = audit_bundle(args.verify, args.gold)
            print(f"{result['status']}: offline metrics and report reproduced")
        else:
            if not args.run_dir or not args.output_dir:
                raise ValueError("Build requires --run-dir and --output-dir")
            output = build_bundle(run_dir=args.run_dir, gold_path=args.gold, output_dir=args.output_dir,
                                  assessments_path=args.assessments, mode=args.mode,
                                  n_bootstrap=args.n_bootstrap, seed=args.seed)
            print(f"{args.mode} bundle saved: {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Report failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
