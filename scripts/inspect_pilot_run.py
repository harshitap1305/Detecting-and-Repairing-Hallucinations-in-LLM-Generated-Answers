"""Summarize saved stage diagnostics; never promote them to human factuality."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.scoring import load_gold, load_run


def summarize(run):
    result = {}
    for system, rows in run['outputs'].items():
        result[system] = {}
        for rid, row in rows.items():
            record = row['record']
            tracking = record.get('claim_tracking') or {}
            final = tracking.get('final') or {}
            result[system][rid] = {
                'candidate': record['repaired_answer'],
                'verification': record.get('verification_diagnostics'),
                'coverage': final.get('coverage'),
                'candidate_only_unverified': final.get('candidate_only_unverified'),
                'repair': record.get('repair_diagnostics'),
                'retrieval_stages': [{ 'stage': s['stage'], 'retrieval': s.get('retrieval')}
                                     for s in tracking.get('stages', [])],
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--baseline-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gold = load_gold(a.gold)
    run = load_run(a.run_dir, gold)
    result = {'run_id': run['manifest']['run_id'],
              'interpretation': 'Model diagnostics, not independent human performance. Historical SCR semantics may differ.',
              'systems': summarize(run)}
    if a.baseline_run:
        baseline = load_run(a.baseline_run, gold)
        result['previous_run_id'] = baseline['manifest']['run_id']
        result['previous_systems'] = summarize(baseline)
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(a.output)


if __name__ == '__main__':
    main()
