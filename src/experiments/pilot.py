"""Postprocess one complete or incomplete dev run without model calls.

Every manifest revision gets immutable products. Partial runs have coverage and
failure diagnostics only, never reference comparisons or review packets.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile

from src.experiments.artifacts import atomic_json, file_hash, fingerprint
from src.experiments.scoring import load_gold, load_run, evaluate_run
from src.experiments.assessment import export_packet
from src.experiments.reporting import build_bundle, audit_bundle
from scripts.inspect_pilot_run import summarize


def failure_diagnostics(run_dir: Path):
    failures = []
    for path in sorted((run_dir / 'attempts').glob('*/*.json')):
        row = json.loads(path.read_text())['result']
        if row['status'] != 'error':
            continue
        events = [json.loads(line) for line in (run_dir / row['trace_path']).read_text().splitlines() if line.strip()]
        failures.append({'system':row['system'], 'record_id':row['input']['id'], 'attempt':row['attempt'],
                         'error':row['error'], 'trace_path':row['trace_path'],
                         'extraction_events':[event for event in events if event.get('event') == 'extraction']})
    return failures


def build_pilot_products(run_dir: Path, gold_path: Path, products_root: Path,
                         previous_run: Path | None = None, n_bootstrap=1000):
    gold = load_gold(gold_path)
    run = load_run(run_dir, gold, allow_partial=True)
    complete = run['manifest']['status'] == 'execution_complete'
    identity = {'run_id':run['manifest']['run_id'], 'manifest_sha256':run['manifest_sha256'],
                'gold_sha256':file_hash(gold_path), 'n_bootstrap':n_bootstrap,
                'previous_manifest_sha256':file_hash(previous_run/'manifest.json') if previous_run and complete else None}
    revision = fingerprint(identity)[:20]
    products_root.mkdir(parents=True, exist_ok=True)
    destination = products_root / revision
    if destination.exists():
        state = json.loads((destination/'state.json').read_text())
        if state['identity'] != identity:
            raise ValueError('Pilot product identity differs')
        inventory={str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file() and p.name!='state.json'}
        if inventory != set(state['files_sha256']) or any(file_hash(destination/name) != sha for name,sha in state['files_sha256'].items()):
            raise ValueError('Existing pilot products changed')
        audit_bundle(destination/'progress',gold_path,recompute=True)
    else:
        staging = Path(tempfile.mkdtemp(prefix='.pilot-products-',dir=products_root))
        try:
            coverage = {system:{'successful':len(rows), 'expected':len(gold),
                               'missing_ids':sorted(set(gold)-set(rows))} for system,rows in run['outputs'].items()}
            diagnostics = {'run_id':identity['run_id'], 'status':run['manifest']['status'], 'coverage':coverage,
                           'interpretation':'Model diagnostics only; no independent human performance claims.',
                           'systems':summarize(run), 'failures':failure_diagnostics(run_dir),
                           'comparisons':{'status':'unavailable','reason':'Incomplete execution' if not complete else 'Human assessments pending'}}
            if complete and previous_run:
                previous = load_run(previous_run,gold)
                diagnostics.update(previous_run_id=previous['manifest']['run_id'], previous_systems=summarize(previous))
            atomic_json(staging/'diagnostics.json',diagnostics)
            if complete:
                from src.experiments.runner import code_identity, environment
                metrics = evaluate_run(run,gold,None,n_bootstrap,42)
                metrics['scoring_provenance'] = {'code':code_identity(),'environment':environment(),
                                                 'n_bootstrap':n_bootstrap,'seed':42}
                metrics['provenance'] = identity
                atomic_json(staging/'reference_metrics.json',metrics)
                export_packet(run,gold,staging/'review')
            build_bundle(run_dir=run_dir,gold_path=gold_path,output_dir=staging/'progress',
                         mode='progress',n_bootstrap=n_bootstrap)
            audit = audit_bundle(staging/'progress',gold_path,recompute=True)
            inventory = {str(p.relative_to(staging)):file_hash(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            state = {'identity':identity, 'execution_complete':complete, 'audit':audit,
                     'review_status':'inspect_outputs_before_independent_review' if complete else 'no_review_incomplete_execution',
                     'files_sha256':inventory}
            atomic_json(staging/'state.json',state)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    latest = {'revision':revision, 'products_dir':str(destination.resolve()),
              'execution_complete':complete, 'review_status':state['review_status'], 'audit':state['audit']}
    atomic_json(products_root/'latest.json',latest)
    return latest
