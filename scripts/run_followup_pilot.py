"""Run the v4 dev pilot, audit progress, and export artifacts even after failures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.runner import ROOT, run_benchmark
from src.experiments.pilot import build_pilot_products
from scripts.export_pilot_results import export_results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--run-name',default='pilot_followup_v4')
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation_halueval_colab.yaml')
    p.add_argument('--options-file',type=Path,default=ROOT/'configs/pilot_followup_v4.options.json')
    p.add_argument('--gold',type=Path,default=ROOT/'data/gold/pilot_dev.jsonl')
    p.add_argument('--splits',type=Path,default=ROOT/'data/gold/pilot_splits.json')
    p.add_argument('--previous-run',type=Path)
    p.add_argument('--output-zip',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--model')
    p.add_argument('--reasoning-effort',choices=('low','medium','high'))
    p.add_argument('--gptoss-min-completion-tokens',type=int)
    a=p.parse_args(argv)
    if not a.run_name or Path(a.run_name).name != a.run_name or a.run_name in ('.','..'):
        p.error('run-name must be a single directory name')
    a.results_root.mkdir(parents=True,exist_ok=True)
    run_dir=a.results_root/a.run_name
    code=1
    try:
        options=json.loads(a.options_file.read_text())
        if not isinstance(options,dict):
            raise ValueError('Options file must be a JSON object')
        for key in ('model','reasoning_effort','gptoss_min_completion_tokens'):
            value=getattr(a,key)
            if value is not None:
                options[key]=value
        print('Starting dev execution; existing runs and the index are preserved.',flush=True)
        manifest=run_benchmark(config_path=a.config,inputs_path=a.gold,splits_path=a.splits,
                               output_dir=run_dir,task='repair',split='dev',options=options,resume=a.resume)
        print(f"Execution: {manifest['status']}; {manifest['successful_outputs']}/{manifest['expected_outputs']}",flush=True)
        products=build_pilot_products(run_dir,a.gold,a.results_root/f'{a.run_name}_products',a.previous_run)
        print(json.dumps(products,indent=2),flush=True)
        if manifest['status'] != 'execution_complete':
            diagnostics=json.loads((Path(products['products_dir'])/'diagnostics.json').read_text())
            for failure in diagnostics['failures']:
                print(json.dumps({key:failure[key] for key in ('system','record_id','attempt','error')},ensure_ascii=False),flush=True)
        code=0 if manifest['status']=='execution_complete' else 1
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        print(f'Pilot stopped: {exc}',file=sys.stderr,flush=True)
    finally:
        if (run_dir/'manifest.json').is_file():
            try:
                export_results(a.results_root,a.run_name,a.output_zip)
                print('Download backup before ending the runtime:',a.output_zip,flush=True)
            except (OSError,ValueError) as exc:
                print(f'Backup failed: {exc}',file=sys.stderr,flush=True)
                code=1
    return code


if __name__=='__main__':
    raise SystemExit(main())
