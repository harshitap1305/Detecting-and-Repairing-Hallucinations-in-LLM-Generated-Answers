"""Complete/incomplete pilot products and export, using synthetic adapters only."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import zipfile

import tests.test_independent_evaluation as fixtures
from src.experiments.pilot import build_pilot_products
from src.experiments.runner import run_benchmark, ROOT, resolve_options
from src.experiments.runtime import recorder
from scripts.run_followup_pilot import main
from scripts.export_pilot_results import export_results


class TestPilotWorkflow(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TestOfflineAssessment(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.run=self.root/'repair'
        self.products=self.root/'repair_products'

    def execute(self, **kwargs):
        return run_benchmark(config_path=self.fixture.config_path,inputs_path=self.fixture.gold_path,
                             splits_path=self.fixture.splits,output_dir=self.run,task='repair',split='dev',**kwargs)

    def test_complete_products_audit_and_repeat_preserves_review_packet(self):
        self.execute()
        state=build_pilot_products(self.run,self.fixture.gold_path,self.products,n_bootstrap=10)
        directory=Path(state['products_dir'])
        self.assertTrue(state['execution_complete'])
        self.assertEqual(state['audit']['status'],'valid_progress_bundle')
        self.assertTrue((directory/'reference_metrics.json').exists())
        key=(directory/'review/private/key.json').read_bytes()
        again=build_pilot_products(self.run,self.fixture.gold_path,self.products,n_bootstrap=10)
        self.assertEqual(state,again)
        self.assertEqual(key,(directory/'review/private/key.json').read_bytes())
        archive=export_results(self.root,'repair',self.root/'results.zip')
        with zipfile.ZipFile(archive) as z:
            self.assertIn('repair_products/latest.json',z.namelist())
            self.assertIn(f"repair_products/{state['revision']}/review/reviewer/items.jsonl",z.namelist())

    def test_failed_extraction_exports_progress_and_findings_without_reviews_or_metrics(self):
        def flaky(system,record,task,options):
            if system['id']=='pipeline' and record.id=='d2':
                callback=recorder.get()
                callback({'event':'extraction','record_id':record.id,
                          'diagnostics':{'status':'unresolved_coverage','reason':'synthetic lost relation'}})
                raise ValueError('Synthetic extraction failure')
            return self.fixture.adapter(system,record,task,options)
        with patch('src.experiments.runner.adapters.execute',side_effect=flaky):
            self.execute()
        state=build_pilot_products(self.run,self.fixture.gold_path,self.products,n_bootstrap=10)
        directory=Path(state['products_dir'])
        self.assertFalse(state['execution_complete'])
        self.assertFalse((directory/'reference_metrics.json').exists())
        self.assertFalse((directory/'review').exists())
        diagnostic=json.loads((directory/'diagnostics.json').read_text())
        self.assertEqual(diagnostic['coverage']['pipeline']['missing_ids'],['d2'])
        self.assertEqual(diagnostic['comparisons']['status'],'unavailable')
        self.assertEqual(diagnostic['failures'][0]['extraction_events'][0]['diagnostics']['reason'],'synthetic lost relation')
        self.execute(resume=True)
        finished=build_pilot_products(self.run,self.fixture.gold_path,self.products,n_bootstrap=10)
        self.assertTrue(finished['execution_complete'])
        self.assertNotEqual(state['revision'],finished['revision'])
        self.assertTrue(directory.exists())
        self.assertFalse((directory/'review').exists())

    def test_changed_products_are_not_silently_reused(self):
        self.execute()
        state=build_pilot_products(self.run,self.fixture.gold_path,self.products,n_bootstrap=10)
        (Path(state['products_dir'])/'diagnostics.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'products changed'):
            build_pilot_products(self.run,self.fixture.gold_path,self.products,n_bootstrap=10)

    def test_command_exports_incomplete_run_and_returns_failure(self):
        def fail(system,record,task,options):
            raise ValueError('Synthetic model extraction failure')
        archive=self.root/'backup.zip'
        with patch('src.experiments.runner.adapters.execute',side_effect=fail):
            code=main(['--results-root',str(self.root),'--run-name','v4',
                       '--config',str(self.fixture.config_path),'--gold',str(self.fixture.gold_path),
                       '--splits',str(self.fixture.splits),'--output-zip',str(archive)])
        self.assertEqual(code,1)
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as z:
            self.assertIn('v4_products/latest.json',z.namelist())
            self.assertFalse(any('/review/' in name for name in z.namelist()))

    def test_profile_locks_same_system_scope_and_checked_extraction(self):
        config=json.loads(json.dumps(self.fixture.config))
        profile=json.loads((ROOT/'configs/pilot_followup_v4.options.json').read_text())
        options=resolve_options(self.fixture.config_path,config,profile)
        self.assertEqual(options['entity_planner_version'],2)
        self.assertEqual(options['extraction_check'],'semantic_v1')
        self.assertEqual(options['extraction_max_attempts'],2)
        self.assertEqual(options['extraction_max_checks'],2)
        self.assertEqual(options['repair_policy'],'rewrite-only')
        self.assertEqual(profile['model'],'openai/gpt-oss-120b')
        with self.assertRaisesRegex(ValueError,'planner version'):
            resolve_options(self.fixture.config_path,config,{**profile,'entity_planner_version':1})
