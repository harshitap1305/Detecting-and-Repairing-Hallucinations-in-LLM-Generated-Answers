"""Snapshot integrity, secret exclusion, and effective model request provenance."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest

from src.experiments.artifacts import fingerprint
from src.experiments.provenance import (freeze_environment, source_hashes,
                                        save_reproduction, validate_reproduction)
from src.experiments.runtime import execution
from src.llm import call, _make_key
from scripts.export_pilot_results import export_results


def snapshot(tmp_path):
    root, out = tmp_path / 'repo', tmp_path / 'out'
    (root / 'src').mkdir(parents=True)
    out.mkdir()
    (root / 'src/code.py').write_text('# exact dirty bytes\nvalue = 7\n')
    (root / '.env').write_text('SECRET_SHOULD_NOT_BE_SNAPSHOTTED')
    identity = {'code': {'source_sha256': fingerprint(source_hashes(root))},
                'reproduction_environment_sha256': fingerprint(freeze_environment())}
    hashes = save_reproduction(root, out, identity)
    return root, out, {'identity': identity, 'reproduction_files_sha256': hashes}


def test_exact_source_and_environment_snapshot_without_secret_files(tmp_path, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'SECRET_NOT_IN_FREEZE')
    root, out, manifest = snapshot(tmp_path)
    validate_reproduction(out, manifest)
    with zipfile.ZipFile(out / 'source.zip') as z:
        assert z.namelist() == ['src/code.py']
        assert z.read('src/code.py') == (root / 'src/code.py').read_bytes()
    assert 'SECRET_NOT_IN_FREEZE' not in (out / 'environment.freeze.json').read_text()


@pytest.mark.parametrize('name', ['source.zip', 'source_manifest.json', 'environment.freeze.json'])
def test_modified_reproduction_artifact_invalidates_run(tmp_path, name):
    _, out, manifest = snapshot(tmp_path)
    (out / name).write_bytes(b'changed')
    with pytest.raises(ValueError):
        validate_reproduction(out, manifest)


def test_source_change_before_freeze_aborts(tmp_path):
    root, out, manifest = snapshot(tmp_path)
    (root / 'src/code.py').write_text('changed after identity capture')
    with pytest.raises(ValueError, match='Source changed'):
        save_reproduction(root, out, manifest['identity'])


def test_pilot_export_excludes_index_env_and_caches(tmp_path):
    root = tmp_path / 'results'
    run = root / 'new_run'
    run.mkdir(parents=True)
    (run / 'manifest.json').write_text('{}')
    (run / 'source.zip').write_bytes(b'snapshot')
    (run / 'llm_cache.db').write_bytes(b'cache')
    (root / 'fever_fts.sqlite').write_bytes(b'large index')
    (root / '.env').write_text('secret')
    out = export_results(root, 'new_run', tmp_path / 'results.zip')
    with zipfile.ZipFile(out) as z:
        assert set(z.namelist()) == {'new_run/manifest.json', 'new_run/source.zip'}


def test_reasoning_settings_change_cache_identity():
    with execution(settings={'reasoning_effort': 'low'}):
        low = _make_key('openai/gpt-oss-120b', 'prompt', 0., 4096, 42)
    with execution(settings={'reasoning_effort': 'high'}):
        high = _make_key('openai/gpt-oss-120b', 'prompt', 0., 4096, 42)
    assert low != high


def test_actual_completion_usage_and_effective_parameters_are_recorded():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Answer'), finish_reason='stop')],
                               usage=SimpleNamespace(prompt_tokens=4, completion_tokens=8, total_tokens=12))
    events = []
    with patch('groq.Groq') as client, execution(record=events.append, settings={
            'reasoning_effort': 'high', 'gptoss_min_completion_tokens': 8192}):
        client.return_value.chat.completions.create.return_value = response
        assert call('Synthetic prompt', model='openai/gpt-oss-120b', use_cache=False) == 'Answer'
        args = client.return_value.chat.completions.create.call_args.kwargs
        assert args['max_tokens'] == 8192
        assert args['reasoning_effort'] == 'high'
    meta = next(e for e in events if e.get('event') == 'completion_metadata')
    assert meta['finish_reason'] == 'stop'
    assert meta['usage']['total_tokens'] == 12
    assert events[-1]['backend_parameters']['reasoning_effort'] == 'high'


def test_options_file_is_versioned_and_explicit_cli_overrides(tmp_path):
    from scripts.run_benchmark import main
    options = tmp_path / 'options.json'
    options.write_text(json.dumps({'model': 'profile-model', 'repair_guard': 'grounded_v1'}))
    with patch('scripts.run_benchmark.run_benchmark', return_value={
            'status': 'execution_complete', 'successful_outputs': 10, 'expected_outputs': 10}) as runner:
        assert main(['--config', 'unused', '--inputs', 'unused', '--splits', 'unused',
                     '--output-dir', 'unused', '--task', 'repair', '--split', 'dev',
                     '--options-file', str(options), '--model', 'explicit-model']) == 0
    assert runner.call_args.kwargs['options'] == {'model': 'explicit-model', 'repair_guard': 'grounded_v1'}
