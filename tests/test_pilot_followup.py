"""Behavioral regressions for undefined SCR and contextual decomposition.

All examples are synthetic; no pilot gold enters production inference.
"""
from dataclasses import asdict
import json
from unittest.mock import patch

import pytest

from src.decompose import decompose
from src.evaluate import compute_scr, evaluate_dataset
from src.experiments.adapters import execute
from src.experiments.runner import DEFAULTS
from src.experiments.runtime import execution
from src.pipeline import run_pipeline
from src.schema import AnswerRecord, Claim
from src.verify import verify


def record(answer='The ferry operates daily.', question='How often does the Brook ferry operate?'):
    return AnswerRecord(id='synthetic', source_dataset='fixture', question=question, answer=answer)


def inventory(text, worthy=True):
    return json.dumps([{'claim': text, 'sentence_id': 0, 'check_worthy': worthy}])


@pytest.mark.parametrize('answer,claims,reason', [
    ('', [], 'empty_answer'),
    ('Beautiful!', [Claim('c', 'Beautiful!', 0, check_worthy=False)], 'no_check_worthy_claims'),
    ('A factual-looking sentence.', [], 'no_check_worthy_claims'),
])
def test_undefined_verification_has_reason_and_counts(answer, claims, reason):
    r = record(answer)
    r.claims = claims
    with patch('src.verify.llm.call', side_effect=AssertionError('No factual targets')):
        verify(r)
    assert r.reliability_score_before is None
    assert r.verification_diagnostics['before']['reason'] == reason
    assert r.verification_diagnostics['before']['check_worthy'] == 0
    assert compute_scr(r) is None


@pytest.mark.parametrize('labels,score', [
    (['Supported'], 1.), (['Unverifiable'], 0.),
    (['Supported', 'Contradicted', 'Unverifiable'], 1/3),
])
def test_defined_verification_counts(labels, score):
    r = record()
    r.claims = [Claim(str(i), 'A synthetic fact.', 0) for i in range(len(labels))]
    def verdict(claim, **kwargs):
        claim.label = labels[int(claim.id)]
        return claim
    with patch('src.verify.verify_claim_llm', side_effect=verdict):
        verify(r)
    assert r.reliability_score_before == score
    counts = r.verification_diagnostics['before']
    assert counts['supported'] == labels.count('Supported')
    assert counts['contradicted'] == labels.count('Contradicted')
    assert counts['unverifiable'] == labels.count('Unverifiable')


@pytest.mark.parametrize('budget', [0, 2])
def test_pipeline_nonfactual_answer_does_not_claim_success(budget, capsys):
    r = record('A wonderful sight!', 'What is your opinion?')
    with patch('src.llm.call', return_value=inventory(r.answer, False)), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r):
        run_pipeline(r, max_repair_iterations=budget, verbose=True)
    assert r.reliability_score_before is r.reliability_score_after is None
    assert r.verification_diagnostics['stop_reason'] == 'no_check_worthy_claims'
    assert r.repair_iterations == 0
    assert 'N/A' in capsys.readouterr().out


@pytest.mark.parametrize('candidate', ['According to reports, the ferry operates daily.',
                                       'In local practice, schedules.', 'Occasionally, daily.'])
def test_reverification_losing_all_targets_cannot_be_perfect(candidate):
    # Exercise the historical failure shape even when an upstream decomposer
    # returns an all-nonfactual candidate. This pipeline guard is independent
    # of the improved prompt and retry safeguards.
    r = record()
    def split(r, **kwargs):
        r.claims = [Claim(r.id + '_c1', r.answer, 0, check_worthy=r.id == 'synthetic')]
        return r
    def repair(r, **kwargs):
        r.repaired_answer = candidate
        return r
    with patch('src.pipeline.decompose', side_effect=split), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.verify.verify_claim_llm', side_effect=lambda c, **kw: setattr(c, 'label', 'Unverifiable')), \
            patch('src.pipeline.repair', side_effect=repair):
        run_pipeline(r)
    assert r.reliability_score_before == 0
    assert r.reliability_score_after is None
    assert r.verification_diagnostics['before']['unverifiable'] == 1
    assert r.verification_diagnostics['after']['check_worthy'] == 0
    assert r.verification_diagnostics['stop_reason'] == 'no_check_worthy_claims'
    assert compute_scr(r, use_after=True) is None  # must not use original claims


def test_noop_is_unresolved_not_convergence_success():
    r = record()
    r.claims = [Claim('c', r.answer, 0)]
    def no_op(r, **kw):
        r.repaired_answer = r.answer
        return r
    with patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.verify.verify_claim_llm', side_effect=lambda c, **kw: setattr(c, 'label', 'Unverifiable')), \
            patch('src.pipeline.repair', side_effect=no_op):
        run_pipeline(r, use_existing_claims=True)
    assert r.verification_diagnostics['stop_reason'] == 'unchanged_answer'
    assert r.reliability_score_after == 0


def test_historical_serialization_and_recorded_score_remain_unchanged():
    r = record()
    old = asdict(r)
    old.pop('verification_diagnostics')
    old.pop('decomposition_diagnostics')
    old.pop('claim_tracking')
    old.pop('retrieval_diagnostics')
    assert r.to_dict() == old
    # Historical stored 1.0 is read as stored, not reinterpreted retroactively.
    r.reliability_score_after = 1.
    assert compute_scr(r, use_after=True) == 1.
    assert AnswerRecord.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_null_diagnostics_roundtrip_and_legacy_aggregate_coverage():
    r = verify(record(''))
    r.verification_diagnostics['after'] = dict(r.verification_diagnostics['before'])
    loaded = AnswerRecord.from_jsonl_line(r.to_jsonl_line())
    assert loaded.verification_diagnostics == r.verification_diagnostics
    result = evaluate_dataset([loaded], n_bootstrap=10)
    assert result['scr']['delta']['mean'] is None
    assert result['scr']['before']['mean'] is None
    assert result['scr']['coverage']['paired_defined'] == 0
    assert result['scr']['raw_after'] == [None]


def test_native_adapter_preserves_versioned_diagnostics():
    r = record('Beautiful!', 'What is your opinion?')
    with patch('src.pipeline.decompose', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r):
        result = execute({'id': 'pipeline', 'retrieval': 'none'}, r, 'repair',
                         {**DEFAULTS, 'index_path': 'unused'})
    assert result['native']['self_verification_scr']['before'] is None
    assert result['native']['self_verification_scr']['details']['semantics_version'] == 2


@pytest.mark.parametrize('question,answer,claim', [
    ('Where was Mira Vale born?', 'Alder Bay', 'Mira Vale was born in Alder Bay.'),
    ('How often is the Northbridge Review published?', 'Monthly', 'The Northbridge Review is published monthly.'),
    ('What does Cedar teach? What activity measures distances?', 'Surveying', 'Surveying involves measuring distances.'),
    ('When did they arrive?', '1998', 'The people referred to in the question arrived in 1998.'),
])
def test_contextual_fragments_retry_and_keep_original_span(question, answer, claim):
    r = record(answer, question)
    with patch('src.llm.call', side_effect=[inventory(answer, False), inventory(claim)]) as call:
        decompose(r)
    assert call.call_count == 2
    assert call.call_args.kwargs['use_cache'] is False
    assert 'RETRY FEEDBACK' in call.call_args.kwargs['prompt']
    assert r.claims[0].text == claim
    assert r.claims[0].span == [0, len(answer)]
    assert r.answer == answer
    assert r.decomposition_diagnostics['status'] == 'complete'


@pytest.mark.parametrize('answer,claim', [
    ('Reportedly, Mira Vale was born in Alder Bay.', 'Reportedly, Mira Vale was born in Alder Bay.'),
    ('Mira Vale might have arrived in 1998.', 'Mira Vale might have arrived in 1998.'),
    ('Usually monthly', 'The Northbridge Review is usually published monthly.'),
])
def test_hedged_factual_payload_remains_checkworthy_without_strengthening(answer, claim):
    r = record(answer)
    with patch('src.llm.call', side_effect=[inventory('Mira Vale was born in Alder Bay.'), inventory(claim)]) as call:
        decompose(r)
    assert call.call_count == 2
    assert r.check_worthy_claims[0].text == claim


def test_exhausted_fragment_coverage_never_invents_a_fact():
    r = record('Monthly')
    with patch('src.llm.call', return_value=inventory('Monthly', False)) as call:
        decompose(r)
    assert call.call_count == 2
    assert r.claims == []
    assert r.decomposition_diagnostics['status'] == 'unresolved_coverage'
    with patch('src.llm.call', return_value=inventory('Monthly', False)), execution():
        with pytest.raises(ValueError, match='unresolved coverage'):
            decompose(record('Monthly'))


@pytest.mark.parametrize('answer', ['I think it is beautiful.', 'Please visit the museum.',
                                    'Have you visited the museum?', 'As an AI, I cannot answer.'])
def test_nonfactual_text_does_not_trigger_factual_fragment_retry(answer):
    r = record(answer, 'What is your opinion?')
    with patch('src.llm.call', return_value=inventory(answer, False)) as call:
        decompose(r)
    assert call.call_count == 1
    assert not r.check_worthy_claims
    assert r.decomposition_diagnostics['status'] == 'no_check_worthy_claims'


def test_demo_displays_na_instead_of_invented_scores():
    from demo.app import format_metrics_banner
    html = format_metrics_banner(record(''))
    assert html.count('N/A') == 3
    assert '50.0%' not in html


def test_after_score_is_not_invented_when_only_initial_verification_ran():
    r = record()
    r.claims = [Claim('c', r.answer, 0)]
    with patch('src.verify.verify_claim_llm', side_effect=lambda c, **kw: setattr(c, 'label', 'Supported')):
        verify(r)
    assert compute_scr(r) == 1.
    assert compute_scr(r, use_after=True) is None


def test_legacy_cli_displays_null_scr_without_crashing(tmp_path, capsys):
    from scripts.run_eval import main
    path = tmp_path / 'records.jsonl'
    path.write_text(verify(record('')).to_jsonl_line() + '\n')
    main(['--results', str(path), '--n_bootstrap', '10'])
    output = capsys.readouterr().out
    assert 'SCR BEFORE: N/A' in output
    assert 'SCR AFTER: N/A' in output


def test_paired_cli_reports_undefined_scr_without_dropping_records(tmp_path, capsys):
    from scripts.compute_significance import main
    path = tmp_path / 'records.jsonl'
    output = tmp_path / 'metrics.json'
    path.write_text(verify(record('')).to_jsonl_line() + '\n')
    main(['--system', str(path), '--baseline', str(path), '--output', str(output), '--n_bootstrap', '10'])
    metrics = json.loads(output.read_text())
    assert metrics['scr']['status'] == 'unavailable'
    assert metrics['scr']['undefined_record_ids'] == ['synthetic']
    assert 'scr: N/A' in capsys.readouterr().out


def test_reverification_invalidates_a_stale_final_score():
    r = record('')
    r.reliability_score_after = 1.
    verify(r)
    assert r.reliability_score_after is None
    assert compute_scr(r, use_after=True) is None
