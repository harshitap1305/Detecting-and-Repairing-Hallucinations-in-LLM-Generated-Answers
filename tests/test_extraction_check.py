"""Bounded semantic checks, failure cases, and pipeline extraction gating."""
import copy
import json
from unittest.mock import patch
import pytest

from src.answer_attribution import build_attribution
from src.decompose import decompose
from src.extraction_check import check_extraction, extraction_coverage
from src.experiments.runtime import execution
from src.pipeline import run_pipeline
from src.schema import AnswerRecord, Claim

Q = 'Which event did Nera win? When did she retire?'
A = 'The Alder Cup'


def proposal(text='The Alder Cup was won by Nera.', clause='q1'):
    return {'claim': text, 'sentence_id': 0, 'check_worthy': True,
        'attribution': {'status': 'bound', 'answer_spans': [[0, len(A)]],
            'bindings': [{'clause_id': clause, 'answer_span': [0, len(A)], 'answer_text': A,
                         'relation': 'event won', 'borrowed_context': []}],
            'rationale': 'This fragment identifies the event she won.'}}


def result(items, verdicts=None, missing=False):
    verdicts = verdicts or ['expressed'] * len(items)
    checks = [{'claim_id': f'r_c{i+1}', 'verdict': verdict, 'clause_ids': [p['attribution']['bindings'][0]['clause_id']],
               'rationale': 'Semantic relationship inspected against the raw answer.'}
              for i, (p, verdict) in enumerate(zip(items, verdicts))]
    clauses = []
    for qid in ('q1', 'q2'):
        expressed = [c['claim_id'] for c in checks if qid in c['clause_ids'] and c['verdict'] == 'expressed']
        bad = [c['claim_id'] for c in checks if qid in c['clause_ids'] and c['verdict'] == 'wrong_relation']
        clauses.append({'clause_id': qid, 'verdict': 'represented' if expressed else 'missing_relation' if bad else 'unanswered',
                        'claim_ids': expressed or bad, 'rationale': 'The retirement clause can legitimately remain unanswered.'})
    return {'claim_checks': checks, 'clause_checks': clauses,
            'missing_content': [{'answer_span': [0, len(A)], 'clause_ids': ['q1'],
                                 'rationale': 'The event-winning relationship was not extracted.'}] if missing else []}


def audit(items, response, **kwargs):
    attrs = build_attribution(items, question=Q, answer=A, sentence_spans=[(A, 0, len(A))], record_id='r')
    with patch('src.llm.call', return_value=json.dumps(response)) as model:
        data = check_extraction(question=Q, answer=A, items=items, attribution=attrs,
                                record_id='r', model='mock', provider='groq', **kwargs)
    return data, model


def test_paraphrased_relationship_and_unanswered_clause_are_valid():
    items = [proposal()]
    check, _ = audit(items, result(items))
    assert check['status'] == 'passed'
    assert check['clause_checks'][1]['verdict'] == 'unanswered'


def test_related_event_description_cannot_substitute_for_winning_relation():
    items = [proposal('The Alder Cup took place in 1998.')]
    check, _ = audit(items, result(items, ['wrong_relation'], missing=True))
    assert check['status'] == 'rejected'
    assert check['missing_content'][0]['clause_ids'] == ['q1']


def test_extra_answer_to_an_unaddressed_question_clause_is_rejected():
    items = [proposal(), proposal('Nera retired in 2001.', 'q2')]
    check, _ = audit(items, result(items, ['expressed', 'unsupported_assertion']))
    assert check['status'] == 'rejected'
    assert check['claim_checks'][1]['claim_id'] == 'r_c2'


@pytest.mark.parametrize('mutate', [
    lambda r: r['claim_checks'].clear(),
    lambda r: r['claim_checks'].append(copy.deepcopy(r['claim_checks'][0])),
    lambda r: r['claim_checks'][0].update(claim_id='unknown'),
    lambda r: r['claim_checks'][0].update(clause_ids=['unknown']),
    lambda r: r['clause_checks'][0].update(verdict='unanswered', claim_ids=[]),
    lambda r: r['clause_checks'][0].update(claim_ids=[]),
    lambda r: r['clause_checks'].pop(),
    lambda r: r.update(approved=True),
    lambda r: r.update(missing_content=[{'answer_span':[0,999],'clause_ids':['q1'],'rationale':'bad span'}]),
    lambda r: r['claim_checks'][0].update(verdict='nonfactual'),
])
def test_malformed_incomplete_or_contradictory_checks_cannot_approve(mutate):
    items = [proposal()]
    response = result(items)
    mutate(response)
    check, model = audit(items, response)
    assert model.call_count == 1
    assert check['status'] == 'unresolved'
    assert check['reason'] == 'malformed_semantic_check'


def test_missing_attribution_and_oversized_inventory_need_no_semantic_call():
    items = [proposal()]
    attrs = build_attribution(items, question=Q, answer=A, sentence_spans=[(A, 0, len(A))], record_id='r')
    with patch('src.llm.call', side_effect=AssertionError('Budget/preconditions')):
        missing = check_extraction(question=Q, answer=A, items=items,
            attribution={'status':'not_provided'}, record_id='r', model='mock', provider='groq')
        capped = check_extraction(question=Q, answer=A, items=items,
            attribution=attrs, record_id='r', model='mock', provider='groq', max_claims=1)
    assert missing['reason'] == 'missing_or_partial_attribution'
    assert capped['reason'] == 'semantic_check_inventory_budget_exceeded'


def test_rejected_extraction_retries_with_findings_and_preserves_offsets():
    bad, good = [proposal('The Alder Cup took place in 1998.')], [proposal()]
    sequence = [json.dumps(bad), json.dumps(result(bad, ['wrong_relation'], missing=True)),
                json.dumps(good), json.dumps(result(good))]
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    with patch('src.llm.call', side_effect=sequence) as model, execution():
        decompose(r, extraction_check='semantic_v1')
    assert model.call_count == 4
    assert 'wrong_relation' in model.call_args_list[2].kwargs['prompt']
    assert r.decomposition_diagnostics['semantic_checks_used'] == 2
    assert extraction_coverage(r)['status'] == 'passed'
    assert r.claims[0].span == [0, len(A)]
    r.claims[0].text = 'An unaudited replacement proposition.'
    assert extraction_coverage(r)['reason'] == 'semantic_check_input_mismatch'


def test_exhausted_check_budget_stops_without_an_extra_generation():
    bad = [proposal('The Alder Cup took place in 1998.')]
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    with patch('src.llm.call', side_effect=[json.dumps(bad), json.dumps(result(bad, ['wrong_relation']))]) as model:
        decompose(r, extraction_check='semantic_v1', max_retries=4, max_checks=1)
    assert model.call_count == 2
    assert r.claims == []
    assert r.decomposition_diagnostics['stop_reason'] == 'semantic_check_budget_exhausted'
    events = []
    with patch('src.llm.call', side_effect=[json.dumps(bad), json.dumps(result(bad, ['wrong_relation']))]), execution(record=events.append):
        with pytest.raises(ValueError, match='Semantic extraction unresolved'):
            decompose(r, extraction_check='semantic_v1', max_checks=1)
    assert events[-1]['event'] == 'extraction'
    assert events[-1]['diagnostics']['stop_reason'] == 'semantic_check_budget_exhausted'


def test_checker_request_has_no_evidence_reference_or_factual_labels():
    items = [proposal()]
    _, model = audit(items, result(items))
    payload = json.loads(model.call_args.args[0].rsplit('\n', 1)[1])
    assert set(payload) == {'question','answer','claims','proposed_attribution'}
    assert all(set(c) == {'claim_id','text','sentence_id','check_worthy'} for c in payload['claims'])


def test_default_pipeline_requires_checked_extraction_for_success():
    items = [proposal()]
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    def verified(candidate, **kwargs):
        for c in candidate.claims:
            c.label = 'Supported'
        return candidate
    with patch('src.llm.call', side_effect=[json.dumps(items), json.dumps(result(items))]), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.verify', side_effect=verified):
        run_pipeline(r)
    assert r.verification_diagnostics['stop_reason'] == 'all_targets_verified'
    assert r.claim_tracking['final']['factual_success']
    assert r.verification_diagnostics['extraction_coverage']['before']['status'] == 'passed'


def test_supported_but_unassessed_inventory_cannot_pass_end_to_end():
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A,
                     claims=[Claim('c','A related event date.',0,label='Supported')])
    with patch('src.pipeline.decompose', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.verify', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.repair', side_effect=AssertionError('Unresolved extraction')):
        run_pipeline(r)
    assert r.reliability_score_after == 1
    assert r.verification_diagnostics['stop_reason'] == 'unresolved_extraction_coverage'
    assert r.claim_tracking['final']['target_verification_success']
    assert not r.claim_tracking['final']['factual_success']


def test_valid_extraction_with_missing_evidence_stays_unchanged_unresolved():
    items = [proposal()]
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    def uncertain(candidate, **kw):
        for c in candidate.claims:
            c.label = 'Unverifiable'
        return candidate
    def unchanged(candidate, **kw):
        candidate.repaired_answer = candidate.answer
        return candidate
    with patch('src.llm.call', side_effect=[json.dumps(items), json.dumps(result(items))]), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.verify', side_effect=uncertain), \
            patch('src.pipeline.repair', side_effect=unchanged):
        run_pipeline(r)
    assert r.verification_diagnostics['stop_reason'] == 'unchanged_answer'
    assert r.verification_diagnostics['extraction_coverage']['after']['status'] == 'passed'
    assert r.reliability_score_after == 0
    assert not r.claim_tracking['final']['factual_success']


def test_explicit_answers_to_both_clauses_can_be_accepted():
    answer = 'The Alder Cup; 2001.'
    items = [proposal(), proposal('Nera retired in 2001.', 'q2')]
    items[1]['attribution']['bindings'][0]['relation'] = 'retirement year'
    for item, span in zip(items, ([0,13], [15,19])):
        item['attribution']['answer_spans'] = [span]
        item['attribution']['bindings'][0].update(answer_span=span, answer_text=answer[span[0]:span[1]])
    attrs = build_attribution(items, question=Q, answer=answer,
                              sentence_spans=[(answer,0,len(answer))], record_id='r')
    with patch('src.llm.call', return_value=json.dumps(result(items))):
        data = check_extraction(question=Q, answer=answer, items=items, attribution=attrs,
                                record_id='r', model='mock', provider='groq')
    assert data['status'] == 'passed'


def test_lost_uncertainty_or_negation_can_be_rejected_as_an_extra_assertion():
    items = [proposal('Nera definitely won the Alder Cup.')]
    # The checker can flag strengthening independently of lexical overlap.
    data, _ = audit(items, result(items, ['unsupported_assertion']))
    assert data['status'] == 'rejected'


def test_candidate_extraction_failure_is_distinct_from_no_factual_content():
    items = [proposal()]
    original = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    with patch('src.llm.call', side_effect=[json.dumps(items), json.dumps(result(items))]):
        decompose(original, extraction_check='semantic_v1')
    def split(candidate, **kw):
        if candidate.id == 'r':
            return original
        candidate.claims = []
        candidate.decomposition_diagnostics = {'status':'unresolved_coverage', 'reason':'wrong candidate relationship',
            'semantic_check': {'status':'rejected', 'reason':'semantic_extraction_findings'}}
        return candidate
    def uncertain(candidate, **kw):
        for c in candidate.claims:
            c.label = 'Unverifiable'
        return candidate
    def repair(candidate, **kw):
        candidate.repaired_answer = 'A related date without the winning relationship.'
        return candidate
    with patch('src.pipeline.decompose', side_effect=split), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.verify', side_effect=uncertain), \
            patch('src.pipeline.repair', side_effect=repair):
        result_record = run_pipeline(original)
    assert result_record.verification_diagnostics['stop_reason'] == 'unresolved_extraction_coverage'
    assert result_record.verification_diagnostics['extraction_coverage']['before']['status'] == 'passed'
    assert result_record.verification_diagnostics['extraction_coverage']['after']['status'] == 'unresolved'
    assert result_record.claim_tracking['stages'][-1]['decomposition']['reason'] == 'wrong candidate relationship'
    assert result_record.reliability_score_after is None


@pytest.mark.parametrize('options', [dict(extraction_check='typo'), dict(extraction_max_attempts=0),
                                     dict(extraction_max_checks=5), dict(extraction_max_claims=129)])
def test_pipeline_rejects_invalid_extraction_settings_before_model_calls(options):
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    with patch('src.llm.call', side_effect=AssertionError('Invalid settings')):
        with pytest.raises(ValueError):
            run_pipeline(r, **options)


def test_fixed_inventory_can_still_be_repaired_without_claiming_extraction_success():
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A,
                     claims=[Claim('supplied','Nera won the Alder Cup.',0,label='Unverifiable')])
    def unchanged(candidate, **kw):
        candidate.repaired_answer = candidate.answer
        return candidate
    with patch('src.pipeline.decompose', side_effect=AssertionError('Supplied original')), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.verify', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.repair', side_effect=unchanged) as repair:
        run_pipeline(r, use_existing_claims=True)
    assert repair.call_count == 1
    assert r.verification_diagnostics['stop_reason'] == 'unchanged_answer'
    assert r.verification_diagnostics['extraction_coverage']['before']['status'] == 'unresolved'
    assert not r.claim_tracking['final']['factual_success']


def test_checked_nonfactual_answer_has_no_factual_success_or_fake_score():
    answer = 'Beautiful!'
    question = 'What is your opinion?'
    raw = {'claim': answer, 'sentence_id': 0, 'check_worthy': False,
           'attribution': {'status':'nonfactual','answer_spans':[[0,len(answer)]],
                           'bindings':[], 'rationale':'A subjective opinion.'}}
    response = {'claim_checks':[{'claim_id':'r_c1', 'verdict':'nonfactual', 'clause_ids':[], 'rationale':'Opinion only.'}],
                'clause_checks':[{'clause_id':'q1','verdict':'unanswered','claim_ids':[], 'rationale':'No factual question answer.'}],
                'missing_content':[]}
    r = AnswerRecord(id='r', source_dataset='synthetic', question=question, answer=answer)
    with patch('src.llm.call', side_effect=[json.dumps([raw]), json.dumps(response)]), \
            patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r):
        run_pipeline(r)
    assert r.verification_diagnostics['stop_reason'] == 'no_check_worthy_claims'
    assert r.reliability_score_before is r.reliability_score_after is None
    assert r.verification_diagnostics['extraction_coverage']['before']['status'] == 'passed'
    assert not r.claim_tracking['final']['factual_success']
