"""The contract checks attribution structure, not the truth of model proposals."""
import copy
import json
from unittest.mock import patch
import pytest

from src.answer_attribution import build_attribution, question_clauses
from src.decompose import decompose
from src.schema import AnswerRecord
from src.experiments.runtime import execution


Q = 'Where was Nera born? What year did she move?'
A = 'Cedar Bay'


def item(question=Q, answer=A, status='bound', clauses=('q1',)):
    context = {'source': 'question', 'span': [question.index('Nera'), question.index('Nera')+4],
               'text': 'Nera', 'role': 'referent'}
    return {'claim': 'Nera was born in Cedar Bay.', 'sentence_id': 0, 'check_worthy': status != 'nonfactual',
            'attribution': {'status': status, 'answer_spans': [[0, len(answer)]],
                           'bindings': [{'clause_id': cid, 'answer_span': [0, len(answer)], 'answer_text': answer,
                                         'relation': 'birthplace', 'borrowed_context': [context]} for cid in clauses],
                           'rationale': 'The location fragment supplies the birthplace value.'}}


def build(items, question=Q, answer=A):
    return build_attribution(items, question=question, answer=answer,
                             sentence_spans=[(answer, 0, len(answer))], record_id='r')


def test_surface_clauses_preserve_question_offsets():
    q = '  Where was Nera born and when did she move?\n Why leave; How far?'
    clauses = question_clauses(q)
    assert [c['text'] for c in clauses] == ['Where was Nera born', 'when did she move', 'Why leave', 'How far']
    assert [c['clause_id'] for c in clauses] == ['q1', 'q2', 'q3', 'q4']
    assert all(q[c['span'][0]:c['span'][1]] == c['text'] for c in clauses)


def test_partial_compound_answer_does_not_automatically_answer_other_clause():
    data = build([item()])
    assert data['status'] == 'proposed'
    assert [c['status'] for c in data['clause_coverage']] == ['proposed_answered', 'unanswered']
    assert data['validation'] == 'structural_only_not_semantic_or_factual'
    assert data['claims'][0]['claim_id'] == 'r_c1'


def test_a_shared_span_can_be_proposed_for_multiple_explicit_relations():
    a = 'Cedar Bay is her birthplace and current home.'
    q = 'Where was Nera born? Where does she live?'
    first = item(q, a)
    second = copy.deepcopy(first)
    second['claim'] = 'Nera lives in Cedar Bay.'
    second['attribution']['bindings'][0].update(clause_id='q2', relation='current residence')
    data = build([first, second], q, a)
    assert [c['status'] for c in data['clause_coverage']] == ['proposed_answered'] * 2


def test_ambiguous_binding_is_explicit_without_guessed_people_or_dates():
    data = build([item(status='ambiguous', clauses=())])
    assert all(c['status'] == 'ambiguous' for c in data['clause_coverage'])
    assert data['claims'][0]['bindings'] == []


def test_answer_only_and_nonfactual_content_can_have_no_question_binding():
    for status in ('answer_only', 'nonfactual'):
        data = build([item(status=status, clauses=())])
        assert data['claims'][0]['status'] == status
        assert all(c['status'] == 'unanswered' for c in data['clause_coverage'])


def test_context_can_resolve_an_answer_pronoun_from_earlier_answer_text():
    answer = 'Nera was born in Cedar Bay. She moved in 1998.'
    start = answer.index('She')
    value = item(answer=answer)
    value.update(claim='Nera moved in 1998.', sentence_id=1)
    attr = value['attribution']
    attr['answer_spans'] = [[start, len(answer)]]
    attr['bindings'][0].update(clause_id='q2', answer_span=[start, len(answer)], answer_text=answer[start:],
        relation='move year', borrowed_context=[{'source':'answer', 'span':[0,4], 'text':'Nera', 'role':'referent'}])
    result = build_attribution([value], question=Q, answer=answer,
        sentence_spans=[(answer[:start],0,start),(answer[start:],start,len(answer))], record_id='r')
    assert result['claims'][0]['bindings'][0]['borrowed_context'][0]['source'] == 'answer'


@pytest.mark.parametrize('mutation', [
    lambda v: v['attribution'].update(answer_spans=[[0, 999]]),
    lambda v: v['attribution']['bindings'][0].update(clause_id='missing'),
    lambda v: v['attribution']['bindings'][0].update(answer_text='Invented location'),
    lambda v: v['attribution']['bindings'][0].update(answer_span=[True, 2]),
    lambda v: v['attribution']['bindings'][0]['borrowed_context'][0].update(text='Invented referent'),
    lambda v: v['attribution']['bindings'][0]['borrowed_context'][0].update(role='factual_evidence'),
    lambda v: v['attribution'].update(bindings=[]),
    lambda v: v['attribution'].update(status='nonfactual'),
    lambda v: v['attribution'].update(typo=True),
    lambda v: v.update(attribution=None),
])
def test_structurally_invalid_attribution_is_rejected(mutation):
    value = item()
    mutation(value)
    with pytest.raises(ValueError):
        build([value])


def test_missing_legacy_attribution_stays_unknown_not_accepted():
    value = item()
    value.pop('attribution')
    data = build([value])
    assert data['status'] == 'not_provided'
    assert all(c['status'] == 'unknown' for c in data['clause_coverage'])


def test_valid_claim_attribution_survives_real_decompose_and_record_roundtrip():
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    with patch('src.llm.call', return_value=json.dumps([item()])):
        decompose(r)
    attribution = r.decomposition_diagnostics['attribution']
    assert r.claims[0].span == [0, len(A)]
    assert AnswerRecord.from_jsonl_line(r.to_jsonl_line()).decomposition_diagnostics['attribution'] == attribution


def test_bad_attribution_retries_with_feedback_and_exhaustion_is_unresolved():
    bad = item()
    bad['attribution']['bindings'][0]['answer_text'] = 'Not in the answer'
    r = AnswerRecord(id='r', source_dataset='synthetic', question=Q, answer=A)
    with patch('src.llm.call', side_effect=[json.dumps([bad]), json.dumps([item()])]) as model:
        decompose(r)
    assert model.call_count == 2
    assert 'Malformed answer attribution' in model.call_args.kwargs['prompt']
    with patch('src.llm.call', return_value=json.dumps([bad])), execution():
        with pytest.raises(ValueError, match='Malformed answer attribution'):
            decompose(r)


def test_contract_is_copied_and_is_not_a_semantic_checker():
    v = item()
    # A deliberately wrong relation descriptor is structurally valid. Phase 3
    # must check semantics rather than trust this model-proposed description.
    v['attribution']['bindings'][0]['relation'] = 'an incorrect relation proposal'
    data = build([v])
    v['attribution']['bindings'].clear()
    assert data['claims'][0]['bindings']
    assert data['validation'] == 'structural_only_not_semantic_or_factual'
