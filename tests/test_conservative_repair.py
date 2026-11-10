"""Guarded corrections, unresolved fallback, and exact preservation of text."""
import json
from unittest.mock import patch
import pytest
from src.schema import AnswerRecord, Claim, EvidencePassage
from src.repair import repair


def source(mixed=False):
    evidence = EvidencePassage('e', 'Mira', 'Mira was born in Alder Bay in 1998.')
    claims = [Claim('wrong', 'Mira was born in Birch Bay.', 0, label='Contradicted', evidence=[evidence])]
    if mixed:
        claims.append(Claim('good', 'Mira was born in 1998.', 0, label='Supported', evidence=[evidence]))
    return AnswerRecord(id='r', source_dataset='synthetic', question='Where was Mira born?',
                        answer='Mira was born in Birch Bay in 1998.', claims=claims)


def verdict(**changes):
    v = dict(correction_grounded=True, supported_preserved=True, no_new_unsupported_facts=True,
             hedge_only=False, supporting_passage_ids=[1], rationale='The passage supports the corrected relation.')
    return json.dumps({**v, **changes})


@pytest.mark.parametrize('backend', ['prompted', 'trained'])
def test_grounded_relation_is_accepted_by_both_entry_points(backend):
    r = source(True)
    calls = [verdict()]
    if backend == 'prompted':
        calls.insert(0, 'Mira was born in Alder Bay in 1998.')
    with patch('src.llm.call', side_effect=calls), \
            patch('src.training.repair_model.generate_edit', return_value='Mira was born in Alder Bay in 1998.'):
        repair(r, guard_mode='grounded_v1', backend=backend,
               trained_model_path='fixture' if backend == 'trained' else None)
    assert r.repaired_answer == 'Mira was born in Alder Bay in 1998.'
    assert r.repair_diagnostics['sentences'][0]['status'] == 'grounded_replacement'


@pytest.mark.parametrize('change', [dict(correction_grounded=False), dict(supported_preserved=False),
                                     dict(no_new_unsupported_facts=False), dict(hedge_only=True),
                                     dict(supporting_passage_ids=[]), dict(supporting_passage_ids=[99])])
def test_entity_presence_does_not_bypass_relation_or_preservation_guard(change):
    r = source(True)
    with patch('src.llm.call', side_effect=['Mira was not born in Alder Bay in 1998.', verdict(**change)]):
        repair(r, guard_mode='grounded_v1', policy='rewrite-only', max_retries=0)
    assert r.repaired_answer == r.answer
    assert r.repair_diagnostics['sentences'][0]['status'] == 'unchanged_unresolved'


def test_unfounded_scope_is_rejected_and_mixed_supported_fact_is_not_hedged():
    r = source(True)
    with patch('src.llm.call', side_effect=['Mira was born in Alder Bay, usually in 1998.',
                                         verdict(no_new_unsupported_facts=False)]):
        repair(r, guard_mode='grounded_v1', policy='rewrite+hedge', max_retries=0)
    assert r.repaired_answer == r.answer


def test_no_evidence_abstains_without_generation_or_critic_calls():
    r = source()
    r.claims[0].evidence = []
    with patch('src.llm.call', side_effect=AssertionError('No usable evidence')):
        repair(r, guard_mode='grounded_v1', policy='rewrite-only')
    assert r.repaired_answer == r.answer
    assert r.repair_diagnostics['sentences'][0]['stop_reason'] == 'no_replacement_evidence'


def test_malformed_guard_response_retries_then_abstains():
    r = source()
    with patch('src.llm.call', side_effect=['Mira was born in Alder Bay in 1998.', 'invalid'] * 2) as model:
        repair(r, guard_mode='grounded_v1', policy='rewrite-only')
    assert model.call_count == 4
    assert r.repaired_answer == r.answer
    assert r.repair_diagnostics['sentences'][0]['stop_reason'] == 'guard_retries_exhausted'


def test_hedging_and_noop_are_not_reported_as_grounded_corrections():
    r = source()
    with patch('src.llm.call', side_effect=['Mira was born in Birch Bay in 1998.']):
        repair(r, guard_mode='grounded_v1')
    assert r.repair_diagnostics['sentences'][0]['status'] == 'unchanged_unresolved'
    r = source()
    with patch('src.llm.call', return_value='An ungrounded new place 2015.'):
        repair(r, guard_mode='grounded_v1', policy='rewrite+hedge')
    assert r.repair_diagnostics['sentences'][0]['status'] == 'hedged_unresolved'


def test_untouched_bytes_and_whitespace_survive_localized_repair():
    r = source()
    r.answer = '  A supported fact.\n\nMira was born in Birch Bay in 1998.  '
    r.claims[0].sentence_id = 1
    r.claims.append(Claim('other', 'A supported fact.', 0, label='Supported'))
    with patch('src.llm.call', side_effect=['Mira was born in Alder Bay in 1998.', verdict()]):
        repair(r, guard_mode='grounded_v1', policy='rewrite-only')
    assert r.repaired_answer == '  A supported fact.\n\nMira was born in Alder Bay in 1998.  '
