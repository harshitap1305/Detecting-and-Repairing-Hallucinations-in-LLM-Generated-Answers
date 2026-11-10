"""Semantic alignment contract and conservative coverage, using synthetic facts."""
import copy
import json
from unittest.mock import patch

import pytest

from src.claim_tracking import track_claims
from src.pipeline import run_pipeline
from src.schema import AnswerRecord, Claim


def answer(text, claims):
    return AnswerRecord(id='r', source_dataset='synthetic', question='Where was Mira born?',
                        answer=text, claims=claims)


def claim(cid, text, label='Supported'):
    return Claim(cid, text, 0, label=label)


def group(oids, cids, relation):
    return {'original_ids': oids, 'candidate_ids': cids, 'relation': relation,
            'rationale': 'Synthetic semantic relationship.'}


def align(original, candidate, links, extras=None):
    with patch('src.claim_tracking.llm.call', return_value=json.dumps(
            {'links': links, 'candidate_only_ids': extras or []})):
        return track_claims(original, candidate, model='mock', provider='groq')


def test_changed_ids_and_paraphrases_are_linked_by_model_not_overlap():
    o = answer('Mira was born in Alder Bay.', [claim('a', 'Mira was born in Alder Bay.')])
    c = answer('Alder Bay is her birthplace.', [claim('z', 'Alder Bay is the birthplace of Mira.')])
    r = align(o, c, [group(['a'], ['z'], 'retained')])
    assert r['coverage'] == 1
    assert r['factual_success']
    assert r['alignment_source'] == 'model_alignment'


def test_many_to_many_groups_and_new_propositions_are_retained():
    o = answer('Two facts.', [claim('a', 'Mira lives in Alder.'), claim('b', 'Mira teaches art.')])
    c = answer('Combined facts plus another.', [claim('x', 'Mira lives in Alder and teaches art.'),
                                              claim('y', 'Mira teaches art in Alder.'),
                                              claim('new', 'Mira won an award.', 'Unverifiable')])
    r = align(o, c, [group(['a', 'b'], ['x', 'y'], 'retained')], ['new'])
    assert r['matched_original_targets'] == 2
    assert r['candidate_only'][0]['id'] == 'new'
    assert r['candidate_only_unverified'] == ['new']
    assert not r['factual_success']


@pytest.mark.parametrize('relation', ['hedged', 'removed', 'unresolved'])
def test_missing_or_weakened_target_never_establishes_factual_success(relation):
    o = answer('Mira was born in Alder.', [claim('a', 'Mira was born in Alder.')])
    c = answer('Possibly Alder.', [claim('x', 'Mira may have been born in Alder.')])
    ids = [] if relation == 'removed' else ['x']
    r = align(o, c, [group(['a'], ids, relation)], ['x'] if not ids else [])
    assert not r['factual_success']
    assert r['links'][0]['relation'] == relation


def test_verified_replacement_of_supported_fact_is_not_preservation():
    o = answer('Mira was born in Alder.', [claim('a', 'Mira was born in Alder.')])
    c = answer('Mira was born in Birch.', [claim('x', 'Mira was born in Birch.')])
    r = align(o, c, [group(['a'], ['x'], 'modified')])
    assert r['grounded_original_targets'] == 1
    assert not r['factual_success']
    o.claims[0].label = 'Unverifiable'
    assert align(o, c, [group(['a'], ['x'], 'modified')])['factual_success']


@pytest.mark.parametrize('bad', [
    {'links': [], 'candidate_only_ids': ['x']},
    {'links': [group(['unknown'], ['x'], 'retained')], 'candidate_only_ids': []},
    {'links': [group(['a'], [], 'retained')], 'candidate_only_ids': ['x']},
    {'links': [group(['a'], ['x'], 'retained')], 'candidate_only_ids': ['x']},
    {'links': [group(['a'], ['x'], 'retained'), group(['a'], ['x'], 'retained')], 'candidate_only_ids': []},
])
def test_malformed_or_incomplete_alignment_stays_unresolved(bad):
    o = answer('Original.', [claim('a', 'Mira was born in Alder.')])
    c = answer('Changed.', [claim('x', 'Mira was born in Alder Bay.')])
    with patch('src.claim_tracking.llm.call', return_value=json.dumps(bad)) as model:
        r = track_claims(o, c, model='mock', provider='groq')
    assert model.call_count == 1
    assert r['links'][0]['relation'] == 'unresolved'
    assert r['coverage'] == 0
    assert not r['factual_success']


def test_high_overlap_negated_target_is_not_matched_without_semantic_decision():
    o = answer('Mira lives in Alder.', [claim('a', 'Mira lives in Alder.')])
    c = answer('Mira does not live in Alder.', [claim('x', 'Mira does not live in Alder.')])
    r = align(o, c, [group(['a'], ['x'], 'unresolved')])
    assert r['coverage'] == 0
    assert not r['factual_success']


def test_empty_candidate_missing_decomposition_and_budget_need_no_model():
    o = answer('Original.', [claim('a', 'Mira lives in Alder.')])
    with patch('src.claim_tracking.llm.call', side_effect=AssertionError('Must not call')):
        empty = track_claims(o, answer('', []), model='mock', provider='groq')
        missing = track_claims(o, answer('A revised factual sentence.', []), model='mock', provider='groq')
        budget = track_claims(o, answer('New.', [claim('x', 'A fact.')]),
                              model='mock', provider='groq', max_claims=1)
    assert empty['links'][0]['relation'] == 'removed'
    assert missing['links'][0]['relation'] == 'unresolved'
    assert budget['reason'] == 'alignment_claim_budget_exceeded'
    assert not any(r['factual_success'] for r in (empty, missing, budget))


def test_pipeline_freezes_original_and_gates_missing_target_success():
    r = answer('Mira lives in Alder and teaches art.',
               [claim('a', 'Mira lives in Alder.'), claim('b', 'Mira teaches art.', 'Unverifiable')])
    initial = copy.deepcopy(r.claims)
    def verify(candidate, **kw):
        return candidate
    def repair(candidate, **kw):
        candidate.claims[0].text = 'CORRUPTED WORKING COPY'
        candidate.repaired_answer = 'Mira lives in Alder.'
        return candidate
    def decompose(candidate, **kw):
        candidate.claims = [claim('x', 'Mira lives in Alder.')]
        return candidate
    with patch('src.pipeline.retrieve', side_effect=lambda r, **kw: r), \
            patch('src.pipeline.verify', side_effect=verify), \
            patch('src.pipeline.repair', side_effect=repair), \
            patch('src.pipeline.decompose', side_effect=decompose), \
            patch('src.claim_tracking.llm.call', return_value=json.dumps({
                'links': [group(['a'], ['x'], 'retained'), group(['b'], [], 'unresolved')],
                'candidate_only_ids': []})) as model:
        result = run_pipeline(r, use_existing_claims=True)
    assert result.claims == initial
    assert result.claim_tracking['original_claims'][0]['text'] == initial[0].text
    assert result.claim_tracking['final']['coverage'] == .5
    assert result.verification_diagnostics['stop_reason'] == 'unresolved_target_coverage'
    assert model.call_count == 1  # verified candidate reused, no duplicate final pass
    assert result.reliability_score_after == 1  # score describes current claims only
    assert len(result.claim_tracking['stages']) == 2


def test_alignment_request_excludes_gold_and_is_bounded_to_one_call():
    o = answer('Original.', [claim('a', 'Mira lives in Alder.')])
    o.gold_repaired_answer = 'PRIVATE GOLD REFERENCE'
    c = answer('Changed.', [claim('x', 'Mira lives in Alder Bay.')])
    with patch('src.claim_tracking.llm.call', return_value='invalid') as model:
        r = track_claims(o, c, model='mock', provider='groq')
    assert model.call_count == 1
    assert 'PRIVATE GOLD REFERENCE' not in model.call_args.args[0]
    assert not r['factual_success']


def test_tracking_diagnostics_survive_record_roundtrip():
    o = answer('Original.', [claim('a', 'Mira lives in Alder.')])
    o.claim_tracking = {'semantics_version': 1, 'final': track_claims(o, copy.deepcopy(o), model='mock', provider='groq')}
    assert AnswerRecord.from_jsonl_line(o.to_jsonl_line()).claim_tracking == o.claim_tracking


def test_delete_baseline_does_not_attach_detector_tracking_to_deleted_candidate():
    from src.experiments.adapters import execute
    from src.experiments.runner import DEFAULTS
    r = answer('Mira lives in Alder.', [claim('a', 'Mira lives in Alder.', 'Unverifiable')])
    def detector(r, **kw):
        r.reliability_score_before = r.reliability_score_after = 0.
        r.claim_tracking = {'final': {'factual_success': False}}
        r.verification_diagnostics = {'semantics_version': 2, 'before': {'score': 0.}, 'after': {'score': 0.}}
        return r
    with patch('src.pipeline.run_pipeline', side_effect=detector):
        result = execute({'id': 'b6', 'retrieval': 'bm25'}, r, 'repair', {**DEFAULTS, 'index_path': 'unused'})
    assert result['record']['repaired_answer'] == ''
    assert result['record']['reliability_score_after'] is None
    assert 'claim_tracking' not in result['record']
    assert result['native']['detector_claim_tracking']['factual_success'] is False
