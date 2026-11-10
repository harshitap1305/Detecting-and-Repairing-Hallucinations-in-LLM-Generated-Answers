"""Bounded retrieval planning against fixed, synthetic evidence."""
import copy
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.experiments.runtime import execution
from src.schema import AnswerRecord, Claim, EvidencePassage

retrieval = importlib.import_module('src.retrieve')


def record(question='How often is the magazine Mira writes for published?'):
    return AnswerRecord(id='r', source_dataset='synthetic', question=question, answer='Monthly',
                        claims=[Claim('c', 'The magazine is published monthly.', 0)])


@pytest.fixture
def index(tmp_path):
    path = tmp_path / 'index'
    path.write_bytes(b'synthetic index')
    return str(path)


def passage(cid='person', title='Mira Vale', text='Mira writes for [[Northbridge Review]].'):
    return EvidencePassage(doc_id=cid, title=title, text=text)


def test_one_hop_entity_attribute_retrieval_records_bridge_and_fixed_target(index):
    r = record()
    original = r.claims[0].text
    calls = []
    def fetch(query, index_path, top_k):
        calls.append(query)
        if query.startswith('Northbridge Review'):
            return [passage('magazine', 'Northbridge Review', 'Northbridge Review is published monthly.')]
        return [passage()]
    events = []
    with patch('src.retrieve.retrieve_bm25', side_effect=fetch), execution(record=events.append):
        retrieval.retrieve(r, index, top_k=2, use_cache=False, query_strategy='contextual_v1')
    assert len(calls) == 3
    assert r.claims[0].text == original
    assert r.claims[0].label is None  # retrieval never proves a claim
    assert {p.doc_id for p in r.claims[0].evidence} == {'person', 'magazine'}
    q = r.retrieval_diagnostics['claims'][0]['queries'][-1]
    assert q['kind'] == 'linked_entity'
    assert q['link']['source_doc_id'] == 'person'
    start, end = q['link']['source_span']
    assert passage().text[start:end] == 'Northbridge Review'
    assert events[-1]['event'] == 'retrieval'


def test_caps_are_hard_even_for_many_entities_and_duplicate_passages(index):
    def fetch(query, index_path, top_k):
        return [passage(str(i), 'Seed', 'Alpha Review and Beta Journal publish news.') for i in range(top_k)]
    with patch('src.retrieve.retrieve_bm25', side_effect=fetch) as backend:
        r = retrieval.retrieve(record(), index, top_k=2, max_queries=2, max_passages=4,
                              use_cache=False, query_strategy='contextual_v1')
    d = r.retrieval_diagnostics['claims'][0]
    assert backend.call_count <= 2
    assert d['fetched_passages'] <= 4
    assert len(r.claims[0].evidence) <= 2
    assert len({p.doc_id for p in r.claims[0].evidence}) == len(r.claims[0].evidence)


def test_missing_evidence_remains_unverifiable_and_claim_is_not_rewritten(index):
    r = record()
    r.claims[0].label = 'Unverifiable'
    with patch('src.retrieve.retrieve_bm25', return_value=[]):
        retrieval.retrieve(r, index, use_cache=False, query_strategy='contextual_v1')
    assert r.claims[0].label == 'Unverifiable'
    assert r.claims[0].evidence == []
    assert r.retrieval_diagnostics['claims'][0]['trigger'] == 'previously_unverifiable'


def test_claim_only_strategy_uses_one_query(index):
    with patch('src.retrieve.retrieve_bm25', return_value=[passage()]) as backend:
        r = retrieval.retrieve(record(), index, use_cache=False, query_strategy='claim_only')
    assert backend.call_count == 1
    assert r.retrieval_diagnostics['claims'][0]['queries'][0]['kind'] == 'claim'


def test_cache_identity_includes_question_strategy_caps_and_index_version(index, monkeypatch):
    cache = {}
    monkeypatch.setattr(retrieval, '_cache_get_passages', lambda key: copy.deepcopy(cache.get(key)))
    monkeypatch.setattr(retrieval, '_cache_set_passages', lambda key, rows: cache.setdefault(key, copy.deepcopy(rows)))
    with patch('src.retrieve.retrieve_bm25', return_value=[]) as backend:
        def run(**kw):
            return retrieval.retrieve(record(), index, **kw)
        run()
        run()
        assert backend.call_count == 1
        run(max_queries=4)
        assert backend.call_count == 2
        run(max_passages=21)
        assert backend.call_count == 3
        retrieval.retrieve(record('A different question'), index)
        assert backend.call_count == 4
        run(query_strategy='contextual_v1')
        assert backend.call_count == 6  # claim and question query
        Path(index).write_bytes(b'a different index size')
        run()
        assert backend.call_count == 7


def test_duplicate_hits_rank_fusion_is_deterministic_and_does_not_mutate_sources(index):
    seed = passage()
    original = copy.deepcopy(seed)
    with patch('src.retrieve.retrieve_bm25', return_value=[seed, seed]):
        a = retrieval.retrieve(record(), index, top_k=2, use_cache=False, query_strategy='contextual_v1')
        b = retrieval.retrieve(record(), index, top_k=2, use_cache=False, query_strategy='contextual_v1')
    assert a.claims[0].evidence == b.claims[0].evidence
    assert len(a.claims[0].evidence) == 1
    assert seed == original


@pytest.mark.parametrize('options', [dict(max_queries=0), dict(max_queries=9),
                                     dict(max_passages=201), dict(max_passages=1),
                                     dict(query_strategy='typo')])
def test_invalid_caps_are_rejected_before_fetch(index, options):
    with patch('src.retrieve.retrieve_bm25', side_effect=AssertionError('Must not fetch')):
        with pytest.raises(ValueError):
            retrieval.retrieve(record(), index, **options)


def test_verifier_gap_continues_with_remaining_budget_and_same_fixed_target(index):
    r = record()
    r.claims[0].text = 'The news magazine Mira writes for is published monthly.'
    # Enough lexical coverage for initial planning, but verification can still
    # find that the passages do not establish the relation.
    seeds = [passage('seed', 'Seed', r.claims[0].text + ' See [[Northbridge Review]].')]
    with patch('src.retrieve.retrieve_bm25', return_value=seeds) as backend:
        retrieval.retrieve(r, index, top_k=1, use_cache=False, query_strategy='contextual_v1', max_passages=3)
        assert backend.call_count == 1
        r.claims[0].label = 'Unverifiable'
        retrieval.retrieve(r, index, top_k=1, use_cache=False, query_strategy='contextual_v1',
                           max_passages=3, continue_unverifiable=True)
        assert backend.call_count == 3
        assert r.retrieval_diagnostics['claims'][0]['fetched_passages'] == 3
        retrieval.retrieve(r, index, top_k=1, use_cache=False, query_strategy='contextual_v1',
                           max_passages=3, continue_unverifiable=True)
        assert backend.call_count == 3  # exhausted cap never resets
    assert r.claims[0].label == 'Unverifiable'


def test_continuation_rejects_changed_scope(index):
    r = record()
    with patch('src.retrieve.retrieve_bm25', return_value=[]):
        retrieval.retrieve(r, index, use_cache=False, query_strategy='contextual_v1')
        r.claims[0].label = 'Unverifiable'
        with pytest.raises(ValueError, match='changed strategy'):
            retrieval.retrieve(r, index, use_cache=False, query_strategy='contextual_v1',
                               max_queries=4, continue_unverifiable=True)


def test_pipeline_verifies_expanded_evidence_before_repair(index):
    from src.pipeline import run_pipeline
    r = record()
    r.claims[0].text = 'The news magazine Mira writes for is published monthly.'
    verdicts = []
    def fetch(query, index_path, top_k):
        if query == r.claims[0].text:
            return [passage('a', 'Other Magazine', r.claims[0].text),
                    passage('b', 'Another Magazine', r.claims[0].text)]
        return [passage('magazine', 'Northbridge Review', 'A source establishes the actual magazine schedule.')]
    def verdict(c, **kw):
        verdicts.append([p.doc_id for p in c.evidence])
        c.label = 'Supported' if 'magazine' in verdicts[-1] else 'Unverifiable'
        return c
    with patch('src.retrieve.retrieve_bm25', side_effect=fetch), \
            patch('src.verify.verify_claim_llm', side_effect=verdict), \
            patch('src.pipeline.repair', side_effect=AssertionError('Evidence first')):
        result = run_pipeline(r, index_path=index, top_k=2, use_existing_claims=True)
    assert len(verdicts) == 2
    assert result.claims[0].label == 'Supported'
    assert result.verification_diagnostics['stop_reason'] == 'all_targets_verified'
    assert len(result.retrieval_diagnostics['claims'][0]['queries']) <= 3


def test_sqlite_fts_context_and_link_queries_use_same_corpus(tmp_path):
    from scripts.build_fever_fts import build_index
    db = tmp_path / 'corpus.sqlite'
    rows = [
        {'doc_id': 'person', 'title': 'Mira Vale', 'chunk_id': 0,
         'text': 'Title: Mira Vale. Mira Vale writes for [[Northbridge Review]].'},
        {'doc_id': 'magazine', 'title': 'Northbridge Review', 'chunk_id': 0,
         'text': 'Title: Northbridge Review. Northbridge Review is a news magazine published monthly.'},
        {'doc_id': 'other', 'title': 'Other Review', 'chunk_id': 0,
         'text': 'Other Review is published every day.'},
    ]
    build_index(iter(rows), db, source='synthetic test corpus')
    r = record()
    retrieval.retrieve(r, str(db), method='fts5', top_k=3, query_strategy='contextual_v1',
                       use_cache=False, max_queries=3, max_passages=9)
    logs = r.retrieval_diagnostics['claims'][0]
    assert any(q['kind'] == 'question_context' for q in logs['queries'])
    assert len(logs['queries']) <= 3
    assert logs['fetched_passages'] <= 9
    assert any(p.doc_id == 'magazine' for p in r.claims[0].evidence)
    assert r.claims[0].label is None


def test_in_memory_lexical_index_is_reloaded_when_file_changes(index):
    with patch('src.retrieve.pickle.load', side_effect=[{'bm25': 'old', 'passages': []},
                                                       {'bm25': 'new', 'passages': []}]) as load:
        assert retrieval._load_bm25(index)[0] == 'old'
        assert retrieval._load_bm25(index)[0] == 'old'
        Path(index).write_bytes(b'changed index file size')
        assert retrieval._load_bm25(index)[0] == 'new'
        assert load.call_count == 2


def test_index_title_header_is_not_an_entity_and_unicode_mention_offsets_are_exact():
    p = passage(title='Mira_Vale', text="Title: Mira_Vale. Mira Vale was a reporter for L’Obs.")
    decisions = []
    leads = list(retrieval._linked_entities([p], {'Mira_Vale'}, 'Mira Vale magazine frequency', decisions))
    assert leads[0]['entity'] == 'L’Obs'
    assert leads[0]['query_entity'] == "L'Obs"
    assert not any(lead['entity'] == 'Title' for lead in leads)
    for lead in leads:
        a, b = lead['source_span']
        assert p.text[a:b] == lead['entity']
    assert any(d['reason'] == 'structured_title_header' for d in decisions)


def test_header_only_and_empty_passages_do_not_invent_an_entity():
    for text in ('', 'Title: Mira_Vale.', 'Title: Mira_Vale.  ', 'Title: Mira_Vale\n', 'Title: Mira_Vale  '):
        assert list(retrieval._linked_entities([passage(title='Mira_Vale', text=text)], set())) == []


def test_actual_word_title_in_article_body_is_not_globally_blacklisted():
    p = passage(title='A novelist', text='Title: A novelist. Her novel was named Title.')
    leads = list(retrieval._linked_entities([p], {'A novelist'}, 'Her novel'))
    assert any(lead['entity'] == 'Title' for lead in leads)


def test_repeated_fever_aliases_do_not_consume_multiple_lead_slots():
    p = passage(text="See [[Élodie_Review_-LRB-magazine-RRB-]]. Also [[Élodie Review (magazine)]].")
    decisions = []
    leads = list(retrieval._linked_entities([p, p], set(), '', decisions))
    explicit = [lead for lead in leads if lead['mention_kind'] == 'explicit_link']
    assert len(explicit) == 1
    assert explicit[0]['query_entity'] == 'Élodie Review (magazine)'
    assert any(d['reason'] == 'duplicate_normalized_mention' for d in decisions)


def test_relevant_unicode_body_name_is_prioritized_over_sentence_start():
    p = passage(text='Title: Mira Vale. Other Things happened. She writes for Élodie-Ann Review.')
    leads = list(retrieval._linked_entities([p], {'Mira Vale'}, 'Which Review publishes the report?'))
    assert leads[0]['entity'] == 'Élodie-Ann Review'


def test_formatted_person_to_publication_query_fits_original_budget(index):
    r = record()
    seed = passage(title='Mira_Vale', text="Title: Mira_Vale. Mira Vale writes for L'Obs.")
    def fetch(query, index_path, top_k):
        if query.startswith("L'Obs"):
            return [passage('publication', "L'Obs", "Title: L'Obs. The magazine is published monthly.")]
        return [seed]
    with patch('src.retrieve.retrieve_bm25', side_effect=fetch) as backend:
        retrieval.retrieve(r, index, top_k=2, use_cache=False, query_strategy='contextual_v1')
    assert backend.call_count == 3
    assert {p.doc_id for p in r.claims[0].evidence} == {'person', 'publication'}
    last = r.retrieval_diagnostics['claims'][0]['queries'][-1]
    assert last['kind'] == 'linked_entity' and last['query'].startswith("L'Obs")
    assert r.retrieval_diagnostics['entity_planner_version'] == 2


def test_invalid_leads_leave_query_slot_unused(index):
    with patch('src.retrieve.retrieve_bm25', return_value=[passage(title='Mira Vale', text='Title: Mira Vale.')]) as backend:
        r = retrieval.retrieve(record(), index, top_k=2, use_cache=False, query_strategy='contextual_v1')
    assert backend.call_count == 2
    assert len(r.retrieval_diagnostics['claims'][0]['queries']) == 2
