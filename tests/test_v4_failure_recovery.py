"""Generic reproductions of the v4 offset, audit-contract and 429 failures."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import pytest
from groq import RateLimitError

from src.answer_attribution import reanchor_context_quotes
from src.decompose import decompose
from src.experiments.runtime import execution
from src.llm import _call_groq
from src.schema import AnswerRecord


def raw_item(question='What is Nera known for?', answer='Surveying'):
    return {'claim':'Nera is known for surveying.','sentence_id':0,'check_worthy':True,
            'attribution':{'status':'bound','answer_spans':[[0,len(answer)]],
                'bindings':[{'clause_id':'q1','answer_span':[0,len(answer)],'answer_text':answer,
                    'relation':'known for','borrowed_context':[{'source':'question','span':[0,4],
                        'text':'Nera','role':'referent'}]}], 'rationale':'A proposed relationship.'}}


def test_unique_exact_quote_repairs_only_its_offsets_and_preserves_raw_input():
    item=raw_item(); original=copy.deepcopy(item)
    data,repairs=reanchor_context_quotes([item],question='What is Nera known for?',answer='Surveying')
    context=data[0]['attribution']['bindings'][0]['borrowed_context'][0]
    assert context['span']==[8,12]
    assert repairs[0]['original_span']==[0,4]
    assert repairs[0]['rule']=='unique_exact_quote'
    assert item==original
    assert data[0]['claim']==item['claim']


@pytest.mark.parametrize('question', ['Nera and Nera', 'What is NERA known for?', 'No matching referent'])
def test_ambiguous_or_nonexact_quotes_are_never_reanchored(question):
    item=raw_item()
    data,repairs=reanchor_context_quotes([item],question=question,answer='Surveying')
    assert not repairs
    assert data==[item]


def test_decompose_records_reanchoring_and_strict_mode_still_rejects_bad_offsets():
    item=raw_item()
    r=AnswerRecord(id='r',source_dataset='synthetic',question='What is Nera known for?',answer='Surveying')
    with patch('src.llm.call',return_value=json.dumps([item])):
        decompose(r)
    assert r.decomposition_diagnostics['attempt_history'][0]['span_repairs']
    with patch('src.llm.call',return_value=json.dumps([item])):
        decompose(r,attribution_anchor_mode='strict')
    assert not r.claims
    assert r.decomposition_diagnostics['status']=='unresolved_coverage'


def test_answer_only_fact_and_unanswered_question_can_pass_a_format_retry():
    answer='Cedar County has many languages.'
    item={'claim':answer,'sentence_id':0,'check_worthy':True,'attribution':{
        'status':'answer_only','answer_spans':[[0,len(answer)]],'bindings':[],
        'rationale':'A factual assertion that does not supply a creation year.'}}
    claim={'claim_id':'r_c1','verdict':'expressed','clause_ids':['q1'],'rationale':'The assertion is present.'}
    clause={'clause_id':'q1','verdict':'unanswered','claim_ids':[],'rationale':'No creation year in the answer.'}
    malformed={'claim_checks':[claim],'clause_checks':[clause],
               'missing_content':[{'answer_span':[0,0],'clause_ids':['q1'],'rationale':'No creation year supplied.'}]}
    valid={'claim_checks':[{**claim,'clause_ids':[]}],'clause_checks':[clause],'missing_content':[]}
    r=AnswerRecord(id='r',source_dataset='synthetic',question='When was Cedar County created?',answer=answer)
    with patch('src.llm.call',side_effect=[json.dumps([item]),json.dumps(malformed),json.dumps(valid)]) as model, execution():
        decompose(r,extraction_check='semantic_v1')
    assert model.call_count==3  # one generator, two audits, same budgets
    assert r.decomposition_diagnostics['semantic_checks_used']==2
    assert r.decomposition_diagnostics['semantic_check']['status']=='passed'
    assert r.decomposition_diagnostics['semantic_check']['prior_format_failure']['reason']=='malformed_semantic_check'
    assert 'Do not turn a genuine' in model.call_args.args[0]


def rate_error(message='Try again in 217.5ms.',headers=None):
    response=httpx.Response(429,headers=headers or {},request=httpx.Request('POST','https://api.groq.com/openai/v1/chat/completions'))
    return RateLimitError(message,response=response,body={'error':{'type':'tokens','code':'rate_limit_exceeded'}})


def completion():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Answer'),finish_reason='stop')],usage=None)


def test_short_rate_limit_uses_provider_delay_and_records_retry_without_hidden_sdk_loop():
    events=[]
    with patch('groq.Groq') as client,patch('src.llm.time.sleep') as sleep,execution(record=events.append):
        client.return_value.chat.completions.create.side_effect=[rate_error(),completion()]
        assert _call_groq('openai/gpt-oss-120b','Synthetic prompt',0,4096,42)=='Answer'
        assert client.call_args.kwargs['max_retries']==0
    assert sleep.call_args.args[0]==pytest.approx(.2675)
    assert events[0]['event']=='rate_limit_retry'
    assert events[0]['source']=='provider_message'


def test_long_provider_wait_is_not_ignored_or_retried_early():
    with patch('groq.Groq') as client,patch('src.llm.time.sleep') as sleep:
        client.return_value.chat.completions.create.side_effect=rate_error(headers={'retry-after':'120'})
        with pytest.raises(RateLimitError):
            _call_groq('openai/gpt-oss-120b','Synthetic prompt',0,4096,42)
        assert client.return_value.chat.completions.create.call_count==1
        sleep.assert_not_called()


def test_rate_limit_retries_and_total_sleep_are_capped():
    with patch('groq.Groq') as client,patch('src.llm.time.sleep') as sleep,execution(settings={
            'rate_limit_retries':2,'rate_limit_wait_budget':1.0}):
        client.return_value.chat.completions.create.side_effect=rate_error(headers={'retry-after':'0.2'})
        with pytest.raises(RateLimitError):
            _call_groq('openai/gpt-oss-120b','Synthetic prompt',0,4096,42)
        assert client.return_value.chat.completions.create.call_count==3
        assert sleep.call_count==2
        assert sum(c.args[0] for c in sleep.call_args_list)<=1.0
