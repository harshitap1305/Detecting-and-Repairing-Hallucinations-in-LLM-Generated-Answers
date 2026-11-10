"""Structural answer/question attribution contract; not a semantic fact checker.

A valid span or declared relation cannot prove that a claim is expressed by an
answer. Missing legacy annotations stay unknown; semantic checking is separate.
"""
from __future__ import annotations

import copy
import re

VERSION = 1
STATUSES = {'bound', 'ambiguous', 'answer_only', 'nonfactual'}


def question_clauses(question: str) -> list[dict]:
    """Surface segmentation with stable original offsets, not a full parse."""
    boundaries = list(re.finditer(r'[?;\n]+|\b(?:and|but|also)\s+(?=(?:who|what|when|where|which|how)\b)', question, re.I))
    result, start = [], 0
    for boundary in [*boundaries, None]:
        end = boundary.start() if boundary else len(question)
        a, b = start, end
        while a < b and question[a].isspace():
            a += 1
        while b > a and question[b - 1].isspace():
            b -= 1
        if a < b:
            result.append({'clause_id': f'q{len(result) + 1}', 'span': [a, b], 'text': question[a:b]})
        start = boundary.end() if boundary else end
    return result


def _span(value, text, where):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(n) is not int for n in value)
            or not 0 <= value[0] < value[1] <= len(text)):
        raise ValueError(f'{where}: invalid original-text span')
    return value


def build_attribution(items, *, question, answer, sentence_spans, record_id):
    clauses = question_clauses(question)
    clause_ids = {c['clause_id'] for c in clauses}
    claims, provided = [], 0
    for i, item in enumerate(items):
        cid = f'{record_id}_c{i + 1}'
        value = item.get('attribution')
        if 'attribution' not in item:
            claims.append({'claim_id': cid, 'status': 'not_provided'})
            continue
        provided += 1
        if (not isinstance(value, dict)
                or set(value) != {'status', 'answer_spans', 'bindings', 'rationale'}
                or not isinstance(value['status'], str) or value['status'] not in STATUSES
                or not isinstance(value['rationale'], str) or not value['rationale'].strip()
                or not isinstance(value['answer_spans'], list) or not value['answer_spans']
                or not isinstance(value['bindings'], list)):
            raise ValueError('Malformed attribution fields/status')
        if (value['status'] == 'nonfactual') != (not item['check_worthy']):
            raise ValueError('Nonfactual attribution must agree with check_worthy')
        _, sent_start, sent_end = sentence_spans[item['sentence_id']]
        for span in value['answer_spans']:
            a, b = _span(span, answer, 'answer_spans')
            if not sent_start <= a < b <= sent_end:
                raise ValueError('Answer attribution is outside its mapped sentence')
        if len({tuple(s) for s in value['answer_spans']}) != len(value['answer_spans']):
            raise ValueError('Duplicate answer attribution span')
        if value['status'] == 'bound' and not value['bindings']:
            raise ValueError('Bound attribution requires a question relationship')
        if value['status'] in {'answer_only', 'nonfactual'} and value['bindings']:
            raise ValueError('Answer-only/nonfactual attribution cannot assert question bindings')
        seen = set()
        for binding in value['bindings']:
            if (not isinstance(binding, dict)
                    or set(binding) != {'clause_id', 'answer_span', 'answer_text', 'relation', 'borrowed_context'}
                    or not isinstance(binding['clause_id'], str) or binding['clause_id'] not in clause_ids
                    or not isinstance(binding['borrowed_context'], list)
                    or not isinstance(binding['relation'], str) or not binding['relation'].strip()):
                raise ValueError('Malformed question relationship binding')
            a, b = _span(binding['answer_span'], answer, 'binding.answer_span')
            if not any(s[0] <= a < b <= s[1] for s in value['answer_spans']):
                raise ValueError('Binding must be supported by a declared answer span')
            if binding['answer_text'] != answer[a:b]:
                raise ValueError('Binding answer_text does not match original answer bytes')
            key = (binding['clause_id'], a, b, binding['relation'])
            if key in seen:
                raise ValueError('Duplicate question relationship binding')
            seen.add(key)
            for context in binding['borrowed_context']:
                if (not isinstance(context, dict) or set(context) != {'source', 'span', 'text', 'role'}
                        or not isinstance(context['source'], str) or context['source'] not in {'question', 'answer'}
                        or not isinstance(context['role'], str) or context['role'] not in {'referent', 'relation_context'}):
                    raise ValueError('Borrowed context must resolve a referent or relationship only')
                text = question if context['source'] == 'question' else answer
                x, y = _span(context['span'], text, 'borrowed_context')
                if context['text'] != text[x:y]:
                    raise ValueError('Borrowed context must reference original source bytes')
        claims.append({'claim_id': cid, **copy.deepcopy(value)})
    coverage = []
    for clause in clauses:
        bindings = [c for c in claims if any(b['clause_id'] == clause['clause_id'] for b in c.get('bindings', []))]
        ambiguous = any(c['status'] == 'ambiguous' and not c['bindings'] for c in claims)
        status = ('proposed_answered' if any(c['status'] == 'bound' for c in bindings) else
                  'ambiguous' if bindings or ambiguous else
                  'unknown' if provided < len(items) else 'unanswered')
        coverage.append({'clause_id': clause['clause_id'], 'status': status,
                         'claim_ids': [c['claim_id'] for c in bindings]})
    return {'semantics_version': VERSION, 'kind': 'model_proposed_attribution',
            'validation': 'structural_only_not_semantic_or_factual',
            'status': ('empty_answer' if not answer.strip() else 'not_provided' if not provided
                       else 'partial' if provided < len(items) else 'proposed'),
            'clause_segmentation': 'surface_v1', 'question_clauses': clauses,
            'claims': claims, 'clause_coverage': coverage}


def reanchor_context_quotes(items, *, question, answer):
    """Repair numeric offsets only when the unchanged quote has one exact match.

    Never fuzzy-match, normalize text, choose among duplicates, or change a
    claim, source selector, question binding, or relation. Strict validation runs
    afterwards; raw model responses and all applied corrections stay traceable.
    """
    normalized, repairs = copy.deepcopy(items), []
    for i, item in enumerate(normalized):
        if not isinstance(item, dict) or not isinstance(item.get('attribution'), dict):
            continue
        bindings = item['attribution'].get('bindings')
        if not isinstance(bindings, list):
            continue
        for j, binding in enumerate(bindings):
            if not isinstance(binding, dict) or not isinstance(binding.get('borrowed_context'), list):
                continue
            for k, context in enumerate(binding['borrowed_context']):
                if (not isinstance(context, dict) or context.get('source') not in ('question', 'answer')
                        or not isinstance(context.get('text'), str) or not context['text']):
                    continue
                source = question if context['source'] == 'question' else answer
                quote, span = context['text'], context.get('span')
                if (isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span)
                        and 0 <= span[0] < span[1] <= len(source) and source[span[0]:span[1]] == quote):
                    continue
                start = source.find(quote)
                if start >= 0 and source.find(quote, start + 1) < 0:
                    corrected = [start, start + len(quote)]
                    repairs.append({'field': f'claims[{i}].attribution.bindings[{j}].borrowed_context[{k}].span',
                                    'source': context['source'], 'text': quote,
                                    'original_span': span, 'corrected_span': corrected,
                                    'rule': 'unique_exact_quote'})
                    context['span'] = corrected
    return normalized, repairs
