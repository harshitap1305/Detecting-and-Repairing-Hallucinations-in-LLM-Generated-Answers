"""Bounded semantic extraction audit. No evidence, gold, or factual labels enter.

This is a fallible model diagnostic, not an independent truth/completeness score.
"""
from __future__ import annotations
import json
import hashlib

import src.llm as llm

CLAIM_VERDICTS = {'expressed', 'nonfactual', 'wrong_relation', 'unsupported_assertion', 'ambiguous'}
CLAUSE_VERDICTS = {'represented', 'unanswered', 'missing_relation', 'ambiguous'}


def extraction_identity(question, answer, claims, attribution):
    value = {"question": question, "answer": answer, "claims": claims, "attribution": attribution}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_extraction(*, question, answer, items, attribution, record_id, model, provider,
                     max_claims=64, use_cache=True):
    if type(max_claims) is not int or not 1 <= max_claims <= 128:
        raise ValueError('max_claims must be between 1 and 128')
    base = {'semantics_version': 1, 'kind': 'model_semantic_extraction_check',
            'status': 'unresolved', 'model_calls': 0}
    if attribution.get('status') != 'proposed':
        return {**base, 'reason': 'missing_or_partial_attribution'}
    if any(c['status'] == 'ambiguous' for c in attribution['claims']):
        return {**base, 'reason': 'ambiguous_attribution'}
    if len(items) + len(attribution['question_clauses']) > max_claims:
        return {**base, 'reason': 'semantic_check_inventory_budget_exceeded'}
    claims = {f'{record_id}_c{i+1}': item for i, item in enumerate(items)}
    clauses = {c['clause_id']: c for c in attribution['question_clauses']}
    audit_claims = [{'claim_id': cid, 'text': c['claim'], 'sentence_id': c['sentence_id'],
                     'check_worthy': c['check_worthy']} for cid, c in claims.items()]
    base['input_sha256'] = extraction_identity(question, answer, audit_claims, attribution)
    prompt = (
        'SEMANTIC EXTRACTION AUDIT: independently inspect the raw question and answer, then the proposed extraction. '
        'Do NOT trust the generator\'s relation strings, rationale, or bound status. Do NOT assess real-world truth. '
        'Decide which propositions the ANSWER actually expresses in question context. Distinguish the requested '
        'relationship from descriptive question premises: describing when/where an event happened cannot replace '
        'linking it to the person/event relationship that the short answer identifies. Do not add a claim about '
        'an institution just because a fragment answers a separate definition clause. A true new fact is still an '
        'unsupported extraction. Check relation/argument roles, negation, uncertainty and referents semantically, '
        'allowing paraphrases. Missing answers to other clauses may legitimately be unanswered; do not force all '
        'question clauses to be answered. Check answer-only facts and nonfactual content too. Missing answer '
        'content must be reported with an original answer span. Return JSON only with every claim and clause once: '
        '{"claim_checks":[{"claim_id":"id","verdict":"expressed|nonfactual|wrong_relation|unsupported_assertion|ambiguous",'
        '"clause_ids":["q1"],"rationale":"explanation"}],'
        '"clause_checks":[{"clause_id":"q1","verdict":"represented|unanswered|missing_relation|ambiguous",'
        '"claim_ids":["id"],"rationale":"explanation"}],'
        '"missing_content":[{"answer_span":[0,5],"clause_ids":["q1"],"rationale":"missing relationship/fact"}]}. '
        'Use empty missing_content only after checking every factual answer assertion. '
        'Represented clauses require expressed claim IDs. Unanswered clauses have no claim IDs.\n'
        + json.dumps({'question': question, 'answer': answer,
            'claims': audit_claims,
            'proposed_attribution': attribution}, ensure_ascii=False)
    )
    response = llm.call(prompt, model=model, provider=provider, max_tokens=2048, use_cache=use_cache)
    base['model_calls'] = 1
    try:
        raw = response.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {'claim_checks', 'clause_checks', 'missing_content'}:
            raise ValueError('Unexpected audit fields')
        def refs(value, allowed):
            if (not isinstance(value, list) or any(not isinstance(v, str) for v in value)
                    or len(set(value)) != len(value) or not set(value) <= set(allowed)):
                raise ValueError('Invalid audit references')
        def entries(values, id_key, allowed, verdicts, ref_key, ref_allowed):
            if not isinstance(values, list):
                raise ValueError('Audit checks must be lists')
            seen, result = set(), {}
            for entry in values:
                if (not isinstance(entry, dict) or set(entry) != {id_key, 'verdict', ref_key, 'rationale'}
                        or not isinstance(entry[id_key], str) or entry[id_key] not in allowed or entry[id_key] in seen
                        or not isinstance(entry['verdict'], str) or entry['verdict'] not in verdicts
                        or not isinstance(entry['rationale'], str) or not entry['rationale'].strip()):
                    raise ValueError('Malformed/duplicate audit finding')
                refs(entry[ref_key], ref_allowed)
                seen.add(entry[id_key]); result[entry[id_key]] = entry
            if seen != set(allowed):
                raise ValueError('Incomplete audit coverage')
            return result
        cc = entries(data['claim_checks'], 'claim_id', claims, CLAIM_VERDICTS, 'clause_ids', clauses)
        qc = entries(data['clause_checks'], 'clause_id', clauses, CLAUSE_VERDICTS, 'claim_ids', claims)
        proposed = {c['claim_id']: c for c in attribution['claims']}
        for cid, check in cc.items():
            if check['verdict'] in {'expressed', 'nonfactual'}:
                expected = 'expressed' if claims[cid]['check_worthy'] else 'nonfactual'
                bindings = {b['clause_id'] for b in proposed[cid]['bindings']}
                if check['verdict'] != expected or set(check['clause_ids']) != bindings:
                    raise ValueError('Claim audit contradicts attribution/check-worthiness')
                if any(qc[qid]['verdict'] != 'represented' or cid not in qc[qid]['claim_ids'] for qid in bindings):
                    raise ValueError('Claim and clause audit contradict each other')
        for qid, check in qc.items():
            if check['verdict'] == 'represented' and (not check['claim_ids'] or any(
                    cc[cid]['verdict'] != 'expressed' or qid not in cc[cid]['clause_ids'] for cid in check['claim_ids'])):
                raise ValueError('Represented clause lacks expressed claims')
            if check['verdict'] == 'unanswered' and check['claim_ids']:
                raise ValueError('Unanswered clause cannot contain claims')
        if not isinstance(data['missing_content'], list):
            raise ValueError('Missing-content findings must be explicit')
        for finding in data['missing_content']:
            if (not isinstance(finding, dict) or set(finding) != {'answer_span', 'clause_ids', 'rationale'}
                    or not isinstance(finding['rationale'], str) or not finding['rationale'].strip()):
                raise ValueError('Malformed missing-content finding')
            span = finding['answer_span']
            if (not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span)
                    or not 0 <= span[0] < span[1] <= len(answer)):
                raise ValueError('Missing content must reference an original answer span')
            refs(finding['clause_ids'], clauses)
        passed = (all(c['verdict'] in {'expressed', 'nonfactual'} for c in cc.values())
                  and all(q['verdict'] in {'represented', 'unanswered'} for q in qc.values())
                  and not data['missing_content'])
        return {**base, **data, 'status': 'passed' if passed else 'rejected',
                'reason': None if passed else 'semantic_extraction_findings'}
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        return {**base, 'reason': 'malformed_semantic_check', 'detail': str(exc)}


def extraction_coverage(record):
    """Never infer a successful extraction from numeric verification scores."""
    diagnostic = record.decomposition_diagnostics or {}
    check = diagnostic.get('semantic_check') or {}
    attribution = diagnostic.get('attribution') or {}
    current = extraction_identity(record.question, record.answer,
        [{'claim_id': c.id, 'text': c.text, 'sentence_id': c.sentence_id,
          'check_worthy': c.check_worthy} for c in record.claims], attribution)
    passed = (diagnostic.get('status') in {'complete', 'no_check_worthy_claims'}
              and attribution.get('status') == 'proposed'
              and check.get('status') == 'passed' and check.get('input_sha256') == current)
    reason = None if passed else (
        'semantic_check_input_mismatch' if check.get('status') == 'passed' and check.get('input_sha256') != current
        else check.get('reason') or diagnostic.get('reason') or 'semantic_check_not_available')
    return {'status': 'passed' if passed else 'unresolved', 'reason': reason, 'semantic_check': check}
