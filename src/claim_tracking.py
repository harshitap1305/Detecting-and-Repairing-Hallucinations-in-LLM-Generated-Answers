"""Conservative, model-diagnostic links between original and candidate claims.

Alignment is not factual assessment. Missing/ambiguous links never establish
correction, preservation, or coverage. No reference repair or gold enters prompts.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from src.schema import AnswerRecord
import src.llm as llm

RELATIONS = {'retained', 'modified', 'hedged', 'removed', 'unresolved'}


def track_claims(original: AnswerRecord, candidate: AnswerRecord, *, model: str,
                 provider: str, max_claims: int = 64) -> dict:
    if type(max_claims) is not int or not 1 <= max_claims <= 512:
        raise ValueError("max_claims must be between 1 and 512")
    before, after = original.check_worthy_claims, candidate.check_worthy_claims
    original_ids, candidate_ids = {c.id for c in before}, {c.id for c in after}
    reason = None
    links = []
    source = 'not_run'
    if not candidate.answer.strip():
        source = 'empty_candidate'
        links = [{'original_ids': [c.id], 'candidate_ids': [], 'relation': 'removed',
                  'rationale': 'The candidate answer is empty.'} for c in before]
    elif (candidate.answer == original.answer
          and [(c.text, c.check_worthy) for c in before] == [(c.text, c.check_worthy) for c in after]):
        source = 'unchanged_inventory'
        links = [{'original_ids': [a.id], 'candidate_ids': [b.id], 'relation': 'retained',
                  'rationale': 'Unchanged answer and identical ordered claim inventory.'}
                 for a, b in zip(before, after)]
    elif not before or not after:
        reason = 'no_original_targets' if not before else 'no_candidate_targets'
    elif len(before) + len(after) > max_claims:
        reason = 'alignment_claim_budget_exceeded'
    else:
        source = 'model_alignment'
        prompt = (
            'Align factual targets across an answer edit. This is semantic alignment, not fact checking. '
            'Do not match merely by claim ID, word overlap, or shared entity. Preserve relations, negation, '
            'dates, scope, uncertainty and attribution. Retained means the same fact with the same force; '
            'hedged means its force was weakened. Modified means a related replacement proposition, '
            'not a verified correction. Removed requires explicit absence; uncertain matches are unresolved. '
            'Allow many-to-many groups. Each original ID must occur in exactly one group; candidate IDs '
            'can participate in multiple groups. List every otherwise unlinked candidate ID separately. '
            'Never treat missing decomposition as proof of deletion. Return only JSON: '
            '{"links":[{"original_ids":["id"],"candidate_ids":["id"],'
            '"relation":"retained|modified|hedged|removed|unresolved","rationale":"explanation"}],'
            '"candidate_only_ids":["id"]}. For removed, candidate_ids must be empty. '
            'Retained/modified/hedged require candidate IDs.\n'
            + json.dumps({'question': original.question, 'original_answer': original.answer,
                          'candidate_answer': candidate.answer,
                          'original_claims': [{'id': c.id, 'text': c.text} for c in before],
                          'candidate_claims': [{'id': c.id, 'text': c.text} for c in after]}, ensure_ascii=False)
        )
        try:
            raw = llm.call(prompt, model=model, provider=provider, max_tokens=4096)
            cleaned = raw.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0]
            data = json.loads(cleaned)
            links = data['links']
            if not isinstance(links, list):
                raise ValueError('links must be a list')
            seen, linked = set(), set()
            for link in links:
                oids, cids = link['original_ids'], link['candidate_ids']
                if (not isinstance(oids, list) or not oids or not isinstance(cids, list)
                        or any(not isinstance(i, str) for i in oids + cids)
                        or len(set(oids)) != len(oids) or len(set(cids)) != len(cids)
                        or not set(oids) <= original_ids or seen.intersection(oids)
                        or not set(cids) <= candidate_ids or link['relation'] not in RELATIONS
                        or not isinstance(link.get('rationale'), str) or not link['rationale'].strip()
                        or (link['relation'] in {'retained', 'modified', 'hedged'} and not cids)
                        or (link['relation'] == 'removed' and cids)):
                    raise ValueError('Invalid alignment group')
                seen.update(oids)
                linked.update(cids)
            extras = data['candidate_only_ids']
            if (seen != original_ids or not isinstance(extras, list)
                    or any(not isinstance(i, str) for i in extras)
                    or len(set(extras)) != len(extras) or set(extras) != candidate_ids - linked):
                raise ValueError('Incomplete alignment coverage')
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            # API transport errors still propagate; malformed model alignment
            # yields an explicit unresolved diagnostic, never a fabricated link.
            reason = f'alignment_unresolved: {type(exc).__name__}: {exc}'
            links = []
    if reason:
        links = [{'original_ids': [c.id], 'candidate_ids': [], 'relation': 'unresolved',
                  'rationale': reason} for c in before]
    linked = {cid for link in links for cid in link['candidate_ids']}
    matched = {oid for link in links if link['relation'] in {'retained', 'modified', 'hedged'}
               for oid in link['original_ids']}
    verified = {c.id for c in after if c.label == 'Supported'}
    grounded = {oid for link in links if link['relation'] in {'retained', 'modified'}
                and link['candidate_ids'] and set(link['candidate_ids']) <= verified
                for oid in link['original_ids']}
    supported_original = {c.id for c in before if c.label == 'Supported'}
    preserved = {oid for link in links if link['relation'] == 'retained'
                 and link['candidate_ids'] and set(link['candidate_ids']) <= verified
                 for oid in link['original_ids']}
    candidate_only = [asdict(c) for c in after if c.id not in linked]
    return {'semantics_version': 1, 'kind': 'model_diagnostic_not_human_assessment',
            'alignment_source': source, 'reason': reason, 'links': links,
            'original_targets': len(before), 'matched_original_targets': len(matched),
            'grounded_original_targets': len(grounded),
            'coverage': len(matched) / len(before) if before else None,
            'candidate_only': candidate_only,
            'candidate_only_unverified': [c['id'] for c in candidate_only if c['label'] != 'Supported'],
            'factual_success': bool(before and after and original_ids <= grounded
                                    and supported_original <= preserved
                                    and all(c.label == 'Supported' for c in after))}
