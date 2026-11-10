"""
decompose.py — Stage 1: Break an answer into atomic, decontextualized claims with span mapping.

Phase 4 Implementation:
  - Robust sentence splitting with character offsets [start_char, end_char]
  - Few-shot decontextualization prompt covering 8 distinct syntactic patterns:
      1. Multi-fact sentences
      2. Cross-sentence pronoun / coreference resolution
      3. List decomposition
      4. Hedged factual statements (check_worthy=True; preserve uncertainty)
      5. Subjective opinions & hyperbole (check_worthy=False)
      6. AI/LLM boilerplate & meta-text (check_worthy=False)
      7. Questions & imperative instructions (check_worthy=False)
      8. Precise temporal and numeric qualifiers
  - Character-level span alignment [start, end] mapping claims back to answer text
  - JSON parse validation with retry and repair heuristics

Public API:
    split_sentences(text: str) -> List[str]
    split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]
    align_claim_span(claim_text: str, sent_text: str, sent_start: int, sent_end: int) -> List[int]
    decompose(record: AnswerRecord, model: str, provider: str) -> AnswerRecord
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from src.schema import AnswerRecord, Claim
import src.llm as llm
from src.experiments.runtime import strict, recorder
from src.answer_attribution import build_attribution, question_clauses, reanchor_context_quotes
from src.extraction_check import check_extraction

# ─── spaCy with regex fallback ────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            # Sentence splitting must also work offline and in test runners.
            # Installing a model is an explicit setup step, never a runtime side effect.
            _nlp = False
    return _nlp if _nlp is not False else None


def split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences preserving character offsets.
    Returns: List of (sentence_text, start_char, end_char)
    """
    if not text.strip():
        return []

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        sents = []
        for s in doc.sents:
            s_text = s.text.strip()
            if s_text:
                # Calculate stripped start and end
                leading_ws = len(s.text) - len(s.text.lstrip())
                start = s.start_char + leading_ws
                end = start + len(s_text)
                sents.append((s_text, start, end))
        if sents:
            return sents

    # Fallback regex splitter preserving exact char offsets
    spans: List[Tuple[str, int, int]] = []
    pattern = re.compile(r'[^.!?\n]+(?:[.!?\n]+|$)')
    for match in pattern.finditer(text):
        raw = match.group()
        stripped = raw.strip()
        if stripped:
            start = match.start() + (len(raw) - len(raw.lstrip()))
            end = start + len(stripped)
            spans.append((stripped, start, end))
    return spans


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy or fallback regex."""
    return [s for s, _, _ in split_sentences_with_spans(text)]


# ─── Span Alignment ───────────────────────────────────────────────────────────

def align_claim_span(
    claim_text: str,
    sent_text: str,
    sent_start: int,
    sent_end: int,
) -> List[int]:
    """
    Find character offsets [start, end] of a claim within its source sentence.
    Falls back to [sent_start, sent_end] if no exact substring or token span is found.
    """
    if not sent_text:
        return [sent_start, sent_end]

    # 1. Direct exact or lower-cased substring match
    pos = sent_text.lower().find(claim_text.lower().rstrip("."))
    if pos != -1:
        match_len = len(claim_text.rstrip("."))
        return [sent_start + pos, sent_start + pos + match_len]

    # 2. Key content tokens matching window
    # Extract substantive alphanumeric tokens (>2 chars)
    tokens = [t.lower() for t in re.findall(r'\b\w+\b', claim_text)
              if len(t) > 2 and t.lower() not in {"the", "and", "was", "were", "are", "for", "that"}]
    
    if len(tokens) >= 2:
        # Search for first and last token in sentence
        sent_lower = sent_text.lower()
        first_pos = sent_lower.find(tokens[0])
        last_pos = sent_lower.rfind(tokens[-1])
        if first_pos != -1 and last_pos != -1 and last_pos >= first_pos:
            end_pos = last_pos + len(tokens[-1])
            return [sent_start + first_pos, sent_start + end_pos]

    # 3. Fallback to full sentence span
    return [sent_start, sent_end]


# ─── Few-Shot Decontextualization Prompt ──────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert NLP fact-checking assistant. Your task is to break an answer into ATOMIC, SELF-CONTAINED, DECONTEXTUALIZED factual claims.

CORE PRINCIPLES:
1. ATOMICITY: Each claim must express exactly ONE independently checkable fact. If a sentence has multiple facts, split it.
2. DECONTEXTUALIZATION: Resolve all pronouns (he, she, it, they, this, that) into explicit named entities so each claim can be verified in isolation.
3. PRESERVATION: Never change names, dates, numbers, measurements, or technical terms.
4. CHECK-WORTHINESS:
   - Mark verifiable real-world facts with check_worthy=true.
   - Mark opinions, subjective judgments, questions, imperative suggestions, and AI meta-boilerplate with check_worthy=false.
   - A hedge does NOT remove factual content. Keep factual assertions check_worthy=true
     and preserve uncertainty, frequency qualifications and attribution in the claim text.
     Never turn "may have" into "did" or invent a source for "Reportedly".
5. SENTENCE MAPPING: Assign each claim its 0-indexed sentence_id in the original answer.
6. SHORT ANSWERS: Use the question ONLY to resolve the relationship and referents
   expressed by the answer. Expand entity-only, date-only, frequency-only and
   definition fragments into full propositions. Do not reduce an answer to a
   generic category claim (for example, merely saying that an event is an event).
7. QUESTION BOUNDARIES: A question is not evidence that its premises are true.
   Do not invent missing answers to other clauses of a compound question. If a
   reference is ambiguous, preserve that ambiguity instead of guessing a fact.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects:
[
  {"claim": "<standalone atomic claim>", "sentence_id": <int>, "check_worthy": <true|false>,
   "attribution": {"status": "bound|ambiguous|answer_only|nonfactual",
     "answer_spans": [[<start>, <end>]], "bindings": [
       {"clause_id": "q1", "answer_span": [<start>, <end>], "answer_text": "<exact answer slice>",
        "relation": "<answer-bearing relationship, not a descriptive question premise>",
        "borrowed_context": [{"source": "question|answer", "span": [<start>, <end>], "text": "<exact source slice>",
                              "role": "referent|relation_context"}]}],
     "rationale": "<why this answer span expresses this relationship>"}},
  ...
]"""

_FEW_SHOT_EXAMPLES = """EXAMPLE 1: Multi-fact sentence
Question: Who was Marie Curie?
Answer: Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity.
Output:
[
  {"claim": "Marie Curie was a Polish-born physicist.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Marie Curie was a chemist.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Marie Curie conducted pioneering research on radioactivity.", "sentence_id": 0, "check_worthy": true}
]

EXAMPLE 2: Coreference and pronoun resolution across sentences
Question: When did Albert Einstein publish his theories?
Answer: Albert Einstein published four groundbreaking papers in 1905. He was later awarded the Nobel Prize in Physics in 1921.
Output:
[
  {"claim": "Albert Einstein published four groundbreaking papers in 1905.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Albert Einstein was awarded the Nobel Prize in Physics in 1921.", "sentence_id": 1, "check_worthy": true}
]

EXAMPLE 3: Entity list decomposition
Question: What are Canada's largest cities?
Answer: Canada's three largest metropolitan areas are Toronto, Montreal, and Vancouver.
Output:
[
  {"claim": "Toronto is one of Canada's three largest metropolitan areas.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Montreal is one of Canada's three largest metropolitan areas.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Vancouver is one of Canada's three largest metropolitan areas.", "sentence_id": 0, "check_worthy": true}
]

EXAMPLE 4: Subjective opinion and hyperbole (non-check-worthy)
Question: Tell me about the Palace of Versailles.
Answer: The Palace of Versailles was built in France in 1631. It is the most breathtaking and glorious royal palace in human history.
Output:
[
  {"claim": "The Palace of Versailles was built in France in 1631.", "sentence_id": 0, "check_worthy": true},
  {"claim": "The Palace of Versailles is the most breathtaking and glorious royal palace in human history.", "sentence_id": 1, "check_worthy": false}
]

EXAMPLE 5: Hedged statements
Question: Who wrote the play Cardenio?
Answer: Some literary scholars believe that William Shakespeare may have co-authored the lost play Cardenio with John Fletcher.
Output:
[
  {"claim": "Some literary scholars believe that William Shakespeare may have co-authored the lost play Cardenio with John Fletcher.", "sentence_id": 0, "check_worthy": true}
]

EXAMPLE 6: AI boilerplate and meta-discourse
Question: How long is a lunar day?
Answer: As an AI language model, I can tell you that the Moon rotates on its axis once every 27.3 Earth days. I hope this helps!
Output:
[
  {"claim": "As an AI language model, I can tell you this information.", "sentence_id": 0, "check_worthy": false},
  {"claim": "The Moon rotates on its axis once every 27.3 Earth days.", "sentence_id": 0, "check_worthy": true},
  {"claim": "I hope this helps!", "sentence_id": 1, "check_worthy": false}
]

EXAMPLE 7: Questions and imperative commands
Question: What should I see in Rome?
Answer: Have you visited Rome? Be sure to explore the Colosseum, which was inaugurated in 80 AD under Emperor Titus.
Output:
[
  {"claim": "Have you visited Rome?", "sentence_id": 0, "check_worthy": false},
  {"claim": "Be sure to explore the Colosseum.", "sentence_id": 1, "check_worthy": false},
  {"claim": "The Colosseum was inaugurated in 80 AD.", "sentence_id": 1, "check_worthy": true},
  {"claim": "The Colosseum was inaugurated under Emperor Titus.", "sentence_id": 1, "check_worthy": true}
]

EXAMPLE 8: Temporal and numerical precision
Question: Describe the Apollo 11 moon landing.
Answer: Apollo 11 landed on the lunar surface on July 20, 1969, carrying commander Neil Armstrong and pilot Buzz Aldrin.
Output:
[
  {"claim": "Apollo 11 landed on the lunar surface on July 20, 1969.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Neil Armstrong was the commander of Apollo 11.", "sentence_id": 0, "check_worthy": true},
  {"claim": "Buzz Aldrin was the pilot of Apollo 11.", "sentence_id": 0, "check_worthy": true}
]"""


# Generic fragment examples; no pilot gold answers are used at inference time.
_FRAGMENT_EXAMPLES = """
EXAMPLE 9: Entity-only answer (synthetic)
Question: Where was researcher Mira Vale born?
Answer: Alder Bay
Output:
[{"claim": "Mira Vale was born in Alder Bay.", "sentence_id": 0, "check_worthy": true}]

EXAMPLE 10: Frequency-only answer (synthetic)
Question: How often is the Northbridge Review published?
Answer: Monthly
Output:
[{"claim": "The Northbridge Review is published monthly.", "sentence_id": 0, "check_worthy": true}]

EXAMPLE 11: Definition fragment and unanswered question (synthetic)
Question: What is the Cedar Institute known for? What activity measures distances?
Answer: Surveying
Output:
[{"claim": "Surveying involves measuring distances.", "sentence_id": 0, "check_worthy": true}]
Do not assert that the Cedar Institute teaches surveying: the answer did not say so.

EXAMPLE 12: Qualified fragment (synthetic)
Question: How often does the Brook ferry operate?
Answer: Usually daily
Output:
[{"claim": "The Brook ferry usually operates daily.", "sentence_id": 0, "check_worthy": true}]

EXAMPLE 13: Ambiguous referent (synthetic)
Question: When did they arrive?
Answer: 1998
Output:
[{"claim": "The people referred to in the question arrived in 1998.", "sentence_id": 0, "check_worthy": true}]
Do not invent names for unspecified people.
"""


_ATTRIBUTION_EXAMPLE_QUESTION = "Where was Nera born? What year did she move?"
_ATTRIBUTION_EXAMPLE = [{
    "claim": "Nera was born in Cedar Bay.", "sentence_id": 0, "check_worthy": True,
    "attribution": {"status": "bound", "answer_spans": [[0, 9]],
        "bindings": [{"clause_id": "q1", "answer_span": [0, 9], "answer_text": "Cedar Bay",
            "relation": "birthplace", "borrowed_context": [{"source": "question",
                "span": [_ATTRIBUTION_EXAMPLE_QUESTION.index("Nera"), _ATTRIBUTION_EXAMPLE_QUESTION.index("Nera") + 4],
                "text": "Nera", "role": "referent"}]}],
        "rationale": "The location fragment answers only the birthplace clause, not the move-year clause."},
}]


def _build_prompt(question: str, answer: str, sentences: List[Tuple[str, int, int]]) -> str:
    numbered_sentences = "\n".join(
        f"[{i}] chars[{start}:{end}] {s_text}" for i, (s_text, start, end) in enumerate(sentences)
    )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n{_FRAGMENT_EXAMPLES}\n\n"
        "The examples above illustrate claim content. In your output also provide attribution for EVERY claim. "
        "Use original character offsets. A bound factual claim requires an answer-bearing relationship, "
        "answer spans and question clause IDs. Borrow only referents/relation context, never a new factual answer. "
        "The same span can support multiple claims only when the answer actually expresses each relationship. "
        "Use ambiguous for uncertain clause/reference binding, with no guessed names or missing answers. "
        "Use answer_only for explicit answer facts not answering a question clause; nonfactual for boilerplate/opinions. "
        "Those last two statuses have empty bindings. Borrowed context can point to question text or earlier answer text. "
        "Do not mark every clause answered by one fragment.\n"
        f"Complete attribution example: Question={_ATTRIBUTION_EXAMPLE_QUESTION!r}, Answer='Cedar Bay', "
        f"Output={json.dumps(_ATTRIBUTION_EXAMPLE)}. The q2 clause remains unanswered.\n"
        f"Available question clauses (surface segmentation): {json.dumps(question_clauses(question), ensure_ascii=False)}\n"
        f"Original answer length: {len(answer)} characters.\n"
        f"NOW DECOMPOSE THE FOLLOWING:\n"
        f"Question: {question}\n"
        f"Numbered sentences in Answer:\n{numbered_sentences}\n\n"
        f"Output JSON array:"
    )


# ─── Parsing and Validation ───────────────────────────────────────────────────

def _parse_claims_json(response: str) -> Optional[List[dict]]:
    """Extract and parse JSON array with fence removal and bracket matching."""
    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return None


def _clean_claim_text(text: str) -> str:
    """Normalize quotation marks, whitespace, and leading bullets."""
    text = re.sub(r'^[\s\-\*•\d\.\)]+', '', text)
    text = text.strip(' "\'')
    if text and not text.endswith(('.', '?', '!')):
        text += '.'
    return text


_MODALITY = re.compile(
    r"\b(reportedly|allegedly|possibly|probably|might|usually|sometimes|"
    r"according to|may (?!\d)|believe[ds]?)\b", re.I,
)
_PREDICATE = re.compile(
    r"\b(is|are|was|were|has|have|had|involves?|includes?|operates?|published|born|"
    r"died|arrived|measures?|occurs?|wrote|won|built|created|founded|released|"
    r"located|lives?|works?|teaches?|runs?|opens?)\b", re.I,
)


def _contextual_fragment(question: str, answer: str) -> bool:
    """Conservative retry signal, not a factuality or grammatical classifier."""
    if not re.match(r"^(who|when|where|which|what|how|at what|in what)\b", question.strip(), re.I):
        return False
    if re.search(r"\b(opinion|beautiful|favorite|favourite|amazing|wonderful)\b", question + " " + answer, re.I):
        return False
    if re.match(r"^(I (think|believe|hope|don.t know)|as an AI|please|perhaps not|unknown|not sure|no idea)\b", answer.strip(), re.I):
        return False
    return len(answer.split()) <= 6 and not _PREDICATE.search(answer)


def _inventory_issue(parsed, question, answer, sentence_spans, min_length):
    if not parsed:
        return "No claim inventory. Extract answer content without inventing missing facts."
    for item in parsed:
        if (not isinstance(item, dict) or not isinstance(item.get("claim"), str)
                or len(_clean_claim_text(item["claim"])) < min_length
                or type(item.get("sentence_id")) is not int
                or not 0 <= item["sentence_id"] < len(sentence_spans)
                or type(item.get("check_worthy")) is not bool):
            return "Malformed decomposition claim fields or unusably short claim."
        source = sentence_spans[item["sentence_id"]][0]
        # A surface safeguard: ask the model to fix loss of modality, never
        # manufacture a stronger proposition or force every sentence factual.
        if _MODALITY.search(source) and (_PREDICATE.search(source) or _contextual_fragment(question, source)):
            if not item["check_worthy"] or not _MODALITY.search(item["claim"]):
                return "Preserve factual uncertainty/attribution in claim text and keep factual payload check-worthy."
    if _contextual_fragment(question, answer):
        worthy = [item for item in parsed if item["check_worthy"]]
        if not worthy or any(
                _clean_claim_text(item["claim"]).casefold() == _clean_claim_text(answer).casefold()
                for item in worthy):
            return "Expand the short answer into a checkable relationship using question context; do not invent other answers."
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def decompose(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    max_retries: int = 2,
    min_claim_length: int = 5,
    extraction_check: str = "structural",
    max_checks: int = 2,
    max_check_claims: int = 64,
    attribution_anchor_mode: str = "unique_quote_v1",
) -> AnswerRecord:
    """
    Decompose record.answer into atomic, decontextualized claims with span mapping.
    Mutates record.claims in place and returns the record.

    Args:
        record:           The AnswerRecord to process.
        model:            LLM model name.
        provider:         LLM provider ("groq" | "openai" | "local").
        max_retries:      Maximum attempts, including retries for unusable inventories.
        min_claim_length: Retry claims shorter than this threshold.
    """
    if type(max_retries) is not int or not 1 <= max_retries <= 4:
        raise ValueError("max_retries must allow 1 to 4 decomposition attempts")
    if attribution_anchor_mode not in ("strict", "unique_quote_v1"):
        raise ValueError("attribution_anchor_mode must be strict or unique_quote_v1")
    if extraction_check not in ("structural", "semantic_v1"):
        raise ValueError("extraction_check must be structural or semantic_v1")
    if type(max_checks) is not int or not 1 <= max_checks <= 4:
        raise ValueError("max_checks must be between 1 and 4")
    if type(max_check_claims) is not int or not 1 <= max_check_claims <= 128:
        raise ValueError("max_check_claims must be between 1 and 128")
    sentence_spans = split_sentences_with_spans(record.answer)
    if not sentence_spans:
        record.claims = []
        record.decomposition_diagnostics = {"semantics_version": 5, "status": "empty_answer", "attempts": 0,
            "attribution": build_attribution([], question=record.question, answer=record.answer,
                                             sentence_spans=[], record_id=record.id)}
        return record

    prompt = _build_prompt(record.question, record.answer, sentence_spans)
    raw_claims: List[dict] = []

    issue = None
    attribution = None
    semantic_check = None
    checks_used = 0
    attempts = []
    for attempt in range(max_retries):
        attribution = None
        temp = 0.0 if attempt == 0 else 0.2
        response = llm.call(
            prompt=prompt if issue is None else prompt + "\nRETRY FEEDBACK: " + issue,
            model=model, provider=provider, temperature=temp, use_cache=(attempt == 0),
        )
        parsed = _parse_claims_json(response)
        span_repairs = []
        if parsed is not None and attribution_anchor_mode == "unique_quote_v1":
            parsed, span_repairs = reanchor_context_quotes(parsed, question=record.question, answer=record.answer)
        issue = ("Malformed JSON; return a JSON array." if parsed is None else
                 _inventory_issue(parsed, record.question, record.answer, sentence_spans, min_claim_length))
        if issue is None:
            try:
                attribution = build_attribution(parsed, question=record.question, answer=record.answer,
                                                sentence_spans=sentence_spans, record_id=record.id)
            except (ValueError, TypeError, KeyError) as exc:
                issue = f"Malformed answer attribution: {exc}"
        proposal = [{**c, "claim": _clean_claim_text(c["claim"])} for c in parsed] if issue is None else []
        semantic_check = None
        if issue is None and extraction_check == "semantic_v1":
            if checks_used >= max_checks:
                semantic_check = {"status": "unresolved", "reason": "semantic_check_budget_exhausted", "model_calls": 0}
            else:
                semantic_check = check_extraction(question=record.question, answer=record.answer,
                    items=proposal, attribution=attribution, record_id=record.id, model=model,
                    provider=provider, max_claims=max_check_claims, use_cache=(attempt == 0))
                checks_used += semantic_check["model_calls"]
                audit_history = [semantic_check]
                # One format/consistency retry uses the existing check budget;
                # rejected semantic findings instead go back to extraction.
                if semantic_check.get("reason") == "malformed_semantic_check" and checks_used < max_checks:
                    semantic_check = check_extraction(question=record.question, answer=record.answer,
                        items=proposal, attribution=attribution, record_id=record.id, model=model,
                        provider=provider, max_claims=max_check_claims, use_cache=False,
                        retry_feedback=audit_history[0])
                    checks_used += semantic_check["model_calls"]
                    semantic_check["prior_format_failure"] = audit_history[0]
            if semantic_check["status"] != "passed":
                issue = "Semantic extraction unresolved: " + json.dumps(semantic_check, ensure_ascii=False)
        attempts.append({"attempt": attempt + 1, "issue": issue, "semantic_check": semantic_check, "span_repairs": span_repairs,
                         "proposed_claims": [{"text": c["claim"], "sentence_id": c["sentence_id"],
                                              "check_worthy": c["check_worthy"]} for c in proposal]})
        if issue is None:
            raw_claims = proposal
            break
        if semantic_check and (semantic_check.get("reason") == "semantic_check_budget_exhausted"
                               or checks_used >= max_checks):
            break
    record.decomposition_diagnostics = {
        "semantics_version": 5, "status": "unresolved_coverage" if issue else "complete",
        "attempts": attempt + 1, "reason": issue,
        "attribution": attribution,
        "check_mode": extraction_check, "attribution_anchor_mode": attribution_anchor_mode, "semantic_check": semantic_check,
        "semantic_checks_used": checks_used, "attempt_history": attempts,
        "budgets": {"attempts": max_retries, "checks": max_checks, "inventory": max_check_claims},
        "stop_reason": ("semantic_check_budget_exhausted" if issue and checks_used >= max_checks else
                        "attempt_budget_exhausted" if issue else "extraction_accepted"),
    }
    callback = recorder.get()
    if callback:
        callback({"event": "extraction", "record_id": record.id, "answer": record.answer,
                  "diagnostics": record.decomposition_diagnostics})
    if issue and strict.get():
        raise ValueError("Decomposition unresolved coverage: " + issue)
    claims: List[Claim] = []
    for i, rc in enumerate(raw_claims):
        if not isinstance(rc, dict):
            continue
        raw_text = str(rc.get("claim", ""))
        cleaned_text = _clean_claim_text(raw_text)
        if len(cleaned_text) < min_claim_length:
            continue

        raw_sid = rc.get("sentence_id", 0)
        try:
            sid = int(raw_sid)
            if sid < 0 or sid >= len(sentence_spans):
                sid = 0
        except (ValueError, TypeError):
            sid = 0

        # Retrieve sentence offsets for span mapping
        sent_text, sent_start, sent_end = sentence_spans[sid]
        span = align_claim_span(cleaned_text, sent_text, sent_start, sent_end)

        claim = Claim(
            id=f"{record.id}_c{len(claims) + 1}",
            text=cleaned_text,
            sentence_id=sid,
            span=span,
            check_worthy=bool(rc.get("check_worthy", True)),
        )
        claims.append(claim)

    record.claims = claims
    if not issue and not record.check_worthy_claims:
        record.decomposition_diagnostics["status"] = "no_check_worthy_claims"
    return record
