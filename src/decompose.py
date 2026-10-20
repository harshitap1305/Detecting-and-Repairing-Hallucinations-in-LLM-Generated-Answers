"""
decompose.py — Stage 1: Break an answer into atomic, decontextualized claims with span mapping.

Phase 4 Implementation:
  - Robust sentence splitting with character offsets [start_char, end_char]
  - Few-shot decontextualization prompt covering 8 distinct syntactic patterns:
      1. Multi-fact sentences
      2. Cross-sentence pronoun / coreference resolution
      3. List decomposition
      4. Hedged statements (check_worthy=False)
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
import subprocess
from typing import List, Optional, Tuple

from src.schema import AnswerRecord, Claim
import src.llm as llm

# ─── spaCy with regex fallback ────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                               check=True)
                _nlp = spacy.load("en_core_web_sm")
        except (ImportError, ModuleNotFoundError, Exception):
            _nlp = None
    return _nlp


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
   - Mark opinions, subjective judgments, questions, imperative suggestions, hedges, and AI meta-boilerplate with check_worthy=false.
5. SENTENCE MAPPING: Assign each claim its 0-indexed sentence_id in the original answer.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects:
[
  {"claim": "<standalone atomic claim>", "sentence_id": <int>, "check_worthy": <true|false>},
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
  {"claim": "William Shakespeare may have co-authored the lost play Cardenio with John Fletcher.", "sentence_id": 0, "check_worthy": false}
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


def _build_prompt(question: str, answer: str, sentences: List[Tuple[str, int, int]]) -> str:
    numbered_sentences = "\n".join(
        f"[{i}] {s_text}" for i, (s_text, _, _) in enumerate(sentences)
    )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n\n"
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


# ─── Public API ───────────────────────────────────────────────────────────────

def decompose(
    record: AnswerRecord,
    model: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
    max_retries: int = 2,
    min_claim_length: int = 5,
) -> AnswerRecord:
    """
    Decompose record.answer into atomic, decontextualized claims with span mapping.
    Mutates record.claims in place and returns the record.

    Args:
        record:           The AnswerRecord to process.
        model:            LLM model name.
        provider:         LLM provider ("groq" | "openai" | "local").
        max_retries:      Retries on malformed JSON response.
        min_claim_length: Discard claims shorter than this threshold.
    """
    sentence_spans = split_sentences_with_spans(record.answer)
    if not sentence_spans:
        record.claims = []
        return record

    prompt = _build_prompt(record.question, record.answer, sentence_spans)
    raw_claims: List[dict] = []

    for attempt in range(max_retries):
        temp = 0.0 if attempt == 0 else 0.2
        response = llm.call(
            prompt=prompt,
            model=model,
            provider=provider,
            temperature=temp,
            use_cache=(attempt == 0),
        )
        parsed = _parse_claims_json(response)
        if parsed is not None:
            raw_claims = parsed
            break
        if attempt == max_retries - 1:
            print(f"[decompose] ⚠ JSON parse failed for record {record.id} after {max_retries} attempts.")

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
    return record
