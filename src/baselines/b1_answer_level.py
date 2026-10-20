"""
Baseline B1: Answer-level detection.
One prompt on the whole answer: "Does this answer contain factual errors? yes/no".
"""
import json
import re
from src.schema import AnswerRecord
from src.llm import call as llm_call

B1_PROMPT = """You are evaluating the factual accuracy of an answer.

Question: {question}
Answer: {answer}

Does this answer contain any factual errors or hallucinated information?
Respond with JSON: {{"has_hallucination": true/false, "confidence": 0.0-1.0, "explanation": "..."}}
"""

def run_b1(record: AnswerRecord, model: str = "llama-3.3-70b-versatile", provider: str = "groq") -> dict:
    prompt = B1_PROMPT.format(question=record.question, answer=record.answer)
    response = llm_call(prompt=prompt, model=model, provider=provider, temperature=0)
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"has_hallucination": False, "confidence": 0.5, "explanation": "Failed to parse"}

