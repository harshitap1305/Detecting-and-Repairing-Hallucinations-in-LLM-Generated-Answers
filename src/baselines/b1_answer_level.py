"""
Baseline B1: Answer-level detection.
One prompt on the whole answer: "Does this answer contain factual errors? yes/no".
"""
import json
import re
from src.schema import AnswerRecord
from src.experiments.runtime import strict
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
            result = json.loads(match.group())
            if strict.get() and (not isinstance(result, dict)
                    or type(result.get("has_hallucination")) is not bool
                    or type(result.get("confidence")) not in (int, float)
                    or not 0 <= result["confidence"] <= 1):
                raise ValueError("Invalid B1 verdict fields")
            return result
        except Exception:
            pass
    if strict.get():
        raise ValueError("Malformed B1 answer-detection verdict")
    return {"has_hallucination": False, "confidence": 0.5, "explanation": "Failed to parse"}

