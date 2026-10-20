"""
Baseline B3: LLM self-repair without external evidence.
Tests the value of external retrieval vs. pure internal parametric knowledge.
"""
from src.schema import AnswerRecord
from src.llm import call as llm_call

B3_PROMPT = """You are a factual editor. Review the answer to the question.
Find and fix any factual errors or hallucinations in the answer using your own knowledge.
Keep the parts that are already correct and fluent. Return ONLY the repaired answer text.

Question: {question}
Answer: {answer}

Repaired answer:"""

def run_b3(record: AnswerRecord, model: str = "llama-3.3-70b-versatile", provider: str = "groq") -> AnswerRecord:
    prompt = B3_PROMPT.format(question=record.question, answer=record.answer)
    response = llm_call(prompt=prompt, model=model, provider=provider, temperature=0).strip()
    record.repaired_answer = response
    record.repair_iterations = 1
    return record

