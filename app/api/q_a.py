from typing import Any
from fastapi import APIRouter
from transformers import pipeline

qa_router = APIRouter()
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@qa_router.post("/answer")
async def answer(question: str, context: str) -> Any:
    return qa_pipeline(question=question, context=context)