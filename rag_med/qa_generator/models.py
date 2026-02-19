"""Pydantic models for QA generation."""

from pydantic import BaseModel, Field


class QAResult(BaseModel):
    """Result model for QA generation."""

    chunk_index: int = Field(..., description="Index of the chunk")
    chunk: str = Field(..., description="Text chunk used for generation")
    chunk_length_chars: int = Field(..., description="Character length of chunk")
    chunk_length_words: int = Field(..., description="Word count of chunk")
    model_used: str = Field(..., description="Model name used")
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Generated answer")
    raw_model_output: str = Field(..., description="Raw output from model")

    # Optional evaluation against external RAG system (ValueAI)
    valueai_answer: str | None = Field(None, description="Answer returned by ValueAI RAG")
    evaluation_metrics: dict | None = Field(
        None, description="Comparison metrics between generated and ValueAI answers"
    )
