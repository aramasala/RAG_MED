"""QA generator module for generating clinical questions and answers from PDF."""

from .generator import generate_qa, generate_qa_from_pdf

__all__ = ["generate_qa", "generate_qa_from_pdf"]
