"""Evaluation utilities (RAGAS metrics; compare reference vs candidate)."""

from .metrics import (
    compare_two_answers,
    evaluate_answer_pair,
    evaluate_answer_pair_ragas_extended,
)

__all__ = [
    "compare_two_answers",
    "evaluate_answer_pair",
    "evaluate_answer_pair_ragas_extended",
]
