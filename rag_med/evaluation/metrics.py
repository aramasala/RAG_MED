from __future__ import annotations

import re
from collections import Counter


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize a string into normalized whitespace tokens."""
    norm = normalize_text(text)
    return [t for t in norm.split(" ") if t]


def exact_match(reference: str, candidate: str) -> bool:
    """Exact match after normalization."""
    return normalize_text(reference) == normalize_text(candidate)


def token_f1(reference: str, candidate: str) -> dict[str, float]:
    """Token overlap F1 (precision/recall/F1) using bag-of-words counts."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens and not cand_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum((ref_counts & cand_counts).values())

    precision = overlap / max(1, len(cand_tokens))
    recall = overlap / max(1, len(ref_tokens))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute LCS length between two token lists (O(n*m) DP)."""
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> dict[str, float]:
    """ROUGE-L (LCS-based) precision/recall/F1 on tokens."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens and not cand_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, cand_tokens)
    precision = lcs / max(1, len(cand_tokens))
    recall = lcs / max(1, len(ref_tokens))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_answer_pair(reference: str, candidate: str) -> dict:
    """Compute a small set of metrics between reference and candidate answers."""
    tf1 = token_f1(reference, candidate)
    rl = rouge_l_f1(reference, candidate)
    return {
        "exact_match": exact_match(reference, candidate),
        "token_f1": tf1,
        "rouge_l": rl,
        "primary_score": float(0.5 * tf1["f1"] + 0.5 * rl["f1"]),
    }
