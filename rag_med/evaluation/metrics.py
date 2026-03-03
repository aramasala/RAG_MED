"""RAG evaluation metrics: RAGAS Faithfulness, FactualCorrectness, cosine_similarity."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections import Counter
from pathlib import Path

from configs.paths import PROJECT_DPATH
from configs.settings import settings

_DEBUG_LOG_PATH = PROJECT_DPATH / "debug" / "debug-3dbdd8.log"
_DEBUG_PROOF_PATH = PROJECT_DPATH / "debug" / "debug_ragas_client.txt"


def _debug_log(hypothesis_id: str, message: str, data: dict, location: str = ""):
    import time as _t
    payload = {"sessionId": "3dbdd8", "hypothesisId": hypothesis_id, "message": message, "data": data, "timestamp": int(_t.time() * 1000), "location": location or "metrics.py"}
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        with open(_DEBUG_PROOF_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass

logger = logging.getLogger(__name__)

#  cosine_similarity (reference vs candidate) 

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return [t for t in text.split() if t]


def _cosine_similarity(reference: str, candidate: str) -> float:
    """Cosine similarity between reference and candidate (bag-of-words). Score in [0, 1]."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    vocab = sorted(set(ref_tokens) | set(cand_tokens))
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    ref_vec = [ref_counts.get(t, 0) for t in vocab]
    cand_vec = [cand_counts.get(t, 0) for t in vocab]
    dot = sum(a * b for a, b in zip(ref_vec, cand_vec))
    norm_ref = sum(a * a for a in ref_vec) ** 0.5
    norm_cand = sum(b * b for b in cand_vec) ** 0.5
    if norm_ref == 0 or norm_cand == 0:
        return 0.0
    cos = dot / (norm_ref * norm_cand)
    return max(0.0, min(1.0, float(cos)))


def compare_two_answers(reference: str, candidate: str) -> dict:
    """Return cosine_similarity and factual_correctness (etalon vs RAG answer)."""
    out: dict = {"cosine_similarity": _cosine_similarity(reference, candidate)}

    # FactualCorrectness (LLM)
    try:
        fc = _get_factual_correctness_scorer()
        result = fc.score(response=candidate, reference=reference)
        out["factual_correctness"] = float(result.value)
    except Exception as e:
        logger.warning("FactualCorrectness failed: %s", e)
        out["factual_correctness"] = None
        out["factual_error"] = str(e)

    return out


#  RAGAS Faithfulness 


def _use_valueai_llm() -> bool:
    """True if ValueAI base URL and credentials are set (use v1/llm for metrics)."""
    base = getattr(settings, "valueai_base_url", None) or ""
    user = getattr(settings, "valueai_username", None) or ""
    pwd = getattr(settings, "valueai_password", None) or ""
    return bool(base.strip() and user and pwd)


def _get_ragas_llm():  # noqa: ANN201
    from ragas.llms import llm_factory

    if _use_valueai_llm():
        from configs.llm_api_client import ValueAIAsyncOpenAI, get_token

        base = (getattr(settings, "valueai_base_url", None) or "").rstrip("/")
        token = get_token(
            base,
            getattr(settings, "valueai_username", "") or "",
            getattr(settings, "valueai_password", "") or "",
        )
        model_name = getattr(settings, "metrics_llm_model_name", "llm_qwen_2_5_coder_32b_instruct_q8")
        max_tokens = getattr(settings, "ragas_max_tokens", 8192)
        poll = getattr(settings, "metrics_llm_poll_interval_seconds", 2.0)
        timeout = getattr(settings, "metrics_llm_timeout_seconds", 120.0)
        client = ValueAIAsyncOpenAI(
            base_url=base,
            token=token,
            model_name=model_name,
            max_tokens=max_tokens,
            poll_interval=poll,
            timeout=timeout,
        )
        import openai as _openai
        create_fn = getattr(getattr(client, "chat", None), "completions", None)
        create_fn = getattr(create_fn, "create", None) if create_fn else None
        _debug_log("H2", "ValueAIAsyncOpenAI after create", {
            "client_type": type(client).__name__,
            "create_is_coro": inspect.iscoroutinefunction(create_fn) if create_fn else None,
            "isinstance_AsyncOpenAI": isinstance(client, _openai.AsyncOpenAI),
            "isinstance_OpenAI": isinstance(client, _openai.OpenAI),
        }, "metrics.py:_get_ragas_llm")
        llm = llm_factory(
            model_name,
            provider="openai",
            client=client,
            max_tokens=max_tokens,
        )
        stored = getattr(llm, "client", None)
        stored_create = getattr(getattr(stored, "chat", None), "completions", None)
        stored_create = getattr(stored_create, "create", None) if stored_create else None
        proof = {
            "isinstance_AsyncOpenAI": isinstance(client, _openai.AsyncOpenAI),
            "isinstance_OpenAI": isinstance(client, _openai.OpenAI),
            "llm_client_type": type(stored).__name__ if stored else None,
            "stored_create_is_coro": inspect.iscoroutinefunction(stored_create) if stored_create else None,
            "llm_is_async": getattr(llm, "is_async", None),
        }
        _debug_log("H1", "After llm_factory", proof, "metrics.py:after_llm_factory")
        try:
            _DEBUG_PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_DEBUG_PROOF_PATH, "w", encoding="utf-8") as f:
                f.write(json.dumps(proof, ensure_ascii=False, indent=2))
        except Exception:
            pass
        return llm
    raise ValueError(
        "ValueAI credentials are required for RAGAS metrics. "
        "Set VALUEAI_BASE_URL, VALUEAI_USERNAME, VALUEAI_PASSWORD in .env or configs/settings."
    )


def _get_factual_correctness_scorer():
    from ragas.metrics.collections import FactualCorrectness

    if not hasattr(_get_factual_correctness_scorer, "_scorer"):
        llm = _get_ragas_llm()
        _get_factual_correctness_scorer._scorer = FactualCorrectness(llm=llm, mode="f1")
    return _get_factual_correctness_scorer._scorer


def _get_faithfulness_scorer():
    from ragas.metrics.collections import Faithfulness

    if not hasattr(_get_faithfulness_scorer, "_scorer"):
        llm = _get_ragas_llm()
        _get_faithfulness_scorer._scorer = Faithfulness(llm=llm)
    return _get_faithfulness_scorer._scorer


def evaluate_answer_pair_ragas_extended(
    question: str,
    response: str,
    retrieved_contexts: list[str],
    reference_answer: str | None = None,
) -> dict:
    """Return only faithfulness (RAGAS)."""
    out = {"faithfulness": 0.0}
    if not retrieved_contexts:
        return out
    try:
        result = _get_faithfulness_scorer().score(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        out["faithfulness"] = float(result.value)
    except Exception as e:
        try:
            scorer = _get_faithfulness_scorer()
            sl = getattr(scorer, "llm", None)
            _debug_log("H3", "On Faithfulness exception", {"scorer_llm_client_type": type(getattr(sl, "client", None)).__name__ if sl else None, "llm_is_async": getattr(sl, "is_async", None), "error": str(e)[:200]}, "metrics.py:except_ragas")
        except Exception as _:
            pass
        logger.exception("RAGAS Faithfulness failed: %s", e)
        out["error"] = str(e)
    return out


def evaluate_answer_pair(question: str, response: str, retrieved_contexts: list[str]) -> dict:
    """Return only faithfulness (RAGAS)."""
    if not retrieved_contexts:
        return {"faithfulness": 0.0}
    out = {"faithfulness": 0.0}
    try:
        result = _get_faithfulness_scorer().score(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        out["faithfulness"] = float(result.value)
    except Exception as e:
        logger.exception("RAGAS Faithfulness failed: %s", e)
        out["error"] = str(e)
    return out


#  LLM judge: alignment (RAG vs etalon), 1–10 


def evaluate_answer_pair_llm_alignment(
    question: str,
    etalon_answer: str,
    rag_answer: str,
    context: str | None = None,
) -> dict:
    """LLM-judge alignment score (1–10) for RAG answer vs etalon.

    Uses ValueAI LLM (metrics_llm_model_name). Prompt and output are fully in Russian.
    """
    out: dict = {
        "alignment_score": None,
        "alignment_comment": None,
    }

    if not etalon_answer or not rag_answer:
        out["alignment_error"] = "missing etalon_answer or rag_answer"
        return out

    question_str = question or ""
    context_str = context or ""

    system_prompt = (
        "Ты — экспертный врач и методист, который оценивает качество ответа модели "
        "по сравнению с эталонным ответом."
    )

    user_prompt_template = """
Тебе даны:
1) Вопрос пациента или клинический вопрос.
2) Фрагмент клинического текста (контекст), на основе которого должны строиться ответы.
3) Эталонный ответ (правильный, проверенный специалистом).
4) Ответ RAG‑системы (модели), который нужно сравнить с эталоном.

Твоя задача — ОЦЕНИТЬ, насколько ответ RAG‑системы соответствует эталонному
по следующим критериям:
- медицинская корректность (нет ли ошибок, противоречий с эталоном и контекстом),
- полнота (насколько хорошо покрыты ключевые моменты эталона),
- ясность и структурированность.

Оцени СООТВЕТСТВИЕ ответа RAG‑системы ЭТАЛОНУ по шкале от 1 до 10:
- 1–3: ответ сильно хуже эталона, есть серьёзные ошибки или упущены важные моменты.
- 4–6: частично совпадает с эталоном, но ответ заметно слабее.
- 7–8: в целом близко к эталону, есть лишь несущественные недочёты.
- 9–10: очень близко к эталону, без существенных расхождений.

Входные данные:

Вопрос:
{question}

Контекст:
{context}

Эталонный ответ:
{etalon_answer}

Ответ RAG‑системы:
{rag_answer}

ОТВЕТЬ ТОЛЬКО В ВИДЕ ВАЛИДНОГО JSON, БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА, В СЛЕДУЮЩЕМ ФОРМАТЕ:

{{
  "alignment_score": <ЦЕЛОЕ_ЧИСЛО_ОТ_1_ДО_10>,
  "comment": "<КРАТКОЕ_ОБОСНОВАНИЕ_НА_РУССКОМ_ЯЗЫКЕ>"
}}
"""

    user_prompt = user_prompt_template.format(
        question=question_str,
        context=context_str,
        etalon_answer=etalon_answer,
        rag_answer=rag_answer,
    )

    try:
        if _use_valueai_llm():
            from configs.llm_api_client import get_token, predict_sync

            base = (getattr(settings, "valueai_base_url", None) or "").rstrip("/")
            token = get_token(
                base,
                getattr(settings, "valueai_username", "") or "",
                getattr(settings, "valueai_password", "") or "",
            )
            model_name = getattr(settings, "metrics_llm_model_name", "llm_qwen_2_5_coder_32b_instruct_q8")
            poll = getattr(settings, "metrics_llm_poll_interval_seconds", 2.0)
            timeout = getattr(settings, "metrics_llm_timeout_seconds", 120.0)
            content = predict_sync(
                base,
                token,
                model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=getattr(settings, "ragas_max_tokens", 8192),
                temperature=0,
                poll_interval=poll,
                timeout=timeout,
            ).strip()
        else:
            out["alignment_error"] = (
                "ValueAI credentials required for LLM judge. "
                "Set VALUEAI_BASE_URL, VALUEAI_USERNAME, VALUEAI_PASSWORD."
            )
            return out
    except Exception as e:
        logger.exception("LLM alignment judge call failed: %s", e)
        out["alignment_error"] = str(e)
        return out

    try:
        data = json.loads(content)
        score = data.get("alignment_score")
        comment = data.get("comment")

        if isinstance(score, (int, float)):
            score_int = int(round(score))
            score_int = max(1, min(10, score_int))
            out["alignment_score"] = score_int
        else:
            out["alignment_score"] = None

        if isinstance(comment, str):
            out["alignment_comment"] = comment
    except Exception as e:
        logger.exception("Failed to parse LLM alignment JSON: %s", e)
        out["alignment_error"] = str(e)
        out["alignment_comment"] = content[:500]

    return out
