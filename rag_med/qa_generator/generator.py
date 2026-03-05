"""QA generation functionality."""

import json
import logging
import random
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from configs.settings import settings
from rag_med.valueai.llm_api_client import get_token, predict_sync
from rag_med.evaluation.metrics import (
    compare_two_answers,
    evaluate_answer_pair_llm_alignment,
    evaluate_answer_pair_ragas_extended,
)
from rag_med.valueai.client import ValueAIRagClient, ValueAIRagClientConfig

from rag_med.qa_generator.models import QAResult

logger = logging.getLogger(__name__)


CHUNKS_PER_QA = 4  


def _extract_qa_from_json_like(text: str) -> tuple[str, str] | None:
    """Extract question and answer from JSON-like text. Handles truncated JSON (missing closing \"})."""
    if not text or "question" not in text.lower() or "answer" not in text.lower():
        return None
    for label in ('"question": "', '"Вопрос": "'):
        q_start = text.find(label)
        if q_start == -1:
            continue
        start = q_start + len(label)
        i = start
        while i < len(text):
            if text[i] == "\\" and i + 1 < len(text):
                i += 2
                continue
            if text[i] == '"':
                question = text[start:i].replace("\\n", "\n").replace('\\"', '"').strip()
                break
            i += 1
        else:
            continue
        for a_label in ('"answer": "', '"Ответ": "'):
            a_start = text.find(a_label, i)
            if a_start == -1:
                continue
            start_a = a_start + len(a_label)
            j = start_a
            while j < len(text):
                if text[j] == "\\" and j + 1 < len(text):
                    j += 2
                    continue
                if text[j] == '"':
                    answer = text[start_a:j]
                    break
                j += 1
            else:
                answer = text[start_a:]  # truncated JSON
            answer = answer.replace("\\n", "\n").replace('\\"', '"').strip()
            if question or answer:
                return (question, answer)
    return None


def generate_qa(chunk: str, chunk_index: int = 1) -> QAResult:
    """Generate QA pair from a text chunk.

    Parameters
    ----------
    chunk : str
        Text chunk to generate QA from
    chunk_index : int
        Index of the chunk

    Returns
    -------
    QAResult
        Generated QA result
    """
    prompt = f"""По медицинскому тексту составь один клинический вопрос и развёрнутый ответ. Без markdown. Ответь только валидным JSON в формате:
{{"question": "текст вопроса", "answer": "текст ответа"}}

Текст:
{chunk}"""

    question = ""
    answer = ""
    generated_text = ""

    model_used = getattr(settings, "metrics_llm_model_name", "llm_qwen_2_5_coder_32b_instruct_q8")
    try:
        base_url = (getattr(settings, "valueai_base_url", None) or "").rstrip("/")
        token = get_token(
            base_url,
            getattr(settings, "valueai_username", "") or "",
            getattr(settings, "valueai_password", "") or "",
        )
        generated_text = predict_sync(
            base_url,
            token,
            model_name=model_used,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=getattr(settings, "max_tokens", 5000),
            temperature=getattr(settings, "temperature", 0.7),
            poll_interval=getattr(settings, "metrics_llm_poll_interval_seconds", 2.0),
            timeout=getattr(settings, "metrics_llm_timeout_seconds", 120.0),
        )

        
        parsed = False
        if generated_text.strip():
            text = generated_text.strip()
            
            if "```" in text:
                start = text.find("```json") + 7 if "```json" in text else text.find("```") + 3
                end = text.find("```", start)
                if end > start:
                    text = text[start:end].strip()
            if "{" in text and "}" in text:
                try:
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    obj = json.loads(text[start:end])
                    if isinstance(obj, dict):
                        q = obj.get("question") or obj.get("Вопрос")
                        a = obj.get("answer") or obj.get("Ответ")
                        if q and a:
                            question = q.strip() if isinstance(q, str) else str(q)
                            answer = a.strip() if isinstance(a, str) else str(a)
                            parsed = True
                except (json.JSONDecodeError, ValueError, TypeError):
                    
                    extracted = _extract_qa_from_json_like(text)
                    if extracted:
                        question, answer = extracted
                        parsed = bool(question and answer)

        
        if not parsed:
            if "Ответ:" in generated_text:
                parts = generated_text.split("Ответ:", 1)
                before_answer = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
                if "Вопрос:" in before_answer:
                    question = before_answer.split("Вопрос:", 1)[-1].strip()
                else:
                    question = before_answer
            else:
                question = ""
                answer = ""

        if not question or not answer:
            lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
            if len(lines) >= 2:
                question = lines[0]
                answer = lines[1]

    except Exception as e:
        logger.exception("Ошибка при вызове LLM (ValueAI)")
        generated_text = f"Ошибка: модель не ответила - {e!s}"
        question = "Ошибка"
        answer = "Ошибка"

    return QAResult(
        chunk_index=chunk_index,
        chunk=chunk,
        chunk_length_chars=len(chunk),
        chunk_length_words=len(chunk.split()),
        model_used=model_used,
        question=question,
        answer=answer,
        raw_model_output=generated_text,
    )


def _prompt_num_questions(max_questions: int, default: int) -> int:
    
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")

    if not sys.stdin.isatty():
        return default

    while True:
        raw = input(
            f"Сколько вопросов-ответов сгенерировать? (1..{max_questions}) "
            f"[по умолчанию: {default}]: "
        ).strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.info("Введите целое число.")
            continue
        if 1 <= value <= max_questions:
            return value
        logger.info("Введите число в диапазоне 1..%s.", max_questions)


def _build_valueai_client() -> ValueAIRagClient:
    
    if not settings.valueai_username or not settings.valueai_password:
        raise ValueError(
            "ValueAI credentials are not configured. "
            "Set VALUEAI_USERNAME and VALUEAI_PASSWORD in environment or .env."
        )
    config = ValueAIRagClientConfig(
        base_url=settings.valueai_base_url,
        username=settings.valueai_username,
        password=settings.valueai_password,
        rag_id=settings.valueai_rag_id,
        model_name=settings.valueai_model_name,
        instructions=settings.valueai_instructions,
        poll_interval_seconds=settings.valueai_poll_interval_seconds,
        timeout_seconds=settings.valueai_timeout_seconds,
    )
    return ValueAIRagClient(config)


def _run_valueai_evaluation(
    results: list[QAResult],
    pdf_path: Path,
    num_questions: int,
    output_file: Path,
    summary_file: Path | None,
) -> None:
    """Run ValueAI RAG evaluation on results and write summary."""
    client = _build_valueai_client()
    aggregate = {
        "count": len(results),
        "avg_faithfulness": 0.0,
        "avg_cosine_similarity": 0.0,
        "avg_factual_correctness": 0.0,
        "avg_llm_alignment_score": 0.0,
    }
    sum_faith = 0.0
    sum_cos = 0.0
    sum_factual = 0.0
    sum_alignment = 0.0
    n_evaluated = 0

    for r in results:
        
        if (not r.question or not r.answer or
                r.question.strip() == "Ошибка" or r.answer.strip() == "Ошибка" or
                "Ошибка при вызове" in (r.answer or "") or "модель не ответила" in (r.raw_model_output or "")):
            logger.warning("Пропуск ValueAI для chunk %s: генерация не удалась", r.chunk_index)
            r.valueai_answer = None
            r.evaluation_metrics = {"skipped": "generation failed"}
            continue
        try:
            valueai_answer = client.ask(r.question)
            metrics: dict = {}

            # RAGAS (etalon = context, ValueAI = response)
            ragas = evaluate_answer_pair_ragas_extended(
                question=r.question,
                response=valueai_answer,
                retrieved_contexts=[r.answer],
                reference_answer=None,
            )
            metrics["ragas_faithfulness"] = ragas.get("faithfulness")
            if ragas.get("error") is not None:
                metrics["ragas_error"] = ragas["error"]

            
            text_metrics = compare_two_answers(reference=r.answer, candidate=valueai_answer)
            metrics["cosine_similarity"] = text_metrics.get("cosine_similarity")
            metrics["factual_correctness"] = text_metrics.get("factual_correctness")
            if text_metrics.get("factual_error") is not None:
                metrics["factual_error"] = text_metrics["factual_error"]

            # LLM judge (1–10) RAG vs etalon
            llm_judge = evaluate_answer_pair_llm_alignment(
                question=r.question,
                etalon_answer=r.answer,
                rag_answer=valueai_answer,
                context=(r.chunk or ""),
            )
            metrics["llm_alignment_score"] = llm_judge.get("alignment_score")
            metrics["llm_alignment_comment"] = llm_judge.get("alignment_comment")
            if llm_judge.get("alignment_error") is not None:
                metrics["llm_alignment_error"] = llm_judge["alignment_error"]

            r.valueai_answer = valueai_answer
            r.evaluation_metrics = metrics

            v_faith = metrics.get("ragas_faithfulness")
            if v_faith is not None:
                sum_faith += float(v_faith)
            v_cos = metrics.get("cosine_similarity")
            if v_cos is not None:
                sum_cos += float(v_cos)
            v_factual = metrics.get("factual_correctness")
            if v_factual is not None:
                sum_factual += float(v_factual)
            v_align = metrics.get("llm_alignment_score")
            if v_align is not None:
                sum_alignment += float(v_align)
            n_evaluated += 1
        except Exception as e:  # noqa: PERF203
            logger.exception("ValueAI error for question: %s", r.question[:50])
            r.valueai_answer = None
            r.evaluation_metrics = {"error": str(e)}

    if results and n_evaluated > 0:
        n = n_evaluated
        aggregate["avg_faithfulness"] = sum_faith / n
        aggregate["avg_cosine_similarity"] = sum_cos / n
        aggregate["avg_factual_correctness"] = sum_factual / n
        aggregate["avg_llm_alignment_score"] = sum_alignment / n
    aggregate["evaluated_count"] = n_evaluated

    if summary_file is None:
        summary_file = output_file.with_name(f"{output_file.stem}_valueai_eval.json")
    summary_payload = {
        "pdf_path": str(pdf_path),
        "num_questions": num_questions,
        "aggregate_metrics": aggregate,
    }
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    logger.info("ValueAI evaluation summary saved to: %s", summary_file)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)


def generate_qa_from_pdf(
    pdf_path: Path,
    output_file: Path | None = None,
    *,
    num_questions: int | None = None,
    evaluate_with_valueai: bool = False,
    summary_file: Path | None = None,
) -> list[QAResult]:
    if not pdf_path.exists():
        msg = f"PDF файл не найден: {pdf_path}"
        raise FileNotFoundError(msg)

    logger.info(
        "Используется модель для генерации Q&A: %s (config: metrics_llm_model_name)",
        getattr(settings, "metrics_llm_model_name", "llm_qwen_2_5_coder_32b_instruct_q8"),
    )
    logger.info(f"Чтение PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    logger.info(f"Загружено {len(text)} символов")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Создано {len(chunks)} chunk-ов")

    text_chunks = [c for c in chunks if len(c.strip().split()) > settings.min_chunk_words]

    
    max_questions = len(text_chunks) // CHUNKS_PER_QA
    if max_questions <= 0:
        raise ValueError(
            f"Не найдено достаточно chunk-ов. Нужно минимум {CHUNKS_PER_QA} chunk-ов для одного вопроса, "
            f"сейчас: {len(text_chunks)}."
        )

    default_questions = min(settings.num_chunks_to_select, max_questions)
    if num_questions is None:
        num_questions = _prompt_num_questions(
            max_questions=max_questions, default=default_questions
        )
    if not (1 <= num_questions <= max_questions):
        msg = f"num_questions must be in range 1..{max_questions}. Got: {num_questions}"
        raise ValueError(msg)

    
    n_chunks_needed = num_questions * CHUNKS_PER_QA
    selected_chunks = random.sample(text_chunks, n_chunks_needed)

    results = []
    for idx in range(num_questions):
        group = selected_chunks[idx * CHUNKS_PER_QA : (idx + 1) * CHUNKS_PER_QA]
        combined_chunk = "\n\n".join(group)
        logger.info(
            f"Обработка группы {idx + 1}/{num_questions} (4 chunk-а, "
            f"{len(combined_chunk)} символов, {len(combined_chunk.split())} слов)..."
        )
        result = generate_qa(combined_chunk, idx + 1)
        results.append(result)

    if output_file is None:
        output_file = Path("qa_result.json")

    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Готово! Результаты сохранены в: {output_file}")

    for r in results:
        logger.info("=" * 50)
        logger.info(f"Chunk {r.chunk_index}:")
        logger.info(f"Вопрос: {r.question}")
        logger.info(f"Ответ: {r.answer}")

    if evaluate_with_valueai:
        _run_valueai_evaluation(
            results=results,
            pdf_path=pdf_path,
            num_questions=num_questions,
            output_file=output_file,
            summary_file=summary_file,
        )

    return results