"""QA generation functionality."""

import json
import logging
import random
import sys
from pathlib import Path

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ..config import settings
from ..evaluation.metrics import evaluate_answer_pair
from ..valueai.client import ValueAIRagClient, ValueAIRagClientConfig
from .models import QAResult

logger = logging.getLogger(__name__)


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
    prompt = f"""На основе медицинского текста создай 1 клинический вопрос и краткий ответ.

ТЕКСТ:
{chunk}

ФОРМАТ ОТВЕТА (строго):
Вопрос: ...
Ответ: ..."""

    question = ""
    answer = ""
    generated_text = ""

    try:
        url = f"{settings.ollama_base_url}/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": settings.model_name,
            "prompt": prompt,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }

        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0].get("text", "")

        for line in generated_text.split("\n"):
            line_clean = line.strip()
            if line_clean.startswith("Вопрос:"):
                question = line_clean.replace("Вопрос:", "").strip()
            elif line_clean.startswith("Ответ:"):
                answer = line_clean.replace("Ответ:", "").strip()

        if not question or not answer:
            lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
            if len(lines) >= 2:
                question = lines[0]
                answer = lines[1]

    except Exception as e:
        logger.error(f"Ошибка при вызове модели Ollama: {e}")
        generated_text = f"Ошибка: модель не ответила - {e!s}"
        question = "Ошибка"
        answer = "Ошибка"

    return QAResult(
        chunk_index=chunk_index,
        chunk=chunk,
        chunk_length_chars=len(chunk),
        chunk_length_words=len(chunk.split()),
        model_used=settings.model_name,
        question=question,
        answer=answer,
        raw_model_output=generated_text,
    )


def _prompt_num_questions(max_questions: int, default: int) -> int:
    """Prompt user for number of QA items to generate."""
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
            print("Введите целое число.")
            continue
        if 1 <= value <= max_questions:
            return value
        print(f"Введите число в диапазоне 1..{max_questions}.")


def _build_valueai_client() -> ValueAIRagClient:
    """Build ValueAI client with credentials."""
    print(f" DEBUG - Building ValueAI client")
    print(f" DEBUG - Username: {settings.valueai_username}")
    print(f" DEBUG - Password exists: {bool(settings.valueai_password)}")
    print(f" DEBUG - Base URL: {settings.valueai_base_url}")
    print(f" DEBUG - RAG ID: {settings.valueai_rag_id}")

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


def generate_qa_from_pdf(
    pdf_path: Path,
    output_file: Path | None = None,
    *,
    num_questions: int | None = None,
    evaluate_with_valueai: bool = False,
    summary_file: Path | None = None,
) -> list[QAResult]:
    """Generate QA pairs from PDF file.

    Parameters
    ----------
    pdf_path : Path
        Path to PDF file
    output_file : Path | None
        Output JSON file path. If None, creates qa_result.json in current directory
    num_questions : int | None
        How many QA items to generate. If None, prompts after chunking (interactive) and
        falls back to default in non-interactive mode.
    evaluate_with_valueai : bool
        If True, sends generated questions to ValueAI RAG and compares answers.
    summary_file : Path | None
        Where to write evaluation summary JSON (only when evaluate_with_valueai is True).

    Returns
    -------
    list[QAResult]
        List of generated QA results
    """
    # DEBUG
    print(f"🔍 DEBUG - evaluate_with_valueai in generator: {evaluate_with_valueai}")
    print(f"🔍 DEBUG - Type: {type(evaluate_with_valueai)}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

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

    max_questions = len(text_chunks)
    if max_questions <= 0:
        raise ValueError("Не найдено достаточно длинных текстовых chunk-ов для генерации вопросов.")

    default_questions = min(settings.num_chunks_to_select, max_questions)
    if num_questions is None:
        num_questions = _prompt_num_questions(
            max_questions=max_questions, default=default_questions
        )
    if not (1 <= num_questions <= max_questions):
        raise ValueError(f"num_questions must be in range 1..{max_questions}. Got: {num_questions}")

    selected_chunks = random.sample(text_chunks, num_questions)

    results = []
    for idx, chunk in enumerate(selected_chunks, start=1):
        logger.info(f"Обработка chunk {idx} ({len(chunk)} символов, {len(chunk.split())} слов)...")
        result = generate_qa(chunk, idx)
        results.append(result)

    if output_file is None:
        output_file = Path("qa_result.json")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
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
        print(" DEBUG - evaluate_with_valueai is True, building client...")
        client = _build_valueai_client()

        aggregate = {
            "count": len(results),
            "exact_match_rate": 0.0,
            "avg_token_f1": 0.0,
            "avg_rouge_l_f1": 0.0,
            "avg_primary_score": 0.0,
        }

        em_count = 0
        sum_tf1 = 0.0
        sum_rl = 0.0
        sum_ps = 0.0

        for r in results:
            try:
                print(f" DEBUG - Asking ValueAI question: {r.question[:50]}...")
                valueai_answer = client.ask(r.question)
                print(f" DEBUG - Got answer: {valueai_answer[:50]}...")
                metrics = evaluate_answer_pair(r.answer, valueai_answer)
                r.valueai_answer = valueai_answer
                r.evaluation_metrics = metrics

                if metrics.get("exact_match"):
                    em_count += 1
                sum_tf1 += float(metrics["token_f1"]["f1"])
                sum_rl += float(metrics["rouge_l"]["f1"])
                sum_ps += float(metrics["primary_score"])
            except Exception as e:
                print(f" DEBUG - Error in ValueAI: {e}")
                r.valueai_answer = None
                r.evaluation_metrics = {"error": str(e)}

        if results:
            aggregate["exact_match_rate"] = em_count / len(results)
            aggregate["avg_token_f1"] = sum_tf1 / len(results)
            aggregate["avg_rouge_l_f1"] = sum_rl / len(results)
            aggregate["avg_primary_score"] = sum_ps / len(results)

        if summary_file is None:
            summary_file = output_file.with_name(f"{output_file.stem}_valueai_eval.json")

        summary_payload = {
            "pdf_path": str(pdf_path),
            "num_questions": num_questions,
            "aggregate_metrics": aggregate,
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"ValueAI evaluation summary saved to: {summary_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)
    else:
        print("🔍 DEBUG - evaluate_with_valueai is False, skipping ValueAI")

    return results
