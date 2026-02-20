"""PDF cleaning functionality."""

import logging
import re
from pathlib import Path

import fitz

from configs.settings import settings

logger = logging.getLogger(__name__)


def find_text_in_pdf(pdf_path: Path, search_text: str) -> int | None:
    """Find page number containing the search text.

    Parameters
    ----------
    pdf_path : Path
        Path to PDF file
    search_text : str
        Text to search for

    Returns
    -------
    int | None
        Page number (1-indexed) or None if not found
    """
    pdf_doc = fitz.open(pdf_path)
    total_pages = pdf_doc.page_count

    for page_num in range(total_pages):
        page = pdf_doc[page_num]
        text = page.get_text()

        if search_text in text:
            lines = text.split("\n")
            for line in lines:
                if search_text in line:
                    numbers = re.findall(r"\b(\d{1,3})\b", line)
                    if numbers:
                        for num in reversed(numbers):
                            page_int = int(num)
                            if 1 <= page_int <= total_pages:
                                pdf_doc.close()
                                return page_int
            break

    pdf_doc.close()
    return None


def clean_pdf(input_pdf: Path, output_pdf: Path | None = None) -> bool:
    """Remove unnecessary sections from PDF file.

    Parameters
    ----------
    input_pdf : Path
        Input PDF file path
    output_pdf : Path | None
        Output PDF file path. If None, creates {input_pdf.stem}_cleaned.pdf

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if not input_pdf.exists():
        logger.error(f"Файл не найден: {input_pdf}")
        return False

    if output_pdf is None:
        output_pdf = input_pdf.with_name(f"{input_pdf.stem}_cleaned.pdf")

    logger.info(f"Обрабатываю: {input_pdf}")

    try:
        doc = fitz.open(input_pdf)
        total_pages = doc.page_count
        logger.debug(f"PDF открыт, всего страниц: {total_pages}")
    except Exception:
        logger.exception("Ошибка открытия PDF")
        return False

    start_page = find_text_in_pdf(input_pdf, settings.start_section_text)
    end_page = find_text_in_pdf(input_pdf, settings.end_section_text)

    if start_page:
        logger.debug(f"Найдена стартовая страница ({settings.start_section_text}): {start_page}")
    else:
        logger.warning(f"Текст '{settings.start_section_text}' не найден")

    if end_page:
        logger.debug(f"Найдена конечная страница ({settings.end_section_text}): {end_page}")
    else:
        logger.warning(f"Текст '{settings.end_section_text}' не найден")

    if not start_page or not end_page:
        logger.warning("Тексты не найдены — файл пропущен")
        doc.close()
        return False

    pages_to_remove = set(range(min(start_page, end_page) - 1, max(start_page, end_page) - 1))
    logger.debug(f"Страницы для удаления: {sorted(pages_to_remove)}")

    for page_num in sorted(pages_to_remove, reverse=True):
        logger.debug(f"Удаляется страница: {page_num + 1}")
        doc.delete_page(page_num)

    try:
        doc.save(output_pdf, garbage=4, deflate=True, clean=True)
    except Exception:
        logger.exception("Ошибка сохранения")
        doc.close()
        return False
    else:
        doc.close()
        logger.info(f"Сохранено: {output_pdf}")
        return True
