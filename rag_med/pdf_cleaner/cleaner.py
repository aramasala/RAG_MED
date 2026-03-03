"""PDF cleaning functionality."""

import logging
import re
from pathlib import Path

import fitz

from configs.settings import settings

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+", re.UNICODE)


def _norm(t: str) -> str:
    return _WS.sub(" ", t).strip()


def _norm_search(t: str) -> str:
    """Нормализация для поиска"""
    return _norm(t).replace(" ", "").lower()


def find_text_in_pdf(
    pdf_path: Path,
    search_text: str,
    use_last: bool = False,
) -> int | None:
    """Найти номер страницы (1-based), где встречается search_text."""
    pdf_doc = fitz.open(pdf_path)
    total_pages = pdf_doc.page_count
    needle = _norm_search(search_text)
    found = None

    for page_num in range(total_pages):
        page = pdf_doc[page_num]
        text_compare = _norm_search(page.get_text())
        if needle in text_compare:
            found = page_num + 1
            if not use_last:
                pdf_doc.close()
                return found
    pdf_doc.close()
    return found


def clean_pdf(input_pdf: Path, output_pdf: Path | None = None) -> bool:
    
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

    start_page = find_text_in_pdf(input_pdf, settings.start_section_text, use_last=True)
    end_page = find_text_in_pdf(input_pdf, settings.end_section_text, use_last=True)
    if not end_page and settings.end_section_text != "Приложение А2":
        end_page = find_text_in_pdf(input_pdf, "Приложение А2", use_last=True)
        if end_page:
            logger.debug("Найдена конечная страница по короткой фразе «Приложение А2»: %s", end_page)
    if not end_page:
        end_page = find_text_in_pdf(input_pdf, "Приложение A2", use_last=True)
        if end_page:
            logger.debug("Найдена конечная страница по фразе «Приложение A2» (латинская A): %s", end_page)

    if start_page:
        logger.debug(f"Найдена стартовая страница ({settings.start_section_text}): {start_page}")
    else:
        logger.warning(f"Текст '{settings.start_section_text}' не найден")

    if end_page:
        logger.debug(f"Найдена конечная страница ({settings.end_section_text}): {end_page}")
    else:
        logger.warning(f"Текст '{settings.end_section_text}' не найден")

    if not start_page or not end_page:
        logger.warning("Тексты не найдены — файл пропущен, в выход не записывается")
        doc.close()
        return False

    if start_page >= end_page:
        logger.warning(
            "Страница «Список литературы» (%s) >= страницы «Приложение А2» (%s) — пропуск, в выход не записывается",
            start_page,
            end_page,
        )
        doc.close()
        return False
    pages_to_remove = set(range(start_page - 1, end_page - 1))
    logger.debug(f"Страницы для удаления: {sorted(pages_to_remove)}")

    for page_num in sorted(pages_to_remove, reverse=True):
        logger.debug(f"Удаляется страница: {page_num + 1}")
        doc.delete_page(page_num)

    # Удалить все изображения 
    for page_num in range(doc.page_count):
        page = doc[page_num]
        img_list = page.get_images()
        for img in img_list:
            xref = img[0]
            try:
                page.delete_image(xref)
            except Exception as e:
                logger.debug("Не удалось удалить изображение xref=%s: %s", xref, e)
        if img_list:
            logger.debug(f"Со страницы {page_num + 1} удалено изображений: {len(img_list)}")

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
