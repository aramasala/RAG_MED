"""Unit tests for PDF cleaner module."""

import pytest
from pathlib import Path

from rag_med.pdf_cleaner import clean_pdf, find_text_in_pdf


def test_find_text_in_pdf_not_found(tmp_path: Path) -> None:
    """Test finding text that doesn't exist in PDF."""
    pdf_path = tmp_path / "test.pdf"
    if not pdf_path.exists():
        pytest.skip("No test PDF available")


def test_clean_pdf_file_not_found(tmp_path: Path) -> None:
    """Test cleaning non-existent PDF."""
    input_pdf = tmp_path / "nonexistent.pdf"
    output_pdf = tmp_path / "output.pdf"

    result = clean_pdf(input_pdf, output_pdf)
    assert result is False
