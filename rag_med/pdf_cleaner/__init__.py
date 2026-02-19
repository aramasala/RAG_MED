"""PDF cleaner module for removing unnecessary sections from PDF files."""

from .cleaner import clean_pdf, find_text_in_pdf

__all__ = ["clean_pdf", "find_text_in_pdf"]
