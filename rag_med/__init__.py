"""RAG_MED - Medical RAG system for PDF processing and QA generation."""

from importlib import metadata as importlib_metadata

from .pdf_cleaner import clean_pdf
from .qa_generator import generate_qa_from_pdf

__all__ = [
    "clean_pdf",
    "generate_qa_from_pdf",
    "__version__",
]


def get_version() -> str:
    """Get package version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__version__: str = get_version()
