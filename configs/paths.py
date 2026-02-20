"""Project paths (repo root). Don't use if you use the package externally."""

from pathlib import Path

# Project root (parent of configs/)
PROJECT_DPATH = Path(__file__).resolve().parents[1]
