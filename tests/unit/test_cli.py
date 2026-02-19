"""Tests for CLI interface."""

from typer.testing import CliRunner
from rag_med.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "RAG_MED" in result.stdout
