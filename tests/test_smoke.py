from app.cli import main


def test_cli_interactive_smoke(monkeypatch, capsys):
    answers = iter(["cells are", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["--corpus", "app/data", "--seed", "2", "--max-words", "5"]) == 0
    output = capsys.readouterr().out
    assert "Educational Content Expander" in output
    assert "cells are" in output.lower()
    assert "Goodbye" in output
