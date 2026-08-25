import os
from pathlib import Path

import pytest

os.environ["USE_MOCK_LLM"] = "1"

from assistant.chains import build_prompt, run_assistant  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_LLM", "1")


@pytest.fixture
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "interactions.jsonl"
    monkeypatch.setenv("LOG_PATH", str(path))
    return path


def test_run_assistant_returns_required_keys(log_file: Path) -> None:
    result = run_assistant("PAC-001", "Protocolo para febre e tosse?")

    assert set(result.keys()) == {
        "response",
        "sources",
        "requires_human_validation",
        "flags",
        "valid",
    }
    assert result["valid"] is True
    assert result["requires_human_validation"] is True
    assert result["sources"]
    assert "validação do médico responsável" in result["response"].lower()
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8").strip()


def test_run_assistant_includes_chart_context_in_prompt() -> None:
    chart = "Prontuário — PAC-001\nPerfil: Adulto 45a"
    prompt = build_prompt(
        "PAC-001",
        "febre",
        chart,
        [{"content": "Protocolo de febre", "source": "protocolo_febre_v2.md", "score": 0.9}],
    )

    assert "PAC-001" in prompt
    assert chart in prompt
    assert "protocolo_febre_v2.md" in prompt
    assert "febre" in prompt


def test_run_assistant_logs_interaction(log_file: Path) -> None:
    run_assistant("PAC-002", "Qual conduta para dor abdominal?")

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    import json

    entry = json.loads(lines[0])
    assert entry["patient_id"] == "PAC-002"
    assert entry["query"] == "Qual conduta para dor abdominal?"
    assert "timestamp" in entry
    assert entry["sources"]
