import json
import os
from pathlib import Path

import pytest

os.environ["USE_MOCK_LLM"] = "1"

from langgraph_flows.clinical_workflow import build_graph, run_workflow  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_LLM", "1")


@pytest.fixture
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "interactions.jsonl"
    monkeypatch.setenv("LOG_PATH", str(path))
    return path


def test_build_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_pac002_has_pending_exam_path(log_file: Path) -> None:
    result = run_workflow("PAC-002", "Qual conduta para dor abdominal?")

    assert "agente_alerta" in result["graph_path"]
    assert "agente_prontuario" not in result["graph_path"]
    assert result["alerts"]
    assert result["pending_exams"]
    assert any(exam["status"] == "pendente" for exam in result["pending_exams"])
    assert result["suggestion"]
    assert result["sources"]
    assert result["requires_human_validation"] is True
    assert log_file.exists()

    entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["patient_id"] == "PAC-002"
    assert "agente_alerta" in entry["graph_path"]


def test_pac001_skips_alert_when_no_pending(log_file: Path) -> None:
    result = run_workflow("PAC-001", "Protocolo para febre e tosse?")

    assert "agente_prontuario" in result["graph_path"]
    assert "agente_alerta" not in result["graph_path"]
    assert not result["pending_exams"]
    assert not result["alerts"]
    assert result["suggestion"]
    assert log_file.exists()
