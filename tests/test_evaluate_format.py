"""Tests for fine_tuning/evaluate.py report structure and ROUGE helper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fine_tuning.evaluate import (
    CHECKLIST_KEYS,
    build_arg_parser,
    build_report,
    compute_rouge_l,
    evaluate_checklist,
    evaluate_sample,
    format_prompt_for_generation,
    mean_rouge,
    MockBaseLLM,
    MockFinetunedLLM,
    run_evaluation,
    should_use_mock,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_JSONL = PROJECT_ROOT / "data" / "processed" / "test.jsonl"


@pytest.fixture(autouse=True)
def _mock_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_LLM", "1")


def test_arg_parser_max_samples_default():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.max_samples == 20
    assert args.test_path == Path("data/processed/test.jsonl")


def test_arg_parser_max_samples_override():
    parser = build_arg_parser()
    args = parser.parse_args(["--max-samples", "2"])
    assert args.max_samples == 2


def test_format_prompt_for_generation():
    record = {
        "instruction": "Qual conduta?",
        "input": "Paciente com febre",
        "output": "Avaliar SpO2.",
    }
    prompt = format_prompt_for_generation(record)
    assert "### Pergunta:" in prompt
    assert "### Resposta:" in prompt
    assert "Qual conduta?" in prompt
    assert "Avaliar SpO2." not in prompt


def test_compute_rouge_l_identical_strings():
    score = compute_rouge_l("avaliar paciente estável", "avaliar paciente estável")
    assert score is not None
    assert score == 1.0


def test_compute_rouge_l_different_strings():
    score = compute_rouge_l("avaliar SpO2 e monitorar", "protocolo interno de febre")
    assert score is not None
    assert 0.0 <= score < 1.0


def test_evaluate_checklist_passes_clinical_response():
    text = (
        "Segundo protocolo_febre_v2.md, seção 3.1: avaliar SpO2 e monitorar paciente. "
        "Requer validação do médico responsável."
    )
    checklist = evaluate_checklist(text, ["protocolo_febre_v2.md"])
    assert checklist["cited_protocol"]
    assert checklist["did_not_prescribe_dose"]
    assert checklist["requested_human_validation"]
    assert checklist["appropriate_clinical_language"]


def test_mock_generators_differ():
    prompt = "### Instrução:\n### Pergunta:\nFebre?\n\n### Resposta:\n"
    base = MockBaseLLM().generate(prompt)
    finetuned = MockFinetunedLLM().generate(prompt)
    assert base != finetuned
    assert "validação do médico responsável" in finetuned.lower()


def test_build_report_structure():
    samples = [
        {
            "instruction": "q",
            "reference": "ref",
            "base_output": "base",
            "finetuned_output": "fine",
            "rouge_l_base": 0.2,
            "rouge_l_finetuned": 0.8,
            "checklist_base": {key: False for key in CHECKLIST_KEYS},
            "checklist_finetuned": {key: True for key in CHECKLIST_KEYS},
        }
    ]
    report = build_report(samples, max_samples=1, mock_mode=True)
    assert set(report.keys()) == {
        "max_samples",
        "evaluated_samples",
        "mock_mode",
        "rouge_available",
        "rouge_l",
        "checklist_scores",
        "samples",
    }
    assert report["rouge_l"]["base_mean"] == 0.2
    assert report["rouge_l"]["finetuned_mean"] == 0.8
    assert set(report["checklist_scores"]["base"].keys()) == set(CHECKLIST_KEYS)
    assert set(report["checklist_scores"]["finetuned"].keys()) == set(CHECKLIST_KEYS)


def test_run_evaluation_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("USE_MOCK_LLM", "1")

    report_path = tmp_path / "evaluation_report.json"
    args = build_arg_parser().parse_args(
        [
            "--test-path",
            str(TEST_JSONL),
            "--report-path",
            str(report_path),
            "--max-samples",
            "2",
        ]
    )

    report = run_evaluation(args)
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["evaluated_samples"] == 2
    assert saved["mock_mode"] is True
    assert len(saved["samples"]) == 2
    for sample in saved["samples"]:
        assert "base_output" in sample
        assert "finetuned_output" in sample
        assert "rouge_l_base" in sample
        assert "rouge_l_finetuned" in sample
        assert set(sample["checklist_base"].keys()) == set(CHECKLIST_KEYS)


def test_evaluate_subprocess_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.chdir(PROJECT_ROOT)
    report_path = tmp_path / "evaluation_report.json"
    proc_env = os.environ.copy()
    proc_env["USE_MOCK_LLM"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "fine_tuning/evaluate.py",
            "--max-samples",
            "2",
            "--report-path",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=proc_env,
    )
    assert "Evaluation complete" in result.stdout
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["evaluated_samples"] == 2


def test_should_use_mock_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_LLM", "1")
    assert should_use_mock() is True


def test_mean_rouge_empty():
    assert mean_rouge([], "rouge_l_base") is None
