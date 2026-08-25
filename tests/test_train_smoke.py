"""Smoke tests for fine_tuning/train.py (no full GPU training)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fine_tuning.train import (
    build_arg_parser,
    build_metrics_stub,
    build_text_dataset,
    gpu_available,
    load_jsonl,
    records_to_texts,
    run_training,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
VAL_JSONL = PROJECT_ROOT / "data" / "processed" / "val.jsonl"


def test_train_module_imports():
    import fine_tuning.train as train_module

    assert hasattr(train_module, "main")
    assert hasattr(train_module, "run_training")


def test_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.train_path == Path("data/processed/train.jsonl")
    assert args.val_path == Path("data/processed/val.jsonl")
    assert args.max_steps is None


def test_arg_parser_max_steps():
    parser = build_arg_parser()
    args = parser.parse_args(["--max-steps", "2"])
    assert args.max_steps == 2


def test_load_jsonl_train_and_val():
    train_records = load_jsonl(TRAIN_JSONL)
    val_records = load_jsonl(VAL_JSONL)
    assert len(train_records) > 0
    assert len(val_records) > 0
    assert "instruction" in train_records[0] or "text" in train_records[0]


def test_records_to_texts_uses_instruction_format():
    records = load_jsonl(TRAIN_JSONL)
    texts = records_to_texts(records[:3])
    assert all(text.startswith("### Instrução:") for text in texts)
    assert all("### Resposta:" in text for text in texts)


def test_build_text_dataset():
    records = load_jsonl(TRAIN_JSONL)[:2]
    dataset = build_text_dataset(records)
    assert len(dataset) == 2
    assert "text" in dataset.column_names


def test_build_metrics_stub_structure():
    stub = build_metrics_stub("no_gpu", train_count=10, val_count=2)
    assert stub["skipped_training"] is True
    assert stub["reason"] == "no_gpu"
    assert stub["train_samples"] == 10
    assert stub["val_samples"] == 2
    assert stub["epochs"] == []
    assert stub["hyperparameters"]["lora_r"] == 16


def test_run_training_skips_without_gpu(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setattr("fine_tuning.train.gpu_available", lambda: False)

    metrics_path = tmp_path / "training_metrics.json"
    args = build_arg_parser().parse_args(
        [
            "--train-path",
            str(TRAIN_JSONL),
            "--val-path",
            str(VAL_JSONL),
            "--metrics-path",
            str(metrics_path),
            "--output-dir",
            str(tmp_path / "lora_adapter"),
        ]
    )

    metrics = run_training(args)
    assert metrics["skipped_training"] is True
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["epochs"] == []
    assert saved["hyperparameters"]["learning_rate"] == 2e-4


@pytest.mark.skipif(gpu_available(), reason="CPU-only path covered when CUDA unavailable")
def test_gpu_available_false_on_cpu_only():
    assert gpu_available() is False


def test_train_help():
    result = subprocess.run(
        [sys.executable, "fine_tuning/train.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max-steps" in result.stdout
    assert "LoRA" in result.stdout
