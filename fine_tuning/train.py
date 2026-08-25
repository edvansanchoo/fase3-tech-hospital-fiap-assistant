"""LoRA fine-tuning for the Hospital FIAP clinical assistant.

Training is intended for Google Colab or a CUDA-enabled machine. On CPU-only
environments the script validates dataset loading and writes a metrics stub
without running the Trainer. Use ``--max-steps`` for a short GPU smoke test.

Example::

    python fine_tuning/train.py
    python fine_tuning/train.py --max-steps 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fine_tuning.formatting import format_sample_text

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

DEFAULT_TRAIN_PATH = Path("data/processed/train.jsonl")
DEFAULT_VAL_PATH = Path("data/processed/val.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/lora_adapter")
DEFAULT_METRICS_PATH = Path("artifacts/training_metrics.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for hospital-fiap-assistant (4-bit, PEFT)."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for LoRA adapter weights",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path for training_metrics.json",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limit training steps for GPU smoke tests (overrides epoch count)",
    )
    return parser


def gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_texts(records: list[dict]) -> list[str]:
    return [format_sample_text(record) for record in records]


def build_text_dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_dict({"text": records_to_texts(records)})


def hyperparameters_dict() -> dict[str, Any]:
    return {
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "num_epochs": NUM_EPOCHS,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "quantization": "4-bit",
    }


def save_metrics(metrics: dict[str, Any], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_metrics_stub(reason: str, train_count: int, val_count: int) -> dict[str, Any]:
    return {
        "hyperparameters": hyperparameters_dict(),
        "train_samples": train_count,
        "val_samples": val_count,
        "epochs": [],
        "skipped_training": True,
        "reason": reason,
    }


def extract_epoch_metrics(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group Trainer log history into per-epoch train/eval loss entries."""
    by_epoch: dict[int, dict[str, Any]] = {}

    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        epoch_num = int(epoch)
        bucket = by_epoch.setdefault(epoch_num, {"epoch": epoch_num})
        if "loss" in entry:
            bucket["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            bucket["eval_loss"] = entry["eval_loss"]

    return [by_epoch[key] for key in sorted(by_epoch)]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    train_records = load_jsonl(args.train_path)
    val_records = load_jsonl(args.val_path)

    if not gpu_available():
        metrics = build_metrics_stub("no_gpu", len(train_records), len(val_records))
        save_metrics(metrics, args.metrics_path)
        print(
            "No CUDA GPU detected — skipping training. "
            f"Wrote metrics stub to {args.metrics_path}"
        )
        return metrics

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from assistant.llm_loader import get_base_model_id, load_llm_for_training

    model, tokenizer = load_llm_for_training()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    train_dataset = build_text_dataset(train_records)
    val_dataset = build_text_dataset(val_records)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    train_tokenized = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
    }
    if args.max_steps is not None:
        training_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {
        "hyperparameters": hyperparameters_dict(),
        "base_model_id": get_base_model_id(),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "epochs": extract_epoch_metrics(trainer.state.log_history),
        "skipped_training": False,
        "output_dir": str(args.output_dir),
    }
    if args.max_steps is not None:
        metrics["max_steps"] = args.max_steps

    save_metrics(metrics, args.metrics_path)
    print(f"Training complete. Adapter saved to {args.output_dir}")
    print(f"Metrics saved to {args.metrics_path}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
