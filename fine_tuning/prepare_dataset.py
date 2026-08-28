"""Dataset preparation pipeline: PubMedQA + protocol pairs → train/val/test JSONL."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fine_tuning.formatting import format_instruction_sample

DEFAULT_PROTOCOLS_DIR = Path("data/synthetic/protocols")
DEFAULT_OUT_DIR = Path("data/processed")


def anonymize_text(text: str) -> str:
    text = re.sub(r"\d{3}\.\d{3}\.\d{3}-\d{2}", "[CPF]", text)
    text = re.sub(r"\d{2}/\d{2}/\d{4}", "[DATA]", text)
    text = re.sub(
        r"(?:Sr\.|Sra\.|Dr\.|Dra\.)\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*",
        "[NOME]",
        text,
    )
    text = re.sub(
        r"Paciente\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)+",
        "Paciente [NOME]",
        text,
    )
    return text


def load_pubmedqa(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def generate_protocol_pairs(protocols_dir: str | Path) -> list[dict]:
    records: list[dict] = []
    protocols_path = Path(protocols_dir)

    for md_path in sorted(protocols_path.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        protocol_name = md_path.stem
        title_line = content.splitlines()[0].lstrip("# ").strip() if content else protocol_name

        sections = re.split(r"^##\s+", content, flags=re.MULTILINE)
        for section in sections[1:]:
            header_end = section.find("\n")
            if header_end == -1:
                continue
            header = section[:header_end].strip()
            body = section[header_end:].strip()
            if not body:
                continue

            section_title = header.split("—", 1)[-1].strip() if "—" in header else header
            instruction = f"Segundo {protocol_name}, o que orienta a seção '{section_title}'?"
            input_text = f"Protocolo: {title_line}"
            output = (
                f"Segundo {protocol_name}, {header}: {body} "
                "Requer validação do médico responsável."
            )
            records.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }
            )

    return records


def _anonymize_record(record: dict) -> dict:
    return {
        key: anonymize_text(value) if isinstance(value, str) else value
        for key, value in record.items()
    }


def split_and_write(
    records: list[dict],
    out_dir: str | Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    shuffled = records.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_records in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in split_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_datasets(
    pubmedqa_path: str | Path,
    protocols_dir: str | Path = DEFAULT_PROTOCOLS_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> None:
    records: list[dict] = []
    records.extend(load_pubmedqa(pubmedqa_path))
    records.extend(generate_protocol_pairs(protocols_dir))

    processed: list[dict] = []
    for record in records:
        anonymized = _anonymize_record(record)
        anonymized["text"] = format_instruction_sample(
            anonymized["instruction"],
            anonymized.get("input", ""),
            anonymized["output"],
        )
        processed.append(anonymized)

    split_and_write(processed, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning datasets.")
    parser.add_argument(
        "--pubmedqa",
        required=True,
        help="Path to PubMedQA-style JSONL sample.",
    )
    parser.add_argument(
        "--protocols-dir",
        default=str(DEFAULT_PROTOCOLS_DIR),
        help="Directory containing synthetic protocol markdown files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for train/val/test JSONL files.",
    )
    args = parser.parse_args()
    build_datasets(args.pubmedqa, args.protocols_dir, args.out_dir)
    print(f"Wrote datasets to {args.out_dir}")


if __name__ == "__main__":
    main()
