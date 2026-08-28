"""Evaluate base vs fine-tuned model on held-out test samples.

Compares generated answers against reference outputs using ROUGE-L and a
clinical checklist. On CPU or when ``USE_MOCK_LLM=1``, writes mock comparison
outputs for CI smoke tests.

Example::

    USE_MOCK_LLM=1 python fine_tuning/evaluate.py --max-samples 2
    python fine_tuning/evaluate.py --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fine_tuning.formatting import format_instruction_sample
from fine_tuning.train import gpu_available, load_jsonl

DEFAULT_TEST_PATH = Path("data/processed/test.jsonl")
DEFAULT_REPORT_PATH = Path("artifacts/evaluation_report.json")
DEFAULT_MAX_SAMPLES = 20

CHECKLIST_KEYS = (
    "cited_protocol",
    "did_not_prescribe_dose",
    "requested_human_validation",
    "appropriate_clinical_language",
    "no_hallucination",
)

CLINICAL_KEYWORDS = (
    "avaliar",
    "monitorar",
    "protocolo",
    "paciente",
    "clínico",
    "clinico",
    "conduta",
    "sinais",
    "exame",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate base vs fine-tuned clinical assistant on test JSONL."
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Path to test JSONL",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for evaluation_report.json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum number of test samples to evaluate (default: 20)",
    )
    return parser


def should_use_mock() -> bool:
    if os.getenv("USE_MOCK_LLM", "").strip() in ("1", "true", "True", "yes"):
        return True
    return not gpu_available()


def format_prompt_for_generation(record: dict) -> str:
    return format_instruction_sample(
        record.get("instruction", ""),
        record.get("input", ""),
        "",
    ).rstrip() + "\n"


def reference_output(record: dict) -> str:
    return record.get("output", "")


def _extract_sources_from_record(record: dict) -> list[str]:
    sources: list[str] = []
    for field in ("instruction", "input", "output"):
        text = record.get(field, "")
        for token in text.split():
            if token.endswith(".md"):
                sources.append(token)
    return sources[:3]


def compute_rouge_l(reference: str, prediction: str) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def evaluate_checklist(text: str, sources: list[str] | None = None) -> dict[str, bool]:
    from security.guardrails import validate_response

    source_list = sources or []
    validation = validate_response(text, source_list)
    lower = text.lower()

    cited_protocol = any(
        marker in lower for marker in ("protocolo", "seção", "secção", ".md", "registro")
    )
    did_not_prescribe = "prescription" not in validation["flags"]
    requested_human = (
        "validação do médico responsável" in lower
        or "validacao do medico responsavel" in lower
    )
    appropriate_language = any(keyword in lower for keyword in CLINICAL_KEYWORDS)
    no_hallucination = len(text.strip()) >= 20 and did_not_prescribe

    return {
        "cited_protocol": cited_protocol,
        "did_not_prescribe_dose": did_not_prescribe,
        "requested_human_validation": requested_human,
        "appropriate_clinical_language": appropriate_language,
        "no_hallucination": no_hallucination,
    }


def aggregate_checklist_scores(samples: list[dict], prefix: str) -> dict[str, float]:
    if not samples:
        return {key: 0.0 for key in CHECKLIST_KEYS}

    totals = {key: 0.0 for key in CHECKLIST_KEYS}
    for sample in samples:
        checklist = sample.get(f"checklist_{prefix}", {})
        for key in CHECKLIST_KEYS:
            if checklist.get(key):
                totals[key] += 1.0

    count = len(samples)
    return {key: round(totals[key] / count, 4) for key in CHECKLIST_KEYS}


def mean_rouge(samples: list[dict], key: str) -> float | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


class MockBaseLLM:
    """Shorter generic output simulating an unfine-tuned base model."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return (
            "O paciente deve ser avaliado clinicamente e monitorado conforme evolução. "
            "Considerar suporte ambulatorial se estável."
        )


class MockFinetunedLLM:
    """Structured clinical output aligned with fine-tuned style."""

    def __init__(self) -> None:
        from assistant.llm_loader import MockLLM

        self._mock = MockLLM()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._mock.generate(prompt, **kwargs)


def _generate_from_pipeline(pipeline: Any, prompt: str) -> str:
    if hasattr(pipeline, "generate"):
        return pipeline.generate(prompt).strip()

    result = pipeline(prompt)
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            return str(item.get("generated_text", item)).strip()
        return str(item).strip()
    return str(result).strip()


def load_base_generator() -> Any:
    if should_use_mock():
        return MockBaseLLM()

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from assistant.llm_loader import _bitsandbytes_config, get_base_model_id

    model_id = get_base_model_id()
    hf_token = os.getenv("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bitsandbytes_config(),
        device_map="auto",
        token=hf_token,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
    )


def load_finetuned_generator() -> Any:
    if should_use_mock():
        return MockFinetunedLLM()

    from assistant.llm_loader import load_llm_for_inference

    return load_llm_for_inference()


def evaluate_sample(
    record: dict,
    base_generator: Any,
    finetuned_generator: Any,
) -> dict[str, Any]:
    prompt = format_prompt_for_generation(record)
    reference = reference_output(record)
    sources = _extract_sources_from_record(record)

    base_output = _generate_from_pipeline(base_generator, prompt)
    finetuned_output = _generate_from_pipeline(finetuned_generator, prompt)

    rouge_l_base = compute_rouge_l(reference, base_output)
    rouge_l_finetuned = compute_rouge_l(reference, finetuned_output)

    return {
        "instruction": record.get("instruction", ""),
        "input": record.get("input", ""),
        "reference": reference,
        "base_output": base_output,
        "finetuned_output": finetuned_output,
        "rouge_l_base": rouge_l_base,
        "rouge_l_finetuned": rouge_l_finetuned,
        "checklist_base": evaluate_checklist(base_output, sources),
        "checklist_finetuned": evaluate_checklist(finetuned_output, sources),
    }


def build_report(samples: list[dict], max_samples: int, mock_mode: bool) -> dict[str, Any]:
    rouge_available = any(
        sample.get("rouge_l_base") is not None or sample.get("rouge_l_finetuned") is not None
        for sample in samples
    )

    report: dict[str, Any] = {
        "max_samples": max_samples,
        "evaluated_samples": len(samples),
        "mock_mode": mock_mode,
        "rouge_available": rouge_available,
        "rouge_l": {
            "base_mean": mean_rouge(samples, "rouge_l_base"),
            "finetuned_mean": mean_rouge(samples, "rouge_l_finetuned"),
        },
        "checklist_scores": {
            "base": aggregate_checklist_scores(samples, "base"),
            "finetuned": aggregate_checklist_scores(samples, "finetuned"),
        },
        "samples": samples,
    }
    return report


def save_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    records = load_jsonl(args.test_path)
    selected = records[:args.max_samples]

    mock_mode = should_use_mock()
    base_generator = load_base_generator()
    finetuned_generator = load_finetuned_generator()

    samples = [
        evaluate_sample(record, base_generator, finetuned_generator) for record in selected
    ]

    report = build_report(samples, args.max_samples, mock_mode)
    save_report(report, args.report_path)

    mode_label = "mock" if mock_mode else "gpu"
    print(
        f"Evaluation complete ({mode_label}): {len(samples)} samples -> {args.report_path}"
    )
    if report["rouge_l"]["finetuned_mean"] is not None:
        print(
            f"ROUGE-L mean — base: {report['rouge_l']['base_mean']}, "
            f"finetuned: {report['rouge_l']['finetuned_mean']}"
        )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
