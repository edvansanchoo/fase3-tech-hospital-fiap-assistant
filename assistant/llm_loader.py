"""Load base and fine-tuned LLMs with swappable BASE_MODEL_ID."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"


def get_base_model_id() -> str:
    return os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID)


def _should_use_mock() -> bool:
    if os.getenv("USE_MOCK_LLM", "").strip() in ("1", "true", "True", "yes"):
        return True
    try:
        import torch
    except ImportError:
        return True
    return not torch.cuda.is_available()


def _extract_sources_from_prompt(prompt: str) -> list[str]:
    sources: list[str] = []
    for match in re.finditer(r"([\w\-]+\.md)", prompt, flags=re.IGNORECASE):
        source = match.group(1)
        if source not in sources:
            sources.append(source)
    for match in re.finditer(r"protocolo[_\w]+", prompt, flags=re.IGNORECASE):
        source = match.group(0)
        if source not in sources:
            sources.append(source)
    return sources[:3]


class MockLLM:
    """Simple mock LLM for CI and CPU-only development."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or get_base_model_id()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        sources = _extract_sources_from_prompt(prompt)
        if not sources:
            sources = ["protocolo_febre_v2.md"]

        source_refs = ", ".join(
            f"{source}, seção 3.1" if not source.endswith(".md") else f"{source}, seção 3.1"
            for source in sources
        )
        return (
            f"Segundo {source_refs}: avaliar sinais de gravidade, monitorar evolução clínica "
            f"e seguir protocolo interno do Hospital FIAP. "
            f"Conduta sugerida com base nos registros e protocolos disponíveis. "
            f"Requer validação do médico responsável."
        )

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)


def _get_adapter_path() -> Path:
    return Path(os.getenv("LORA_ADAPTER_PATH", "./artifacts/lora_adapter"))


def _adapter_available(adapter_path: Path) -> bool:
    return adapter_path.is_dir() and any(adapter_path.iterdir())


def _bitsandbytes_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )


def load_llm_for_inference(device_map: str = "auto") -> Any:
    """Load HuggingFace text-generation pipeline or MockLLM when mock/CPU-only."""
    if _should_use_mock():
        return MockLLM()

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    model_id = get_base_model_id()
    adapter_path = _get_adapter_path()
    hf_token = os.getenv("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bitsandbytes_config(),
        device_map=device_map,
        token=hf_token,
    )

    if _adapter_available(adapter_path):
        model = PeftModel.from_pretrained(model, str(adapter_path))

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
    )


def load_langchain_llm() -> Any:
    """Return a LangChain LLM wrapper for chains."""
    llm = load_llm_for_inference()
    if isinstance(llm, MockLLM):
        from langchain_core.language_models.llms import LLM
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun
        from typing import Optional, List

        mock = llm

        class MockLangChainLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "mock_clinical_llm"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                return mock.generate(prompt, **kwargs)

        return MockLangChainLLM()

    from langchain_community.llms import HuggingFacePipeline

    return HuggingFacePipeline(pipeline=llm)


def load_llm_for_training() -> tuple[Any, Any]:
    """Load base model and tokenizer for LoRA training (used by train.py)."""
    if _should_use_mock():
        raise RuntimeError(
            "Training requires GPU and USE_MOCK_LLM must not be set. "
            "Run fine-tuning on Colab or a CUDA-enabled machine."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return model, tokenizer
