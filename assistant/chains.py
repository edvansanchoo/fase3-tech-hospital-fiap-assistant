"""Main LangChain assistant pipeline: chart lookup, RAG, LLM, guardrails, audit log."""

from __future__ import annotations

from typing import Any

from assistant.llm_loader import load_langchain_llm
from assistant.prompts import SYSTEM_PROMPT
from assistant.tools import buscar_protocolo, consultar_prontuario
from security.guardrails import validate_response
from security.logger import log_interaction


def _format_protocol_context(protocol_chunks: list[dict[str, Any]]) -> str:
    if not protocol_chunks:
        return "Nenhum protocolo relevante encontrado."

    sections: list[str] = []
    for index, chunk in enumerate(protocol_chunks, start=1):
        sections.append(
            f"### Protocolo {index}: {chunk['source']} (score={chunk.get('score', 'n/a')})\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(sections)


def build_prompt(patient_id: str, query: str, chart: str, protocol_chunks: list[dict[str, Any]]) -> str:
    """Build the full prompt sent to the LLM."""
    protocol_context = _format_protocol_context(protocol_chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Paciente: {patient_id}\n\n"
        f"### Prontuário:\n{chart}\n\n"
        f"### Pergunta:\n{query}\n\n"
        f"### Protocolos relevantes:\n{protocol_context}\n\n"
        "### Resposta:"
    )


def _invoke_llm(llm: Any, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        result = llm.invoke(prompt)
    else:
        result = llm(prompt)

    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "content"):
        return str(result.content).strip()
    return str(result).strip()


def run_assistant(patient_id: str, query: str, llm: Any | None = None) -> dict[str, Any]:
    """Run the clinical assistant pipeline for a patient query."""
    chart = consultar_prontuario(patient_id)
    protocol_chunks = buscar_protocolo(query)
    sources = [chunk["source"] for chunk in protocol_chunks]

    prompt = build_prompt(patient_id, query, chart, protocol_chunks)
    model = llm if llm is not None else load_langchain_llm()
    raw_response = _invoke_llm(model, prompt)

    validation = validate_response(raw_response, sources)
    response = validation["sanitized_text"]

    log_interaction(
        {
            "patient_id": patient_id,
            "query": query,
            "response": response,
            "sources": sources,
            "flags": validation["flags"],
            "valid": validation["valid"],
            "requires_human_validation": validation["requires_human_validation"],
        }
    )

    return {
        "response": response,
        "sources": sources,
        "requires_human_validation": validation["requires_human_validation"],
        "flags": validation["flags"],
        "valid": validation["valid"],
    }
