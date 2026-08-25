"""LangGraph node functions for the clinical workflow."""

from __future__ import annotations

from typing import Any

from assistant.chains import build_prompt
from assistant.llm_loader import load_langchain_llm
from assistant.tools import (
    _connect_db,
    buscar_protocolo,
    consultar_prontuario,
    verificar_exames_pendentes,
)
from langgraph_flows.state import ClinicalState
from security.guardrails import validate_response


def _append_path(state: ClinicalState, node_name: str) -> list[str]:
    return [*state.get("graph_path", []), node_name]


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


def _load_patient_record(patient_id: str) -> dict[str, Any]:
    with _connect_db() as conn:
        patient = conn.execute(
            "SELECT patient_id, perfil, queixa, status FROM pacientes WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()

    if patient is None:
        return {"patient_id": patient_id, "found": False}

    return {
        "patient_id": patient["patient_id"],
        "perfil": patient["perfil"],
        "queixa": patient["queixa"],
        "status": patient["status"],
        "found": True,
    }


def triagem(state: ClinicalState) -> ClinicalState:
    """Classify the query and load initial patient context."""
    patient_data = _load_patient_record(state["patient_id"])
    query_lower = state["query"].lower()

    if any(word in query_lower for word in ("conduta", "protocolo", "tratamento")):
        triage_type = "conduta"
    elif any(word in query_lower for word in ("exame", "pendente", "alerta")):
        triage_type = "alerta"
    else:
        triage_type = "informacao"

    patient_data["triage_type"] = triage_type

    return {
        **state,
        "patient_data": patient_data,
        "graph_path": _append_path(state, "triagem"),
    }


def verificar_exames(state: ClinicalState) -> ClinicalState:
    """Check for pending exams in the patient record."""
    pending = verificar_exames_pendentes(state["patient_id"])
    return {
        **state,
        "pending_exams": pending,
        "graph_path": _append_path(state, "verificar_exames"),
    }


def route_after_exams(state: ClinicalState) -> str:
    """Route to alert branch when pending exams exist."""
    if state.get("pending_exams"):
        return "alerta"
    return "prontuario"


def agente_alerta(state: ClinicalState) -> ClinicalState:
    """Generate team alerts for pending exams."""
    alerts: list[str] = []
    for exam in state.get("pending_exams", []):
        alerts.append(
            f"ALERTA: Exame pendente — {exam['tipo']} "
            f"(status={exam['status']}, data={exam['data']}). "
            f"Priorizar acompanhamento clínico."
        )

    if not alerts:
        alerts.append("ALERTA: Verificar exames pendentes no prontuário.")

    chart = consultar_prontuario(state["patient_id"])
    patient_data = {**state.get("patient_data", {}), "chart": chart}

    return {
        **state,
        "alerts": alerts,
        "patient_data": patient_data,
        "graph_path": _append_path(state, "agente_alerta"),
    }


def agente_prontuario(state: ClinicalState) -> ClinicalState:
    """Load full chart from SQLite when no pending-exam alert branch is taken."""
    chart = consultar_prontuario(state["patient_id"])
    patient_data = {**state.get("patient_data", {}), "chart": chart}

    return {
        **state,
        "patient_data": patient_data,
        "graph_path": _append_path(state, "agente_prontuario"),
    }


def agente_protocolo(state: ClinicalState) -> ClinicalState:
    """Retrieve protocols via RAG and generate a clinical suggestion with the LLM."""
    chart = state.get("patient_data", {}).get("chart")
    if not chart:
        chart = consultar_prontuario(state["patient_id"])

    protocol_chunks = buscar_protocolo(state["query"])
    sources = [chunk["source"] for chunk in protocol_chunks]

    prompt = build_prompt(state["patient_id"], state["query"], chart, protocol_chunks)
    llm = load_langchain_llm()
    suggestion = _invoke_llm(llm, prompt)

    return {
        **state,
        "retrieved_protocols": protocol_chunks,
        "sources": sources,
        "suggestion": suggestion,
        "graph_path": _append_path(state, "agente_protocolo"),
    }


def agente_auditoria(state: ClinicalState) -> ClinicalState:
    """Apply guardrails and explainability checks to the generated suggestion."""
    validation = validate_response(state.get("suggestion", ""), state.get("sources", []))

    return {
        **state,
        "suggestion": validation["sanitized_text"],
        "requires_human_validation": validation["requires_human_validation"],
        "log_entry": {
            "flags": validation["flags"],
            "valid": validation["valid"],
        },
        "graph_path": _append_path(state, "agente_auditoria"),
    }


def registrar_log(state: ClinicalState) -> ClinicalState:
    """Append an audit log entry with the full workflow trace."""
    from security.logger import log_interaction

    graph_path = _append_path(state, "registrar_log")
    audit_flags = {
        key: value
        for key, value in state.get("log_entry", {}).items()
        if key not in {"graph_path", "patient_id", "query", "response", "sources"}
    }
    log_entry = {
        "patient_id": state["patient_id"],
        "query": state["query"],
        "response": state.get("suggestion", ""),
        "sources": state.get("sources", []),
        "alerts": state.get("alerts", []),
        "requires_human_validation": state.get("requires_human_validation", False),
        "graph_path": graph_path,
        **audit_flags,
    }
    log_interaction(log_entry)

    return {
        **state,
        "log_entry": log_entry,
        "graph_path": graph_path,
    }
