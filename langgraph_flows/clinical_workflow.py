"""LangGraph clinical workflow: triage → exams → alert/chart → protocol → audit → log."""

from __future__ import annotations

import argparse
import json
from typing import Any

from langgraph.graph import END, StateGraph

from langgraph_flows.nodes import (
    agente_alerta,
    agente_auditoria,
    agente_prontuario,
    agente_protocolo,
    registrar_log,
    route_after_exams,
    triagem,
    verificar_exames,
)
from langgraph_flows.state import ClinicalState


def build_graph() -> Any:
    """Build and compile the clinical workflow StateGraph."""
    graph = StateGraph(ClinicalState)

    graph.add_node("triagem", triagem)
    graph.add_node("verificar_exames", verificar_exames)
    graph.add_node("agente_alerta", agente_alerta)
    graph.add_node("agente_prontuario", agente_prontuario)
    graph.add_node("agente_protocolo", agente_protocolo)
    graph.add_node("agente_auditoria", agente_auditoria)
    graph.add_node("registrar_log", registrar_log)

    graph.set_entry_point("triagem")
    graph.add_edge("triagem", "verificar_exames")
    graph.add_conditional_edges(
        "verificar_exames",
        route_after_exams,
        {"alerta": "agente_alerta", "prontuario": "agente_prontuario"},
    )
    graph.add_edge("agente_alerta", "agente_protocolo")
    graph.add_edge("agente_prontuario", "agente_protocolo")
    graph.add_edge("agente_protocolo", "agente_auditoria")
    graph.add_edge("agente_auditoria", "registrar_log")
    graph.add_edge("registrar_log", END)

    return graph.compile()


def run_workflow(patient_id: str, query: str) -> ClinicalState:
    """Execute the clinical workflow for a patient query."""
    initial: ClinicalState = {
        "patient_id": patient_id,
        "query": query,
        "patient_data": {},
        "pending_exams": [],
        "retrieved_protocols": [],
        "alerts": [],
        "suggestion": "",
        "sources": [],
        "requires_human_validation": False,
        "log_entry": {},
        "graph_path": [],
    }
    graph = build_graph()
    return graph.invoke(initial)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hospital FIAP LangGraph clinical workflow")
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. PAC-002)")
    parser.add_argument("--query", required=True, help="Clinical question")
    parser.add_argument("--json", action="store_true", help="Print full state as JSON")
    args = parser.parse_args()

    result = run_workflow(args.patient, args.query)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(result.get("suggestion", ""))
    if result.get("alerts"):
        print("\nAlertas:")
        for alert in result["alerts"]:
            print(f"  - {alert}")
    print("\nFontes:", result.get("sources", []))
    print("Fluxo:", " -> ".join(result.get("graph_path", [])))
    if result.get("requires_human_validation"):
        print("\n[Requer validação do médico responsável]")


if __name__ == "__main__":
    main()
