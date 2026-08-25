"""Streamlit demo for Hospital FIAP Assistant."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_llm_mode() -> None:
    """Use mock LLM on CPU-only machines unless explicitly overridden."""
    if os.getenv("USE_MOCK_LLM", "").strip() in ("1", "true", "True", "yes"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            os.environ["USE_MOCK_LLM"] = "1"
    except ImportError:
        os.environ["USE_MOCK_LLM"] = "1"


_configure_llm_mode()

import streamlit as st

from assistant.chains import run_assistant
from langgraph_flows.clinical_workflow import run_workflow

PATIENTS = [f"PAC-{i:03d}" for i in range(1, 6)]
LOG_PATH = Path(os.getenv("LOG_PATH", str(ROOT / "logs" / "interactions.jsonl")))


def tail_log_lines(log_path: Path, count: int = 10) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return text.splitlines()[-count:]


def _mock_mode_active() -> bool:
    return os.getenv("USE_MOCK_LLM", "").strip() in ("1", "true", "True", "yes")


def display_validation_badge(requires_human_validation: bool, valid: bool | None = None) -> None:
    if valid is False:
        st.error("Resposta bloqueada pelos guardrails")
    if requires_human_validation:
        st.warning("Requer validação do médico responsável")
    else:
        st.success("Sem exigência de validação humana adicional")


def display_result(
    response: str,
    sources: list[str],
    requires_human_validation: bool,
    valid: bool | None = None,
) -> None:
    st.subheader("Resposta")
    st.markdown(response)
    display_validation_badge(requires_human_validation, valid)
    st.subheader("Fontes")
    if sources:
        for source in sources:
            st.markdown(f"- `{source}`")
    else:
        st.caption("Nenhuma fonte registrada.")


def render_logs_tab() -> None:
    st.subheader("Últimas interações")
    st.caption(f"Arquivo: `{LOG_PATH}`")

    if st.button("Atualizar logs", key="refresh_logs"):
        st.rerun()

    lines = tail_log_lines(LOG_PATH, count=10)
    if not lines:
        st.info("Nenhum registro em logs/interactions.jsonl.")
        return

    for line in lines:
        try:
            entry: dict[str, Any] = json.loads(line)
            st.json(entry)
        except json.JSONDecodeError:
            st.code(line)


def main() -> None:
    st.set_page_config(page_title="Hospital FIAP Assistant", layout="wide")
    st.title("Hospital FIAP Assistant")
    st.caption("Demo Tech Challenge — LangChain, LangGraph, guardrails e auditoria")

    if _mock_mode_active():
        st.sidebar.info("Modo mock LLM (CPU / USE_MOCK_LLM=1)")
    else:
        st.sidebar.success("Modo inferência com GPU")

    patient_id = st.selectbox("Paciente", PATIENTS, index=0)
    query = st.text_input(
        "Pergunta clínica",
        placeholder="Ex.: Protocolo para febre e tosse?",
    )

    tab_langchain, tab_langgraph, tab_logs = st.tabs(
        ["Assistente LangChain", "Fluxo LangGraph", "Logs"]
    )

    with tab_langchain:
        st.markdown("Pipeline LangChain: prontuário → RAG → LLM → guardrails → log.")
        if st.button("Executar LangChain", key="run_langchain"):
            if not query.strip():
                st.warning("Informe uma pergunta clínica.")
            else:
                with st.spinner("Processando assistente LangChain..."):
                    result = run_assistant(patient_id, query.strip())
                display_result(
                    result["response"],
                    result["sources"],
                    result["requires_human_validation"],
                    result.get("valid"),
                )

    with tab_langgraph:
        st.markdown("Workflow LangGraph: triagem → exames → alerta/prontuário → protocolo → auditoria.")
        if st.button("Executar LangGraph", key="run_langgraph"):
            if not query.strip():
                st.warning("Informe uma pergunta clínica.")
            else:
                with st.spinner("Executando fluxo LangGraph..."):
                    result = run_workflow(patient_id, query.strip())
                display_result(
                    result.get("suggestion", ""),
                    result.get("sources", []),
                    result.get("requires_human_validation", False),
                )
                if result.get("alerts"):
                    st.subheader("Alertas")
                    for alert in result["alerts"]:
                        st.warning(alert)
                graph_path = result.get("graph_path", [])
                if graph_path:
                    st.subheader("Fluxo percorrido")
                    st.code(" → ".join(graph_path))

    with tab_logs:
        render_logs_tab()


if __name__ == "__main__":
    main()
