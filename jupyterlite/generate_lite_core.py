"""One-off script to regenerate lite_core.py from project data."""
import json
from pathlib import Path

proj = Path(__file__).resolve().parent.parent
embedded_path = Path(__file__).resolve().parent / "_embedded_data.json"
with embedded_path.open(encoding="utf-8") as f:
    data = json.load(f)

protocols_repr = json.dumps(data["protocols"], ensure_ascii=False, indent=4)
samples_repr = json.dumps(data["samples"], ensure_ascii=False, indent=4)

TEMPLATE = r'''"""Hospital FIAP Assistant — versão JupyterLite (Python puro).

Compatível com https://jupyter.org/try-jupyter/lab/
Sem torch, LangChain ou LangGraph — reproduz os conceitos das aulas da Fase 3.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable

PROTOCOLS: dict[str, str] = __PROTOCOLS__

DATASET_SAMPLES: dict[str, list[dict[str, Any]]] = __SAMPLES__

PATIENTS = [
    ("PAC-001", "Adulto 45a", "febre, tosse 3 dias", "estável"),
    ("PAC-002", "Adulto 62a", "dor abdominal", "estável"),
    ("PAC-003", "Adulto 52a", "follow-up oncologia mama", "estável"),
    ("PAC-004", "Adulto 70a", "polifarmácia", "estável"),
    ("PAC-005", "Adulto 30a", "consulta geral", "estável"),
]

EXAMS = [
    ("PAC-001", "hemograma", "concluido", "2026-01-10"),
    ("PAC-002", "tomografia_abdominal", "pendente", "2026-01-20"),
    ("PAC-003", "mamografia", "pendente", "2026-01-18"),
    ("PAC-004", "função_renal", "concluido", "2026-01-12"),
    ("PAC-005", "consulta_geral", "concluido", "2026-01-15"),
]

PRESCRIPTIONS = [
    ("PAC-004", "warfarina 5mg", "ativo"),
    ("PAC-004", "amoxicilina 500mg", "ativo"),
]

SYSTEM_PROMPT = """Você é assistente de apoio clínico do Hospital FIAP.
- NÃO substitui o médico.
- NUNCA prescreva medicamentos ou doses finais.
- SEMPRE cite a fonte (protocolo, seção, registro).
- SEMPRE indique "Requer validação do médico responsável" em condutas.
- Se dados insuficientes, diga explicitamente."""

HUMAN_VALIDATION_PHRASE = "validação do médico responsável"
PRESCRIPTION_PATTERNS = [r"\bprescrev", r"\d+\s*mg", r"\d+\s*ml", r"\d+/\d+h"]

AUDIT_LOG: list[dict[str, Any]] = []


def anonymize_text(text: str) -> str:
    text = re.sub(r"\d{3}\.\d{3}\.\d{3}-\d{2}", "[CPF]", text)
    text = re.sub(r"\d{2}/\d{2}/\d{4}", "[DATA]", text)
    text = re.sub(
        r"(?:Sr\.|Sra\.|Dr\.|Dra\.)\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*",
        "[NOME]",
        text,
    )
    return text


def format_instruction_sample(instruction: str, input_text: str, output: str) -> dict[str, str]:
    text = (
        f"### Instrução:\n### Pergunta:\n{instruction}\n"
        f"### Contexto:\n{input_text}\n\n### Resposta:\n{output}"
    )
    return {"instruction": instruction, "input": input_text, "output": output, "text": text}


def generate_protocol_pairs() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for filename, content in sorted(PROTOCOLS.items()):
        protocol_name = filename.replace(".md", "")
        title = content.splitlines()[0].lstrip("# ").strip() if content else protocol_name
        sections = re.split(r"^##\s+", content, flags=re.MULTILINE)
        for section in sections[1:]:
            header_end = section.find("\n")
            if header_end == -1:
                continue
            header = section[:header_end].strip()
            body = section[header_end:].strip()
            if len(body) < 20:
                continue
            instruction = f"Segundo {protocol_name}, o que orienta a seção '{header}'?"
            input_text = f"Protocolo: {title}"
            output = (
                f"Segundo {protocol_name}, {header}: {body[:280].strip()} "
                f"Requer validação do médico responsável."
            )
            records.append(format_instruction_sample(instruction, input_text, output))
    return records


def prepare_dataset_demo() -> dict[str, list[dict[str, str]]]:
    records = generate_protocol_pairs()
    total = len(records)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def simulate_lora_training(epochs: int = 3, steps_per_epoch: int = 10) -> dict[str, Any]:
    import random

    random.seed(42)
    loss_history: list[dict[str, float]] = []
    loss = 2.4
    step = 0
    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            step += 1
            loss = max(0.35, loss * 0.92 + random.uniform(-0.03, 0.05))
            loss_history.append({"step": step, "epoch": epoch, "loss": round(loss, 4)})

    return {
        "base_model": "meta-llama/Llama-2-7b-hf (simulado)",
        "method": "LoRA rank=16, alpha=32, 4-bit (conceitual)",
        "epochs": epochs,
        "final_loss": loss_history[-1]["loss"],
        "loss_history": loss_history,
        "adapter_path": "artifacts/lora_adapter/ (simulado)",
        "note": "No JupyterLite não há GPU/torch. Treino real: use Colab ou máquina local.",
    }


def create_in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE pacientes (patient_id TEXT PRIMARY KEY, perfil TEXT, queixa TEXT, status TEXT);
        CREATE TABLE exames (patient_id TEXT, tipo TEXT, status TEXT, data TEXT);
        CREATE TABLE prescricoes (patient_id TEXT, medicamento TEXT, status TEXT);
        """
    )
    cur.executemany("INSERT INTO pacientes VALUES (?,?,?,?)", PATIENTS)
    cur.executemany("INSERT INTO exames VALUES (?,?,?,?)", EXAMS)
    cur.executemany("INSERT INTO prescricoes VALUES (?,?,?)", PRESCRIPTIONS)
    conn.commit()
    return conn


_DB: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        _DB = create_in_memory_db()
    return _DB


def consultar_prontuario(patient_id: str) -> str:
    conn = get_db()
    patient = conn.execute(
        "SELECT patient_id, perfil, queixa, status FROM pacientes WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    if patient is None:
        return f"Paciente {patient_id} não encontrado."
    exams = conn.execute(
        "SELECT tipo, status, data FROM exames WHERE patient_id = ? ORDER BY data",
        (patient_id,),
    ).fetchall()
    prescriptions = conn.execute(
        "SELECT medicamento, status FROM prescricoes WHERE patient_id = ?",
        (patient_id,),
    ).fetchall()

    lines = [
        f"Prontuário — {patient['patient_id']}",
        f"Perfil: {patient['perfil']}",
        f"Queixa: {patient['queixa']}",
        f"Status: {patient['status']}",
    ]
    if exams:
        lines.append("Exames:")
        for exam in exams:
            lines.append(f"  - {exam['tipo']} ({exam['status']}, {exam['data']})")
    else:
        lines.append("Exames: nenhum registrado.")
    if prescriptions:
        lines.append("Prescrições ativas:")
        for rx in prescriptions:
            lines.append(f"  - {rx['medicamento']} ({rx['status']})")
    else:
        lines.append("Prescrições: nenhuma ativa.")
    return "\n".join(lines)


def verificar_exames_pendentes(patient_id: str) -> list[dict[str, str]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT tipo, status, data FROM exames
        WHERE patient_id = ? AND status = 'pendente' ORDER BY data
        """,
        (patient_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def buscar_protocolo(query: str, k: int = 3) -> list[dict[str, Any]]:
    keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not keywords:
        keywords = [query.lower()]

    scored: list[dict[str, Any]] = []
    for filename, content in PROTOCOLS.items():
        haystack = f"{filename} {content}".lower()
        matches = sum(1 for kw in keywords if kw in haystack)
        if matches == 0:
            continue
        scored.append({
            "content": content[:1200],
            "source": filename,
            "score": round(matches / len(keywords), 4),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


class MockLLM:
    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-hf (mock)") -> None:
        self.model_id = model_id

    def _extract_sources(self, prompt: str) -> list[str]:
        sources: list[str] = []
        for match in re.finditer(r"([\w\-]+\.md)", prompt, flags=re.IGNORECASE):
            if match.group(1) not in sources:
                sources.append(match.group(1))
        return sources[:3] or ["protocolo_febre_v2.md"]

    def invoke(self, prompt: str) -> str:
        sources = self._extract_sources(prompt)
        refs = ", ".join(f"{s}, seção 3.1" for s in sources)
        return (
            f"Segundo {refs}: avaliar sinais de gravidade, monitorar evolução clínica "
            f"e seguir protocolo interno do Hospital FIAP. "
            f"Conduta sugerida com base nos registros e protocolos disponíveis. "
            f"Requer validação do médico responsável."
        )


def validate_response(text: str, sources: list[str]) -> dict[str, Any]:
    flags: list[str] = []
    lower = text.lower()
    for pat in PRESCRIPTION_PATTERNS:
        if re.search(pat, lower):
            flags.append("prescription")
    if not sources:
        flags.append("missing_source")
    source_markers = ("protocolo", "secção", "seção", "registro", ".md")
    if not any(m in lower for m in source_markers) and not sources:
        flags.append("missing_source")
    requires_human = any(k in lower for k in ("conduta", "recomend", "protocolo", "seguir"))
    if requires_human and HUMAN_VALIDATION_PHRASE not in lower:
        flags.append("missing_human_validation")
    sanitized = text
    if "prescription" in flags:
        sanitized = "Não posso prescrever medicamentos. Requer validação do médico responsável."
    return {
        "valid": "prescription" not in flags,
        "requires_human_validation": requires_human or "missing_human_validation" in flags,
        "flags": flags,
        "sanitized_text": sanitized,
    }


def log_interaction(entry: dict[str, Any]) -> dict[str, Any]:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    AUDIT_LOG.append(record)
    return record


def build_prompt(patient_id: str, query: str, chart: str, protocol_chunks: list[dict]) -> str:
    protocol_context = "\n\n".join(
        f"### Protocolo {i}: {c['source']} (score={c.get('score', 'n/a')})\n{c['content']}"
        for i, c in enumerate(protocol_chunks, 1)
    ) or "Nenhum protocolo relevante encontrado."
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Paciente: {patient_id}\n\n"
        f"### Prontuário:\n{chart}\n\n"
        f"### Pergunta:\n{query}\n\n"
        f"### Protocolos relevantes:\n{protocol_context}\n\n"
        "### Resposta:"
    )


def run_assistant(patient_id: str, query: str, llm: MockLLM | None = None) -> dict[str, Any]:
    chart = consultar_prontuario(patient_id)
    protocol_chunks = buscar_protocolo(query)
    sources = [c["source"] for c in protocol_chunks]
    prompt = build_prompt(patient_id, query, chart, protocol_chunks)
    model = llm or MockLLM()
    raw = model.invoke(prompt)
    validation = validate_response(raw, sources)
    response = validation["sanitized_text"]
    log_interaction({
        "patient_id": patient_id,
        "query": query,
        "response": response,
        "sources": sources,
        "flags": validation["flags"],
        "valid": validation["valid"],
        "requires_human_validation": validation["requires_human_validation"],
        "mode": "langchain_chain",
    })
    return {
        "response": response,
        "sources": sources,
        "requires_human_validation": validation["requires_human_validation"],
        "flags": validation["flags"],
        "valid": validation["valid"],
        "prompt_preview": prompt[:500] + "...",
    }


ClinicalState = dict[str, Any]


def _append_path(state: ClinicalState, node: str) -> list[str]:
    return [*state.get("graph_path", []), node]


def node_triagem(state: ClinicalState) -> ClinicalState:
    query_lower = state["query"].lower()
    if any(w in query_lower for w in ("conduta", "protocolo", "tratamento")):
        triage_type = "conduta"
    elif any(w in query_lower for w in ("exame", "pendente", "alerta")):
        triage_type = "alerta"
    else:
        triage_type = "informacao"
    return {**state, "triage_type": triage_type, "graph_path": _append_path(state, "triagem")}


def node_verificar_exames(state: ClinicalState) -> ClinicalState:
    pending = verificar_exames_pendentes(state["patient_id"])
    return {**state, "pending_exams": pending, "graph_path": _append_path(state, "verificar_exames")}


def route_after_exams(state: ClinicalState) -> str:
    return "alerta" if state.get("pending_exams") else "prontuario"


def node_agente_alerta(state: ClinicalState) -> ClinicalState:
    alerts = [
        f"ALERTA: Exame pendente — {e['tipo']} (status={e['status']}, data={e['data']})."
        for e in state.get("pending_exams", [])
    ]
    chart = consultar_prontuario(state["patient_id"])
    return {**state, "alerts": alerts, "chart": chart, "graph_path": _append_path(state, "agente_alerta")}


def node_agente_prontuario(state: ClinicalState) -> ClinicalState:
    chart = consultar_prontuario(state["patient_id"])
    return {**state, "chart": chart, "graph_path": _append_path(state, "agente_prontuario")}


def node_agente_protocolo(state: ClinicalState) -> ClinicalState:
    chart = state.get("chart") or consultar_prontuario(state["patient_id"])
    protocol_chunks = buscar_protocolo(state["query"])
    sources = [c["source"] for c in protocol_chunks]
    prompt = build_prompt(state["patient_id"], state["query"], chart, protocol_chunks)
    suggestion = MockLLM().invoke(prompt)
    return {
        **state,
        "retrieved_protocols": protocol_chunks,
        "sources": sources,
        "suggestion": suggestion,
        "graph_path": _append_path(state, "agente_protocolo"),
    }


def node_agente_auditoria(state: ClinicalState) -> ClinicalState:
    validation = validate_response(state.get("suggestion", ""), state.get("sources", []))
    return {
        **state,
        "suggestion": validation["sanitized_text"],
        "requires_human_validation": validation["requires_human_validation"],
        "audit_flags": validation["flags"],
        "graph_path": _append_path(state, "agente_auditoria"),
    }


def node_registrar_log(state: ClinicalState) -> ClinicalState:
    graph_path = _append_path(state, "registrar_log")
    log_entry = log_interaction({
        "patient_id": state["patient_id"],
        "query": state["query"],
        "response": state.get("suggestion", ""),
        "sources": state.get("sources", []),
        "alerts": state.get("alerts", []),
        "requires_human_validation": state.get("requires_human_validation", False),
        "graph_path": graph_path,
        "flags": state.get("audit_flags", []),
        "mode": "langgraph_workflow",
    })
    return {**state, "log_entry": log_entry, "graph_path": graph_path}


WORKFLOW_NODES: dict[str, Callable[[ClinicalState], ClinicalState]] = {
    "triagem": node_triagem,
    "verificar_exames": node_verificar_exames,
    "agente_alerta": node_agente_alerta,
    "agente_prontuario": node_agente_prontuario,
    "agente_protocolo": node_agente_protocolo,
    "agente_auditoria": node_agente_auditoria,
    "registrar_log": node_registrar_log,
}


def run_workflow(patient_id: str, query: str) -> ClinicalState:
    state: ClinicalState = {
        "patient_id": patient_id,
        "query": query,
        "pending_exams": [],
        "alerts": [],
        "sources": [],
        "graph_path": [],
    }
    state = WORKFLOW_NODES["triagem"](state)
    state = WORKFLOW_NODES["verificar_exames"](state)
    branch = route_after_exams(state)
    state = WORKFLOW_NODES["agente_alerta" if branch == "alerta" else "agente_prontuario"](state)
    state = WORKFLOW_NODES["agente_protocolo"](state)
    state = WORKFLOW_NODES["agente_auditoria"](state)
    state = WORKFLOW_NODES["registrar_log"](state)
    return state
'''

core = TEMPLATE.replace("__PROTOCOLS__", protocols_repr).replace("__SAMPLES__", samples_repr)
out = Path(__file__).resolve().parent / "lite_core.py"
out.write_text(core, encoding="utf-8")
print(f"Generated {out} ({len(core.splitlines())} lines)")
