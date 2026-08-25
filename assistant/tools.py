"""LangChain tools for SQL chart lookup and protocol RAG retrieval."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_URL = "sqlite:///data/synthetic/hospital.db"
DEFAULT_PROTOCOLS_DIR = PROJECT_ROOT / "data" / "synthetic" / "protocols"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "artifacts" / "chroma_db"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _get_db_path() -> Path:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    if not url.startswith("sqlite:///"):
        raise ValueError(f"Unsupported DATABASE_URL: {url}")
    return _resolve_path(url.removeprefix("sqlite:///"))


def _get_protocols_dir() -> Path:
    env_dir = os.getenv("PROTOCOLS_DIR")
    if env_dir:
        return _resolve_path(env_dir)
    return DEFAULT_PROTOCOLS_DIR


def _get_chroma_path() -> Path:
    return _resolve_path(os.getenv("CHROMA_PATH", "./artifacts/chroma_db"))


def _connect_db() -> sqlite3.Connection:
    db_path = _get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run: python data/synthetic/seed_db.py"
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def consultar_prontuario(patient_id: str) -> str:
    """Return a formatted chart summary for the given patient ID."""
    with _connect_db() as conn:
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
            lines.append(
                f"  - {exam['tipo']} ({exam['status']}, {exam['data']})"
            )
    else:
        lines.append("Exames: nenhum registrado.")

    if prescriptions:
        lines.append("Prescrições ativas:")
        for rx in prescriptions:
            lines.append(f"  - {rx['medicamento']} ({rx['status']})")
    else:
        lines.append("Prescrições: nenhuma ativa.")

    return "\n".join(lines)


def verificar_exames_pendentes(patient_id: str) -> list[dict[str, Any]]:
    """Return pending exams for the given patient."""
    with _connect_db() as conn:
        rows = conn.execute(
            """
            SELECT tipo, status, data
            FROM exames
            WHERE patient_id = ? AND status = 'pendente'
            ORDER BY data
            """,
            (patient_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _keyword_search_protocols(query: str, k: int, protocols_dir: Path) -> list[dict[str, Any]]:
    if not protocols_dir.is_dir():
        return []

    keywords = [word for word in re.findall(r"\w+", query.lower()) if len(word) > 2]
    if not keywords:
        keywords = [query.lower()]

    scored: list[dict[str, Any]] = []
    for md_path in sorted(protocols_dir.glob("**/*.md")):
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError:
            continue

        haystack = f"{md_path.name} {content}".lower()
        matches = sum(1 for keyword in keywords if keyword in haystack)
        if matches == 0:
            continue

        score = matches / len(keywords)
        scored.append(
            {
                "content": content[:1200],
                "source": md_path.name,
                "score": round(score, 4),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


def _chroma_search_protocols(query: str, k: int, chroma_path: Path) -> list[dict[str, Any]]:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    vector_store = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

    results: list[dict[str, Any]] = []
    for doc, distance in docs_with_scores:
        source = Path(doc.metadata.get("source", "unknown")).name
        similarity = 1.0 / (1.0 + float(distance))
        results.append(
            {
                "content": doc.page_content,
                "source": source,
                "score": round(similarity, 4),
            }
        )
    return results


def buscar_protocolo(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Search clinical protocols by query, using Chroma when available or keyword fallback."""
    chroma_path = _get_chroma_path()
    protocols_dir = _get_protocols_dir()

    if chroma_path.is_dir() and any(chroma_path.iterdir()):
        try:
            return _chroma_search_protocols(query, k, chroma_path)
        except Exception:
            pass

    return _keyword_search_protocols(query, k, protocols_dir)


def build_vector_store(protocols_dir: str, chroma_path: str) -> None:
    """Index protocol markdown files into a Chroma vector store."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    resolved_protocols = _resolve_path(protocols_dir)
    resolved_chroma = _resolve_path(chroma_path)
    resolved_chroma.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        str(resolved_protocols),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    Chroma.from_documents(docs, embeddings, persist_directory=str(resolved_chroma))


def get_tools() -> list[StructuredTool]:
    """Return LangChain tools for chart lookup and protocol retrieval."""
    return [
        StructuredTool.from_function(
            func=consultar_prontuario,
            name="consultar_prontuario",
            description="Consulta o prontuário sintético de um paciente pelo ID (ex: PAC-001).",
        ),
        StructuredTool.from_function(
            func=verificar_exames_pendentes,
            name="verificar_exames_pendentes",
            description="Lista exames com status pendente para um paciente.",
        ),
        StructuredTool.from_function(
            func=buscar_protocolo,
            name="buscar_protocolo",
            description="Busca trechos relevantes em protocolos clínicos internos.",
        ),
    ]
