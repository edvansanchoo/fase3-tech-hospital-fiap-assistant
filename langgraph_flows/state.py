"""Shared LangGraph state for the clinical workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class ClinicalState(TypedDict):
    patient_id: str
    query: str
    patient_data: dict[str, Any]
    pending_exams: list[dict[str, Any]]
    retrieved_protocols: list[dict[str, Any]]
    alerts: list[str]
    suggestion: str
    sources: list[str]
    requires_human_validation: bool
    log_entry: dict[str, Any]
    graph_path: list[str]
