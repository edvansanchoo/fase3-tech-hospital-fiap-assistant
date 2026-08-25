"""LangGraph clinical workflow package."""

from langgraph_flows.state import ClinicalState

__all__ = ["ClinicalState", "build_graph", "run_workflow"]


def __getattr__(name: str):
    if name == "build_graph":
        from langgraph_flows.clinical_workflow import build_graph

        return build_graph
    if name == "run_workflow":
        from langgraph_flows.clinical_workflow import run_workflow

        return run_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
