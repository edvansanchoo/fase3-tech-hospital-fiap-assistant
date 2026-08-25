import re
from typing import List

PRESCRIPTION_PATTERNS = [
    r"\bprescrev",
    r"\d+\s*mg",
    r"\d+\s*ml",
    r"\d+/\d+h",
]
HUMAN_VALIDATION_PHRASE = "validação do médico responsável"
SOURCE_MARKERS = ("protocolo", "secção", "seção", "registro", ".md")


def validate_response(text: str, sources: List[str]) -> dict:
    flags: List[str] = []
    lower = text.lower()
    for pat in PRESCRIPTION_PATTERNS:
        if re.search(pat, lower):
            flags.append("prescription")
    if not sources:
        flags.append("missing_source")
    has_source_in_text = any(m in lower for m in SOURCE_MARKERS) or bool(sources)
    if not has_source_in_text:
        flags.append("missing_source")
    requires_human = any(k in lower for k in ("conduta", "recomend", "protocolo", "seguir"))
    if requires_human and HUMAN_VALIDATION_PHRASE not in lower:
        flags.append("missing_human_validation")
    sanitized = text
    if "prescription" in flags:
        sanitized = "Não posso prescrever medicamentos. Requer validação do médico responsável."
    valid = "prescription" not in flags
    return {
        "valid": valid,
        "requires_human_validation": requires_human or "missing_human_validation" in flags,
        "flags": flags,
        "sanitized_text": sanitized,
    }
