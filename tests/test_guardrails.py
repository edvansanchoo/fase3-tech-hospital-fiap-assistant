from security.guardrails import validate_response


def test_blocks_direct_prescription():
    result = validate_response("Prescreva amoxicilina 500mg 8/8h por 7 dias.", sources=["protocolo_febre_v2.md"])
    assert result["valid"] is False
    assert "prescription" in result["flags"]


def test_requires_human_validation_for_conduct():
    result = validate_response(
        "Segundo protocolo_febre_v2.md secção 3.1: avaliar SpO2.",
        sources=["protocolo_febre_v2.md"],
    )
    assert result["requires_human_validation"] is True


def test_flags_missing_sources():
    result = validate_response("Paciente estável.", sources=[])
    assert "missing_source" in result["flags"]
