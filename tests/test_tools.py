from assistant.tools import consultar_prontuario, verificar_exames_pendentes


def test_consultar_prontuario_pac001():
    text = consultar_prontuario("PAC-001")
    assert "PAC-001" in text


def test_exames_pendentes_pac002():
    pending = verificar_exames_pendentes("PAC-002")
    assert any(e["status"] == "pendente" for e in pending)
