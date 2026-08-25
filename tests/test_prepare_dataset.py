from fine_tuning.prepare_dataset import anonymize_text


def test_anonymize_removes_cpf():
    text = "Paciente CPF 123.456.789-00"
    assert "123" not in anonymize_text(text)
