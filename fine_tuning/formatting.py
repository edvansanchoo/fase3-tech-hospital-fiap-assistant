def format_instruction_sample(instruction: str, input_text: str, output: str) -> str:
    user = f"### Pergunta:\n{instruction}"
    if input_text.strip():
        user += f"\n### Contexto:\n{input_text}"
    return f"### Instrução:\n{user}\n\n### Resposta:\n{output}"
