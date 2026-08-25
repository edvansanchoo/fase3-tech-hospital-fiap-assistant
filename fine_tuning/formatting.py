def format_instruction_sample(instruction: str, input_text: str, output: str) -> str:
    user = f"### Pergunta:\n{instruction}"
    if input_text.strip():
        user += f"\n### Contexto:\n{input_text}"
    return f"### Instrução:\n{user}\n\n### Resposta:\n{output}"


def format_sample_text(record: dict) -> str:
    """Return the training text for a JSONL record (uses pre-built text or formats fields)."""
    if record.get("text"):
        return record["text"]
    return format_instruction_sample(
        record.get("instruction", ""),
        record.get("input", ""),
        record.get("output", ""),
    )


def tokenize_texts(tokenizer, texts: list[str], max_length: int = 512) -> dict:
    """Tokenize instruction samples for causal LM training."""
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
