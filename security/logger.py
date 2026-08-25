import json
import os
from datetime import datetime, timezone
from pathlib import Path


def log_interaction(entry: dict, log_path: str | None = None) -> None:
    path = Path(log_path or os.getenv("LOG_PATH", "./logs/interactions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
