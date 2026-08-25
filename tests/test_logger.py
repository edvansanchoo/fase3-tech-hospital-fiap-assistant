import json

from security.logger import log_interaction


def test_log_interaction_writes_jsonl_with_timestamp(tmp_path):
    log_file = tmp_path / "interactions.jsonl"
    log_interaction({"patient_id": "PAC-001", "query": "test query"}, log_path=str(log_file))

    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "timestamp" in entry
    assert entry["patient_id"] == "PAC-001"
    assert entry["query"] == "test query"
