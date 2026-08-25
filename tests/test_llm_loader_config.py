import os

from assistant.llm_loader import get_base_model_id


def test_default_model_id():
    os.environ.pop("BASE_MODEL_ID", None)
    assert "llama" in get_base_model_id().lower()


def test_override_model_id():
    os.environ["BASE_MODEL_ID"] = "mistralai/Mistral-7B-v0.1"
    assert get_base_model_id() == "mistralai/Mistral-7B-v0.1"
