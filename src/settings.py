"""Environment-backed settings for the Tau2 purple baseline agent."""

import os

DEFAULT_MODEL = "qwen/qwen3.6-plus:free"


def get_openai_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def get_openai_model() -> str:
    model = (os.environ.get("AGENT_LLM") or "").strip()
    return model or DEFAULT_MODEL


def get_openai_base_url() -> str | None:
    url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    return url or None
