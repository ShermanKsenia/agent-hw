"""Environment-backed settings for the Tau2 purple baseline agent."""

import os

# LiteLLM OpenRouter ids use the `openrouter/` prefix (see LiteLLM OpenRouter docs).
DEFAULT_MODEL = "openrouter/qwen/qwen3.6-plus:free"

# Fixed OpenRouter API base (do not use OPENAI_BASE_URL for switching providers).
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def get_openai_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def get_openai_model() -> str:
    model = (os.environ.get("AGENT_LLM") or "").strip()
    return model or DEFAULT_MODEL


def litellm_model_for_openrouter(model: str) -> str:
    """Ensure LiteLLM knows to use the OpenRouter provider (avoids 'LLM Provider NOT provided')."""
    m = model.strip()
    if not m:
        return DEFAULT_MODEL
    if m.startswith("openrouter/"):
        return m
    return f"openrouter/{m}"


def get_agent_llm_max_retries() -> int:
    return int(os.environ.get("AGENT_LLM_MAX_RETRIES", "5"))


def get_agent_llm_backoff_base() -> int:
    return int(os.environ.get("AGENT_LLM_BACKOFF_BASE", "2"))


def get_max_completion_tokens() -> int:
    return int(os.environ.get("AGENT_MAX_TOKENS", "2048"))


def agent_debug_logging() -> bool:
    return (os.environ.get("AGENT_DEBUG") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or (os.environ.get("DEBUG") or "").strip() == "1"
