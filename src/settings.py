"""Environment-backed settings for the Tau2 purple baseline agent."""

import logging
import os

logger = logging.getLogger(__name__)

# LiteLLM OpenRouter ids use the `openrouter/` prefix (see LiteLLM OpenRouter docs).
DEFAULT_MODEL = "openrouter/openai/gpt-4o-mini"

# Fixed OpenRouter API base (do not use OPENAI_BASE_URL for switching providers).
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

DOMAIN_AIRLINE = "airline"
DOMAIN_RETAIL = "retail"
DOMAIN_TELECOM = "telecom"
VALID_DOMAINS = frozenset({DOMAIN_AIRLINE, DOMAIN_RETAIL, DOMAIN_TELECOM})
DEFAULT_DOMAIN = DOMAIN_AIRLINE


def _normalize_domain_raw(raw: str) -> str:
    s = raw.strip().lower()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip().lower()
    return s


def get_domain() -> str:
    """Customer-service vertical from DOMAIN env: airline, retail, or telecom."""
    raw_env = (os.environ.get("DOMAIN") or "").strip()
    raw = _normalize_domain_raw(raw_env)
    if raw in VALID_DOMAINS:
        return raw
    if raw_env:
        logger.warning(
            "Invalid DOMAIN=%r (expected one of %s); using %r",
            raw_env,
            ", ".join(sorted(VALID_DOMAINS)),
            DEFAULT_DOMAIN,
        )
    return DEFAULT_DOMAIN


def get_openai_api_key() -> str:
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()


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


def get_agent_rules_reminder_message_threshold() -> int:
    """When len(agent._messages) >= this value, reasoning/JSON/repair LLM calls include an extra ephemeral user recap. 0 disables."""
    return int(os.environ.get("AGENT_RULES_REMINDER_MESSAGE_THRESHOLD", "12"))


def agent_debug_logging() -> bool:
    return (os.environ.get("AGENT_DEBUG") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or (os.environ.get("DEBUG") or "").strip() == "1"
