"""Tau2 purple agent: LiteLLM + JSON tool/respond output (RemoteA2AAgent protocol)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

import settings

logger = logging.getLogger(__name__)

OPENROUTER_API_BASE = settings.OPENROUTER_API_BASE

RETRYABLE_EXCEPTIONS = (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)

_SYSTEM_PROMPT_AIRLINE = """You are a customer service agent for an airline. Help passengers with bookings and itinerary changes, check-in and boarding, baggage (lost, damaged, allowances), flight delays and cancellations, loyalty programs, and special assistance needs. Prioritize time-sensitive travel problems, be precise about fare rules and rebooking options, and remain calm and professional when customers are stressed.

Prioritize time-critical travel issues (e.g., same-day flights, missed connections) and proactively offer rebooking options based on fare rules and availability. Clearly explain restrictions (change fees, fare classes, standby rules) and set accurate expectations.

Handle stressed or urgent travelers with calm, reassuring communication. When applicable, provide next steps such as compensation policies, vouchers, or escalation to airport staff. Always aim to minimize travel disruption and keep the passenger moving."""

_SYSTEM_PROMPT_RETAIL = """You are a customer service agent for a retail and e-commerce business. Assist customers with product discovery, detailed product information, availability, orders and order status, shipping and delivery options, returns and exchanges, refunds, promotions, and account or checkout issues.

Help customers make confident purchase decisions by clarifying product details, comparing options, and addressing concerns. Balance empathy with clear enforcement of return, refund, and promotion policies.

Focus on fast, frictionless resolution while maintaining a positive shopping experience. When issues arise (e.g., delayed shipments or damaged items), offer practical solutions such as replacements, refunds, or store credit, following company policy."""

_SYSTEM_PROMPT_TELECOM = """You are a customer service agent for a telecommunications provider. Support customers with mobile, broadband, and TV services, including plan selection, billing and usage questions, network coverage and outages, device setup, and technical troubleshooting.

Guide users through step-by-step diagnostics in clear, simple language, adapting to their technical level. Identify whether issues are device-related, account-related, or network-related, and take appropriate action.

Handle sensitive account actions (e.g., SIM swaps, plan changes, security updates) with proper verification. Escalate to technical teams or field service when issues cannot be resolved remotely. Aim to restore service quickly while ensuring customer confidence and security."""

_SYSTEM_PROMPT_SHARED = """Domain policies, tool definitions, and scenario details are provided in user messages—treat them as the source of truth.

Output contract (strict):
- Respond with exactly one JSON object. No extra text.
- The object must include:
  - "name": string
  - "arguments": object
- Must be valid JSON with no markdown or commentary.

Tool vs respond:
- Use "name": "respond" with arguments.content for direct user replies.
- Use a tool only when required to access or modify system state.
- Call at most one tool per turn and wait for results before proceeding.

Tool results:
- Any "Tool '...' result: ..." is authoritative. Do not contradict or extend beyond it.

Arguments:
- Use only defined parameters from the tool schema.
- Include all required fields with correct types.

Behavior:
- Follow escalation, refusal, and transfer policies when specified.
- If information is missing, prefer calling a tool instead of guessing.
- Keep responses concise, clear, and solution-oriented unless detail is required."""

_DOMAIN_SYSTEM_PROMPTS: dict[str, str] = {
    settings.DOMAIN_AIRLINE: _SYSTEM_PROMPT_AIRLINE,
    settings.DOMAIN_RETAIL: _SYSTEM_PROMPT_RETAIL,
    settings.DOMAIN_TELECOM: _SYSTEM_PROMPT_TELECOM,
}

_REASONING_INSTRUCTIONS = """You are performing an internal analysis step before the domain-specific assistant responds.

Do not produce JSON, tool calls, or any formatted output. Write in plain prose only.

Briefly analyze the conversation by covering:
- The customer’s main request and any secondary needs
- Relevant context (e.g., prior messages, time sensitivity, sentiment)
- Missing or unclear information required to proceed
- Which domain (airline, retail, telecom, or unknown) and what type of scenario this falls under
- Whether policies, constraints, or tools are likely needed

Conclude with the most appropriate next step (direct reply, ask clarification, or tool call).

Keep the analysis concise and focused, using no more than 5–7 short sentences."""


def domain_role_intro(domain: str) -> str:
    """Domain-specific role text (shared by JSON system prompt and reasoning system prompt)."""
    return _DOMAIN_SYSTEM_PROMPTS.get(domain, _SYSTEM_PROMPT_AIRLINE)


def reasoning_system_for_domain(domain: str) -> str:
    """System message for the plain-text reasoning pass (no Tau2 JSON contract)."""
    return f"{domain_role_intro(domain)}\n\n{_REASONING_INSTRUCTIONS}"


def system_prompt_for_domain(domain: str) -> str:
    """Full system message: domain-specific role plus shared Tau2 JSON/tool rules."""
    return f"{domain_role_intro(domain)}\n\n{_SYSTEM_PROMPT_SHARED}"

JSON_REPAIR_USER = (
    "Your previous reply was not valid JSON or did not match the required shape "
    '(object with string "name" and object "arguments"). '
    'Reply again with exactly one JSON object only, no markdown.'
)

FALLBACK_RESPOND_JSON = json.dumps(
    {
        "name": "respond",
        "arguments": {
            "content": "I encountered an error processing your request.",
        },
    }
)

ACompletionFn = Callable[..., Awaitable[Any]]

TAU2_JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

# Ephemeral user message appended to LLM calls only (not stored in _messages) when history is long.
_EPHEMERAL_RULES_REMINDER_USER = """[Rules reminder]
- Output (JSON pass): exactly one JSON object with string "name" and object "arguments"; no markdown or extra text.
- Use "name": "respond" and arguments.content for direct replies; use a tool only when needed for state.
- At most one tool per turn; wait for tool results before continuing.
- Lines like Tool '...' result: ... are authoritative—do not contradict them.
- Use only parameters defined in the tool schema, with correct types."""


def _strip_markdown_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> str | None:
    """Best-effort: pull the first {...} span if the model added extra prose."""
    s = text.strip()
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def normalize_model_json_text(raw: str) -> str:
    s = _strip_markdown_fences(raw)
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    extracted = _extract_first_json_object(s)
    if extracted:
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            pass
    return s


def parse_tau2_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None
        if "name" not in data or "arguments" not in data:
            return None
        if not isinstance(data["name"], str):
            return None
        if not isinstance(data["arguments"], dict):
            return None
        return data
    except (json.JSONDecodeError, TypeError):
        return None


class Agent:
    """One LLM-backed agent instance per A2A context (see Executor)."""

    def __init__(
        self,
        *,
        acompletion_fn: ACompletionFn | None = None,
    ):
        self._acompletion_fn: ACompletionFn | None = acompletion_fn
        self._domain = settings.get_domain()
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt_for_domain(self._domain)},
        ]
        self.model = settings.litellm_model_for_openrouter(settings.get_openai_model())
        self.api_key = settings.get_openai_api_key()
        self.max_retries = settings.get_agent_llm_max_retries()
        self.backoff_base = settings.get_agent_llm_backoff_base()
        self.max_tokens = settings.get_max_completion_tokens()
        self._rules_reminder_message_threshold = (
            settings.get_agent_rules_reminder_message_threshold()
        )

    def _llm_messages_with_rules_reminder(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Copy of ``messages`` plus a short rules recap when ``_messages`` is long; never mutates ``_messages``."""
        out = list(messages)
        th = self._rules_reminder_message_threshold
        if th <= 0 or len(self._messages) < th:
            return out
        out.append({"role": "user", "content": _EPHEMERAL_RULES_REMINDER_USER})
        return out

    async def _litellm_acompletion(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_base": OPENROUTER_API_BASE,
            "temperature": 1.0,
            "reasoning_effort": "high",
            "max_tokens": self.max_tokens,
            # OpenRouter / many models ignore reasoning_effort; LiteLLM errors unless dropped.
            "drop_params": True,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return await acompletion(**kwargs)

    async def _call_llm_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
    ) -> Any:
        # Snapshot list so callers (e.g. tests) do not see later appends to self._messages.
        msgs = list(messages)

        if self._acompletion_fn is not None:
            return await self._acompletion_fn(msgs)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._litellm_acompletion(
                    msgs, response_format=response_format
                )
                if attempt > 1:
                    logger.info("LLM call succeeded on attempt %s", attempt)
                return resp
            except RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                if attempt >= self.max_retries:
                    logger.error("LLM call failed after %s attempts", self.max_retries)
                    raise
                backoff = self.backoff_base**attempt
                logger.warning(
                    "LLM failed (%s/%s): %s: %s",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    str(e)[:200],
                )
                await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    async def _generate_json_output(self, updater: TaskUpdater | None = None) -> str:
        if not self.api_key and self._acompletion_fn is None:
            raise RuntimeError("OPENAI_API_KEY is not set")

        reasoning_messages: list[dict[str, Any]] = [
            {"role": "system", "content": reasoning_system_for_domain(self._domain)},
            *self._messages[1:],
        ]
        reasoning_resp = await self._call_llm_with_retry(
            self._llm_messages_with_rules_reminder(reasoning_messages),
            response_format=None,
        )
        reasoning_text = (reasoning_resp.choices[0].message.content or "").strip()
        if settings.agent_debug_logging():
            logger.info("LLM reasoning (truncated): %s...", reasoning_text[:500])

        self._messages.append({"role": "assistant", "content": reasoning_text})

        if updater is not None:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Generating response…"),
            )

        response = await self._call_llm_with_retry(
            self._llm_messages_with_rules_reminder(self._messages),
            response_format=TAU2_JSON_RESPONSE_FORMAT,
        )
        raw = (response.choices[0].message.content or "").strip()

        if settings.agent_debug_logging():
            logger.info("LLM raw (truncated): %s...", raw[:500])

        normalized = normalize_model_json_text(raw)
        if parse_tau2_json(normalized):
            return normalized

        self._messages.append({"role": "assistant", "content": raw})
        self._messages.append({"role": "user", "content": JSON_REPAIR_USER})

        try:
            response2 = await self._call_llm_with_retry(
                self._llm_messages_with_rules_reminder(self._messages),
                response_format=TAU2_JSON_RESPONSE_FORMAT,
            )
        except Exception:
            self._messages.pop()
            self._messages.pop()
            raise

        raw2 = (response2.choices[0].message.content or "").strip()
        normalized2 = normalize_model_json_text(raw2)

        if settings.agent_debug_logging():
            logger.info("LLM repair raw (truncated): %s...", raw2[:500])

        if parse_tau2_json(normalized2):
            return normalized2

        print("Invalid JSON after repair; using fallback respond message")
        print('normalized2', normalized2)
        logger.warning("Invalid JSON after repair; using fallback respond message")
        return FALLBACK_RESPOND_JSON

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reasoning…"),
        )

        self._messages.append({"role": "user", "content": input_text})

        try:
            assistant_content = await self._generate_json_output(updater)
        except Exception as e:
            print("LLM call failed: %s: %s", type(e).__name__, e)
            print("Full traceback:")
            print(e)
            logger.error("LLM call failed: %s: %s", type(e).__name__, e)
            if settings.agent_debug_logging():
                logger.exception("Full traceback:")
            assistant_content = FALLBACK_RESPOND_JSON

        self._messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=assistant_content))],
            name="Response",
        )
