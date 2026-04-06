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

SYSTEM_PROMPT = """You are a customer service agent. Domain policy, tool definitions, and scenario details appear in the user messages—follow those sources as the authority for what you may do or say.

Output contract (required):
- Reply with exactly one JSON object and nothing else. No markdown fences, no commentary before or after the JSON.
- The object must have keys "name" (string) and "arguments" (JSON object). This must be parseable by a standard JSON parser.

Tool versus respond:
- Use "name": "respond" with arguments.content only when sending a direct message to the user.
- For anything that requires database or environment state, call a tool by using its exact name from the tool list in the current user message. Use at most one tool per turn; wait for tool results in subsequent messages before your next action.

Using tool results:
- Lines like Tool '...' result: ... are ground truth. Do not contradict them or invent facts they do not support.

Arguments:
- Pass only parameters defined in the tool schema; use the types required (strings, numbers, booleans); include all required fields.

If policy in the messages requires escalation, refusal, or transfer, follow that policy. When unsure whether you have enough information, prefer reading state via a tool over guessing.

Keep user-facing respond text clear and concise unless the situation requires more detail."""

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
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.model = settings.litellm_model_for_openrouter(settings.get_openai_model())
        self.api_key = settings.get_openai_api_key()
        self.max_retries = settings.get_agent_llm_max_retries()
        self.backoff_base = settings.get_agent_llm_backoff_base()
        self.max_tokens = settings.get_max_completion_tokens()

    async def _litellm_acompletion(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_base": OPENROUTER_API_BASE,
            "temperature": 1.0,
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            # OpenRouter / many models ignore reasoning_effort; LiteLLM errors unless dropped.
            "drop_params": True,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return await acompletion(**kwargs)

    async def _call_llm_with_retry(self, messages: list[dict[str, Any]]) -> Any:
        # Snapshot list so callers (e.g. tests) do not see later appends to self._messages.
        msgs = list(messages)

        if self._acompletion_fn is not None:
            return await self._acompletion_fn(msgs)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._litellm_acompletion(msgs)
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

    async def _generate_json_output(self) -> str:
        if not self.api_key and self._acompletion_fn is None:
            raise RuntimeError("OPENAI_API_KEY is not set")

        response = await self._call_llm_with_retry(self._messages)
        raw = (response.choices[0].message.content or "").strip()

        if settings.agent_debug_logging():
            logger.info("LLM raw (truncated): %s...", raw[:500])

        normalized = normalize_model_json_text(raw)
        if parse_tau2_json(normalized):
            return normalized

        self._messages.append({"role": "assistant", "content": raw})
        self._messages.append({"role": "user", "content": JSON_REPAIR_USER})

        try:
            response2 = await self._call_llm_with_retry(self._messages)
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
            new_agent_text_message("Calling model…"),
        )

        self._messages.append({"role": "user", "content": input_text})

        try:
            assistant_content = await self._generate_json_output()
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
