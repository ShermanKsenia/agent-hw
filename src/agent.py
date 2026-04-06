from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

import settings

SYSTEM_PROMPT = (
    "You are a customer-service agent. Follow the policy and tool instructions in "
    "the user messages. Reply with a single JSON object only (no markdown): "
    '{"name": "<function_name_or_respond>", "arguments": { ... }} matching the '
    "format described in the conversation."
)


class Agent:
    """Tau2 purple baseline: OpenAI chat with JSON output, history per A2A context."""

    def __init__(self, client: AsyncOpenAI | None = None):
        self._client_override = client
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.model = settings.get_openai_model()
        self.base_url = "https://openrouter.ai/api/v1"

    def _client(self) -> AsyncOpenAI:
        if self._client_override is not None:
            return self._client_override
        api_key = settings.get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        kwargs: dict[str, str] = {"api_key": api_key}
        kwargs["base_url"] = self.base_url 
        return AsyncOpenAI(**kwargs)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Calling model…"),
        )

        self._messages.append({"role": "user", "content": input_text})

        client = self._client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0].message
        content = (choice.content or "").strip()
        self._messages.append({"role": "assistant", "content": content})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=content))],
            name="Response",
        )
