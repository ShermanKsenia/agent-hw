import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

from agent import Agent

# Non-streaming A2A waits for the full response (LLM + task completion). 10s often times out in CI.
_A2A_HTTP_TIMEOUT = float(os.environ.get("A2A_TEST_HTTP_TIMEOUT", "60"))

# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=_A2A_HTTP_TIMEOUT) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)

@pytest.mark.asyncio
async def test_tau2_agent_openai_json_artifact():
    json_out = '{"name": "respond", "arguments": {"content": "ok"}}'
    reasoning_text = "The user greeted me; I should respond politely."

    def make_response(content: str):
        msg_obj = MagicMock()
        msg_obj.message.content = content
        resp = MagicMock()
        resp.choices = [msg_obj]
        return resp

    mock_completion = AsyncMock(
        side_effect=[make_response(reasoning_text), make_response(json_out)]
    )

    agent = Agent(acompletion_fn=mock_completion)
    updater = MagicMock()
    updater.update_status = AsyncMock()
    updater.add_artifact = AsyncMock()

    user_msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(text="Hello"))],
        message_id="m1",
        context_id=None,
    )

    await agent.run(user_msg, updater)

    assert mock_completion.await_count == 2
    first_msgs = mock_completion.await_args_list[0].args[0]
    assert len(first_msgs) == 2
    assert first_msgs[0]["role"] == "system"
    assert "internal analysis" in first_msgs[0]["content"].lower()
    assert first_msgs[1]["content"] == "Hello"
    second_msgs = mock_completion.await_args_list[1].args[0]
    assert second_msgs[0]["role"] == "system"
    assert second_msgs[1]["content"] == "Hello"
    assert second_msgs[2]["role"] == "assistant"
    assert second_msgs[2]["content"] == reasoning_text
    assert len(agent._messages) == 4

    updater.add_artifact.assert_awaited_once()
    call_kw = updater.add_artifact.await_args.kwargs
    assert call_kw["name"] == "Response"
    text = call_kw["parts"][0].root.text
    assert text == json_out


@pytest.mark.asyncio
async def test_tau2_agent_accumulates_history_across_turns():
    def make_response(content: str):
        msg_obj = MagicMock()
        msg_obj.message.content = content
        resp = MagicMock()
        resp.choices = [msg_obj]
        return resp

    mock_completion = AsyncMock(
        side_effect=[
            make_response("Reasoning for first turn."),
            make_response('{"name": "respond", "arguments": {"content": "a"}}'),
            make_response("Reasoning for second turn."),
            make_response('{"name": "respond", "arguments": {"content": "b"}}'),
        ]
    )

    agent = Agent(acompletion_fn=mock_completion)
    updater = MagicMock()
    updater.update_status = AsyncMock()
    updater.add_artifact = AsyncMock()

    await agent.run(
        Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text="First"))],
            message_id="m1",
            context_id=None,
        ),
        updater,
    )
    await agent.run(
        Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text="Second"))],
            message_id="m2",
            context_id=None,
        ),
        updater,
    )

    assert mock_completion.await_count == 4
    second_turn_json_msgs = mock_completion.await_args_list[3].args[0]
    assert any("First" in m.get("content", "") for m in second_turn_json_msgs)
    assert any(
        '{"name": "respond", "arguments": {"content": "a"}}' == m.get("content", "")
        for m in second_turn_json_msgs
    )


@pytest.mark.asyncio
async def test_ephemeral_rules_reminder_appended_when_threshold_met(monkeypatch):
    """Reminder is added to LLM payloads only, not persisted in _messages."""
    monkeypatch.setenv("AGENT_RULES_REMINDER_MESSAGE_THRESHOLD", "2")

    def make_response(content: str):
        msg_obj = MagicMock()
        msg_obj.message.content = content
        resp = MagicMock()
        resp.choices = [msg_obj]
        return resp

    json_out = '{"name": "respond", "arguments": {"content": "ok"}}'
    reasoning_text = "Thinking."
    mock_completion = AsyncMock(
        side_effect=[make_response(reasoning_text), make_response(json_out)]
    )

    agent = Agent(acompletion_fn=mock_completion)
    updater = MagicMock()
    updater.update_status = AsyncMock()
    updater.add_artifact = AsyncMock()

    await agent.run(
        Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text="Hi"))],
            message_id="m1",
            context_id=None,
        ),
        updater,
    )

    first_msgs = mock_completion.await_args_list[0].args[0]
    assert first_msgs[-1]["role"] == "user"
    assert "[Rules reminder]" in first_msgs[-1]["content"]

    second_msgs = mock_completion.await_args_list[1].args[0]
    assert second_msgs[-1]["role"] == "user"
    assert "[Rules reminder]" in second_msgs[-1]["content"]

    assert not any(
        isinstance(m.get("content"), str) and "[Rules reminder]" in m["content"]
        for m in agent._messages
    )


@pytest.mark.asyncio
async def test_ephemeral_rules_reminder_on_json_repair_call(monkeypatch):
    monkeypatch.setenv("AGENT_RULES_REMINDER_MESSAGE_THRESHOLD", "2")

    def make_response(content: str):
        msg_obj = MagicMock()
        msg_obj.message.content = content
        resp = MagicMock()
        resp.choices = [msg_obj]
        return resp

    fixed = '{"name": "respond", "arguments": {"content": "fixed"}}'
    mock_completion = AsyncMock(
        side_effect=[
            make_response("Reasoning."),
            make_response("not json"),
            make_response(fixed),
        ]
    )

    agent = Agent(acompletion_fn=mock_completion)
    updater = MagicMock()
    updater.update_status = AsyncMock()
    updater.add_artifact = AsyncMock()

    await agent.run(
        Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text="Hi"))],
            message_id="m1",
            context_id=None,
        ),
        updater,
    )

    assert mock_completion.await_count == 3
    repair_msgs = mock_completion.await_args_list[2].args[0]
    assert repair_msgs[-1]["role"] == "user"
    assert "[Rules reminder]" in repair_msgs[-1]["content"]
