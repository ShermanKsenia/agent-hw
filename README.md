# A2A Agent Template (Tau2 purple baseline)

A minimal template for building [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) agents. This repository includes a **Tau2 / AgentBeats purple baseline**: an OpenAI chat model with **JSON** tool-or-respond replies, matching the green agent’s `RemoteA2AAgent` protocol (see [tau2-agentbeats](https://github.com/RDI-Foundation/tau2-agentbeats)).

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Tau2 purple baseline (OpenAI + JSON)
├─ settings.py    # OPENAI_* environment helpers
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # Agent tests
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
amber-manifest.json5  # Amber manifest
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Getting Started

1. **Create your repository** - Click "Use this template" to create your own repository from this template

2. **Implement your agent** - Add your agent logic to [`src/agent.py`](src/agent.py)

3. **Configure your agent card** - Fill in your agent's metadata (name, skills, description) in [`src/server.py`](src/server.py)

4. **Fill out your [Amber](https://github.com/RDI-Foundation/amber) manifest** - Update [`amber-manifest.json5`](amber-manifest.json5) to use your agent in Amber scenarios

5. **Write your tests** - Add custom tests for your agent in [`tests/test_agent.py`](tests/test_agent.py)

For a concrete example of implementing an agent using this template, see this [draft PR](https://github.com/RDI-Foundation/agent-template/pull/8).

## Configuration (OpenAI)

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENT_RULES_REMINDER_MESSAGE_THRESHOLD` | no | when stored message count reaches this value, append a short rules recap to reasoning/JSON/repair LLM calls only (default `12`; `0` disables) |
| `AGENT_LLM` | yes | agent model name |
| `OPENAI_API_KEY` | no | api key to openai |
| `GEMINI_API_KEY` | no | api key to gemini |
| `DEEPSEEK_API_KEY` | no | api key to deepseek |

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server (default http://127.0.0.1:9009)
export OPENAI_API_KEY=sk-...
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

## Testing

```bash
uv sync --extra test

# Unit tests (mocked OpenAI; no agent process or API key)
uv run pytest tests/test_agent.py -k tau2_agent -v
```

Run A2A conformance tests against a **running** agent:

```bash
export OPENAI_API_KEY=sk-...
uv run src/server.py
# other terminal:
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
