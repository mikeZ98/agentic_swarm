"""CLI and server entrypoints: wiring, artefacts and exit codes — no LLM, Docker or network."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from mcp.server.mcpserver import MCPServer
from pydantic import SecretStr

import src.main as cli
import src.sandbox.server as sandbox_server
from src.agents import FrontendAgent, SwarmAgents
from src.config import McpSettings, Settings, get_settings
from src.graph import build_topology_graph
from src.sandbox import CodeTools, SandboxError
from src.state import GuardrailReport, Status, SwarmState
from tests.fakes import FakeCodeTools, clean_report, execution
from tests.test_critic_guardrail import _critic, _verdict
from tests.test_graph import SPEC, Recorder, verdict


def _settings() -> Settings:
    return Settings(_env_file=None, LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[McpSettings]:
    """Route `run()` to scripted agents and an in-memory sandbox; records the MCP settings."""
    opened: list[McpSettings] = []

    @asynccontextmanager
    async def fake_tools(settings: McpSettings) -> AsyncIterator[CodeTools]:
        opened.append(settings)
        yield FakeCodeTools()

    def fake_agents(settings: Settings, tools: CodeTools) -> SwarmAgents:
        assert isinstance(tools, FakeCodeTools)
        return Recorder([verdict(accepted=True)]).agents()

    monkeypatch.setattr(cli, "open_code_tools", fake_tools)
    monkeypatch.setattr(cli, "build_agents", fake_agents)
    monkeypatch.setattr(cli, "get_settings", _settings)
    return opened


async def test_run_executes_the_graph_inside_the_sandbox_session(
    offline: list[McpSettings],
) -> None:
    state, elapsed = await cli.run("build it", _settings(), session_id="s-1")

    assert state.status == Status.ACCEPTED
    assert elapsed >= 0
    assert offline == [McpSettings()]


def test_main_writes_every_artefact_and_exits_zero_on_acceptance(
    offline: list[McpSettings], tmp_path: Path
) -> None:
    code = cli.main(["--prompt", "build it", "--output-dir", str(tmp_path)])

    assert code == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "spec.json",
        "backend.py",
        "index.html",
        "eval_report.json",
    }
    assert json.loads((tmp_path / "spec.json").read_text())["title"] == "URL shortener"
    assert (tmp_path / "backend.py").read_text() == "app = 1\n"
    assert json.loads((tmp_path / "eval_report.json").read_text())["is_accepted"] is True


def test_main_exits_one_when_the_swarm_does_not_converge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = SwarmState(user_prompt="x", status=Status.MAX_ITERATIONS_REACHED, iteration=3)

    async def unconverged(*_: Any, **__: Any) -> tuple[SwarmState, float]:
        return final, 1.0

    monkeypatch.setattr(cli, "run", unconverged)
    monkeypatch.setattr(cli, "get_settings", _settings)

    assert cli.main(["--output-dir", str(tmp_path), "-v"]) == 1
    # Nothing to record without a spec or a report — only the code files are written.
    assert {p.name for p in tmp_path.iterdir()} == {"backend.py", "index.html"}


@pytest.mark.parametrize(
    "error",
    [SandboxError("cannot reach the sandbox MCP server"), RuntimeError("OPENAI_API_KEY missing")],
)
def test_main_exits_two_on_configuration_or_sandbox_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    error: RuntimeError,
) -> None:
    async def failing(*_: Any, **__: Any) -> tuple[SwarmState, float]:
        raise error

    monkeypatch.setattr(cli, "run", failing)
    monkeypatch.setattr(cli, "get_settings", _settings)

    assert cli.main(["--output-dir", str(tmp_path / "out")]) == 2
    assert str(error) in caplog.text
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- sandbox server


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(self: MCPServer, transport: str = "stdio", **kwargs: Any) -> None:
        calls.append((transport, kwargs))

    monkeypatch.setattr(MCPServer, "run", fake_run)
    return calls


def test_server_defaults_to_stdio(served: list[tuple[str, dict[str, Any]]]) -> None:
    assert sandbox_server.main([]) == 0
    assert served == [("stdio", {})]


def test_server_over_http_enables_the_dns_rebinding_guard(
    served: list[tuple[str, dict[str, Any]]],
) -> None:
    args = ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9000"]  # noqa: S104

    assert sandbox_server.main([*args, "--allowed-host", "sandbox-mcp:*"]) == 0

    ((transport, kwargs),) = served
    security = kwargs["transport_security"]
    assert transport == "streamable-http"
    assert (kwargs["host"], kwargs["port"]) == ("0.0.0.0", 9000)  # noqa: S104
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["sandbox-mcp:*", "127.0.0.1:*", "localhost:*"]
    assert security.allowed_origins == []


def test_server_settings_never_load_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-load")
    monkeypatch.setenv("SWARM_SANDBOX__TIMEOUT_S", "7")

    settings = sandbox_server.ServerSettings(_env_file=None)

    assert settings.sandbox.timeout_s == 7.0
    assert "sk-ant-should-not-load" not in repr(settings.model_dump())


# --------------------------------------------------------------------------- small seams


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_settings_keep_provider_keys_secret() -> None:
    settings = Settings(_env_file=None, OPENAI_API_KEY=SecretStr("sk-very-secret"))

    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in settings.model_dump_json()


async def test_topology_graph_nodes_refuse_to_execute() -> None:
    with pytest.raises(RuntimeError, match="topology-only graph"):
        await build_topology_graph().ainvoke(SwarmState(user_prompt="x"))


async def test_frontend_revision_prompt_carries_its_previous_code_and_feedback() -> None:
    prompts: list[str] = []

    def llm(messages: Any) -> AIMessage:
        prompts.append(str(messages[-1].content))
        return AIMessage(content="```html\n<main>v2</main>\n```")

    report = _critic().decide(
        SwarmState(user_prompt="x"),
        _verdict(accepted=False, score=30, owner="frontend"),
        GuardrailReport(static_analysis=clean_report(), execution=execution()),
    )["eval_report"]
    state = SwarmState(
        user_prompt="x",
        architecture_spec=SPEC,
        backend_code="app = 1",
        frontend_code="<main>v1</main>",
        eval_report=report,
        iteration=1,
        status=Status.REVISING,
    )

    update = await FrontendAgent(RunnableLambda(llm))(state)

    assert update["frontend_code"] == "<main>v2</main>"
    assert "Revision #1" in prompts[0]
    assert "<main>v1</main>" in prompts[0]
    assert "XSS: stats table uses innerHTML" in prompts[0]
