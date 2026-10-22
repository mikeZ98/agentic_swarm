"""The MCP boundary: the real protocol, in-process, against a fake sandbox.

`mcp.Client(server)` connects to an `MCPServer` instance without a subprocess or socket,
so these tests exercise tool listing, argument validation, structured results and error
propagation exactly as the agents see them — still without Docker.
"""

from __future__ import annotations

import os
import sys

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from src.config import PROJECT_ROOT, McpSettings
from src.sandbox import McpCodeTools, SandboxError, open_code_tools
from src.sandbox.client import server_environment, server_target
from src.sandbox.server import SERVER_NAME, build_server
from src.state import ExecutionResult, StaticAnalysisReport
from tests.fakes import FakeCodeTools, dirty_report, execution

MAX_CODE = 64


class BrokenSandbox:
    async def execute_python(self, code: str) -> ExecutionResult:
        raise SandboxError("Cannot connect to the Docker daemon")

    async def static_analysis(self, code: str) -> StaticAnalysisReport:
        raise SandboxError("Cannot connect to the Docker daemon")


def _server(tools: object | None = None) -> MCPServer:
    return build_server(tools or FakeCodeTools(), max_code_chars=MAX_CODE)  # type: ignore[arg-type]


async def test_server_lists_both_tools_as_read_only_and_closed_world() -> None:
    async with Client(_server()) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(listed) == {"execute_python", "static_analysis"}
    for tool in listed.values():
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False
        assert tool.input_schema["properties"]["code"]["maxLength"] == MAX_CODE
        assert tool.output_schema is not None
        assert tool.output_schema["additionalProperties"] is False


async def test_tool_results_round_trip_as_strict_contracts() -> None:
    fake = FakeCodeTools(executions=[execution(1, stderr="boom")], reports=[dirty_report()])

    async with open_code_tools(McpSettings(), client=Client(_server(fake))) as tools:
        result = await tools.execute_python("raise SystemExit(1)")
        report = await tools.static_analysis("import subprocess")

    assert result == execution(1, stderr="boom")
    assert report == dirty_report()
    assert fake.executed == ["raise SystemExit(1)"]
    assert fake.analysed == ["import subprocess"]


async def test_sandbox_failure_surfaces_as_sandbox_error_with_its_reason() -> None:
    async with open_code_tools(McpSettings(), client=Client(_server(BrokenSandbox()))) as tools:
        with pytest.raises(SandboxError, match="Docker daemon"):
            await tools.execute_python("x = 1")
        with pytest.raises(SandboxError, match="Docker daemon"):
            await tools.static_analysis("x = 1")


@pytest.mark.parametrize("code", ["", "x" * (MAX_CODE + 1)])
async def test_out_of_bounds_code_is_rejected_by_the_server(code: str) -> None:
    fake = FakeCodeTools()

    async with open_code_tools(McpSettings(), client=Client(_server(fake))) as tools:
        with pytest.raises(SandboxError, match="execute_python"):
            await tools.execute_python(code)

    assert fake.executed == []


async def test_a_malformed_tool_response_is_rejected() -> None:
    rogue = MCPServer("rogue")

    @rogue.tool()
    async def execute_python(code: str) -> dict[str, object]:
        return {"exit_code": "0", "stdout": "", "stderr": "", "timed_out": False}

    async with open_code_tools(McpSettings(), client=Client(rogue)) as tools:
        with pytest.raises(SandboxError, match="malformed ExecutionResult"):
            await tools.execute_python("x = 1")


async def test_an_unstructured_tool_response_is_rejected() -> None:
    rogue = MCPServer("rogue")

    @rogue.tool(structured_output=False)
    async def static_analysis(code: str) -> str:
        return "looks fine to me"

    async with open_code_tools(McpSettings(), client=Client(rogue)) as tools:
        with pytest.raises(SandboxError, match="no structured content"):
            await tools.static_analysis("x = 1")


async def test_an_unknown_tool_is_a_sandbox_error() -> None:
    async with open_code_tools(McpSettings(), client=Client(MCPServer("empty"))) as tools:
        with pytest.raises(SandboxError, match="execute_python"):
            await tools.execute_python("x = 1")


async def test_transport_failure_during_a_call_is_a_sandbox_error() -> None:
    class DeadClient:
        async def call_tool(self, name: str, arguments: dict[str, object]) -> None:
            raise ConnectionResetError("peer closed")

    tools = McpCodeTools(DeadClient())  # type: ignore[arg-type]

    with pytest.raises(SandboxError, match="peer closed"):
        await tools.execute_python("x = 1")


async def test_an_unreachable_server_fails_fast() -> None:
    unreachable = McpSettings(transport="streamable-http", url="http://127.0.0.1:9/mcp")

    with pytest.raises(SandboxError, match="cannot reach the sandbox MCP server"):
        async with open_code_tools(unreachable):
            pytest.fail("must not connect")  # pragma: no cover


# --------------------------------------------------------------------------- targets


def test_http_transport_targets_the_configured_url() -> None:
    settings = McpSettings(transport="streamable-http", url="http://sandbox-mcp:8765/mcp")

    assert server_target(settings) == "http://sandbox-mcp:8765/mcp"


def test_stdio_transport_spawns_the_project_server_module() -> None:
    target = server_target(McpSettings())

    assert isinstance(target, StdioServerParameters)
    assert target.command == sys.executable
    assert target.args == ["-m", "src.sandbox.server"]
    assert target.cwd == PROJECT_ROOT
    assert target.env is not None
    assert target.env.keys() == server_environment(os.environ).keys()


def test_the_server_process_never_inherits_provider_or_tracing_secrets() -> None:
    env = server_environment(
        {
            "PATH": "/usr/bin",
            "DOCKER_HOST": "tcp://dind:2375",
            "SWARM_SANDBOX__IMAGE": "sbx:1",
            "ANTHROPIC_API_KEY": "sk-ant",
            "OPENAI_API_KEY": "sk",
            "LANGFUSE_SECRET_KEY": "lf",
            "SWARM_CRITIC__MODEL": "gpt-4o",
        }
    )

    assert env == {
        "PATH": "/usr/bin",
        "DOCKER_HOST": "tcp://dind:2375",
        "SWARM_SANDBOX__IMAGE": "sbx:1",
    }


def test_server_name_is_stable() -> None:
    assert _server().name == SERVER_NAME


async def test_stdio_server_process_starts_and_serves_both_tools() -> None:
    """The production path: spawn `python -m src.sandbox.server` and handshake over stdio."""
    async with Client(server_target(McpSettings()), read_timeout_seconds=60) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}

    assert names == {"execute_python", "static_analysis"}
