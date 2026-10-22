"""The agents' MCP client: `CodeTools` over an external sandbox server.

Every tool response is re-validated as a strict contract, and every failure (unreachable
server, tool error, malformed payload) surfaces as `SandboxError`, so a broken sandbox can
never be mistaken for clean code.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from src.config import PROJECT_ROOT, McpSettings
from src.sandbox.base import SandboxError
from src.state import Contract, ExecutionResult, StaticAnalysisReport

#: The only variables the stdio server inherits: provider keys and tracing secrets stay here.
_FORWARDED_ENV_PREFIXES = ("DOCKER_", "SWARM_SANDBOX__", "LC_")
_FORWARDED_ENV_KEYS = frozenset({"PATH", "HOME", "LANG", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV"})


def server_environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in environ.items()
        if key in _FORWARDED_ENV_KEYS or key.startswith(_FORWARDED_ENV_PREFIXES)
    }


def server_target(settings: McpSettings) -> StdioServerParameters | str:
    """A URL for a running server, or the command that spawns one over stdio."""
    if settings.transport == "streamable-http":
        return settings.url
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.sandbox.server"],
        env=server_environment(os.environ),
        cwd=PROJECT_ROOT,
    )


def _error_text(result: CallToolResult) -> str:
    text = " ".join(block.text for block in result.content if isinstance(block, TextContent))
    return text.strip()[:500] or "no detail"


class McpCodeTools:
    """`CodeTools` backed by an MCP `Client` session."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def _call[M: Contract](self, tool: str, code: str, contract: type[M]) -> M:
        try:
            result = await self._client.call_tool(tool, {"code": code})
        except Exception as exc:  # transport / protocol failure of any kind
            raise SandboxError(f"MCP call '{tool}' failed: {exc}") from exc
        if result.is_error:
            raise SandboxError(f"MCP tool '{tool}' failed: {_error_text(result)}")
        if result.structured_content is None:
            raise SandboxError(f"MCP tool '{tool}' returned no structured content")
        try:
            return contract.model_validate_json(json.dumps(result.structured_content))
        except ValidationError as exc:
            raise SandboxError(
                f"MCP tool '{tool}' returned a malformed {contract.__name__}"
            ) from exc

    async def execute_python(self, code: str) -> ExecutionResult:
        return await self._call("execute_python", code, ExecutionResult)

    async def static_analysis(self, code: str) -> StaticAnalysisReport:
        return await self._call("static_analysis", code, StaticAnalysisReport)


@asynccontextmanager
async def open_code_tools(
    settings: McpSettings, *, client: Client | None = None
) -> AsyncIterator[McpCodeTools]:
    """Connect to the sandbox server for the duration of a run.

    `client` lets tests connect in-process; production builds one from `settings`.
    """
    session = client or Client(
        server_target(settings), read_timeout_seconds=settings.call_timeout_s
    )
    async with AsyncExitStack() as stack:
        try:
            await stack.enter_async_context(session)
        except Exception as exc:  # spawn failure, refused connection, failed handshake
            raise SandboxError(f"cannot reach the sandbox MCP server: {exc}") from exc
        yield McpCodeTools(session)
