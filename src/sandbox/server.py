"""MCP server exposing the Docker sandbox as two tools: `execute_python`, `static_analysis`.

    python -m src.sandbox.server                               # stdio (spawned by the agents)
    python -m src.sandbox.server --transport streamable-http \\
        --host 0.0.0.0 --port 8765 --allowed-host sandbox-mcp:*  # compose service

The agents never execute generated code themselves: they call these tools, and this
process is the only one that talks to Docker.
"""

# No `from __future__ import annotations`: MCP derives each tool's input schema from its
# signature, and the `Code` constraint is built from runtime config inside `build_server`.
import argparse
import logging
import sys
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import SandboxSettings
from src.sandbox.base import CodeTools, SandboxError
from src.sandbox.docker import DockerSandbox
from src.state import ExecutionResult, StaticAnalysisReport

SERVER_NAME = "agentic-swarm-sandbox"

log = logging.getLogger("sandbox.server")


class ServerSettings(BaseSettings):
    """Only the sandbox section of the config: provider keys never enter this process."""

    model_config = SettingsConfigDict(
        env_prefix="SWARM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sandbox: SandboxSettings = SandboxSettings()


def build_server(tools: CodeTools, *, max_code_chars: int) -> MCPServer:
    server = MCPServer(
        SERVER_NAME,
        instructions="Runs untrusted Python in a network-less, read-only, non-root container.",
    )
    sandboxed = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
    code_field = Field(min_length=1, max_length=max_code_chars, description="Python source")

    @server.tool(
        description=(
            "Execute a Python module inside the sandbox container. It runs under the module "
            "name `__sandbox__`, so an `if __name__ == '__main__':` block does not run."
        ),
        annotations=sandboxed,
    )
    async def execute_python(code: Annotated[str, code_field]) -> ExecutionResult:
        try:
            return await tools.execute_python(code)
        except SandboxError as exc:
            raise ToolError(f"sandbox unavailable: {exc}") from exc

    @server.tool(
        description="Lint (ruff) and security-scan (bandit) Python source in the sandbox.",
        annotations=sandboxed,
    )
    async def static_analysis(code: Annotated[str, code_field]) -> StaticAnalysisReport:
        try:
            return await tools.static_analysis(code)
        except SandboxError as exc:
            raise ToolError(f"sandbox unavailable: {exc}") from exc

    return server


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sandbox-server", description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Host header accepted over HTTP (DNS-rebinding guard), e.g. 'sandbox-mcp:*'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # stdout carries the stdio protocol: logs must go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    sandbox = ServerSettings().sandbox
    server = build_server(DockerSandbox(sandbox), max_code_chars=sandbox.max_code_chars)
    if args.transport == "stdio":
        server.run("stdio")
        return 0
    allowed = [*args.allowed_host, "127.0.0.1:*", "localhost:*"]
    log.info("serving MCP on %s:%d (allowed hosts: %s)", args.host, args.port, allowed)
    server.run(
        "streamable-http",
        host=args.host,
        port=args.port,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=allowed, allowed_origins=[]
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
