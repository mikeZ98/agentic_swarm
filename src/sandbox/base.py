"""The code-tool surface the agents depend on, independent of transport."""

from __future__ import annotations

from typing import Protocol

from src.state import ExecutionResult, StaticAnalysisReport


class SandboxError(RuntimeError):
    """The sandbox could not run the request (infrastructure, not a defect in the code).

    Raised for an unreachable MCP server, a missing Docker daemon or a malformed tool
    response. It aborts the run: unverified code is never accepted.
    """


class CodeTools(Protocol):
    """Runs untrusted, generated code. Implemented by `DockerSandbox` (inside the MCP server)
    and by `McpCodeTools` (the agents' client of that server)."""

    async def execute_python(self, code: str) -> ExecutionResult:
        """Execute `code` as a module named `__sandbox__` (its `__main__` block does not run)."""
        ...

    async def static_analysis(self, code: str) -> StaticAnalysisReport:
        """Lint (ruff) and security-scan (bandit) `code` without executing it."""
        ...
