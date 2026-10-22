"""The real sandbox on a real Docker daemon: isolation holds, analysers parse, MCP works.

Opt-in (`SWARM_DOCKER_TESTS=1`) because it needs a daemon and the sandbox image; the CI
`image` job builds the image and runs these. Locally:

    docker build -t agentic-swarm-sandbox:local docker/sandbox
    SWARM_DOCKER_TESTS=1 uv run pytest -m docker
"""

from __future__ import annotations

import asyncio
import os

import pytest

from src.config import McpSettings, SandboxSettings
from src.sandbox import open_code_tools
from src.sandbox.docker import DockerSandbox, run_process

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        os.environ.get("SWARM_DOCKER_TESTS") != "1",
        reason="needs a Docker daemon and the sandbox image (set SWARM_DOCKER_TESTS=1)",
    ),
]

SETTINGS = SandboxSettings(timeout_s=15.0)


@pytest.fixture
def sandbox() -> DockerSandbox:
    return DockerSandbox(SETTINGS)


async def test_a_snippet_runs_and_reports_its_output(sandbox: DockerSandbox) -> None:
    result = await sandbox.execute_python("print(sum(range(10)))")

    assert result.succeeded, result.stderr
    assert result.stdout.strip() == "45"


async def test_the_main_block_does_not_run(sandbox: DockerSandbox) -> None:
    code = "print('imported')\nif __name__ == '__main__':\n    print('served')\n"

    result = await sandbox.execute_python(code)

    assert result.stdout.split() == ["imported"]


async def test_the_container_has_no_network(sandbox: DockerSandbox) -> None:
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)\n"

    result = await sandbox.execute_python(code)

    assert result.exit_code == 1
    assert "OSError" in result.stderr or "unreachable" in result.stderr.lower()


async def test_the_container_is_unprivileged_and_read_only(sandbox: DockerSandbox) -> None:
    code = """\
import os
print(os.getuid(), os.getgid())
open("/tmp/scratch", "w").write("ok")
try:
    open("/usr/local/lib/escape", "w")
except OSError as exc:
    print("rootfs-blocked", type(exc).__name__)
with open("/proc/self/status") as status:
    print([line.split()[1] for line in status if line.startswith("CapEff")][0])
"""
    result = await sandbox.execute_python(code)

    assert result.succeeded, result.stderr
    uid_gid, blocked, capabilities = result.stdout.splitlines()
    assert uid_gid == "65534 65534"
    assert blocked.startswith("rootfs-blocked")
    assert int(capabilities, 16) == 0


async def test_a_snippet_cannot_impersonate_a_docker_failure(sandbox: DockerSandbox) -> None:
    result = await sandbox.execute_python("import sys\nsys.exit(125)\n")

    assert result.exit_code == 1


async def test_an_endless_snippet_is_killed_and_its_container_removed() -> None:
    sandbox = DockerSandbox(SandboxSettings(timeout_s=3.0))

    result = await sandbox.execute_python("while True:\n    pass\n")

    assert result.timed_out
    await asyncio.sleep(2)
    leftover = await run_process(
        ["docker", "ps", "-aq", "--filter", "label=agentic_swarm.sandbox=1"],
        stdin=b"",
        timeout_s=30,
        output_limit=4096,
    )
    assert leftover.returncode == 0, leftover.stderr
    assert leftover.stdout.strip() == ""


async def test_ruff_and_bandit_flag_dangerous_code(sandbox: DockerSandbox) -> None:
    code = (
        "import subprocess\n\n"
        "def run(cmd):\n"
        "    subprocess.call(cmd, shell=True)\n"
        "    return missing\n"
    )

    report = await sandbox.static_analysis(code)

    rules = {(f.tool, f.rule) for f in report.findings}
    assert not report.passed
    assert ("ruff", "F821") in rules
    assert ("bandit", "B602") in rules


async def test_clean_code_passes_static_analysis(sandbox: DockerSandbox) -> None:
    report = await sandbox.static_analysis("def add(a: int, b: int) -> int:\n    return a + b\n")

    assert report.passed, report.findings


async def test_the_full_mcp_path_runs_code_in_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agents' view: stdio MCP server process -> DockerSandbox -> container."""
    monkeypatch.setenv("SWARM_SANDBOX__IMAGE", SETTINGS.image)

    async with open_code_tools(McpSettings()) as tools:
        result = await tools.execute_python("import fastapi\nprint(fastapi.__name__)")
        report = await tools.static_analysis("x = 1\n")

    assert result.succeeded, result.stderr
    assert result.stdout.strip() == "fastapi"
    assert report.passed
